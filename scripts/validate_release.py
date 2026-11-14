#!/usr/bin/env python3
"""Run release checks that do not require CUDA, COCO, or model weights."""

from __future__ import annotations

import ast
import pathlib
import re
import sys

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


def check_python() -> list[str]:
    errors: list[str] = []
    for path in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"{path}: {exc}")
    return errors


def check_yaml() -> list[str]:
    errors: list[str] = []
    for path in sorted((ROOT / "configs").glob("*.y*ml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or "model" not in data or "run" not in data:
                errors.append(f"{path}: expected model and run mappings")
        except yaml.YAMLError as exc:
            errors.append(f"{path}: {exc}")
    return errors


def check_paths_and_secrets() -> list[str]:
    errors: list[str] = []
    forbidden = re.compile(r"(?:sshpass|BEGIN (?:RSA|OPENSSH) PRIVATE KEY|password\s*=)", re.I)
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".png", ".pdf", ".zip"}:
            continue
        if path.resolve() == pathlib.Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if forbidden.search(text):
            errors.append(f"{path}: private path or credential-like text")
    for expected in [
        "README.md",
        "LICENSE",
        "src/rvt_core.py",
        "scripts/train.py",
        "scripts/evaluate.py",
        "configs/opt67b_cross_backbone.yaml",
        "configs/flant5xl_cross_backbone.yaml",
    ]:
        if not (ROOT / expected).exists():
            errors.append(f"missing required file: {expected}")
    return errors


def main() -> int:
    errors = check_python() + check_yaml() + check_paths_and_secrets()
    if errors:
        print("RELEASE_CHECK_FAILED")
        print("\n".join(errors))
        return 1
    sys.path.insert(0, str(ROOT / "src"))
    from rvt_core import attention_entropy_scores, grad_act_importance, residual_merge, select_visual_tokens
    import torch

    attn = torch.rand(2, 4, 8, 16)
    attn = attn / attn.sum(dim=-1, keepdim=True)
    scores = attention_entropy_scores(attn)
    selected, indices = select_visual_tokens(torch.randn(2, 16, 32), scores, 0.45, "merge")
    assert scores.shape == (2, 16) and selected.shape == (2, 7, 32) and indices.shape == (2, 7)
    # Verify entropy-weighted scoring against the production formula.
    p = attn.clamp_min(1e-12)
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-12)
    qdist = p.mean(dim=1)
    entropy = -(qdist * qdist.clamp_min(1e-12).log()).sum(dim=-1)
    weight = torch.softmax(-entropy, dim=-1).clamp(0.5 / 8, 1.5 / 8)
    weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    reference = (weight.unsqueeze(-1) * qdist).sum(dim=1)
    assert torch.allclose(scores, reference, atol=1e-7, rtol=1e-6)

    # Verify normalized, score-weighted residual merge and log1p scaling.
    inp = torch.tensor([[[0.0, 1.0], [3.0, 2.0],
                         [7.0, -2.0], [8.0, 5.0]]])
    keep = torch.tensor([[0, 3]])
    tok_score = torch.tensor([[0.5, 0.2, 0.4, 0.9]])
    got = residual_merge(inp, keep, tok_score)
    tail = inp[0, 1:3]
    normed = (tail - tail.mean(0, keepdim=True)) / \
             ((tail - tail.mean(0, keepdim=True)).square().mean(0, keepdim=True) + 1e-6).sqrt()
    expected_tail = (normed * torch.tensor([1/3, 2/3]).unsqueeze(-1)).sum(0)
    expected_tail = expected_tail * expected_tail.new_tensor(__import__('math').log1p(2.0))
    assert torch.allclose(got[0, 0], inp[0, 0])
    assert torch.allclose(got[0, 1], 0.5 * inp[0, 3] + 0.5 * expected_tail)
    grad_scores = grad_act_importance(torch.ones(2, 12, 64), torch.ones(2, 12, 64))
    assert grad_scores.shape == (12,) and torch.allclose(grad_scores, torch.full((12,), 128.0))
    print("RELEASE_CHECK_OK")
    print(f"python_files={len(list((ROOT / 'src').rglob('*.py')))}")
    print(f"config_files={len(list((ROOT / 'configs').glob('*.y*ml')))}")
    print("rvt_core_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
