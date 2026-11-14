#!/usr/bin/env python3
"""Check source syntax, configurations and RVT calculations."""
from pathlib import Path
import ast
import math
import sys

import torch
import yaml

root = Path(__file__).resolve().parents[1]
for file in sorted((root / "src").rglob("*.py")) + sorted((root / "scripts").glob("*.py")):
    ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
for file in sorted((root / "configs").glob("*.yaml")):
    cfg = yaml.safe_load(file.read_text(encoding="utf-8"))
    assert isinstance(cfg, dict) and "model" in cfg and "run" in cfg, str(file)

sys.path.insert(0, str(root / "src"))
from rvt_core import attention_entropy_scores, grad_act_importance, select_visual_tokens, residual_merge

torch.manual_seed(42)
attn = torch.rand(2, 4, 8, 16)
attn = attn / attn.sum(dim=-1, keepdim=True)
score = attention_entropy_scores(attn)
p = attn.clamp_min(1e-12)
p = p / (p.sum(-1, keepdim=True) + 1e-12)
mean_attention = p.mean(1)
entropy = -(mean_attention * mean_attention.clamp_min(1e-12).log()).sum(-1)
w = torch.softmax(-entropy, -1).clamp(0.5/8, 1.5/8)
w = w / w.sum(-1, keepdim=True)
expected = (w.unsqueeze(-1) * mean_attention).sum(1)
assert torch.allclose(score, expected, atol=1e-7)
tokens, idx = select_visual_tokens(torch.randn(2, 16, 32), score, 0.45)
assert tokens.shape == (2, 7, 32) and idx.shape == (2, 7)
g = grad_act_importance(torch.ones(2,12,64),torch.ones(2,12,64))
assert g.shape == (12,) and torch.allclose(g,torch.full((12,),128.0))
features = torch.tensor([[[0.,1.],[3.,2.],[7.,-2.],[8.,5.]]])
keep = torch.tensor([[0,3]])
scores = torch.tensor([[0.5,0.2,0.4,0.9]])
merged = residual_merge(features,keep,scores)
tail = features[0,1:3]
center = tail - tail.mean(0,keepdim=True)
normed = center / (center.square().mean(0,keepdim=True)+1e-6).sqrt()
expected_tail = ((normed[0]/3 + normed[1]*2/3)*math.log1p(2.0))
assert torch.allclose(merged[0,0],features[0,0])
assert torch.allclose(merged[0,1], 0.5*features[0,3]+0.5*expected_tail)
print("Python files, configurations and RVT calculations passed.")
