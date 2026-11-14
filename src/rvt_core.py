"""Small, dependency-light reference kernels for the RVT protocol.

The production integration lives in ``lavis.models.blip2_models.blip2_opt``.
These functions make the scoring and structural decisions easy to inspect and
unit-test without loading a multi-billion-parameter checkpoint.
"""

from __future__ import annotations

from typing import Iterable, Sequence
import math

import torch


def attention_entropy_scores(attn: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute RVT visual-token scores from cross-attention probabilities.

    The query weighting matches the ``attn_entropy`` path in the released
    ``blip2_opt.py`` implementation when optional head weights are disabled.

    Args:
        attn: Cross-attention probabilities with shape [B, H, Q, N].
    Returns:
        Tensor of visual-token scores with shape [B, N].
    """
    if attn.ndim != 4:
        raise ValueError(f"expected [B,H,Q,N], got {tuple(attn.shape)}")
    p = attn.clamp_min(eps)
    p = p / (p.sum(dim=-1, keepdim=True) + eps)
    query_dist = p.mean(dim=1)  # [B, Q, N]
    entropy = -(query_dist * query_dist.clamp_min(eps).log()).sum(dim=-1)
    weights = torch.softmax(-entropy, dim=-1)
    q = query_dist.shape[1]
    if q > 0:
        weights = weights.clamp(min=0.5 / q, max=1.5 / q)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(eps)
    return (weights.unsqueeze(-1) * query_dist).sum(dim=1)


def residual_merge(tokens: torch.Tensor, keep_indices: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Merge the normalized score-weighted residual into the last kept token.

    Mirrors the ``merge`` tail strategy in the production OPT model's
    visual-token filtering path. Output keeps the same [B, K, C] shape.
    """
    if tokens.ndim != 3 or keep_indices.ndim != 2:
        raise ValueError("tokens must be [B,N,C] and keep_indices must be [B,K]")
    b, n, c = tokens.shape
    if keep_indices.shape[0] != b or scores.shape != (b, n):
        raise ValueError("batch or score dimensions do not match")
    k = keep_indices.shape[1]
    selected = tokens.gather(1, keep_indices.unsqueeze(-1).expand(-1, -1, c)).clone()
    if k >= n:
        return selected
    dropped = torch.ones((b, n), device=tokens.device, dtype=torch.bool)
    dropped.scatter_(1, keep_indices, False)
    score_copy = scores.detach()
    for sample in range(b):
        residual_tokens = tokens[sample, dropped[sample]]
        t = residual_tokens.shape[0]
        if t == 0:
            continue
        mean = residual_tokens.mean(dim=0, keepdim=True)
        center = residual_tokens - mean
        var = center.square().mean(dim=0, keepdim=True)
        normed = center / (var + 1e-6).sqrt()
        residual_scores = score_copy[sample, dropped[sample]]
        weights = residual_scores.clamp_min(0)
        if float(weights.sum()) <= 0.0:
            weights = torch.ones_like(weights)
        weights = weights / weights.sum()
        merged = (weights.view(-1, 1) * normed).sum(dim=0)
        merged = merged * merged.new_tensor(math.log1p(float(t)))
        selected[sample, -1] = 0.5 * selected[sample, -1] + 0.5 * merged
    return selected


def select_visual_tokens(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    keep_ratio: float,
    tail_strategy: str = "merge",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top-scoring visual tokens in descending score order."""
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    if tokens.ndim != 3 or scores.shape != tokens.shape[:2]:
        raise ValueError("tokens must be [B,N,C] and scores must be [B,N]")
    k = max(1, round(tokens.shape[1] * keep_ratio))
    indices = scores.topk(k, dim=-1, largest=True, sorted=True).indices
    if tail_strategy == "merge" and k < tokens.shape[1]:
        return residual_merge(tokens, indices, scores), indices
    selected = tokens.gather(1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
    return selected, indices


def global_ffn_keep_mask(
    gate_logits: Sequence[torch.Tensor], keep_ratio: float, group_size: int = 16, min_groups: int = 1
) -> list[torch.Tensor]:
    """Select FFN channel groups globally while retaining a per-layer minimum."""
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    scores = [x.detach().sigmoid().flatten() for x in gate_logits]
    counts = [int(x.numel()) for x in scores]
    if any(c < min_groups for c in counts):
        raise ValueError("a layer has fewer groups than min_groups")
    total = sum(counts)
    target = max(sum(min_groups for _ in counts), round(total * keep_ratio))
    masks = [torch.zeros_like(x, dtype=torch.bool) for x in scores]
    for mask, score in zip(masks, scores):
        mask[score.topk(min_groups).indices] = True
    remaining = target - sum(int(x.sum()) for x in masks)
    candidates = [(float(score[i]), layer, i) for layer, score in enumerate(scores) for i in range(len(score)) if not masks[layer][i]]
    candidates.sort(reverse=True)
    for _, layer, i in candidates[: max(0, remaining)]:
        masks[layer][i] = True
    return masks


def grad_act_importance(head_outputs: torch.Tensor, head_grads: torch.Tensor) -> torch.Tensor:
    """Compute one Grad×Act importance score per attention head."""
    if head_outputs.shape != head_grads.shape or head_outputs.ndim < 2:
        raise ValueError("head outputs and gradients must have matching shapes")
    product = (head_outputs * head_grads).abs()
    reduce_dims = (0,) + tuple(range(2, product.ndim))
    return product.sum(dim=reduce_dims)


def intersection_and_fill(
    fold_scores: Sequence[torch.Tensor], keep_ratio: float, min_heads_per_layer: int = 1, layer_ids: Iterable[int] | None = None
) -> torch.Tensor:
    """Build a five-fold pruning mask by intersection followed by global fill."""
    if len(fold_scores) < 2:
        raise ValueError("intersection-and-fill expects multiple folds")
    stacked = torch.stack([x.detach().float() for x in fold_scores])
    n = stacked.shape[1]
    prune_count = n - round(n * keep_ratio)
    order = torch.argsort(stacked, dim=1)
    fold_candidates = torch.zeros_like(stacked, dtype=torch.bool)
    for f in range(stacked.shape[0]):
        fold_candidates[f, order[f, :prune_count]] = True
    selected = fold_candidates.all(dim=0)
    if layer_ids is None:
        layer_ids = [0] * n
    layer_ids = list(layer_ids)
    selected_count = int(selected.sum())
    for idx in torch.argsort(stacked.sum(dim=0)).tolist():
        if selected_count >= prune_count:
            break
        if not selected[idx]:
            selected[idx] = True
            selected_count += 1
    for layer in set(layer_ids):
        members = [i for i, lid in enumerate(layer_ids) if lid == layer]
        if sum(not bool(selected[i]) for i in members) < min_heads_per_layer:
            for i in sorted(members, key=lambda j: float(stacked[:, j].sum())):
                if selected[i]:
                    selected[i] = False
                    selected_count -= 1
                    break
    return selected
