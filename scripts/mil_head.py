#!/usr/bin/env python3
"""Study-level head for 2.5D MRI models: zero-initialised window mixer + per-finding
attention-MIL pooling (SKILL Section 5). Requires torch.

Input: window features from a shared 2D backbone, shape (B, W, D), plus a mask
(B, W) for padded windows and optional integer tags per window (plane, slot,
relative-position bin, contrast class) that get zero-initialised embeddings.

    backbone = timm.create_model("convnext_tiny.in12k_ft_in1k", pretrained=True, num_classes=0)
    feats = backbone(windows.flatten(0, 1)).view(B, W, -1)       # windows: (B, W, 3, H, W)
    head = StudyHead(dim=feats.shape[-1], n_findings=12, tag_sizes={"plane": 3, "pos": 16})
    logits = head(feats, mask, tags={"plane": plane_ids, "pos": pos_bins})   # (B, 12)

Design choices from the write-ups:
  * new modules start as identity (zero-init output projection / embeddings), so training
    begins from plain attention pooling; one gold team reported collapse otherwise.
  * each finding has its own attention over windows (it picks its own slices).
  * tag dropout (p = 0.15) on contrast-type tags keeps the model from leaning on them.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ZeroInitMixer(nn.Module):
    """Transformer layers over windows whose contribution starts at exactly zero."""

    def __init__(self, dim: int, depth: int = 2, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.enc(x, src_key_padding_mask=~mask)
        return x + self.out(h)


class PerFindingAttentionPool(nn.Module):
    """score_l(k) = w_l . tanh(V h_k); softmax over valid windows; z_l = sum a_lk h_k;
    logit_l = u_l . z_l + c_l  (one attention and one classifier per finding)."""

    def __init__(self, dim: int, n_findings: int, hidden: int = 256, prior_logit: torch.Tensor | None = None):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.w = nn.Parameter(torch.randn(n_findings, hidden) * 0.02)
        self.u = nn.Parameter(torch.randn(n_findings, dim) * 0.02)
        self.c = nn.Parameter(torch.zeros(n_findings) if prior_logit is None else prior_logit.clone())

    def forward(self, h: torch.Tensor, mask: torch.Tensor, return_attention: bool = False):
        s = torch.einsum("bwk,fk->bfw", torch.tanh(self.V(h)), self.w)        # (B, F, W)
        s = s.masked_fill(~mask[:, None, :], float("-inf"))
        a = torch.softmax(s, dim=-1)
        z = torch.einsum("bfw,bwd->bfd", a, h)                                # (B, F, D)
        logits = (z * self.u).sum(-1) + self.c
        return (logits, a) if return_attention else logits


class StudyHead(nn.Module):
    def __init__(self, dim: int, n_findings: int, tag_sizes: dict[str, int] | None = None,
                 mixer_depth: int = 2, mixer_heads: int = 8, tag_dropout: dict[str, float] | None = None,
                 prior_logit: torch.Tensor | None = None):
        super().__init__()
        self.tags = nn.ModuleDict()
        for name, size in (tag_sizes or {}).items():
            emb = nn.Embedding(size + 1, dim)            # last index = "unknown / dropped"
            nn.init.zeros_(emb.weight)
            self.tags[name] = emb
        self.tag_dropout = tag_dropout or {}
        self.mixer = ZeroInitMixer(dim, mixer_depth, mixer_heads) if mixer_depth > 0 else None
        self.pool = PerFindingAttentionPool(dim, n_findings, prior_logit=prior_logit)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor, tags: dict[str, torch.Tensor] | None = None,
                return_attention: bool = False):
        h = feats
        for name, ids in (tags or {}).items():
            emb = self.tags[name]
            ids = ids.clone()
            p = self.tag_dropout.get(name, 0.0)
            if self.training and p > 0:
                ids[torch.rand_like(ids, dtype=torch.float) < p] = emb.num_embeddings - 1
            h = h + emb(ids)
        if self.mixer is not None:
            h = self.mixer(h, mask)
        return self.pool(h, mask, return_attention)


def prior_logits(train_targets: torch.Tensor) -> torch.Tensor:
    """Initialise the per-finding bias to logit(mean training label)."""
    p = train_targets.float().nanmean(0).clamp(1e-3, 1 - 1e-3)
    return torch.log(p / (1 - p))


def soft_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None,
             mask: torch.Tensor | None = None) -> torch.Tensor:
    """BCE on soft targets with per-cell weight and mask (never binarise the targets)."""
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target.clamp(0, 1), reduction="none")
    w = torch.ones_like(loss) if weight is None else weight
    if mask is not None:
        w = w * mask
    return (loss * w).sum() / w.sum().clamp_min(1.0)


if __name__ == "__main__":  # shape smoke test: python scripts/mil_head.py
    B, W, D, F = 2, 40, 64, 12
    head = StudyHead(D, F, tag_sizes={"plane": 3, "pos": 16}, mixer_heads=4, tag_dropout={"plane": 0.15})
    feats = torch.randn(B, W, D)
    mask = torch.ones(B, W, dtype=torch.bool); mask[1, 30:] = False
    tags = {"plane": torch.randint(0, 3, (B, W)), "pos": torch.randint(0, 16, (B, W))}
    head.eval()
    base = PerFindingAttentionPool(D, F); base.load_state_dict(head.pool.state_dict())
    out, att = head(feats, mask, tags, return_attention=True)
    same = torch.allclose(out, base(feats, mask), atol=1e-6)
    print("logits", tuple(out.shape), "| attention", tuple(att.shape),
          "| padded windows get zero attention:", bool(att[1, :, 30:].abs().max() == 0),
          "| identical to plain pooling at init:", bool(same))
    loss = soft_bce(out, torch.rand(B, F)); loss.backward(); print("loss", float(loss))
