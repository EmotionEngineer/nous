# nous/zoo_v2/common.py
"""
Shared utilities for zoo_v2 models.
"""
from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(x.clamp_min(eps))


@torch.no_grad()
def init_thresholds_from_quantiles(
    th_tensor_2d: torch.Tensor,
    X: np.ndarray,
    q_lo: float = 0.10,
    q_hi: float = 0.90,
    max_q: int = 9,
) -> None:
    """
    Initialize a 2D threshold tensor [R,D] from per-feature quantiles of X.
    For each feature j, compute min(R, max_q) quantiles then tile to length R.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be 2D [N,D]")
    R, D = th_tensor_2d.shape
    for j in range(D):
        qn = min(R, max_q)
        qs = np.linspace(q_lo, q_hi, num=qn, dtype=np.float32)
        vals = np.quantile(X[:, j], qs).astype(np.float32)
        tiled = np.resize(vals, R).astype(np.float32)
        th_tensor_2d[:, j].copy_(torch.tensor(tiled, device=th_tensor_2d.device))


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Sparsemax along `dim` (sum=1, sparse, non-negative).
    """
    z = logits - logits.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    z_cumsum = torch.cumsum(z_sorted, dim=dim)

    k = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype)
    view = [1] * z.dim()
    view[dim] = -1
    k = k.view(view)

    support = (1 + k * z_sorted) > z_cumsum
    k_z = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = (z_cumsum.gather(dim, k_z - 1) - 1) / k_z
    return torch.clamp(z - tau, min=0.0)


def straight_through_topk_mask(logits: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    """
    Straight-through estimator for hard top-k.
    Returns a tensor in [0,1] with hard forward pass and soft backward pass.
    """
    soft = torch.softmax(logits, dim=dim)
    topk_idx = torch.topk(logits, k=min(int(k), logits.shape[dim]), dim=dim).indices
    hard = torch.zeros_like(logits).scatter(dim, topk_idx, 1.0)
    return hard.detach() - soft.detach() + soft


def default_groups_for_D(D: int, G: int = 4) -> List[List[int]]:
    """
    Default grouping heuristic: if D==8 use a fixed pattern (CaliforniaHousing),
    else split into G roughly equal chunks.
    """
    if D == 8:
        return [[0], [1, 2, 3], [4, 5], [6, 7]]
    chunks = np.array_split(np.arange(D), int(G))
    return [list(c) for c in chunks]


def build_feature_to_fact_mask(D: int, n_thresh_per_feat: int) -> torch.Tensor:
    """
    Assumes ThresholdFactBank orders facts per feature contiguously,
    count = D * n_thresh_per_feat.  Augmented with (1-f), so Fin = 2*F0.
    Returns [D, Fin].
    """
    F0 = int(D) * int(n_thresh_per_feat)
    Fin = 2 * F0
    m = torch.zeros(int(D), int(Fin), dtype=torch.float32)
    for j in range(int(D)):
        a = j * int(n_thresh_per_feat)
        b = (j + 1) * int(n_thresh_per_feat)
        m[j, a:b] = 1.0
        m[j, F0 + a : F0 + b] = 1.0
    return m


def corner_product_z(
    x: torch.Tensor,
    th: torch.Tensor,
    sign_param: torch.Tensor,
    mask_logit: torch.Tensor,
    log_kappa: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One-sided predicates + masked product aggregation.
    z_r(x) = prod_j [ (1-m_{rj}) + m_{rj}*sigmoid(kappa*sign*(x-th)) ]
    """
    kappa = torch.exp(log_kappa).clamp(0.5, 50.0)
    ineq = torch.tanh(sign_param)  # [-1, 1]
    c = torch.sigmoid(kappa * ineq[None, :, :] * (x[:, None, :] - th[None, :, :]))  # [B,R,D]
    m = torch.sigmoid(mask_logit)  # [R,D]
    term = (1.0 - m[None, :, :]) + m[None, :, :] * c
    z = torch.exp(safe_log(term).sum(dim=2)).clamp(0.0, 1.0)  # [B,R]
    return z, c, m