from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..zoo import ThresholdFactBank


class IntervalFactBank(nn.Module):
    """
    Per-feature interval facts:
      f(x) = sigmoid(k*(x-a)) * sigmoid(k*(b-x))
    Produces D*m facts (m intervals per feature).
    """

    def __init__(self, input_dim: int, n_intervals_per_feat: int = 3, init_kappa: float = 3.0) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.m = int(n_intervals_per_feat)
        self.num_facts = self.input_dim * self.m

        feat_idx = torch.arange(self.num_facts) // self.m
        self.register_buffer("feat_idx", feat_idx)

        self.a = nn.Parameter(torch.zeros(self.num_facts))
        self.log_width = nn.Parameter(torch.zeros(self.num_facts))
        self.log_kappa = nn.Parameter(torch.full((self.num_facts,), float(math.log(init_kappa))))

    @torch.no_grad()
    def init_from_data_quantiles(self, X: np.ndarray, q_lo: float = 0.05, q_hi: float = 0.95) -> None:
        Xt = torch.tensor(np.asarray(X), dtype=torch.float32, device=self.a.device)
        for j in range(self.input_dim):
            mask = (self.feat_idx == j)
            n = int(mask.sum().item())
            if n <= 0:
                continue
            qs = torch.linspace(q_lo, q_hi, steps=n + 2, device=Xt.device)
            qv = torch.quantile(Xt[:, j], qs)
            left = qv[:-2]
            right = qv[2:]
            self.a[mask] = left
            self.log_width[mask] = torch.log((right - left).clamp_min(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xj = x[:, self.feat_idx]
        kappa = torch.exp(self.log_kappa).clamp(0.5, 50.0)
        a = self.a
        b = a + torch.exp(self.log_width)
        return torch.sigmoid(kappa * (xj - a)) * torch.sigmoid(kappa * (b - xj))


class RelationalFactBank(nn.Module):
    """
    Pairwise relational facts on differences (xi - xj):
      f_p(x) = sigmoid(kappa * ((xi - xj) - th_p))
    """

    def __init__(self, input_dim: int, pairs: Optional[Sequence[Tuple[int, int]]] = None, init_kappa: float = 2.0) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        if pairs is None:
            pairs = [(i, j) for i in range(self.input_dim) for j in range(i + 1, self.input_dim)]
        self.pairs = list(pairs)
        self.num_facts = len(self.pairs)

        self.register_buffer("i_idx", torch.tensor([p[0] for p in self.pairs], dtype=torch.long))
        self.register_buffer("j_idx", torch.tensor([p[1] for p in self.pairs], dtype=torch.long))

        self.th = nn.Parameter(torch.zeros(self.num_facts))
        self.log_kappa = nn.Parameter(torch.full((self.num_facts,), float(math.log(init_kappa))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xi = x[:, self.i_idx]
        xj = x[:, self.j_idx]
        kappa = torch.exp(self.log_kappa).clamp(0.5, 50.0)
        return torch.sigmoid(kappa * ((xi - xj) - self.th))


class RelationalDiffFactBank(nn.Module):
    """
    Pair-diff facts with multiple thresholds per pair:
      f_{p,k}(x) = sigmoid(kappa * ((xi-xj) - th[p,k]))

    Output is flattened to [B, P*K].
    """

    def __init__(
        self,
        input_dim: int,
        n_pairs: int = 256,
        n_thresh_per_pair: int = 3,
        init_kappa: float = 6.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.D = int(input_dim)
        self.P = int(n_pairs)
        self.K = int(n_thresh_per_pair)

        rng = np.random.RandomState(seed)
        all_pairs = [(i, j) for i in range(self.D) for j in range(i + 1, self.D)]
        if len(all_pairs) == 0:
            all_pairs = [(0, 0)]
        if len(all_pairs) > self.P:
            all_pairs = [all_pairs[t] for t in rng.choice(len(all_pairs), size=self.P, replace=False)]
        self.pairs = all_pairs
        self.P = int(len(self.pairs))

        self.register_buffer("i_idx", torch.tensor([p[0] for p in self.pairs], dtype=torch.long))
        self.register_buffer("j_idx", torch.tensor([p[1] for p in self.pairs], dtype=torch.long))

        self.th = nn.Parameter(torch.zeros(self.P, self.K))
        self.log_kappa = nn.Parameter(torch.tensor(float(math.log(init_kappa))))

    @property
    def num_facts(self) -> int:
        return int(self.P * self.K)

    @torch.no_grad()
    def init_from_data_quantiles(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float32)
        for p in range(self.P):
            i = int(self.i_idx[p].item())
            j = int(self.j_idx[p].item())
            diff = (X[:, i] - X[:, j]).astype(np.float32)
            qs = np.linspace(0.2, 0.8, self.K, dtype=np.float32)
            self.th[p].copy_(torch.tensor(np.quantile(diff, qs), device=self.th.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kappa = torch.exp(self.log_kappa).clamp(0.5, 50.0)
        diffs = x[:, self.i_idx] - x[:, self.j_idx]  # [B,P]
        f = torch.sigmoid(kappa * (diffs[:, :, None] - self.th[None, :, :]))  # [B,P,K]
        return f.reshape(x.shape[0], -1)


class ARFactBank(nn.Module):
    """
    Axis (ThresholdFactBank) + relational diffs (RelationalDiffFactBank).
    """

    def __init__(
        self,
        input_dim: int,
        n_thresh_per_feat: int = 6,
        n_pairs: int = 256,
        n_thresh_per_pair: int = 3,
        init_kappa: float = 6.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.axis = ThresholdFactBank(input_dim, n_thresh_per_feat=n_thresh_per_feat)
        self.rel = RelationalDiffFactBank(
            input_dim, n_pairs=n_pairs, n_thresh_per_pair=n_thresh_per_pair, init_kappa=init_kappa, seed=seed
        )

    @property
    def num_facts(self) -> int:
        return int(self.axis.num_facts + self.rel.num_facts)

    @torch.no_grad()
    def init_from_data_quantiles(self, X: np.ndarray) -> None:
        self.axis.init_from_data_quantiles(X)
        self.rel.init_from_data_quantiles(X)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.axis(x), self.rel(x)], dim=1)


class MultiResAxisFactBank(nn.Module):
    """
    Multi-resolution axis facts: same thresholds but multiple kappas.

    Outputs: [B, S*D*K]
    """

    def __init__(self, input_dim: int, n_thresh_per_feat: int = 6, kappas: Sequence[float] = (1.5, 5.0, 15.0)) -> None:
        super().__init__()
        self.D = int(input_dim)
        self.K = int(n_thresh_per_feat)
        self.S = int(len(kappas))

        self.th = nn.Parameter(torch.zeros(self.D, self.K))
        self.log_kappa = nn.Parameter(torch.tensor(np.log(np.array(kappas, dtype=np.float32))))  # [S]

    @property
    def num_facts(self) -> int:
        return int(self.D * self.K * self.S)

    @torch.no_grad()
    def init_from_data_quantiles(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float32)
        for j in range(self.D):
            qs = np.linspace(0.05, 0.95, self.K, dtype=np.float32)
            self.th[j].copy_(torch.tensor(np.quantile(X[:, j], qs), device=self.th.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kappas = torch.exp(self.log_kappa).clamp(0.5, 50.0)  # [S]
        base = x[:, :, None] - self.th[None, :, :]  # [B,D,K]
        f = torch.sigmoid(kappas[None, :, None, None] * base[:, None, :, :])  # [B,S,D,K]
        return f.reshape(x.shape[0], -1)