# nous/zoo_v2/groups.py
from __future__ import annotations

from typing import Optional, Sequence, List

import numpy as np
import torch
import torch.nn as nn

from .common import default_groups_for_D


class FixedGroupIndexer(nn.Module):
    """
    Fixed feature grouping utility.

    Stores:
      - g_idx:  [G, Lmax] indices (padded with -1)
      - g_mask: [G, Lmax] 1 for valid positions else 0
    """

    def __init__(self, input_dim: int, groups: Optional[Sequence[Sequence[int]]] = None) -> None:
        super().__init__()
        self.input_dim = int(input_dim)

        # default grouping
        if groups is None:
            groups = default_groups_for_D(self.input_dim, G=4)

        # sanitize / freeze groups as python lists of ints
        self.groups: List[List[int]] = []
        for g in groups:
            gg = [int(i) for i in g]
            # Optional: validate range (helps catch silent bugs)
            for i in gg:
                if i < 0 or i >= self.input_dim:
                    raise ValueError(f"Group index {i} out of range for input_dim={self.input_dim}")
            self.groups.append(gg)

        self.G = int(len(self.groups))

        max_len = max((len(g) for g in self.groups), default=1)
        gi = np.full((self.G, max_len), fill_value=-1, dtype=np.int64)
        gm = np.zeros((self.G, max_len), dtype=np.float32)

        for a, g in enumerate(self.groups):
            if len(g) == 0:
                continue
            gi[a, : len(g)] = np.asarray(g, dtype=np.int64)
            gm[a, : len(g)] = 1.0

        self.register_buffer("g_idx", torch.tensor(gi, dtype=torch.long))
        self.register_buffer("g_mask", torch.tensor(gm, dtype=torch.float32))

    def gather(self, x_brd: torch.Tensor) -> torch.Tensor:
        """
        x_brd is typically [B, R, D].
        Returns [B, R, G, Lmax] (gathered + masked).
        """
        idx = self.g_idx.clamp_min(0)
        out = x_brd[:, :, idx]  # [B, R, G, Lmax]
        return out * self.g_mask[None, None, :, :]


class GroupKofNGate(nn.Module):
    """
    Fixed group K-of-N gate used in multiple group-evidence models.

    Inputs:
      eg: [B, R, G] group evidence (or group values)
    Output:
      z:  [B, R] gate activations in [0,1]
    """

    def __init__(self, n_rules: int, n_groups: int, beta_group: float = 6.0, beta_k: float = 8.0) -> None:
        super().__init__()
        self.n_rules = int(n_rules)
        self.G = int(n_groups)
        self.beta_group = float(beta_group)
        self.beta_k = float(beta_k)

        self.tg = nn.Parameter(torch.zeros(self.n_rules, self.G))
        self.gmask_logit = nn.Parameter(torch.full((self.n_rules, self.G), -1.0))
        self.k_frac_param = nn.Parameter(torch.zeros(self.n_rules))

    @torch.no_grad()
    def init_params(self) -> None:
        self.tg.zero_()
        self.gmask_logit.copy_(torch.full_like(self.gmask_logit, -1.0) + 0.01 * torch.randn_like(self.gmask_logit))
        self.k_frac_param.zero_()

    def forward(self, eg: torch.Tensor) -> torch.Tensor:
        pg = torch.sigmoid(self.beta_group * (eg - self.tg[None, :, :]))  # [B,R,G]
        gmask = torch.sigmoid(self.gmask_logit)                           # [R,G]
        score = (pg * gmask[None, :, :]).sum(dim=2)                       # [B,R]
        enabled = (gmask.sum(dim=1)[None, :] + 1e-6)
        k = torch.sigmoid(self.k_frac_param)[None, :] * enabled
        z = torch.sigmoid(self.beta_k * (score - k))
        return z
