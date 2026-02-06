from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from ..zoo import ThresholdFactBank
from .common import safe_log, sparsemax


class PredicateRouterTree(nn.Module):
    """
    Router/diagram over threshold facts.

    Output:
      leaf_p: [B, L] where L = 2^depth
    """

    def __init__(
        self,
        input_dim: int,
        depth: int = 3,
        n_thresh_per_feat: int = 6,
        selector: Literal["sparsemax", "softmax"] = "sparsemax",
        tau_select: float = 0.7,
    ) -> None:
        super().__init__()
        self.depth = int(depth)
        self.selector = str(selector)
        self.tau_select = float(tau_select)

        self.n_nodes = (2**self.depth) - 1
        self.n_leaves = 2**self.depth

        self.facts = ThresholdFactBank(input_dim, n_thresh_per_feat=n_thresh_per_feat)
        F0 = self.facts.num_facts
        self.Fin = 2 * F0

        self.sel_logits = nn.Parameter(0.01 * torch.randn(self.n_nodes, self.Fin))

        used = torch.zeros(self.n_leaves, self.n_nodes, dtype=torch.float32)
        direc = torch.zeros(self.n_leaves, self.n_nodes, dtype=torch.float32)
        for leaf in range(self.n_leaves):
            node = 0
            for d in range(self.depth):
                bit = (leaf >> (self.depth - 1 - d)) & 1
                used[leaf, node] = 1.0
                direc[leaf, node] = float(bit)
                node = 2 * node + 1 + bit
        self.register_buffer("leaf_used", used)
        self.register_buffer("leaf_dir", direc)

    def _sel(self, logits: torch.Tensor) -> torch.Tensor:
        if self.selector == "sparsemax":
            return sparsemax(logits, dim=1)
        return torch.softmax(logits / max(self.tau_select, 1e-6), dim=1)

    @torch.no_grad()
    def init_from_data(self, X):
        self.facts.init_from_data_quantiles(X)
        nn = __import__("torch.nn", fromlist=["init"]).init  # avoid reimport warning
        nn.normal_(self.sel_logits, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.facts(x)
        f_aug = torch.cat([f, 1.0 - f], dim=1)
        sel = self._sel(self.sel_logits)  # [N,Fin]
        t = (f_aug[:, None, :] * sel[None, :, :]).sum(dim=2).clamp(1e-6, 1 - 1e-6)

        term = self.leaf_used[None, :, :] * (
            self.leaf_dir[None, :, :] * t[:, None, :] + (1.0 - self.leaf_dir[None, :, :]) * (1.0 - t[:, None, :])
        ) + (1.0 - self.leaf_used[None, :, :]) * 1.0

        return torch.exp(safe_log(term).sum(dim=2))  # [B,L]