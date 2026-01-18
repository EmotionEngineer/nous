from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import NousNet


def _logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


class ThresholdFactBank(nn.Module):
    """
    D*T threshold facts:
      fact_i(x) = sigmoid(kappa_i * (x[feat_i] - th_i))

    Notes
    -----
    - Thresholds are in the same feature space as input X (often standardized).
    - This bank is intended for interpretable, monotone threshold facts.
    """
    def __init__(self, input_dim: int, n_thresh_per_feat: int = 6, init_kappa: float = 2.0) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_thresh_per_feat = int(n_thresh_per_feat)
        self.num_facts = self.input_dim * self.n_thresh_per_feat

        feat_idx = torch.arange(self.num_facts) // self.n_thresh_per_feat
        self.register_buffer("feat_idx", feat_idx)

        self.th = nn.Parameter(torch.zeros(self.num_facts))
        self.log_kappa = nn.Parameter(torch.zeros(self.num_facts))
        with torch.no_grad():
            self.th.normal_(0.0, 0.8)
            self.log_kappa.fill_(float(np.log(init_kappa)))

    @torch.no_grad()
    def init_from_data_quantiles(self, X: np.ndarray, q_lo: float = 0.05, q_hi: float = 0.95) -> None:
        Xt = torch.tensor(np.asarray(X), dtype=torch.float32, device=self.th.device)
        D = int(Xt.shape[1])
        for j in range(D):
            mask = (self.feat_idx == j)
            n = int(mask.sum().item())
            if n <= 0:
                continue
            qs = torch.linspace(q_lo, q_hi, steps=n, device=Xt.device)
            self.th[mask] = torch.quantile(Xt[:, j], qs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xj = x[:, self.feat_idx]
        kappa = torch.exp(self.log_kappa).clamp(0.5, 50.0)
        return torch.sigmoid(kappa * (xj - self.th))


class LogitANDRuleLayer(nn.Module):
    """
    Logit-AND over facts (+ optional negations):
      facts_aug = [facts, 1-facts]
      p = softmax(sel_logits/tau)
      score = logit(facts_aug) @ p^T - bias
      rule = sigmoid(score)

    Returns
    -------
    z : [B, R] rule activations in [0,1]
    p : [R, Fin] selector probabilities
    """
    def __init__(self, n_rules: int, n_facts: int, tau: float = 0.7, use_negations: bool = True) -> None:
        super().__init__()
        self.n_rules = int(n_rules)
        self.n_facts = int(n_facts)
        self.tau = float(tau)
        self.use_negations = bool(use_negations)

        self.fin = self.n_facts * (2 if self.use_negations else 1)
        self.sel_logits = nn.Parameter(torch.zeros(self.n_rules, self.fin))
        self.bias = nn.Parameter(torch.zeros(self.n_rules))
        with torch.no_grad():
            self.sel_logits.normal_(0.0, 0.2)
            self.bias.zero_()

    def selector_probs(self) -> torch.Tensor:
        return torch.softmax(self.sel_logits / max(self.tau, 1e-6), dim=1)

    def forward(self, facts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        facts_aug = torch.cat([facts, 1.0 - facts], dim=1) if self.use_negations else facts
        p = self.selector_probs()  # [R, Fin]
        score = _logit(facts_aug) @ p.t() - self.bias.unsqueeze(0)
        z = torch.sigmoid(score)
        return z, p


class LowRankInteractionFacts(nn.Module):
    """
    Interaction facts computed from base facts in logit-space:
      h = sigmoid(logit(facts) @ U + b)
    """
    def __init__(self, n_base_facts: int, n_inter: int = 24) -> None:
        super().__init__()
        self.U = nn.Parameter(torch.randn(int(n_base_facts), int(n_inter)) * 0.05)
        self.b = nn.Parameter(torch.zeros(int(n_inter)))

    def forward(self, facts: torch.Tensor) -> torch.Tensor:
        l = _logit(facts)
        return torch.sigmoid(l @ self.U + self.b)


class SoftLogitAND(nn.Module):
    """
    SoftLogitAND classifier:
      - threshold fact bank
      - logit-AND rules
      - linear head on rules -> binary logit
    """
    def __init__(
        self,
        input_dim: int,
        n_rules: int = 64,
        n_thresh_per_feat: int = 6,
        tau: float = 0.7,
        use_negations: bool = True,
    ) -> None:
        super().__init__()
        self.facts = ThresholdFactBank(input_dim, n_thresh_per_feat=n_thresh_per_feat)
        self.rules = LogitANDRuleLayer(n_rules=n_rules, n_facts=self.facts.num_facts, tau=tau, use_negations=use_negations)
        self.head = nn.Linear(n_rules, 1)

    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        self.facts.init_from_data_quantiles(X)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.facts(x)
        z, _ = self.rules(f)
        return self.head(z).squeeze(-1)


class SoftLogicInteraction(nn.Module):
    """
    SoftLogicInteraction classifier:
      - threshold fact bank
      - low-rank interaction facts
      - logit-AND rules over [facts, interactions]
      - linear head -> binary logit
    """
    def __init__(
        self,
        input_dim: int,
        n_rules: int = 64,
        n_thresh_per_feat: int = 6,
        n_inter: int = 24,
        tau: float = 0.7,
    ) -> None:
        super().__init__()
        self.facts = ThresholdFactBank(input_dim, n_thresh_per_feat=n_thresh_per_feat)
        self.inter = LowRankInteractionFacts(self.facts.num_facts, n_inter=n_inter)
        self.rules = LogitANDRuleLayer(n_rules=n_rules, n_facts=self.facts.num_facts + n_inter, tau=tau, use_negations=True)
        self.head = nn.Linear(n_rules, 1)

    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        self.facts.init_from_data_quantiles(X)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.facts(x)
        h = self.inter(f)
        fh = torch.cat([f, h], dim=1)
        z, _ = self.rules(fh)
        return self.head(z).squeeze(-1)


def _default_feature_groups(feature_names: Sequence[str]) -> Dict[str, List[int]]:
    """
    Heuristic grouping inspired by WDBC ('mean', 'error', 'worst').
    Falls back to a single group 'all' if no pattern is detected.
    """
    groups: Dict[str, List[int]] = {"mean": [], "error": [], "worst": []}
    for i, n in enumerate(feature_names):
        nl = str(n).lower()
        if nl.startswith("mean "):
            groups["mean"].append(i)
        elif nl.endswith(" error"):
            groups["error"].append(i)
        elif nl.startswith("worst "):
            groups["worst"].append(i)
        else:
            groups.setdefault("other", []).append(i)

    out = {k: v for k, v in groups.items() if len(v) > 0}
    if not out:
        return {"all": list(range(len(feature_names)))}
    return out


def _build_group_fact_indices(feature_names: Sequence[str], n_thresh_per_feat: int) -> Tuple[List[str], List[torch.Tensor]]:
    groups = _default_feature_groups(feature_names)
    keys = list(groups.keys())
    idx_tensors: List[torch.Tensor] = []
    for key in keys:
        feat_idx = groups[key]
        idxs: List[int] = []
        for j in feat_idx:
            idxs.extend(list(range(j * n_thresh_per_feat, (j + 1) * n_thresh_per_feat)))
        idx_tensors.append(torch.tensor(idxs, dtype=torch.long))
    return keys, idx_tensors


class AdditiveThresholdExpert(nn.Module):
    """Additive expert: linear model on facts producing a logit."""
    def __init__(self, n_facts: int) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(int(n_facts), 1) * 0.02)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return (f @ self.w + self.b).squeeze(-1)


class SegmentMoE(nn.Module):
    """
    SegmentMoE:
      gate(x) computed by group-wise Logit-AND rule layers, combined multiplicatively,
      experts are additive models on full facts, mixed by gate probabilities.
    """
    def __init__(
        self,
        input_dim: int,
        feature_names: Sequence[str],
        n_segments: int = 6,
        n_thresh_per_feat: int = 6,
        tau: float = 0.7,
    ) -> None:
        super().__init__()
        self.feature_names = list(feature_names)
        self.n_segments = int(n_segments)
        self.n_thresh_per_feat = int(n_thresh_per_feat)

        self.facts = ThresholdFactBank(input_dim, n_thresh_per_feat=n_thresh_per_feat)
        self.group_keys, self.group_fact_idx = _build_group_fact_indices(self.feature_names, n_thresh_per_feat)

        self.group_layers = nn.ModuleList([
            LogitANDRuleLayer(n_rules=self.n_segments, n_facts=int(idx.numel()), tau=tau, use_negations=True)
            for idx in self.group_fact_idx
        ])
        self.pi_logits = nn.Parameter(torch.zeros(self.n_segments))
        self.experts = nn.ModuleList([AdditiveThresholdExpert(self.facts.num_facts) for _ in range(self.n_segments)])

    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        self.facts.init_from_data_quantiles(X)

    def gate_probs_from_facts(self, f: torch.Tensor) -> torch.Tensor:
        parts = []
        for idx, layer in zip(self.group_fact_idx, self.group_layers):
            fg = f[:, idx.to(f.device)]
            zg, _ = layer(fg)               # [B, K]
            parts.append(zg.unsqueeze(-1))  # [B, K, 1]
        A = torch.cat(parts, dim=-1)        # [B, K, G]
        m = A.prod(dim=-1)                  # [B, K]
        pi = torch.softmax(self.pi_logits, dim=0).unsqueeze(0)  # [1, K]
        w = (m * pi).clamp_min(1e-12)
        return w / (w.sum(dim=1, keepdim=True) + 1e-12)         # [B, K]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.facts(x)
        gprob = self.gate_probs_from_facts(f)  # [B, K]
        expert_logits = torch.stack([ex(f) for ex in self.experts], dim=1)  # [B, K]
        return (gprob * expert_logits).sum(dim=1)  # [B]


class SoftLogicExpert(nn.Module):
    """Soft logic expert used inside HierarchicalMoE."""
    def __init__(self, n_facts: int, n_rules: int = 24, tau: float = 0.7) -> None:
        super().__init__()
        self.rules = LogitANDRuleLayer(n_rules=n_rules, n_facts=n_facts, tau=tau, use_negations=True)
        self.head = nn.Linear(n_rules, 1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        z, _ = self.rules(f)
        return self.head(z).squeeze(-1)


class HierarchicalMoE(nn.Module):
    """
    HierarchicalMoE:
      - gate is SegmentMoE-like (group-wise rule gates),
      - experts are SoftLogicExpert models.
    """
    def __init__(
        self,
        input_dim: int,
        feature_names: Sequence[str],
        n_segments: int = 5,
        n_rules_expert: int = 24,
        n_thresh_per_feat: int = 6,
        tau: float = 0.7,
    ) -> None:
        super().__init__()
        self.feature_names = list(feature_names)
        self.n_segments = int(n_segments)
        self.n_thresh_per_feat = int(n_thresh_per_feat)

        self.facts = ThresholdFactBank(input_dim, n_thresh_per_feat=n_thresh_per_feat)
        self.group_keys, self.group_fact_idx = _build_group_fact_indices(self.feature_names, n_thresh_per_feat)

        self.group_layers = nn.ModuleList([
            LogitANDRuleLayer(n_rules=self.n_segments, n_facts=int(idx.numel()), tau=tau, use_negations=True)
            for idx in self.group_fact_idx
        ])
        self.pi_logits = nn.Parameter(torch.zeros(self.n_segments))
        self.experts = nn.ModuleList([
            SoftLogicExpert(self.facts.num_facts, n_rules=n_rules_expert, tau=tau) for _ in range(self.n_segments)
        ])

    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        self.facts.init_from_data_quantiles(X)

    def gate_probs_from_facts(self, f: torch.Tensor) -> torch.Tensor:
        parts = []
        for idx, layer in zip(self.group_fact_idx, self.group_layers):
            fg = f[:, idx.to(f.device)]
            zg, _ = layer(fg)
            parts.append(zg.unsqueeze(-1))
        A = torch.cat(parts, dim=-1)
        m = A.prod(dim=-1)
        pi = torch.softmax(self.pi_logits, dim=0).unsqueeze(0)
        w = (m * pi).clamp_min(1e-12)
        return w / (w.sum(dim=1, keepdim=True) + 1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.facts(x)
        gprob = self.gate_probs_from_facts(f)  # [B, K]
        expert_logits = torch.stack([ex(f) for ex in self.experts], dim=1)  # [B, K]
        return (gprob * expert_logits).sum(dim=1)


class FixedWidthThresholdFacts(nn.Module):
    """
    Exactly num_facts threshold facts (for NousFamilies backbone):
      fact_i(x) = sigmoid(kappa_i * (x[feat_i] - th_i))

    The feature index for fact i is (i % input_dim).
    """
    def __init__(self, input_dim: int, num_facts: int, init_kappa: float = 2.0) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_facts = int(num_facts)

        feat_idx = torch.arange(self.num_facts) % self.input_dim
        self.register_buffer("feat_idx", feat_idx)

        self.th = nn.Parameter(torch.zeros(self.num_facts))
        self.log_kappa = nn.Parameter(torch.zeros(self.num_facts))
        with torch.no_grad():
            self.th.normal_(0.0, 0.8)
            self.log_kappa.fill_(float(np.log(init_kappa)))

    @torch.no_grad()
    def init_from_data_quantiles(self, X: np.ndarray, q_lo: float = 0.05, q_hi: float = 0.95) -> None:
        Xt = torch.tensor(np.asarray(X), dtype=torch.float32, device=self.th.device)
        D = int(Xt.shape[1])
        for j in range(D):
            mask = (self.feat_idx == j)
            n = int(mask.sum().item())
            if n <= 0:
                continue
            qs = torch.linspace(q_lo, q_hi, steps=n, device=Xt.device)
            self.th[mask] = torch.quantile(Xt[:, j], qs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xj = x[:, self.feat_idx]
        kappa = torch.exp(self.log_kappa).clamp(0.5, 50.0)
        return torch.sigmoid(kappa * (xj - self.th))


class FamiliesHead(nn.Module):
    """
    Families head:
      - soft assignment A: [R, F] via softmax
      - family activations: fam = (Z @ A) / sum(A)  (per-family normalization)
      - output logit: fam @ W + b
    """
    def __init__(self, n_rules: int, n_families: int, tau: float = 0.7) -> None:
        super().__init__()
        self.n_rules = int(n_rules)
        self.n_families = int(n_families)
        self.tau = float(tau)

        self.A_logits = nn.Parameter(torch.zeros(self.n_rules, self.n_families))
        self.W = nn.Parameter(torch.randn(self.n_families, 1) * 0.05)
        self.b = nn.Parameter(torch.zeros(1))
        with torch.no_grad():
            self.A_logits.normal_(0.0, 0.2)

    def assignment(self) -> torch.Tensor:
        return torch.softmax(self.A_logits / max(self.tau, 1e-6), dim=1)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        A = self.assignment()                       # [R, F]
        denom = A.sum(dim=0, keepdim=True) + 1e-12
        fam = (Z @ A) / denom                       # [B, F]
        return (fam @ self.W + self.b).squeeze(-1)  # [B]


class NousFamilies(nn.Module):
    """
    NousFamilies:
      - NousNet backbone produces internal rule activations (gated_activations),
      - FamiliesHead aggregates all rules into stable "families",
      - Output is a single binary logit.

    Notes
    -----
    By default uses NousNet 'soft_fact' layers. We replace backbone.fact with FixedWidthThresholdFacts
    to align with threshold-based interpretable inputs.
    """
    def __init__(
        self,
        input_dim: int,
        feature_names: Sequence[str],
        num_facts: int = 64,
        rules_per_layer: Sequence[int] = (48, 24),
        n_families: int = 12,
    ) -> None:
        super().__init__()
        self.feature_names = list(feature_names)
        self.num_facts = int(num_facts)
        self.rules_per_layer = tuple(int(x) for x in rules_per_layer if int(x) > 0)
        self.n_rules_total = int(sum(self.rules_per_layer))
        self.n_families = int(n_families)

        self.backbone = NousNet(
            input_dim=input_dim,
            num_outputs=1,
            task_type="classification",
            feature_names=feature_names,
            num_facts=self.num_facts,
            rules_per_layer=self.rules_per_layer,
            rule_selection_method="soft_fact",
            use_calibrators=False,
            use_prototypes=False,
        )
        # Replace beta facts with threshold facts (keeps output shape [B, num_facts])
        self.backbone.fact = FixedWidthThresholdFacts(input_dim=input_dim, num_facts=self.num_facts)

        # Freeze backbone head (we do not use it for the final logit)
        if hasattr(self.backbone, "head") and self.backbone.head is not None:
            for p in self.backbone.head.parameters():
                p.requires_grad = False

        self.head = FamiliesHead(self.n_rules_total, self.n_families, tau=0.7)

    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        self.backbone.fact.init_from_data_quantiles(X)

    def extract_rule_matrix(self, x: torch.Tensor, detach_internals: bool = False) -> torch.Tensor:
        """
        Return concatenated gated rule activations across all blocks: [B, sum(rules_per_layer)].
        """
        _, internals = self.backbone(x, return_internals=True, detach_internals=detach_internals)
        acts: List[torch.Tensor] = []
        for b_idx in range(len(self.backbone.blocks)):
            blk = internals.get(f"block_{b_idx}", {})
            a = blk.get("gated_activations", None)
            if a is None:
                raise RuntimeError("Expected 'gated_activations' in backbone internals.")
            acts.append(a)
        return torch.cat(acts, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.extract_rule_matrix(x, detach_internals=False)
        return self.head(Z)