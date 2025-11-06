from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class BaseRuleGater(nn.Module):
    """
    Base class for per-sample differentiable rule selection over R rules.

    Forward contract:
      inputs:
        - s: [B, R] rule scores (e.g., rule_activations before gating)
        - k: int target cardinality budget
        - mask: optional [B, R] binary mask (1=allowed, 0=forbidden)
      returns:
        - g: [B, R] in [0, 1]; typically sum(g) ~= k (or exactly k)
    """
    def __init__(self, tau: float = 0.5) -> None:
        super().__init__()
        self.tau = float(tau)

    def forward(self, s: torch.Tensor, k: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def extra_loss(self) -> torch.Tensor:
        """
        Optional regularization term (e.g., L0 or budget penalty).
        Default: zero.
        """
        p = next(self.parameters(), None)
        dev = p.device if p is not None else torch.device("cpu")
        return torch.tensor(0.0, device=dev)

def _apply_mask_to_scores(s: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Apply a binary restrict mask to scores by setting disallowed positions to a large negative value.
    """
    if mask is None:
        return s
    big_neg = -1e9
    return torch.where(mask > 0.5, s, torch.full_like(s, big_neg))

class GateSoftRank(BaseRuleGater):
    """
    SoftRank Top-k (pure PyTorch):
      ranks_i = 1 + sum_j sigmoid((s_j - s_i)/tau)
      g_i = sigmoid((k + 0.5 - ranks_i)/tau)

    Smoothly approximates the indicator of being in top-k by rank.
    """
    def __init__(self, tau: float = 0.5) -> None:
        super().__init__(tau)

    def forward(self, s: torch.Tensor, k: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        s = _apply_mask_to_scores(s, mask)                 # [B, R]
        diffs = s.unsqueeze(2) - s.unsqueeze(1)            # [B, R, R] (s_j - s_i)
        ranks = 1.0 + torch.sigmoid(diffs / max(self.tau, 1e-6)).sum(dim=2)  # [B, R]
        g = torch.sigmoid((k + 0.5 - ranks) / max(self.tau, 1e-6)).clamp(0.0, 1.0)
        return g

class GateLogisticThresholdExactK(BaseRuleGater):
    """
    Exact-k via logistic threshold with implicit differentiation solved by Newton updates:
      g_i = sigmoid((s_i - t)/tau), choose t s.t. sum_i g_i = k.

    Returns g in [0, 1] with sum(g) ≈ k (to numerical tolerance).
    """
    def __init__(self, tau: float = 0.5, iters: int = 30) -> None:
        super().__init__(tau)
        self.iters = int(iters)

    def forward(self, s: torch.Tensor, k: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        s = _apply_mask_to_scores(s, mask)
        B, R = s.shape
        k_eff = min(int(k), R)
        # Initialize threshold t at the kth largest score (per sample)
        t = torch.topk(s, k=k_eff, dim=1).values[:, -1]
        for _ in range(self.iters):
            g = torch.sigmoid((s - t.unsqueeze(1)) / max(self.tau, 1e-6))
            Fk = g.sum(dim=1) - float(k_eff)
            dFdt = - (g * (1.0 - g) / max(self.tau, 1e-6)).sum(dim=1)
            t = t - Fk / (dFdt + 1e-8)
        g = torch.sigmoid((s - t.unsqueeze(1)) / max(self.tau, 1e-6))
        return g.clamp(0.0, 1.0)

class GateCappedSimplex(BaseRuleGater):
    """
    Euclidean projection onto the capped simplex:
      argmin_p ||p - s||^2 s.t. sum p = k, 0 <= p <= 1.

    Solved by bisection on the threshold; returns p in [0,1] with sum(p)=k up to tolerance.
    """
    def __init__(self) -> None:
        super().__init__(tau=0.0)

    def forward(self, s: torch.Tensor, k: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        s = _apply_mask_to_scores(s, mask)
        B, R = s.shape
        k_eff = min(int(k), R)
        lo = (s - 1.0).min(dim=1).values
        hi = s.max(dim=1).values
        for _ in range(30):
            t = (lo + hi) / 2.0
            g = torch.clamp(s - t.unsqueeze(1), 0.0, 1.0)
            too_big = (g.sum(dim=1) > float(k_eff))
            lo = torch.where(too_big, t, lo)
            hi = torch.where(too_big, hi, t)
        g = torch.clamp(s - ((lo + hi) / 2.0).unsqueeze(1), 0.0, 1.0)
        return g

def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Standard sparsemax activation (sum=1) along the given dimension.
    """
    z_sorted, _ = torch.sort(z, descending=True, dim=dim)
    z_cumsum = torch.cumsum(z_sorted, dim=dim)
    r = torch.arange(1, z.shape[dim] + 1, device=z.device).view(
        [1]*((dim) % z.dim()) + [-1] + [1]*(z.dim() - dim - 1)
    )
    cond = 1.0 + r * z_sorted > z_cumsum
    k = cond.sum(dim=dim, keepdim=True)
    tau = (z_cumsum.gather(dim, k - 1) - 1.0) / k
    p = torch.clamp(z - tau, min=0.0)
    return p

class GateSparsemaxK(BaseRuleGater):
    """
    Sparsemax-k:
      p = sparsemax(s / tau) gives sum=1, non-negative and sparse.
      Return g = clip(k * p, 0, 1) (sum approx k unless many entries clip at 1).
    """
    def __init__(self, tau: float = 0.7) -> None:
        super().__init__(tau)

    def forward(self, s: torch.Tensor, k: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        s = _apply_mask_to_scores(s, mask)
        p = sparsemax(s / max(self.tau, 1e-6), dim=1)  # sum=1
        g = torch.clamp(float(k) * p, max=1.0)
        return g

class GateHardConcreteBudget(BaseRuleGater):
    """
    Hard-Concrete gates with a soft budget:
      z = clamp(sigmoid((logit + logit(u) - logit(1-u))/tau)*(zeta-gamma)+gamma, 0, 1)

    Adds extra_loss = l0_lambda * sum E[z] + mu_budget * (sum E[z] - target_k)^2
    """
    def __init__(self, R: int, tau: float = 0.5, l0_lambda: float = 1e-3, mu_budget: float = 1e-2, target_k: int = 8) -> None:
        super().__init__(tau)
        self.bias = nn.Parameter(torch.zeros(R))
        self.l0_lambda = float(l0_lambda)
        self.mu_budget = float(mu_budget)
        self.target_k = float(target_k)
        self.gamma, self.zeta = -0.1, 1.1

    def _sample_hard_concrete(self, logits: torch.Tensor) -> torch.Tensor:
        if self.training:
            U = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
            s = torch.sigmoid((torch.log(U) - torch.log1p(-U) + logits) / max(self.tau, 1e-6))
        else:
            s = torch.sigmoid(logits / max(self.tau, 1e-6))
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        return torch.clamp(s_bar, 0.0, 1.0)

    def forward(self, s: torch.Tensor, k: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        s = _apply_mask_to_scores(s, mask)
        logits = s + self.bias.view(1, -1)
        z = self._sample_hard_concrete(logits)
        return z

    def extra_loss(self) -> torch.Tensor:
        with torch.no_grad():
            ez = torch.sigmoid((self.bias) / max(self.tau, 1e-6)) * (self.zeta - self.gamma) + self.gamma
            ez = ez.clamp(0.0, 1.0)
        l0 = self.l0_lambda * ez.sum()
        budget = self.mu_budget * (ez.sum() - self.target_k) ** 2
        return l0 + budget

def make_rule_gater(name: str, R: int, **kwargs: Dict) -> BaseRuleGater:
    """
    Factory for rule gaters by name.

    Supported:
      - "soft_rank"
      - "logistic_threshold"
      - "capped_simplex"
      - "sparsemax_k"
      - "hard_concrete_budget"
    """
    key = (name or "").strip().lower()
    if key in ("soft_rank", "softrank"):
        return GateSoftRank(tau=kwargs.get("tau", 0.5))
    if key in ("logistic_threshold", "logistic", "exact_k"):
        return GateLogisticThresholdExactK(tau=kwargs.get("tau", 0.5), iters=kwargs.get("iters", 30))
    if key in ("capped_simplex", "projection", "proj"):
        return GateCappedSimplex()
    if key in ("sparsemax_k", "sparsemax"):
        return GateSparsemaxK(tau=kwargs.get("tau", 0.7))
    if key in ("hard_concrete_budget", "hardconcrete", "hc_budget"):
        return GateHardConcreteBudget(R=R,
                                      tau=kwargs.get("tau", 0.5),
                                      l0_lambda=kwargs.get("l0_lambda", 1e-3),
                                      mu_budget=kwargs.get("mu_budget", 1e-2),
                                      target_k=kwargs.get("target_k", kwargs.get("k", 8)))
    raise ValueError(f"Unknown rule gater: {name}")

def is_gate_param(name: str) -> bool:
    """
    Check if a parameter name corresponds to a gate parameter in rule layers.
    """
    return ("fact_logits" in name) or ("aggregator_logits" in name) or ("rule_strength_raw" in name)