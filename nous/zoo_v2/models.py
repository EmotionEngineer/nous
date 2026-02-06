# nous/zoo_v2/models.py
from __future__ import annotations
import math
from typing import Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..zoo import ThresholdFactBank, LogitANDRuleLayer
from .common import (
    build_feature_to_fact_mask,
    corner_product_z,
    default_groups_for_D,
    init_thresholds_from_quantiles,
    safe_log,
    sparsemax,
    straight_through_topk_mask,
)
from .facts import ARFactBank, IntervalFactBank, MultiResAxisFactBank, RelationalFactBank
from .groups import FixedGroupIndexer, GroupKofNGate
from .trees import PredicateRouterTree

# -----------------------
# Simple baseline
# -----------------------
class MLP(nn.Module):
    """Standard MLP baseline."""
    def __init__(self, input_dim: int, hidden: int = 256, depth: int = 3, output_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        self.output_dim = int(output_dim)
        layers = []
        d = int(input_dim)
        for _ in range(int(depth)):
            layers += [nn.Linear(d, int(hidden)), nn.ReLU(), nn.Dropout(float(dropout))]
            d = int(hidden)
        layers += [nn.Linear(d, self.output_dim)]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return y.squeeze(-1) if self.output_dim == 1 else y

# -----------------------
# Soft boxes
# -----------------------
class SoftBoxBank(nn.Module):
    """Differentiable soft box (hyperrectangle) fact bank."""
    def __init__(self, input_dim: int, n_boxes: int = 128, init_kappa: float = 6.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_boxes = int(n_boxes)
        self.center = nn.Parameter(torch.zeros(self.n_boxes, self.input_dim))
        self.log_width = nn.Parameter(torch.zeros(self.n_boxes, self.input_dim))
        self.mask_logit = nn.Parameter(torch.randn(self.n_boxes, self.input_dim) * 0.01)
        self.log_kappa = nn.Parameter(torch.full((1,), float(math.log(init_kappa))))
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float32)
        idx = np.random.choice(len(X), size=min(self.n_boxes, len(X)), replace=False)
        self.center.copy_(torch.tensor(X[idx], dtype=torch.float32, device=self.center.device))
        std = X.std(axis=0, keepdims=True) + 1e-3
        self.log_width.copy_(
            torch.log(torch.tensor(std, dtype=torch.float32, device=self.log_width.device)).repeat(self.n_boxes, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kappa = torch.exp(self.log_kappa).clamp(0.5, 50.0)
        width = torch.exp(self.log_width).clamp(1e-3, 50.0)
        a = self.center - 0.5 * width
        b = self.center + 0.5 * width
        m = torch.sigmoid(self.mask_logit)
        I = torch.sigmoid(kappa * (x[:, None, :] - a[None, :, :])) * torch.sigmoid(kappa * (b[None, :, :] - x[:, None, :]))
        term = (1.0 - m[None, :, :]) + m[None, :, :] * I
        z = torch.exp(safe_log(term).sum(dim=2))
        return z.clamp(0.0, 1.0)

class BoxNet(nn.Module):
    """Box-based rule network."""
    def __init__(self, input_dim: int, n_boxes: int = 128, init_kappa: float = 6.0, output_dim: int = 1):
        super().__init__()
        self.output_dim = int(output_dim)
        self.boxes = SoftBoxBank(input_dim=input_dim, n_boxes=n_boxes, init_kappa=init_kappa)
        self.head = nn.Linear(n_boxes, self.output_dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        self.boxes.init_from_data(X)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.boxes(x)
        y = self.head(z)
        return y.squeeze(-1) if self.output_dim == 1 else y

# -----------------------
# Interval facts + LogitAND
# -----------------------
class IntervalLogitAND(nn.Module):
    """Interval facts combined with logit-AND rules."""
    def __init__(self, input_dim: int, n_rules: int = 256, n_intervals_per_feat: int = 3, tau: float = 0.7, output_dim: int = 1):
        super().__init__()
        self.output_dim = int(output_dim)
        self.facts = IntervalFactBank(input_dim, n_intervals_per_feat=n_intervals_per_feat)
        self.rules = LogitANDRuleLayer(n_rules=n_rules, n_facts=self.facts.num_facts, tau=tau, use_negations=True)
        self.head = nn.Linear(n_rules, self.output_dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        self.facts.init_from_data_quantiles(X)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.facts(x)
        z, _ = self.rules(f)
        y = self.head(z)
        return y.squeeze(-1) if self.output_dim == 1 else y

# -----------------------
# Relational facts + LogitAND
# -----------------------
class RelationalLogitAND(nn.Module):
    """Axis facts + relational difference facts with logit-AND rules."""
    def __init__(
        self,
        input_dim: int,
        n_rules: int = 256,
        n_thresh_per_feat: int = 6,
        tau: float = 0.7,
        output_dim: int = 1,
        max_pairs: Optional[int] = None,
        seed: int = 0,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.base_facts = ThresholdFactBank(input_dim, n_thresh_per_feat=n_thresh_per_feat)
        pairs = None
        if max_pairs is not None:
            rng = np.random.RandomState(int(seed))
            all_pairs = [(i, j) for i in range(int(input_dim)) for j in range(i + 1, int(input_dim))]
            if len(all_pairs) > int(max_pairs):
                pairs = [all_pairs[i] for i in rng.choice(len(all_pairs), size=int(max_pairs), replace=False)]
        self.rel_facts = RelationalFactBank(input_dim, pairs=pairs)
        self.rules = LogitANDRuleLayer(
            n_rules=n_rules, n_facts=(self.base_facts.num_facts + self.rel_facts.num_facts), tau=tau, use_negations=True
        )
        self.head = nn.Linear(n_rules, self.output_dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        self.base_facts.init_from_data_quantiles(X)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0 = self.base_facts(x)
        fr = self.rel_facts(x)
        f = torch.cat([f0, fr], dim=1)
        z, _ = self.rules(f)
        y = self.head(z)
        return y.squeeze(-1) if self.output_dim == 1 else y

# -----------------------
# Template rules (AND/soft-logit-mix/k-of-n)
# -----------------------
class TemplateRuleLayer(nn.Module):
    """Rule layer with learnable aggregator templates (AND/OR/k-of-n mixtures)."""
    def __init__(self, n_rules: int, n_facts: int, n_templates: int = 8, tau: float = 0.7, use_negations: bool = True):
        super().__init__()
        self.base = LogitANDRuleLayer(n_rules=n_rules, n_facts=n_facts, tau=tau, use_negations=use_negations)
        self.n_templates = int(n_templates)
        self.template_logits = nn.Parameter(torch.randn(self.n_templates, 3) * 0.01)
        self.rule_to_template = nn.Parameter(torch.randn(n_rules, self.n_templates) * 0.01)
    
    def forward(self, facts: torch.Tensor) -> torch.Tensor:
        z_and, P = self.base(facts)
        facts_aug = torch.cat([facts, 1.0 - facts], dim=1) if self.base.use_negations else facts
        kofn = (facts_aug[:, None, :] * P[None, :, :]).sum(dim=2)
        fa = facts_aug.clamp(1e-6, 1 - 1e-6)
        lit_logit = torch.log(fa) - torch.log1p(-fa)
        z_logit_mix = torch.sigmoid((lit_logit[:, None, :] * P[None, :, :]).sum(dim=2))
        A = torch.stack([z_and, z_logit_mix, kofn], dim=2)  # [B,R,3]
        T = torch.softmax(self.template_logits, dim=1)  # [T,3]
        Rt = torch.softmax(self.rule_to_template, dim=1)  # [R,T]
        agg_w = Rt @ T  # [R,3]
        z = (A * agg_w[None, :, :]).sum(dim=2)
        return z

class TemplateNet(nn.Module):
    """Network with template-based rule aggregation."""
    def __init__(self, input_dim: int, n_rules: int = 256, n_thresh_per_feat: int = 6, n_templates: int = 8, tau: float = 0.7, output_dim: int = 1):
        super().__init__()
        self.output_dim = int(output_dim)
        self.facts = ThresholdFactBank(input_dim, n_thresh_per_feat=n_thresh_per_feat)
        self.rules = TemplateRuleLayer(n_rules=n_rules, n_facts=self.facts.num_facts, n_templates=n_templates, tau=tau, use_negations=True)
        self.head = nn.Linear(n_rules, self.output_dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        self.facts.init_from_data_quantiles(X)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.facts(x)
        z = self.rules(f)
        y = self.head(z)
        return y.squeeze(-1) if self.output_dim == 1 else y

# -----------------------
# Decision list from rules
# -----------------------
class RuleListNet(nn.Module):
    """Ordered rule list with stopping probabilities."""
    def __init__(self, input_dim: int, n_rules: int = 128, n_thresh_per_feat: int = 6, tau: float = 0.7, output_dim: int = 1):
        super().__init__()
        self.output_dim = int(output_dim)
        self.facts = ThresholdFactBank(input_dim, n_thresh_per_feat=n_thresh_per_feat)
        self.rules = LogitANDRuleLayer(n_rules=n_rules, n_facts=self.facts.num_facts, tau=tau, use_negations=True)
        if self.output_dim == 1:
            self.v = nn.Parameter(torch.zeros(n_rules))
            self.v_default = nn.Parameter(torch.zeros(1))
        else:
            self.v = nn.Parameter(torch.zeros(n_rules, self.output_dim))
            self.v_default = nn.Parameter(torch.zeros(self.output_dim))
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        self.facts.init_from_data_quantiles(X)
        self.v.zero_()
        self.v_default.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.facts(x)
        z, _ = self.rules(f)
        z = z.clamp(0.0, 1.0)
        one_minus = (1.0 - z + 1e-6)
        prefix = torch.cumprod(one_minus, dim=1)
        prefix_excl = torch.cat([torch.ones_like(prefix[:, :1]), prefix[:, :-1]], dim=1)
        stop = z * prefix_excl
        remaining = prefix[:, -1]
        if self.output_dim == 1:
            return (stop * self.v[None, :]).sum(dim=1) + remaining * self.v_default
        return stop @ self.v + remaining[:, None] * self.v_default[None, :]

# -----------------------
# Fact decision diagram (leaf mixture) over threshold facts
# -----------------------
class FactDiagram(nn.Module):
    """Decision diagram router over threshold facts with leaf values."""
    def __init__(self, input_dim: int, depth: int = 4, n_thresh_per_feat: int = 6, tau_select: float = 0.7, output_dim: int = 1):
        super().__init__()
        self.output_dim = int(output_dim)
        self.depth = int(depth)
        self.n_nodes = (2**self.depth) - 1
        self.n_leaves = 2**self.depth
        self.facts = ThresholdFactBank(input_dim, n_thresh_per_feat=n_thresh_per_feat)
        fin = 2 * self.facts.num_facts
        self.tau_select = float(tau_select)
        self.sel_logits = nn.Parameter(torch.randn(self.n_nodes, fin) * 0.01)
        if self.output_dim == 1:
            self.leaf_value = nn.Parameter(torch.zeros(self.n_leaves))
        else:
            self.leaf_value = nn.Parameter(torch.zeros(self.n_leaves, self.output_dim))
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
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        self.facts.init_from_data_quantiles(X)
        self.leaf_value.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.facts(x)
        f_aug = torch.cat([f, 1.0 - f], dim=1)
        sel = torch.softmax(self.sel_logits / max(self.tau_select, 1e-6), dim=1)
        t = (f_aug[:, None, :] * sel[None, :, :]).sum(dim=2).clamp(1e-6, 1 - 1e-6)
        term = self.leaf_used[None, :, :] * (
            self.leaf_dir[None, :, :] * t[:, None, :] + (1.0 - self.leaf_dir[None, :, :]) * (1.0 - t[:, None, :])
        ) + (1.0 - self.leaf_used[None, :, :]) * 1.0
        leaf_p = torch.exp(safe_log(term).sum(dim=2))  # [B,L]
        if self.output_dim == 1:
            return (leaf_p * self.leaf_value[None, :]).sum(dim=1)
        return leaf_p @ self.leaf_value

# -----------------------
# Corner-family rules
# -----------------------
class CornerNet(nn.Module):
    """Corner predicate rules (one-sided thresholds with masking)."""
    def __init__(self, input_dim: int, n_rules: int = 256, init_kappa: float = 6.0, output_dim: int = 1):
        super().__init__()
        self.output_dim = int(output_dim)
        self.th = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.sign_param = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.mask_logit = nn.Parameter(torch.full((n_rules, input_dim), -2.0))
        self.log_kappa = nn.Parameter(torch.tensor(float(math.log(init_kappa))))
        self.head = nn.Linear(n_rules, self.output_dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        init_thresholds_from_quantiles(self.th, X)
        nn.init.normal_(self.sign_param, std=0.3)
        self.mask_logit.copy_(torch.full_like(self.mask_logit, -2.0) + 0.01 * torch.randn_like(self.mask_logit))
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, _, _ = corner_product_z(x, self.th, self.sign_param, self.mask_logit, self.log_kappa)
        y = self.head(z)
        return y.squeeze(-1) if self.output_dim == 1 else y

class SoftMinCornerNet(nn.Module):
    """Corner rules with soft-min aggregation instead of product."""
    def __init__(self, input_dim: int, n_rules: int = 256, init_kappa: float = 6.0, tau_min: float = 0.15, output_dim: int = 1):
        super().__init__()
        self.output_dim = int(output_dim)
        self.tau_min = float(tau_min)
        self.th = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.sign_param = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.mask_logit = nn.Parameter(torch.full((n_rules, input_dim), -2.0))
        self.log_kappa = nn.Parameter(torch.tensor(float(math.log(init_kappa))))
        self.head = nn.Linear(n_rules, self.output_dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        init_thresholds_from_quantiles(self.th, X)
        nn.init.normal_(self.sign_param, std=0.3)
        self.mask_logit.copy_(torch.full_like(self.mask_logit, -2.0) + 0.01 * torch.randn_like(self.mask_logit))
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kappa = torch.exp(self.log_kappa).clamp(0.5, 50.0)
        ineq = torch.tanh(self.sign_param)
        c = torch.sigmoid(kappa * ineq[None, :, :] * (x[:, None, :] - self.th[None, :, :]))
        m = torch.sigmoid(self.mask_logit)
        term = (1.0 - m[None, :, :]) + m[None, :, :] * c
        w = torch.softmax(-term / max(self.tau_min, 1e-6), dim=2)
        z = (w * term).sum(dim=2)
        y = self.head(z)
        return y.squeeze(-1) if self.output_dim == 1 else y

class KofNCornerNet(nn.Module):
    """Corner rules with k-of-n aggregation."""
    def __init__(self, input_dim: int, n_rules: int = 256, init_kappa: float = 6.0, beta: float = 8.0, output_dim: int = 1):
        super().__init__()
        self.output_dim = int(output_dim)
        self.beta = float(beta)
        self.th = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.sign_param = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.mask_logit = nn.Parameter(torch.full((n_rules, input_dim), -2.0))
        self.k_frac_param = nn.Parameter(torch.zeros(n_rules))
        self.log_kappa = nn.Parameter(torch.tensor(float(math.log(init_kappa))))
        self.head = nn.Linear(n_rules, self.output_dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        init_thresholds_from_quantiles(self.th, X)
        nn.init.normal_(self.sign_param, std=0.3)
        self.mask_logit.copy_(torch.full_like(self.mask_logit, -2.0) + 0.01 * torch.randn_like(self.mask_logit))
        self.k_frac_param.zero_()
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kappa = torch.exp(self.log_kappa).clamp(0.5, 50.0)
        ineq = torch.tanh(self.sign_param)
        c = torch.sigmoid(kappa * ineq[None, :, :] * (x[:, None, :] - self.th[None, :, :]))
        m = torch.sigmoid(self.mask_logit)
        s = (c * m[None, :, :]).sum(dim=2)
        msum = (m.sum(dim=1)[None, :] + 1e-6)
        k = torch.sigmoid(self.k_frac_param)[None, :] * msum
        z = torch.sigmoid(self.beta * (s - k))
        y = self.head(z)
        return y.squeeze(-1) if self.output_dim == 1 else y

class RingCornerNet(nn.Module):
    """Ring-shaped regions defined by outer corner minus inner corner."""
    def __init__(self, input_dim: int, n_rings: int = 256, init_kappa: float = 6.0, output_dim: int = 1):
        super().__init__()
        self.output_dim = int(output_dim)
        self.th_o = nn.Parameter(torch.zeros(n_rings, input_dim))
        self.sign_o = nn.Parameter(torch.randn(n_rings, input_dim) * 0.1)
        self.mask_o = nn.Parameter(torch.full((n_rings, input_dim), -2.0))
        self.th_i = nn.Parameter(torch.zeros(n_rings, input_dim))
        self.sign_i = nn.Parameter(torch.randn(n_rings, input_dim) * 0.1)
        self.mask_i = nn.Parameter(torch.full((n_rings, input_dim), -2.0))
        self.log_kappa = nn.Parameter(torch.tensor(float(math.log(init_kappa))))
        self.head = nn.Linear(n_rings, self.output_dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        init_thresholds_from_quantiles(self.th_o, X)
        init_thresholds_from_quantiles(self.th_i, X)
        nn.init.normal_(self.sign_o, std=0.3)
        nn.init.normal_(self.sign_i, std=0.3)
        self.mask_o.copy_(torch.full_like(self.mask_o, -2.0) + 0.01 * torch.randn_like(self.mask_o))
        self.mask_i.copy_(torch.full_like(self.mask_i, -2.0) + 0.01 * torch.randn_like(self.mask_i))
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zo, _, _ = corner_product_z(x, self.th_o, self.sign_o, self.mask_o, self.log_kappa)
        zi, _, _ = corner_product_z(x, self.th_i, self.sign_i, self.mask_i, self.log_kappa)
        z = zo * (1.0 - zi)
        y = self.head(z)
        return y.squeeze(-1) if self.output_dim == 1 else y

class HybridCornerIntervalNet(nn.Module):
    """Hybrid corner/interval facts per dimension with type selection."""
    def __init__(self, input_dim: int, n_rules: int = 256, init_kappa: float = 6.0, output_dim: int = 1):
        super().__init__()
        self.output_dim = int(output_dim)
        self.n_rules = int(n_rules)
        self.input_dim = int(input_dim)
        self.th_c = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.sign_c = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.center = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.log_width = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.type_logits = nn.Parameter(torch.zeros(n_rules, input_dim, 2))
        self.mask_logit = nn.Parameter(torch.full((n_rules, input_dim), -2.0))
        self.log_kappa = nn.Parameter(torch.tensor(float(math.log(init_kappa))))
        self.head = nn.Linear(n_rules, self.output_dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float32)
        init_thresholds_from_quantiles(self.th_c, X)
        self.center.copy_(
            torch.tensor(np.tile(X.mean(axis=0, keepdims=True), (self.n_rules, 1)), dtype=torch.float32, device=self.center.device)
        )
        std = X.std(axis=0, keepdims=True) + 1e-3
        self.log_width.copy_(
            torch.log(torch.tensor(np.tile(std, (self.n_rules, 1)), dtype=torch.float32, device=self.log_width.device))
        )
        nn.init.normal_(self.sign_c, std=0.3)
        self.type_logits.zero_()
        self.mask_logit.copy_(torch.full_like(self.mask_logit, -2.0) + 0.01 * torch.randn_like(self.mask_logit))
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kappa = torch.exp(self.log_kappa).clamp(0.5, 50.0)
        sgn = torch.tanh(self.sign_c)
        c_corner = torch.sigmoid(kappa * sgn[None, :, :] * (x[:, None, :] - self.th_c[None, :, :]))
        width = torch.exp(self.log_width).clamp(1e-3, 50.0)
        a = self.center - 0.5 * width
        b = self.center + 0.5 * width
        c_int = torch.sigmoid(kappa * (x[:, None, :] - a[None, :, :])) * torch.sigmoid(kappa * (b[None, :, :] - x[:, None, :]))
        t = torch.softmax(self.type_logits, dim=2)  # [R,D,2]
        c = t[None, :, :, 0] * c_corner + t[None, :, :, 1] * c_int
        m = torch.sigmoid(self.mask_logit)
        term = (1.0 - m[None, :, :]) + m[None, :, :] * c
        z = torch.exp(safe_log(term).sum(dim=2)).clamp(0.0, 1.0)
        y = self.head(z)
        return y.squeeze(-1) if self.output_dim == 1 else y

# -----------------------
# Priority mixture (rules compete)
# -----------------------
class PriorityMixtureNet(nn.Module):
    """Rules compete via priority-weighted mixture."""
    def __init__(self, input_dim: int, n_rules: int = 256, init_kappa: float = 6.0, output_dim: int = 1):
        super().__init__()
        self.output_dim = int(output_dim)
        self.n_rules = int(n_rules)
        self.th = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.sign_param = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.mask_logit = nn.Parameter(torch.full((n_rules, input_dim), -2.0))
        self.log_kappa = nn.Parameter(torch.tensor(float(math.log(init_kappa))))
        self.priority = nn.Parameter(torch.zeros(n_rules))
        if self.output_dim == 1:
            self.v = nn.Parameter(torch.zeros(n_rules))
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.v = nn.Parameter(torch.zeros(n_rules, self.output_dim))
            self.bias = nn.Parameter(torch.zeros(self.output_dim))
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        init_thresholds_from_quantiles(self.th, X)
        nn.init.normal_(self.sign_param, std=0.3)
        self.mask_logit.copy_(torch.full_like(self.mask_logit, -2.0) + 0.01 * torch.randn_like(self.mask_logit))
        nn.init.normal_(self.priority, std=0.01)
        nn.init.normal_(self.v, std=0.01)
        self.bias.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, _, _ = corner_product_z(x, self.th, self.sign_param, self.mask_logit, self.log_kappa)
        logits = self.priority[None, :] + torch.log(z.clamp_min(1e-6))
        pi = torch.softmax(logits, dim=1)  # [B,R]
        if self.output_dim == 1:
            return (pi * self.v[None, :]).sum(dim=1) + self.bias
        return pi @ self.v + self.bias[None, :]

# -----------------------
# Evidence family
# -----------------------
class EvidenceNet(nn.Module):
    """Signed evidence aggregation per rule."""
    def __init__(self, input_dim: int, n_rules: int = 256, init_kappa: float = 6.0, beta: float = 6.0, output_dim: int = 1):
        super().__init__()
        self.output_dim = int(output_dim)
        self.beta = float(beta)
        self.th = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.ineq_sign_param = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.e_sign_param = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.mask_logit = nn.Parameter(torch.full((n_rules, input_dim), -2.0))
        self.log_kappa = nn.Parameter(torch.tensor(float(math.log(init_kappa))))
        self.t = nn.Parameter(torch.zeros(n_rules))
        self.head = nn.Linear(n_rules, self.output_dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        init_thresholds_from_quantiles(self.th, X)
        nn.init.normal_(self.ineq_sign_param, std=0.3)
        nn.init.normal_(self.e_sign_param, std=0.3)
        self.mask_logit.copy_(torch.full_like(self.mask_logit, -2.0) + 0.01 * torch.randn_like(self.mask_logit))
        self.t.zero_()
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kappa = torch.exp(self.log_kappa).clamp(0.5, 50.0)
        ineq = torch.tanh(self.ineq_sign_param)
        c = torch.sigmoid(kappa * ineq[None, :, :] * (x[:, None, :] - self.th[None, :, :]))
        m = torch.sigmoid(self.mask_logit)[None, :, :]
        e_sign = torch.tanh(self.e_sign_param)[None, :, :]
        evidence = (m * e_sign * (2.0 * c - 1.0)).sum(dim=2)
        z = torch.sigmoid(self.beta * (evidence - self.t[None, :]))
        y = self.head(z)
        return y.squeeze(-1) if self.output_dim == 1 else y

class MarginEvidenceNet(nn.Module):
    """Evidence with learnable margins per feature."""
    def __init__(self, input_dim: int, n_rules: int = 256, init_kappa: float = 6.0, beta: float = 6.0, output_dim: int = 1):
        super().__init__()
        self.output_dim = int(output_dim)
        self.beta = float(beta)
        self.th = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.ineq_sign_param = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.margin_param = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.e_sign_param = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.mask_logit = nn.Parameter(torch.full((n_rules, input_dim), -2.0))
        self.log_kappa = nn.Parameter(torch.tensor(float(math.log(init_kappa))))
        self.t = nn.Parameter(torch.zeros(n_rules))
        self.head = nn.Linear(n_rules, self.output_dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        init_thresholds_from_quantiles(self.th, X)
        nn.init.normal_(self.ineq_sign_param, std=0.3)
        nn.init.normal_(self.e_sign_param, std=0.3)
        self.margin_param.zero_()
        self.mask_logit.copy_(torch.full_like(self.mask_logit, -2.0) + 0.01 * torch.randn_like(self.mask_logit))
        self.t.zero_()
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kappa = torch.exp(self.log_kappa).clamp(0.5, 50.0)
        ineq = torch.tanh(self.ineq_sign_param)
        margin = F.softplus(self.margin_param)
        c = torch.sigmoid(kappa * (ineq[None, :, :] * (x[:, None, :] - self.th[None, :, :]) - margin[None, :, :]))
        m = torch.sigmoid(self.mask_logit)[None, :, :]
        e_sign = torch.tanh(self.e_sign_param)[None, :, :]
        evidence = (m * e_sign * (2.0 * c - 1.0)).sum(dim=2)
        z = torch.sigmoid(self.beta * (evidence - self.t[None, :]))
        y = self.head(z)
        return y.squeeze(-1) if self.output_dim == 1 else y

class PerFeatureKappaEvidenceNet(nn.Module):
    """Evidence with per-feature learnable kappas."""
    def __init__(self, input_dim: int, n_rules: int = 256, init_kappa: float = 6.0, beta: float = 6.0, output_dim: int = 1):
        super().__init__()
        self.output_dim = int(output_dim)
        self.beta = float(beta)
        self.th = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.ineq = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.esign = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.mask = nn.Parameter(torch.full((n_rules, input_dim), -2.0))
        self.log_kappa = nn.Parameter(torch.full((n_rules, input_dim), float(math.log(init_kappa))))
        self.t = nn.Parameter(torch.zeros(n_rules))
        self.head = nn.Linear(n_rules, self.output_dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        init_thresholds_from_quantiles(self.th, X)
        nn.init.normal_(self.ineq, std=0.3)
        nn.init.normal_(self.esign, std=0.3)
        self.mask.copy_(torch.full_like(self.mask, -2.0) + 0.01 * torch.randn_like(self.mask))
        self.log_kappa.fill_(float(math.log(6.0)))
        self.t.zero_()
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kappa = torch.exp(self.log_kappa).clamp(0.5, 50.0)
        ineq = torch.tanh(self.ineq)
        c = torch.sigmoid(kappa[None, :, :] * ineq[None, :, :] * (x[:, None, :] - self.th[None, :, :]))
        m = torch.sigmoid(self.mask)[None, :, :]
        e = torch.tanh(self.esign)[None, :, :]
        evidence = (m * e * (2.0 * c - 1.0)).sum(dim=2)
        z = torch.sigmoid(self.beta * (evidence - self.t[None, :]))
        y = self.head(z)
        return y.squeeze(-1) if self.output_dim == 1 else y

class LadderEvidenceNet(nn.Module):
    """Multi-threshold ladder evidence per feature."""
    def __init__(self, input_dim: int, n_rules: int = 256, n_levels: int = 3, init_kappa: float = 6.0, beta: float = 6.0, output_dim: int = 1):
        super().__init__()
        self.output_dim = int(output_dim)
        self.L = int(n_levels)
        assert self.L >= 2
        self.beta = float(beta)
        self.th0 = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.delta = nn.Parameter(torch.zeros(n_rules, input_dim, self.L - 1))
        self.ineq = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.esign = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.mask = nn.Parameter(torch.full((n_rules, input_dim), -2.0))
        self.log_kappa = nn.Parameter(torch.tensor(float(math.log(init_kappa))))
        self.level_w_param = nn.Parameter(torch.zeros(self.L))
        self.t = nn.Parameter(torch.zeros(n_rules))
        self.head = nn.Linear(n_rules, self.output_dim)
    
    def _thresholds(self) -> torch.Tensor:
        inc = F.softplus(self.delta)
        th = torch.cat([self.th0[:, :, None], self.th0[:, :, None] + torch.cumsum(inc, dim=2)], dim=2)
        return th
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        init_thresholds_from_quantiles(self.th0, X)
        self.delta.zero_()
        nn.init.normal_(self.ineq, std=0.3)
        nn.init.normal_(self.esign, std=0.3)
        self.mask.copy_(torch.full_like(self.mask, -2.0) + 0.01 * torch.randn_like(self.mask))
        self.level_w_param.zero_()
        self.t.zero_()
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        th = self._thresholds()
        kappa = torch.exp(self.log_kappa).clamp(0.5, 50.0)
        ineq = torch.tanh(self.ineq)
        c = torch.sigmoid(kappa * ineq[None, :, :, None] * (x[:, None, :, None] - th[None, :, :, :]))  # [B,R,D,L]
        w = F.softplus(self.level_w_param) + 1e-6
        sev = (c * w[None, None, None, :]).sum(dim=3) / w.sum()  # [B,R,D]
        m = torch.sigmoid(self.mask)[None, :, :]
        e = torch.tanh(self.esign)[None, :, :]
        evidence = (m * e * (2.0 * sev - 1.0)).sum(dim=2)
        z = torch.sigmoid(self.beta * (evidence - self.t[None, :]))
        y = self.head(z)
        return y.squeeze(-1) if self.output_dim == 1 else y

class BiEvidenceNet(nn.Module):
    """Bidirectional evidence (low/high thresholds per feature)."""
    def __init__(self, input_dim: int, n_rules: int = 256, init_kappa: float = 6.0, beta: float = 6.0, output_dim: int = 1):
        super().__init__()
        self.output_dim = int(output_dim)
        self.beta = float(beta)
        self.center = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.log_width = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.e_low = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.e_high = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.mask = nn.Parameter(torch.full((n_rules, input_dim), -2.0))
        self.log_kappa = nn.Parameter(torch.tensor(float(math.log(init_kappa))))
        self.t = nn.Parameter(torch.zeros(n_rules))
        self.head = nn.Linear(n_rules, self.output_dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float32)
        med = np.quantile(X, 0.50, axis=0, keepdims=True).astype(np.float32)
        q25 = np.quantile(X, 0.25, axis=0, keepdims=True).astype(np.float32)
        q75 = np.quantile(X, 0.75, axis=0, keepdims=True).astype(np.float32)
        width = (q75 - q25).astype(np.float32)
        width = np.maximum(width, 0.5 * X.std(axis=0, keepdims=True).astype(np.float32) + 1e-3)
        self.center.copy_(torch.tensor(np.tile(med, (self.center.shape[0], 1)), device=self.center.device))
        self.log_width.copy_(torch.log(torch.tensor(np.tile(width, (self.log_width.shape[0], 1)), device=self.log_width.device)))
        nn.init.normal_(self.e_low, std=0.3)
        nn.init.normal_(self.e_high, std=0.3)
        self.mask.copy_(torch.full_like(self.mask, -2.0) + 0.01 * torch.randn_like(self.mask))
        self.t.zero_()
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        width = torch.exp(self.log_width).clamp(1e-3, 50.0)
        t_low = self.center - 0.5 * width
        t_high = self.center + 0.5 * width
        kappa = torch.exp(self.log_kappa).clamp(0.5, 50.0)
        low = torch.sigmoid(kappa * (t_low[None, :, :] - x[:, None, :]))
        high = torch.sigmoid(kappa * (x[:, None, :] - t_high[None, :, :]))
        m = torch.sigmoid(self.mask)[None, :, :]
        el = torch.tanh(self.e_low)[None, :, :]
        eh = torch.tanh(self.e_high)[None, :, :]
        evidence = (m * (el * (2.0 * low - 1.0) + eh * (2.0 * high - 1.0))).sum(dim=2)
        z = torch.sigmoid(self.beta * (evidence - self.t[None, :]))
        y = self.head(z)
        return y.squeeze(-1) if self.output_dim == 1 else y

class EvidenceKofNNet(nn.Module):
    """Evidence rules with combined evidence and count thresholds."""
    def __init__(self, input_dim: int, n_rules: int = 256, init_kappa: float = 6.0, alpha: float = 5.0, beta: float = 8.0, output_dim: int = 1):
        super().__init__()
        self.output_dim = int(output_dim)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.th = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.ineq_sign_param = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.e_sign_param = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.mask_logit = nn.Parameter(torch.full((n_rules, input_dim), -2.0))
        self.log_kappa = nn.Parameter(torch.tensor(float(math.log(init_kappa))))
        self.t_e = nn.Parameter(torch.zeros(n_rules))
        self.k_frac_param = nn.Parameter(torch.zeros(n_rules))
        self.head = nn.Linear(n_rules, self.output_dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        init_thresholds_from_quantiles(self.th, X)
        nn.init.normal_(self.ineq_sign_param, std=0.3)
        nn.init.normal_(self.e_sign_param, std=0.3)
        self.mask_logit.copy_(torch.full_like(self.mask_logit, -2.0) + 0.01 * torch.randn_like(self.mask_logit))
        self.t_e.zero_()
        self.k_frac_param.zero_()
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kappa = torch.exp(self.log_kappa).clamp(0.5, 50.0)
        ineq = torch.tanh(self.ineq_sign_param)
        c = torch.sigmoid(kappa * ineq[None, :, :] * (x[:, None, :] - self.th[None, :, :]))
        m = torch.sigmoid(self.mask_logit)
        e_sign = torch.tanh(self.e_sign_param)
        evidence = (m[None, :, :] * e_sign[None, :, :] * (2.0 * c - 1.0)).sum(dim=2)
        count = (m[None, :, :] * c).sum(dim=2)
        msum = (m.sum(dim=1)[None, :] + 1e-6)
        k = torch.sigmoid(self.k_frac_param)[None, :] * msum
        z = torch.sigmoid(self.alpha * (evidence - self.t_e[None, :])) * torch.sigmoid(self.beta * (count - k))
        y = self.head(z)
        return y.squeeze(-1) if self.output_dim == 1 else y

# -----------------------
# Group-evidence family
# -----------------------
class GroupEvidenceKofNNet(nn.Module):
    """Group-based evidence aggregation with k-of-n gating."""
    def __init__(
        self,
        input_dim: int,
        groups: Optional[Sequence[Sequence[int]]] = None,
        n_rules: int = 256,
        init_kappa: float = 6.0,
        beta_group: float = 6.0,
        beta_k: float = 8.0,
        output_dim: int = 1,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.gi = FixedGroupIndexer(input_dim, groups=groups)
        G = self.gi.G
        self.th = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.ineq_sign_param = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.e_sign_param = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.mask_logit = nn.Parameter(torch.full((n_rules, input_dim), -2.0))
        self.log_kappa = nn.Parameter(torch.tensor(float(math.log(init_kappa))))
        self.gate = GroupKofNGate(n_rules, G, beta_group=beta_group, beta_k=beta_k)
        self.head = nn.Linear(n_rules, self.output_dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        init_thresholds_from_quantiles(self.th, X)
        nn.init.normal_(self.ineq_sign_param, std=0.3)
        nn.init.normal_(self.e_sign_param, std=0.3)
        self.mask_logit.copy_(torch.full_like(self.mask_logit, -2.0) + 0.01 * torch.randn_like(self.mask_logit))
        self.gate.init_params()
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kappa = torch.exp(self.log_kappa).clamp(0.5, 50.0)
        ineq = torch.tanh(self.ineq_sign_param)
        c = torch.sigmoid(kappa * ineq[None, :, :] * (x[:, None, :] - self.th[None, :, :]))
        m = torch.sigmoid(self.mask_logit)
        e = torch.tanh(self.e_sign_param)
        sym = (m[None, :, :] * e[None, :, :] * (2.0 * c - 1.0))
        eg = self.gi.gather(sym).sum(dim=3)  # [B,R,G]
        z = self.gate(eg)
        y = self.head(z)
        return y.squeeze(-1) if self.output_dim == 1 else y

class SoftGroupEvidenceKofNNet(nn.Module):
    """Group evidence with learned soft grouping matrix."""
    def __init__(
        self,
        input_dim: int,
        n_rules: int = 256,
        n_groups: int = 4,
        init_kappa: float = 6.0,
        beta_group: float = 6.0,
        beta_k: float = 8.0,
        output_dim: int = 1,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.input_dim = int(input_dim)
        self.n_rules = int(n_rules)
        self.G = int(n_groups)
        self.beta_group = float(beta_group)
        self.beta_k = float(beta_k)
        self.th = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.ineq_sign_param = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.e_sign_param = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.mask_logit = nn.Parameter(torch.full((n_rules, input_dim), -2.0))
        self.log_kappa = nn.Parameter(torch.tensor(float(math.log(init_kappa))))
        self.group_logits = nn.Parameter(torch.zeros(self.G, self.input_dim))
        self.tg = nn.Parameter(torch.zeros(n_rules, self.G))
        self.gmask_logit = nn.Parameter(torch.full((n_rules, self.G), -1.0))
        self.k_frac_param = nn.Parameter(torch.zeros(n_rules))
        self.head = nn.Linear(n_rules, self.output_dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        init_thresholds_from_quantiles(self.th, X)
        nn.init.normal_(self.ineq_sign_param, std=0.3)
        nn.init.normal_(self.e_sign_param, std=0.3)
        self.mask_logit.copy_(torch.full_like(self.mask_logit, -2.0) + 0.01 * torch.randn_like(self.mask_logit))
        self.group_logits.zero_()
        self.tg.zero_()
        self.gmask_logit.copy_(torch.full_like(self.gmask_logit, -1.0) + 0.01 * torch.randn_like(self.gmask_logit))
        self.k_frac_param.zero_()
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
    
    def group_matrix(self) -> torch.Tensor:
        return torch.softmax(self.group_logits, dim=0)  # [G,D]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kappa = torch.exp(self.log_kappa).clamp(0.5, 50.0)
        ineq = torch.tanh(self.ineq_sign_param)
        c = torch.sigmoid(kappa * ineq[None, :, :] * (x[:, None, :] - self.th[None, :, :]))
        m = torch.sigmoid(self.mask_logit)
        e_sign = torch.tanh(self.e_sign_param)
        sym = (m[None, :, :] * e_sign[None, :, :] * (2.0 * c - 1.0))  # [B,R,D]
        A = self.group_matrix()  # [G,D]
        eg = torch.einsum("brd,gd->brg", sym, A)  # [B,R,G]
        pg = torch.sigmoid(self.beta_group * (eg - self.tg[None, :, :]))
        gmask = torch.sigmoid(self.gmask_logit)
        score = (pg * gmask[None, :, :]).sum(dim=2)
        enabled = (gmask.sum(dim=1)[None, :] + 1e-6)
        k = torch.sigmoid(self.k_frac_param)[None, :] * enabled
        z = torch.sigmoid(self.beta_k * (score - k))
        y = self.head(z)
        return y.squeeze(-1) if self.output_dim == 1 else y

class GroupSoftMinNet(nn.Module):
    """Group evidence with soft-min aggregation per group."""
    def __init__(
        self,
        input_dim: int,
        groups: Optional[Sequence[Sequence[int]]] = None,
        n_rules: int = 256,
        init_kappa: float = 6.0,
        tau_min: float = 0.15,
        beta_group: float = 6.0,
        beta_k: float = 8.0,
        output_dim: int = 1,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.tau_min = float(tau_min)
        self.gi = FixedGroupIndexer(input_dim, groups=groups)
        G = self.gi.G
        self.th = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.sign = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.mask = nn.Parameter(torch.full((n_rules, input_dim), -2.0))
        self.log_kappa = nn.Parameter(torch.tensor(float(math.log(init_kappa))))
        self.gate = GroupKofNGate(n_rules, G, beta_group=beta_group, beta_k=beta_k)
        self.head = nn.Linear(n_rules, self.output_dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        init_thresholds_from_quantiles(self.th, X)
        nn.init.normal_(self.sign, std=0.3)
        self.mask.copy_(torch.full_like(self.mask, -2.0) + 0.01 * torch.randn_like(self.mask))
        self.gate.init_params()
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kappa = torch.exp(self.log_kappa).clamp(0.5, 50.0)
        ineq = torch.tanh(self.sign)
        c = torch.sigmoid(kappa * ineq[None, :, :] * (x[:, None, :] - self.th[None, :, :]))
        m = torch.sigmoid(self.mask)
        term = (1.0 - m[None, :, :]) + m[None, :, :] * c  # [B,R,D]
        tg = self.gi.gather(term)  # [B,R,G,L]
        missing = (self.gi.g_mask[None, None, :, :] < 0.5)
        tg = torch.where(missing, torch.ones_like(tg), tg)
        logits = -tg / max(self.tau_min, 1e-6) + (self.gi.g_mask[None, None, :, :] - 1.0) * 1e6
        w = torch.softmax(logits, dim=3)
        group_val = (w * tg).sum(dim=3)  # [B,R,G]
        z = self.gate(group_val)
        y = self.head(z)
        return y.squeeze(-1) if self.output_dim == 1 else y

class GroupContrastNet(nn.Module):
    """Group contrast: difference between two learned group selections."""
    def __init__(
        self,
        input_dim: int,
        groups: Optional[Sequence[Sequence[int]]] = None,
        n_rules: int = 256,
        init_kappa: float = 6.0,
        beta: float = 8.0,
        output_dim: int = 1,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.beta = float(beta)
        self.gi = FixedGroupIndexer(input_dim, groups=groups)
        G = self.gi.G
        self.th = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.ineq = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.esign = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.mask = nn.Parameter(torch.full((n_rules, input_dim), -2.0))
        self.log_kappa = nn.Parameter(torch.tensor(float(math.log(init_kappa))))
        self.selA = nn.Parameter(torch.zeros(n_rules, G))
        self.selB = nn.Parameter(torch.zeros(n_rules, G))
        self.t = nn.Parameter(torch.zeros(n_rules))
        self.head = nn.Linear(n_rules, self.output_dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        init_thresholds_from_quantiles(self.th, X)
        nn.init.normal_(self.ineq, std=0.3)
        nn.init.normal_(self.esign, std=0.3)
        self.mask.copy_(torch.full_like(self.mask, -2.0) + 0.01 * torch.randn_like(self.mask))
        self.selA.zero_()
        self.selB.zero_()
        self.t.zero_()
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kappa = torch.exp(self.log_kappa).clamp(0.5, 50.0)
        ineq = torch.tanh(self.ineq)
        c = torch.sigmoid(kappa * ineq[None, :, :] * (x[:, None, :] - self.th[None, :, :]))
        m = torch.sigmoid(self.mask)
        e = torch.tanh(self.esign)
        sym = (m[None, :, :] * e[None, :, :] * (2.0 * c - 1.0))
        eg = self.gi.gather(sym).sum(dim=3)  # [B,R,G]
        wa = torch.softmax(self.selA, dim=1)
        wb = torch.softmax(self.selB, dim=1)
        Ea = (eg * wa[None, :, :]).sum(dim=2)
        Eb = (eg * wb[None, :, :]).sum(dim=2)
        z = torch.sigmoid(self.beta * ((Ea - Eb) - self.t[None, :]))
        y = self.head(z)
        return y.squeeze(-1) if self.output_dim == 1 else y

class GroupRingNet(nn.Module):
    """Group ring: outer group region minus inner group region."""
    def __init__(
        self,
        input_dim: int,
        groups: Optional[Sequence[Sequence[int]]] = None,
        n_rings: int = 256,
        init_kappa: float = 6.0,
        beta_group: float = 6.0,
        beta_k: float = 8.0,
        output_dim: int = 1,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.gi = FixedGroupIndexer(input_dim, groups=groups)
        G = self.gi.G
        self.n_rings = int(n_rings)
        self.th_o = nn.Parameter(torch.zeros(n_rings, input_dim))
        self.ineq_o = nn.Parameter(torch.randn(n_rings, input_dim) * 0.1)
        self.esign_o = nn.Parameter(torch.randn(n_rings, input_dim) * 0.1)
        self.mask_o = nn.Parameter(torch.full((n_rings, input_dim), -2.0))
        self.th_i = nn.Parameter(torch.zeros(n_rings, input_dim))
        self.ineq_i = nn.Parameter(torch.randn(n_rings, input_dim) * 0.1)
        self.esign_i = nn.Parameter(torch.randn(n_rings, input_dim) * 0.1)
        self.mask_i = nn.Parameter(torch.full((n_rings, input_dim), -2.0))
        self.log_kappa = nn.Parameter(torch.tensor(float(math.log(init_kappa))))
        self.outer_gate = GroupKofNGate(n_rings, G, beta_group=beta_group, beta_k=beta_k)
        self.inner_gate = GroupKofNGate(n_rings, G, beta_group=beta_group, beta_k=beta_k)
        self.head = nn.Linear(n_rings, self.output_dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        init_thresholds_from_quantiles(self.th_o, X)
        init_thresholds_from_quantiles(self.th_i, X)
        nn.init.normal_(self.ineq_o, std=0.3)
        nn.init.normal_(self.ineq_i, std=0.3)
        nn.init.normal_(self.esign_o, std=0.3)
        nn.init.normal_(self.esign_i, std=0.3)
        self.mask_o.copy_(torch.full_like(self.mask_o, -2.0) + 0.01 * torch.randn_like(self.mask_o))
        self.mask_i.copy_(torch.full_like(self.mask_i, -2.0) + 0.01 * torch.randn_like(self.mask_i))
        self.outer_gate.init_params()
        self.inner_gate.init_params()
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
    
    def _group_evidence(self, x: torch.Tensor, th: torch.Tensor, ineq_p: torch.Tensor, esign_p: torch.Tensor, mask_logit: torch.Tensor) -> torch.Tensor:
        kappa = torch.exp(self.log_kappa).clamp(0.5, 50.0)
        ineq = torch.tanh(ineq_p)
        c = torch.sigmoid(kappa * ineq[None, :, :] * (x[:, None, :] - th[None, :, :]))
        m = torch.sigmoid(mask_logit)
        e = torch.tanh(esign_p)
        sym = (m[None, :, :] * e[None, :, :] * (2.0 * c - 1.0))
        eg = self.gi.gather(sym).sum(dim=3)
        return eg
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eg_o = self._group_evidence(x, self.th_o, self.ineq_o, self.esign_o, self.mask_o)
        eg_i = self._group_evidence(x, self.th_i, self.ineq_i, self.esign_i, self.mask_i)
        zo = self.outer_gate(eg_o)
        zi = self.inner_gate(eg_i)
        z = zo * (1.0 - zi)
        y = self.head(z)
        return y.squeeze(-1) if self.output_dim == 1 else y

# -----------------------
# Regime -> Rules
# -----------------------
class SignedEvidenceRuleLayer(nn.Module):
    """Reusable signed evidence rule layer."""
    def __init__(self, input_dim: int, n_rules: int = 256, init_kappa: float = 6.0, beta: float = 6.0):
        super().__init__()
        self.beta = float(beta)
        self.th = nn.Parameter(torch.zeros(n_rules, input_dim))
        self.ineq_sign_param = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.e_sign_param = nn.Parameter(torch.randn(n_rules, input_dim) * 0.1)
        self.mask_logit = nn.Parameter(torch.full((n_rules, input_dim), -2.0))
        self.log_kappa = nn.Parameter(torch.tensor(float(math.log(init_kappa))))
        self.t = nn.Parameter(torch.zeros(n_rules))
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        init_thresholds_from_quantiles(self.th, X)
        nn.init.normal_(self.ineq_sign_param, std=0.3)
        nn.init.normal_(self.e_sign_param, std=0.3)
        self.mask_logit.copy_(torch.full_like(self.mask_logit, -2.0) + 0.01 * torch.randn_like(self.mask_logit))
        self.t.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kappa = torch.exp(self.log_kappa).clamp(0.5, 50.0)
        ineq = torch.tanh(self.ineq_sign_param)
        c = torch.sigmoid(kappa * ineq[None, :, :] * (x[:, None, :] - self.th[None, :, :]))
        m = torch.sigmoid(self.mask_logit)[None, :, :]
        e_sign = torch.tanh(self.e_sign_param)[None, :, :]
        evidence = (m * e_sign * (2.0 * c - 1.0)).sum(dim=2)
        z = torch.sigmoid(self.beta * (evidence - self.t[None, :]))
        return z

class RegimeRulesNet(nn.Module):
    """Two-stage model: regime detection followed by regime-specific rule heads."""
    def __init__(
        self,
        input_dim: int,
        groups: Optional[Sequence[Sequence[int]]] = None,
        n_regimes: int = 6,
        n_rules: int = 256,
        init_kappa: float = 6.0,
        beta_group: float = 6.0,
        beta_k: float = 8.0,
        beta_rules: float = 6.0,
        output_dim: int = 1,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.K = int(n_regimes)
        self.R = int(n_rules)
        self.gi = FixedGroupIndexer(input_dim, groups=groups)
        G = self.gi.G
        self.th_r = nn.Parameter(torch.zeros(self.K, input_dim))
        self.ineq_r = nn.Parameter(torch.randn(self.K, input_dim) * 0.1)
        self.esign_r = nn.Parameter(torch.randn(self.K, input_dim) * 0.1)
        self.mask_r = nn.Parameter(torch.full((self.K, input_dim), -2.0))
        self.log_kappa = nn.Parameter(torch.tensor(float(math.log(init_kappa))))
        self.regime_gate = GroupKofNGate(self.K, G, beta_group=beta_group, beta_k=beta_k)
        self.regime_prior = nn.Parameter(torch.zeros(self.K))
        self.rules = SignedEvidenceRuleLayer(input_dim, n_rules=n_rules, init_kappa=init_kappa, beta=beta_rules)
        if self.output_dim == 1:
            self.W = nn.Parameter(torch.zeros(self.K, self.R))
            self.b = nn.Parameter(torch.zeros(self.K))
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.W = nn.Parameter(torch.zeros(self.K, self.R, self.output_dim))
            self.b = nn.Parameter(torch.zeros(self.K, self.output_dim))
            self.bias = nn.Parameter(torch.zeros(self.output_dim))
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        init_thresholds_from_quantiles(self.th_r, X)
        nn.init.normal_(self.ineq_r, std=0.3)
        nn.init.normal_(self.esign_r, std=0.3)
        self.mask_r.copy_(torch.full_like(self.mask_r, -2.0) + 0.01 * torch.randn_like(self.mask_r))
        self.regime_gate.init_params()
        self.regime_prior.zero_()
        self.rules.init_from_data(X)
        nn.init.normal_(self.W, std=0.01)
        self.b.zero_()
        self.bias.zero_()
    
    def _regime_group_evidence(self, x: torch.Tensor) -> torch.Tensor:
        kappa = torch.exp(self.log_kappa).clamp(0.5, 50.0)
        ineq = torch.tanh(self.ineq_r)
        c = torch.sigmoid(kappa * ineq[None, :, :] * (x[:, None, :] - self.th_r[None, :, :]))  # [B,K,D]
        m = torch.sigmoid(self.mask_r)  # [K,D]
        e = torch.tanh(self.esign_r)    # [K,D]
        sym = (m[None, :, :] * e[None, :, :] * (2.0 * c - 1.0))
        eg = self.gi.gather(sym).sum(dim=3)  # [B,K,G]
        return eg
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eg = self._regime_group_evidence(x)
        z_reg = self.regime_gate(eg).clamp_min(1e-6)
        logits = self.regime_prior[None, :] + torch.log(z_reg)
        pi = torch.softmax(logits, dim=1)  # [B,K]
        z_rules = self.rules(x)  # [B,R]
        if self.output_dim == 1:
            yk = (z_rules[:, None, :] * self.W[None, :, :]).sum(dim=2) + self.b[None, :]  # [B,K]
            y = (pi * yk).sum(dim=1) + self.bias
            return y
        yk = (z_rules[:, None, :, None] * self.W[None, :, :, :]).sum(dim=2) + self.b[None, :, :]  # [B,K,C]
        y = (pi[:, :, None] * yk).sum(dim=1) + self.bias[None, :]
        return y

# -----------------------
# Router/forest family
# -----------------------
class PredicateForest(nn.Module):
    """Shared predicate forest: multiple trees sharing the same fact bank."""
    def __init__(
        self,
        input_dim: int,
        n_trees: int = 32,
        depth: int = 4,
        n_thresh_per_feat: int = 6,
        tau_select: float = 0.7,
        selector: str = "sparsemax",
        output_dim: int = 1,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.T = int(n_trees)
        self.depth = int(depth)
        self.tau_select = float(tau_select)
        self.selector = str(selector)
        self.facts = ThresholdFactBank(input_dim, n_thresh_per_feat=n_thresh_per_feat)
        F0 = self.facts.num_facts
        self.Fin = 2 * F0
        self.sel_logits = nn.Parameter(0.01 * torch.randn(self.T * self.depth, self.Fin))
        L = 2**self.depth
        if self.output_dim == 1:
            self.leaf_value = nn.Parameter(torch.zeros(self.T, L))
        else:
            self.leaf_value = nn.Parameter(torch.zeros(self.T, L, self.output_dim))
    
    def _sel(self, logits: torch.Tensor) -> torch.Tensor:
        if self.selector == "sparsemax":
            return sparsemax(logits, dim=1)
        return torch.softmax(logits / max(self.tau_select, 1e-6), dim=1)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        self.facts.init_from_data_quantiles(X)
        nn.init.normal_(self.sel_logits, std=0.01)
        nn.init.normal_(self.leaf_value, std=0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        f = self.facts(x)
        f_aug = torch.cat([f, 1.0 - f], dim=1)
        sel = self._sel(self.sel_logits).view(self.T, self.depth, self.Fin)
        p = torch.einsum("bf,tlf->btl", f_aug, sel).clamp(1e-6, 1 - 1e-6)
        probs = x.new_ones(B, self.T, 1)
        for d in range(self.depth):
            pd = p[:, :, d].unsqueeze(-1)
            probs = torch.cat([probs * (1.0 - pd), probs * pd], dim=2)
        if self.output_dim == 1:
            return (probs * self.leaf_value[None, :, :]).sum(dim=2).sum(dim=1)
        return torch.einsum("btl,tlc->btc", probs, self.leaf_value).sum(dim=1)

class ObliviousForest(nn.Module):
    """Fact-oblivious forest (independent tree selectors)."""
    def __init__(self, input_dim: int, n_trees: int = 32, depth: int = 4, n_thresh_per_feat: int = 6, selector: str = "sparsemax", output_dim: int = 1):
        super().__init__()
        self.output_dim = int(output_dim)
        self.T = int(n_trees)
        self.depth = int(depth)
        self.selector = str(selector)
        self.facts = ThresholdFactBank(input_dim, n_thresh_per_feat=n_thresh_per_feat)
        F0 = self.facts.num_facts
        self.Fin = 2 * F0
        self.sel_logits = nn.Parameter(0.01 * torch.randn(self.T * self.depth, self.Fin))
        L = 2**self.depth
        if self.output_dim == 1:
            self.leaf_value = nn.Parameter(torch.zeros(self.T, L))
        else:
            self.leaf_value = nn.Parameter(torch.zeros(self.T, L, self.output_dim))
    
    def _sel(self, logits: torch.Tensor) -> torch.Tensor:
        return sparsemax(logits, dim=1) if self.selector == "sparsemax" else torch.softmax(logits, dim=1)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        self.facts.init_from_data_quantiles(X)
        nn.init.normal_(self.sel_logits, std=0.01)
        nn.init.normal_(self.leaf_value, std=0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        f = self.facts(x)
        f_aug = torch.cat([f, 1.0 - f], dim=1)
        sel = self._sel(self.sel_logits).view(self.T, self.depth, self.Fin)
        p = torch.einsum("bf,tlf->btl", f_aug, sel).clamp(1e-6, 1 - 1e-6)
        probs = x.new_ones(B, self.T, 1)
        for d in range(self.depth):
            pd = p[:, :, d].unsqueeze(-1)
            probs = torch.cat([probs * (1.0 - pd), probs * pd], dim=2)
        if self.output_dim == 1:
            return (probs * self.leaf_value[None, :, :]).sum(dim=2).sum(dim=1)
        return torch.einsum("btl,tlc->btc", probs, self.leaf_value).sum(dim=1)

class LeafLinearForest(nn.Module):
    """Forest with linear models at leaves (calibrated to input features)."""
    def __init__(
        self,
        input_dim: int,
        n_trees: int = 32,
        depth: int = 4,
        n_thresh_per_feat: int = 6,
        leaf_k: int = 3,
        selector: str = "sparsemax",
        output_dim: int = 1,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.D = int(input_dim)
        self.T = int(n_trees)
        self.depth = int(depth)
        self.L = 2**self.depth
        self.K = int(leaf_k)
        self.selector = str(selector)
        self.facts = ThresholdFactBank(self.D, n_thresh_per_feat=n_thresh_per_feat)
        F0 = self.facts.num_facts
        self.Fin = 2 * F0
        self.sel_logits = nn.Parameter(0.01 * torch.randn(self.T * self.depth, self.Fin))
        self.leaf_sel_logits = nn.Parameter(0.01 * torch.randn(self.T, self.L, self.K, self.D))
        if self.output_dim == 1:
            self.leaf_w = nn.Parameter(torch.zeros(self.T, self.L, self.K))
            self.leaf_b = nn.Parameter(torch.zeros(self.T, self.L))
        else:
            self.leaf_w = nn.Parameter(torch.zeros(self.T, self.L, self.K, self.output_dim))
            self.leaf_b = nn.Parameter(torch.zeros(self.T, self.L, self.output_dim))
    
    def _sel(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return sparsemax(logits, dim=dim) if self.selector == "sparsemax" else torch.softmax(logits, dim=dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        self.facts.init_from_data_quantiles(X)
        nn.init.normal_(self.sel_logits, std=0.01)
        nn.init.normal_(self.leaf_sel_logits, std=0.01)
        nn.init.normal_(self.leaf_w, std=0.01)
        self.leaf_b.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        f = self.facts(x)
        f_aug = torch.cat([f, 1.0 - f], dim=1)
        sel = self._sel(self.sel_logits, dim=1).view(self.T, self.depth, self.Fin)
        p = torch.einsum("bf,tlf->btl", f_aug, sel).clamp(1e-6, 1 - 1e-6)
        probs = x.new_ones(B, self.T, 1)
        for d in range(self.depth):
            pd = p[:, :, d].unsqueeze(-1)
            probs = torch.cat([probs * (1.0 - pd), probs * pd], dim=2)  # [B,T,L]
        leaf_sel = self._sel(self.leaf_sel_logits, dim=3)  # [T,L,K,D]
        xleaf = torch.einsum("bd,tlkd->btlk", x, leaf_sel)  # [B,T,L,K]
        if self.output_dim == 1:
            leaf_y = self.leaf_b[None, :, :, None] + xleaf * self.leaf_w[None, :, :, :]
            leaf_y = leaf_y.sum(dim=3)  # [B,T,L]
            return (probs * leaf_y).sum(dim=2).sum(dim=1)
        leaf_y = self.leaf_b[None, :, :, None, :] + xleaf[:, :, :, :, None] * self.leaf_w[None, :, :, :, :]
        leaf_y = leaf_y.sum(dim=3)  # [B,T,L,C]
        yT = torch.einsum("btl,btlc->btc", probs, leaf_y)
        return yT.sum(dim=1)

class AttentiveForest(nn.Module):
    """Forest with sample-dependent attention over trees."""
    def __init__(
        self,
        input_dim: int,
        n_trees: int = 64,
        depth: int = 3,
        n_thresh_per_feat: int = 6,
        selector: str = "sparsemax",
        output_dim: int = 1,
        att_hidden: int = 0,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.T = int(n_trees)
        self.depth = int(depth)
        self.selector = str(selector)
        self.facts = ThresholdFactBank(input_dim, n_thresh_per_feat=n_thresh_per_feat)
        F0 = self.facts.num_facts
        self.Fin = 2 * F0
        self.sel_logits = nn.Parameter(0.01 * torch.randn(self.T * self.depth, self.Fin))
        L = 2**self.depth
        if self.output_dim == 1:
            self.leaf_value = nn.Parameter(torch.zeros(self.T, L))
        else:
            self.leaf_value = nn.Parameter(torch.zeros(self.T, L, self.output_dim))
        if att_hidden and att_hidden > 0:
            self.att = nn.Sequential(nn.Linear(self.Fin, att_hidden), nn.ReLU(), nn.Linear(att_hidden, self.T))
        else:
            self.att = nn.Linear(self.Fin, self.T)
    
    def _sel(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return sparsemax(logits, dim=dim) if self.selector == "sparsemax" else torch.softmax(logits, dim=dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        self.facts.init_from_data_quantiles(X)
        nn.init.normal_(self.sel_logits, std=0.01)
        nn.init.normal_(self.leaf_value, std=0.01)
        for m in self.att.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        f = self.facts(x)
        f_aug = torch.cat([f, 1.0 - f], dim=1)
        sel = self._sel(self.sel_logits, dim=1).view(self.T, self.depth, self.Fin)
        p = torch.einsum("bf,tlf->btl", f_aug, sel).clamp(1e-6, 1 - 1e-6)
        probs = x.new_ones(B, self.T, 1)
        for d in range(self.depth):
            pd = p[:, :, d].unsqueeze(-1)
            probs = torch.cat([probs * (1.0 - pd), probs * pd], dim=2)  # [B,T,L]
        if self.output_dim == 1:
            y_t = (probs * self.leaf_value[None, :, :]).sum(dim=2)  # [B,T]
        else:
            y_t = torch.einsum("btl,tlc->btc", probs, self.leaf_value)  # [B,T,C]
        a_logits = self.att(f_aug)  # [B,T]
        a = self._sel(a_logits, dim=1)
        if self.output_dim == 1:
            return (a * y_t).sum(dim=1)
        return (a[:, :, None] * y_t).sum(dim=1)

# -----------------------
# Tree of rules (router -> leaf-specific head on evidence rules)
# -----------------------
class RuleTree(nn.Module):
    """Decision tree router over threshold facts feeding into evidence rules."""
    def __init__(
        self,
        input_dim: int,
        depth: int = 3,
        n_thresh_per_feat: int = 6,
        n_rules: int = 256,
        init_kappa: float = 6.0,
        beta_rules: float = 6.0,
        output_dim: int = 1,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.router = PredicateRouterTree(input_dim, depth=depth, n_thresh_per_feat=n_thresh_per_feat, selector="sparsemax")
        self.rules = SignedEvidenceRuleLayer(input_dim, n_rules=n_rules, init_kappa=init_kappa, beta=beta_rules)
        L = self.router.n_leaves
        R = int(n_rules)
        if self.output_dim == 1:
            self.W_leaf = nn.Parameter(torch.zeros(L, R))
            self.b_leaf = nn.Parameter(torch.zeros(L))
        else:
            self.W_leaf = nn.Parameter(torch.zeros(L, R, self.output_dim))
            self.b_leaf = nn.Parameter(torch.zeros(L, self.output_dim))
        # expose mask for feature importances
        self.mask_logit = self.rules.mask_logit
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        self.router.init_from_data(X)
        self.rules.init_from_data(X)
        nn.init.normal_(self.W_leaf, std=0.01)
        self.b_leaf.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leaf_p = self.router(x)  # [B,L]
        z = self.rules(x)        # [B,R]
        if self.output_dim == 1:
            y_leaf = z @ self.W_leaf.t() + self.b_leaf[None, :]  # [B,L]
            return (leaf_p * y_leaf).sum(dim=1)
        y_leaf = torch.einsum("br,lrc->blc", z, self.W_leaf) + self.b_leaf[None, :, :]
        return (leaf_p[:, :, None] * y_leaf).sum(dim=1)

class SparseRuleTree(nn.Module):
    """Rule tree with sparse leaf-specific rule selection."""
    def __init__(
        self,
        input_dim: int,
        depth: int = 3,
        n_thresh_per_feat: int = 6,
        n_rules: int = 256,
        init_kappa: float = 6.0,
        beta_rules: float = 6.0,
        selector: str = "sparsemax",
        output_dim: int = 1,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.selector = str(selector)
        self.router = PredicateRouterTree(input_dim, depth=depth, n_thresh_per_feat=n_thresh_per_feat, selector="sparsemax")
        self.rules = SignedEvidenceRuleLayer(input_dim, n_rules=n_rules, init_kappa=init_kappa, beta=beta_rules)
        self.mask_logit = self.rules.mask_logit
        self.L = self.router.n_leaves
        self.R = int(n_rules)
        self.leaf_att_logits = nn.Parameter(0.01 * torch.randn(self.L, self.R))
        if self.output_dim == 1:
            self.leaf_v = nn.Parameter(torch.zeros(self.L, self.R))
            self.leaf_b = nn.Parameter(torch.zeros(self.L))
        else:
            self.leaf_v = nn.Parameter(torch.zeros(self.L, self.R, self.output_dim))
            self.leaf_b = nn.Parameter(torch.zeros(self.L, self.output_dim))
    
    def _sel(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return sparsemax(logits, dim=dim) if self.selector == "sparsemax" else torch.softmax(logits, dim=dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        self.router.init_from_data(X)
        self.rules.init_from_data(X)
        nn.init.normal_(self.leaf_att_logits, std=0.01)
        nn.init.normal_(self.leaf_v, std=0.01)
        self.leaf_b.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leaf_p = self.router(x)  # [B,L]
        z = self.rules(x)        # [B,R]
        a = self._sel(self.leaf_att_logits, dim=1)  # [L,R]
        if self.output_dim == 1:
            W = a * self.leaf_v
            y_leaf = z @ W.t() + self.leaf_b[None, :]
            return (leaf_p * y_leaf).sum(dim=1)
        contrib = torch.einsum("br,lr,lrc->blc", z, a, self.leaf_v) + self.leaf_b[None, :, :]
        return (leaf_p[:, :, None] * contrib).sum(dim=1)

class RuleDiagram(nn.Module):
    """Decision diagram router with rule-based leaf heads."""
    def __init__(
        self,
        input_dim: int,
        diagram_depth: int = 4,
        n_thresh_per_feat: int = 6,
        n_rules: int = 256,
        init_kappa: float = 6.0,
        beta_rules: float = 6.0,
        selector: str = "sparsemax",
        output_dim: int = 1,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.selector = str(selector)
        self.router = PredicateRouterTree(input_dim, depth=diagram_depth, n_thresh_per_feat=n_thresh_per_feat, selector="sparsemax")
        self.rules = SignedEvidenceRuleLayer(input_dim, n_rules=n_rules, init_kappa=init_kappa, beta=beta_rules)
        self.mask_logit = self.rules.mask_logit
        L = self.router.n_leaves
        R = int(n_rules)
        self.leaf_att_logits = nn.Parameter(0.01 * torch.randn(L, R))
        if self.output_dim == 1:
            self.leaf_W = nn.Parameter(torch.zeros(L, R))
            self.leaf_b = nn.Parameter(torch.zeros(L))
        else:
            self.leaf_W = nn.Parameter(torch.zeros(L, R, self.output_dim))
            self.leaf_b = nn.Parameter(torch.zeros(L, self.output_dim))
    
    def _sel(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return sparsemax(logits, dim=dim) if self.selector == "sparsemax" else torch.softmax(logits, dim=dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        self.router.init_from_data(X)
        self.rules.init_from_data(X)
        nn.init.normal_(self.leaf_att_logits, std=0.01)
        nn.init.normal_(self.leaf_W, std=0.01)
        self.leaf_b.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leaf_p = self.router(x)  # [B,L]
        z = self.rules(x)        # [B,R]
        a = self._sel(self.leaf_att_logits, dim=1)  # [L,R]
        if self.output_dim == 1:
            W = a * self.leaf_W
            y_leaf = z @ W.t() + self.leaf_b[None, :]
            return (leaf_p * y_leaf).sum(dim=1)
        y_leaf = torch.einsum("br,lrc,lr->blc", z, self.leaf_W, a) + self.leaf_b[None, :, :]
        return (leaf_p[:, :, None] * y_leaf).sum(dim=1)

# -----------------------
# Clause features
# -----------------------
class ClauseNet(nn.Module):
    """Clause-based features (conjunctions over literals) as input to linear head."""
    def __init__(self, input_dim: int, n_clauses: int = 64, n_thresh_per_feat: int = 6, selector_temp: float = 1.0, output_dim: int = 1):
        super().__init__()
        self.output_dim = int(output_dim)
        self.K = int(n_clauses)
        self.selector_temp = float(selector_temp)
        self.facts = ThresholdFactBank(input_dim, n_thresh_per_feat=n_thresh_per_feat)
        F0 = self.facts.num_facts
        self.Fin = 2 * F0
        self.mask_logit_lit = nn.Parameter(torch.full((self.K, self.Fin), -2.0))
        self.head = nn.Linear(self.K, self.output_dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        self.facts.init_from_data_quantiles(X)
        self.mask_logit_lit.copy_(torch.full_like(self.mask_logit_lit, -2.0) + 0.01 * torch.randn_like(self.mask_logit_lit))
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.facts(x)
        f_aug = torch.cat([f, 1.0 - f], dim=1)  # [B,Fin]
        m = torch.sigmoid(self.mask_logit_lit / max(self.selector_temp, 1e-6))  # [K,Fin]
        term = (1.0 - m[None, :, :]) + m[None, :, :] * f_aug[:, None, :]
        clauses = torch.exp(safe_log(term).sum(dim=2)).clamp(0.0, 1.0)  # [B,K]
        y = self.head(clauses)
        return y.squeeze(-1) if self.output_dim == 1 else y

# -----------------------
# Axis+Relational facts + AND rules
# -----------------------
class ARLogitAND(nn.Module):
    """Axis facts + relational difference facts with logit-AND rules."""
    def __init__(
        self,
        input_dim: int,
        n_rules: int = 256,
        n_thresh_per_feat: int = 6,
        n_pairs: int = 256,
        n_thresh_per_pair: int = 3,
        tau: float = 0.7,
        init_kappa: float = 6.0,
        output_dim: int = 1,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.facts = ARFactBank(
            input_dim, n_thresh_per_feat=n_thresh_per_feat, n_pairs=n_pairs, n_thresh_per_pair=n_thresh_per_pair, init_kappa=init_kappa
        )
        self.rules = LogitANDRuleLayer(n_rules=n_rules, n_facts=self.facts.num_facts, tau=tau, use_negations=True)
        self.head = nn.Linear(n_rules, self.output_dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        self.facts.init_from_data_quantiles(X)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.facts(x)
        z, _ = self.rules(f)
        y = self.head(z)
        return y.squeeze(-1) if self.output_dim == 1 else y

# -----------------------
# Multi-res facts forest
# -----------------------
class MultiResForest(nn.Module):
    """Forest using multi-resolution axis facts (multiple kappas per threshold)."""
    def __init__(
        self,
        input_dim: int,
        n_trees: int = 32,
        depth: int = 4,
        n_thresh_per_feat: int = 6,
        kappas: Sequence[float] = (1.5, 5.0, 15.0),
        selector: str = "sparsemax",
        output_dim: int = 1,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.T = int(n_trees)
        self.depth = int(depth)
        self.selector = str(selector)
        self.facts = MultiResAxisFactBank(input_dim, n_thresh_per_feat=n_thresh_per_feat, kappas=kappas)
        F0 = self.facts.num_facts
        self.Fin = 2 * F0
        self.sel_logits = nn.Parameter(0.01 * torch.randn(self.T * self.depth, self.Fin))
        L = 2**self.depth
        if self.output_dim == 1:
            self.leaf_value = nn.Parameter(torch.zeros(self.T, L))
        else:
            self.leaf_value = nn.Parameter(torch.zeros(self.T, L, self.output_dim))
    
    def _sel(self, logits: torch.Tensor) -> torch.Tensor:
        return sparsemax(logits, dim=1) if self.selector == "sparsemax" else torch.softmax(logits, dim=1)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        self.facts.init_from_data_quantiles(X)
        nn.init.normal_(self.sel_logits, std=0.01)
        nn.init.normal_(self.leaf_value, std=0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        f = self.facts(x)
        f_aug = torch.cat([f, 1.0 - f], dim=1)
        sel = self._sel(self.sel_logits).view(self.T, self.depth, self.Fin)
        p = torch.einsum("bf,tlf->btl", f_aug, sel).clamp(1e-6, 1 - 1e-6)
        probs = x.new_ones(B, self.T, 1)
        for d in range(self.depth):
            pd = p[:, :, d].unsqueeze(-1)
            probs = torch.cat([probs * (1.0 - pd), probs * pd], dim=2)
        if self.output_dim == 1:
            return (probs * self.leaf_value[None, :, :]).sum(dim=2).sum(dim=1)
        return torch.einsum("btl,tlc->btc", probs, self.leaf_value).sum(dim=1)

# -----------------------
# Group-first forest
# -----------------------
class GroupFirstForest(nn.Module):
    """Forest with explicit group-first selection before fact selection."""
    def __init__(
        self,
        input_dim: int,
        groups: Optional[Sequence[Sequence[int]]] = None,
        n_trees: int = 32,
        depth: int = 4,
        n_thresh_per_feat: int = 6,
        selector: str = "sparsemax",
        output_dim: int = 1,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.D = int(input_dim)
        self.T = int(n_trees)
        self.depth = int(depth)
        self.selector = str(selector)
        if groups is None:
            groups = default_groups_for_D(self.D, G=4)
        self.groups = [list(g) for g in groups]
        self.G = len(self.groups)
        self.facts = ThresholdFactBank(self.D, n_thresh_per_feat=n_thresh_per_feat)
        self.n_thresh_per_feat = int(n_thresh_per_feat)
        F0 = self.facts.num_facts
        self.Fin = 2 * F0
        self.group_logits = nn.Parameter(0.01 * torch.randn(self.T * self.depth, self.G))
        self.fact_logits = nn.Parameter(0.01 * torch.randn(self.T * self.depth, self.Fin))
        mask = torch.zeros(self.G, self.Fin, dtype=torch.float32)
        for gi, feats in enumerate(self.groups):
            for j in feats:
                a = j * self.n_thresh_per_feat
                b = (j + 1) * self.n_thresh_per_feat
                mask[gi, a:b] = 1.0
                mask[gi, F0 + a : F0 + b] = 1.0
        self.register_buffer("group_fact_mask", mask)
        L = 2**self.depth
        if self.output_dim == 1:
            self.leaf_value = nn.Parameter(torch.zeros(self.T, L))
        else:
            self.leaf_value = nn.Parameter(torch.zeros(self.T, L, self.output_dim))
    
    def _sel(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return sparsemax(logits, dim=dim) if self.selector == "sparsemax" else torch.softmax(logits, dim=dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        self.facts.init_from_data_quantiles(X)
        nn.init.normal_(self.group_logits, std=0.01)
        nn.init.normal_(self.fact_logits, std=0.01)
        nn.init.normal_(self.leaf_value, std=0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        f = self.facts(x)
        f_aug = torch.cat([f, 1.0 - f], dim=1)
        pg = self._sel(self.group_logits.view(self.T, self.depth, self.G), dim=2)
        pf_raw = self._sel(self.fact_logits, dim=1).view(self.T, self.depth, self.Fin)
        gm = torch.einsum("tdg,gf->tdf", pg, self.group_fact_mask)  # [T,depth,Fin]
        pf = pf_raw * gm
        pf = pf / (pf.sum(dim=2, keepdim=True) + 1e-12)
        p = torch.einsum("bf,tdf->btd", f_aug, pf).clamp(1e-6, 1 - 1e-6)
        probs = x.new_ones(B, self.T, 1)
        for d in range(self.depth):
            pd = p[:, :, d].unsqueeze(-1)
            probs = torch.cat([probs * (1.0 - pd), probs * pd], dim=2)
        if self.output_dim == 1:
            return (probs * self.leaf_value[None, :, :]).sum(dim=2).sum(dim=1)
        return torch.einsum("btl,tlc->btc", probs, self.leaf_value).sum(dim=1)

# -----------------------
# Scorecard + rule correction
# -----------------------
class Scorecard(nn.Module):
    """Soft-binned scorecard model (piecewise constant per feature)."""
    def __init__(self, input_dim: int, n_bins: int = 8, init_temp: float = 0.35, output_dim: int = 1):
        super().__init__()
        self.output_dim = int(output_dim)
        self.D = int(input_dim)
        self.K = int(n_bins)
        self.centers = nn.Parameter(torch.zeros(self.D, self.K))
        self.log_temp = nn.Parameter(torch.full((self.D,), float(math.log(init_temp))))
        if self.output_dim == 1:
            self.w = nn.Parameter(torch.zeros(self.D, self.K))
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.w = nn.Parameter(torch.zeros(self.D, self.K, self.output_dim))
            self.bias = nn.Parameter(torch.zeros(self.output_dim))
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float32)
        for j in range(self.D):
            qs = np.linspace(0.05, 0.95, self.K, dtype=np.float32)
            self.centers[j].copy_(torch.tensor(np.quantile(X[:, j], qs), device=self.centers.device))
        nn.init.normal_(self.w, std=0.01)
        self.bias.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temp = torch.exp(self.log_temp).clamp(0.05, 5.0)
        logits = -(x[:, :, None] - self.centers[None, :, :]).abs() / temp[None, :, None]
        b = torch.softmax(logits, dim=2)
        if self.output_dim == 1:
            return (b * self.w[None, :, :]).sum(dim=2).sum(dim=1) + self.bias
        return (b[:, :, :, None] * self.w[None, :, :, :]).sum(dim=2).sum(dim=1) + self.bias[None, :]

class ScorecardWithRules(nn.Module):
    """Scorecard baseline + evidence rule correction term."""
    def __init__(
        self,
        input_dim: int,
        n_bins: int = 8,
        n_rules: int = 256,
        init_kappa: float = 6.0,
        beta_rules: float = 6.0,
        selector: str = "sparsemax",
        output_dim: int = 1,
        att_scale: float = 2.0,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.selector = str(selector)
        self.att_scale = float(att_scale)
        self.scorecard = Scorecard(input_dim, n_bins=n_bins, output_dim=output_dim)
        self.rules = SignedEvidenceRuleLayer(input_dim, n_rules=n_rules, init_kappa=init_kappa, beta=beta_rules)
        self.mask_logit = self.rules.mask_logit
        self.priority = nn.Parameter(torch.zeros(n_rules))
        if self.output_dim == 1:
            self.v = nn.Parameter(torch.zeros(n_rules))
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.v = nn.Parameter(torch.zeros(n_rules, self.output_dim))
            self.bias = nn.Parameter(torch.zeros(self.output_dim))
    
    def _sel(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return sparsemax(logits, dim=dim) if self.selector == "sparsemax" else torch.softmax(logits, dim=dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        self.scorecard.init_from_data(X)
        self.rules.init_from_data(X)
        nn.init.normal_(self.priority, std=0.01)
        nn.init.normal_(self.v, std=0.01)
        self.bias.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_sc = self.scorecard(x)
        z = self.rules(x).clamp(0.0, 1.0)  # [B,R]
        a = self._sel(self.priority[None, :] + self.att_scale * z, dim=1)  # [B,R]
        if self.output_dim == 1:
            corr = (a * self.v[None, :]).sum(dim=1) + self.bias
            return y_sc + corr
        corr = a @ self.v + self.bias[None, :]
        return y_sc + corr

# -----------------------
# Budgeted interaction forest
# -----------------------
class BudgetedForest(nn.Module):
    """Forest with explicit feature budget per tree (top-k feature selection)."""
    def __init__(
        self,
        input_dim: int,
        k_features: int = 3,
        n_trees: int = 32,
        depth: int = 4,
        n_thresh_per_feat: int = 6,
        selector: str = "sparsemax",
        output_dim: int = 1,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.D = int(input_dim)
        self.k_features = int(k_features)
        self.T = int(n_trees)
        self.depth = int(depth)
        self.selector = str(selector)
        self.facts = ThresholdFactBank(self.D, n_thresh_per_feat=n_thresh_per_feat)
        self.n_thresh_per_feat = int(n_thresh_per_feat)
        F0 = self.facts.num_facts
        self.Fin = 2 * F0
        self.sel_logits = nn.Parameter(0.01 * torch.randn(self.T * self.depth, self.Fin))
        self.feat_logits = nn.Parameter(0.01 * torch.randn(self.T, self.D))
        self.register_buffer("feat_to_fact_mask", build_feature_to_fact_mask(self.D, self.n_thresh_per_feat))  # [D,Fin]
        L = 2**self.depth
        if self.output_dim == 1:
            self.leaf_value = nn.Parameter(torch.zeros(self.T, L))
        else:
            self.leaf_value = nn.Parameter(torch.zeros(self.T, L, self.output_dim))
    
    def _sel(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return sparsemax(logits, dim=dim) if self.selector == "sparsemax" else torch.softmax(logits, dim=dim)
    
    @torch.no_grad()
    def init_from_data(self, X: np.ndarray) -> None:
        self.facts.init_from_data_quantiles(X)
        nn.init.normal_(self.sel_logits, std=0.01)
        nn.init.normal_(self.feat_logits, std=0.01)
        nn.init.normal_(self.leaf_value, std=0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        f = self.facts(x)
        f_aug = torch.cat([f, 1.0 - f], dim=1)
        feat_mask = straight_through_topk_mask(self.feat_logits, k=self.k_features, dim=1)  # [T,D]
        allowed_fact = (feat_mask @ self.feat_to_fact_mask).clamp_min(1e-6)                 # [T,Fin]
        raw = self.sel_logits.view(self.T, self.depth, self.Fin)
        masked_logits = raw + torch.log(allowed_fact[:, None, :] + 1e-12)
        sel = self._sel(masked_logits.reshape(self.T * self.depth, self.Fin), dim=1).view(self.T, self.depth, self.Fin)
        p = torch.einsum("bf,tlf->btl", f_aug, sel).clamp(1e-6, 1 - 1e-6)
        probs = x.new_ones(B, self.T, 1)
        for d in range(self.depth):
            pd = p[:, :, d].unsqueeze(-1)
            probs = torch.cat([probs * (1.0 - pd), probs * pd], dim=2)
        if self.output_dim == 1:
            return (probs * self.leaf_value[None, :, :]).sum(dim=2).sum(dim=1)
        return torch.einsum("btl,tlc->btc", probs, self.leaf_value).sum(dim=1)