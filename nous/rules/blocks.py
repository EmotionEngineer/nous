from __future__ import annotations
import torch
import torch.nn as nn
from .fixed import FixedPairRuleLayer

class SimpleNousBlock(nn.Module):
    """
    Residual block over FixedPairRuleLayer with honest interventions and explain flags.
    Options:
    - drop_rule_idx: zero a single rule pre-gating
    - restrict_mask: 0/1 mask over rules (frozen active set)
    - prune_below: threshold prune post-gating
    - explain_disable_norm / explain_exclude_proj for clean contributions
    """
    def __init__(self, in_dim: int, n_rules: int) -> None:
        super().__init__()
        self.rule = FixedPairRuleLayer(in_dim, n_rules)
        self.proj = nn.Identity() if in_dim == n_rules else nn.Linear(in_dim, n_rules, bias=False)
        self.norm = nn.LayerNorm(n_rules)
    
    def forward(
        self,
        x: torch.Tensor,
        return_details: bool = False,
        drop_rule_idx: int | None = None,
        restrict_mask: torch.Tensor | None = None,
        prune_below: float | None = None,
        explain_disable_norm: bool = False,
        explain_exclude_proj: bool = False,
        detach_details: bool = True
    ):
        """
        Added detach_details parameter to control whether details tensors are detached.
        """
        rule_activations = self.rule(x)
        
        if restrict_mask is not None:
            rule_activations = rule_activations * restrict_mask
        
        if drop_rule_idx is not None:
            rule_activations[:, drop_rule_idx] = 0.0
        
        gated_activations = rule_activations
        
        if prune_below is not None:
            mask = (gated_activations.abs() >= prune_below).float()
            gated_activations = gated_activations * mask
        
        proj_contrib = torch.zeros_like(gated_activations)
        if not isinstance(self.proj, nn.Identity):
            proj_contrib = self.proj(x)
        
        pre_sum = (0.0 * proj_contrib) + gated_activations if explain_exclude_proj else (proj_contrib + gated_activations)
        output = pre_sum if explain_disable_norm else self.norm(pre_sum)
        
        if return_details:
            details = {
                "pre_rule_activations": rule_activations.detach() if detach_details else rule_activations,
                "gated_activations": gated_activations.detach() if detach_details else gated_activations,
                "gate_mask": torch.ones_like(gated_activations, dtype=torch.float32),
                "aggregator_weights": None,
                "selected_indices": torch.arange(gated_activations.shape[1], device=x.device).unsqueeze(0).repeat(x.size(0), 1),
                "facts_used": self.rule.idx.detach(),
                "pre_norm_sum": pre_sum.detach() if detach_details else pre_sum,
                "proj_contrib": proj_contrib.detach() if detach_details else proj_contrib
            }
            return output, details
        return output