from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .gaters import BaseRuleGater

class SoftFactRuleLayer(nn.Module):
    """
    Soft fact selection via temperature-controlled softmax gating.
    Enables gradient flow through fact selection weights.
    Optionally supports a differentiable rule gater (per-sample gating over rules).
    """
    def __init__(
        self,
        input_dim: int,
        num_rules: int,
        top_k_facts: int = 2,
        top_k_rules: int = 8,
        fact_temperature: float = 0.7,
        rule_gater: Optional[BaseRuleGater] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_rules = num_rules
        self.top_k_facts = top_k_facts
        self.top_k_rules = top_k_rules
        self.fact_temperature = fact_temperature
        self.rule_gater = rule_gater

        self.fact_logits = nn.Parameter(torch.randn(num_rules, input_dim) * 0.01)
        self.num_aggregators = 3
        self.aggregator_logits = nn.Parameter(torch.zeros(num_rules, self.num_aggregators))
        self.rule_strength_raw = nn.Parameter(torch.zeros(num_rules))
        self.proj = nn.Linear(input_dim, num_rules, bias=False) if input_dim != num_rules else nn.Identity()
        self.norm = nn.LayerNorm(num_rules)

    def _soft_k_mask(self) -> torch.Tensor:
        p = F.softmax(self.fact_logits / self.fact_temperature, dim=1)
        s = self.top_k_facts * p
        return torch.clamp(s, max=1.0)

    def forward(
        self,
        facts: torch.Tensor,
        return_details: bool = False,
        drop_rule_idx: Optional[int] = None,
        restrict_mask: Optional[torch.Tensor] = None,
        prune_below: Optional[float] = None,
        explain_disable_norm: bool = False,
        explain_exclude_proj: bool = False,
    ):
        # Soft mask over facts
        mask = self._soft_k_mask()                          # [R, F]
        facts_exp = facts.unsqueeze(1)                      # [B, 1, F]
        mask_exp = mask.unsqueeze(0)                        # [1, R, F]
        selected = facts_exp * mask_exp                     # [B, R, F]
    
        # Aggregators
        and_agg = torch.prod(selected + (1.0 - mask_exp), dim=2)                 # [B, R]
        or_agg = 1.0 - torch.prod(1.0 - selected + 1e-8, dim=2)                  # [B, R]
        denom = mask_exp.sum(dim=2) + 1e-8                                       # [1, R]
        k_of_n_agg = selected.sum(dim=2) / denom                                  # [B, R]
    
        agg_weights = F.softmax(self.aggregator_logits, dim=1)                    # [R, A]
        aggregators = torch.stack([and_agg, or_agg, k_of_n_agg], dim=2)           # [B, R, 3]
        mixed_agg = (aggregators * agg_weights.unsqueeze(0)).sum(dim=2)           # [B, R]
    
        # Rule strength and activations
        rule_strength = torch.sigmoid(self.rule_strength_raw)                     # [R]
        rule_activations = mixed_agg * rule_strength.unsqueeze(0)                 # [B, R]
    
        # Rule gating: either differentiable gater or legacy per-sample hard top-k
        if self.rule_gater is None:
            # Legacy hard top-k gating (default, backward-compatible)
            pre_for_topk = rule_activations.clone()
            if restrict_mask is not None:
                pre_for_topk = pre_for_topk + (restrict_mask - 1) * 1e9
            if drop_rule_idx is not None:
                pre_for_topk[:, drop_rule_idx] = -1e9
    
            k = min(self.top_k_rules, self.num_rules)
            _, topk_rule_idx = torch.topk(pre_for_topk, k=k, dim=1)                   # [B, k]
            gate_mask = torch.zeros_like(rule_activations)                            # [B, R]
            gate_mask.scatter_(1, topk_rule_idx, 1.0)
    
            if restrict_mask is not None:
                gate_mask = gate_mask * restrict_mask.unsqueeze(0).to(gate_mask.dtype)
            if drop_rule_idx is not None:
                gate_mask[:, drop_rule_idx] = 0.0
    
            gated_activations = rule_activations * gate_mask                          # [B, R]
        else:
            # Differentiable gating over rules; returns continuous mask in [0,1]
            gate_mask = self.rule_gater(rule_activations, k=self.top_k_rules, mask=restrict_mask)  # [B, R]
            if drop_rule_idx is not None:
                gate_mask = gate_mask.clone()
                gate_mask[:, drop_rule_idx] = 0.0
            gated_activations = rule_activations * gate_mask
    
        # Optional pruning
        if prune_below is not None:
            keep = (gated_activations.abs() >= prune_below).float()
            gated_activations = gated_activations * keep
            gate_mask = gate_mask * keep
    
        # Residual/projection and normalization
        proj_contrib = self.proj(facts) if not isinstance(self.proj, nn.Identity) else facts
        pre_sum = gated_activations if explain_exclude_proj else (proj_contrib + gated_activations)
        output = pre_sum if explain_disable_norm else self.norm(pre_sum)
    
        if return_details:
            with torch.no_grad():
                _, topk_fact_idx = torch.topk(mask, k=min(self.top_k_facts, self.input_dim), dim=1)  # [R, k]
                # For selected_indices, use top-k by the (soft/hard) gate mask consistently
                sel_k = min(self.top_k_rules, self.num_rules)
                _, sel_idx = torch.topk(gate_mask, k=sel_k, dim=1)
    
            details = {
                "pre_rule_activations": rule_activations.detach(),
                "gated_activations": gated_activations.detach(),
                "gate_mask": gate_mask.detach(),
                "aggregator_weights": agg_weights.detach(),
                "selected_indices": sel_idx.detach(),
                "fact_weights": mask.detach(),
                "facts_used": topk_fact_idx.detach(),
                "pre_norm_sum": pre_sum.detach(),
                "proj_contrib": proj_contrib.detach(),
            }
            return output, details
    
        return output