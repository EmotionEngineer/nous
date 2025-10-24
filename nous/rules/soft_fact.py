from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SoftFactRuleLayer(nn.Module):
    """
    Soft fact selection via temperature-controlled softmax gating.
    Enables gradient flow through fact selection weights.
    """
    def __init__(
        self,
        input_dim: int,
        num_rules: int,
        top_k_facts: int = 2,
        top_k_rules: int = 8,
        fact_temperature: float = 0.7,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_rules = num_rules
        self.top_k_facts = top_k_facts
        self.top_k_rules = top_k_rules
        self.fact_temperature = fact_temperature

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
        mask = self._soft_k_mask()
        facts_exp = facts.unsqueeze(1)
        mask_exp = mask.unsqueeze(0)
        selected = facts_exp * mask_exp

        and_agg = torch.prod(selected + (1.0 - mask_exp), dim=2)
        or_agg = 1.0 - torch.prod(1.0 - selected + 1e-8, dim=2)
        denom = mask_exp.sum(dim=2) + 1e-8
        k_of_n_agg = selected.sum(dim=2) / denom

        agg_weights = F.softmax(self.aggregator_logits, dim=1)
        aggregators = torch.stack([and_agg, or_agg, k_of_n_agg], dim=2)
        mixed_agg = (aggregators * agg_weights.unsqueeze(0)).sum(dim=2)

        rule_strength = torch.sigmoid(self.rule_strength_raw)
        rule_activations = mixed_agg * rule_strength.unsqueeze(0)

        pre_for_topk = rule_activations.clone()
        if restrict_mask is not None:
            pre_for_topk = pre_for_topk + (restrict_mask - 1) * 1e9
        if drop_rule_idx is not None:
            pre_for_topk[:, drop_rule_idx] = -1e9

        k = min(self.top_k_rules, self.num_rules)
        _, topk_rule_idx = torch.topk(pre_for_topk, k=k, dim=1)
        gate_mask = torch.zeros_like(rule_activations)
        gate_mask.scatter_(1, topk_rule_idx, 1.0)

        if restrict_mask is not None:
            gate_mask = gate_mask * restrict_mask.unsqueeze(0).to(gate_mask.dtype)
        if drop_rule_idx is not None:
            gate_mask[:, drop_rule_idx] = 0.0

        gated_activations = rule_activations * gate_mask

        if prune_below is not None:
            keep = (gated_activations.abs() >= prune_below).float()
            gated_activations = gated_activations * keep
            gate_mask = gate_mask * keep

        proj_contrib = self.proj(facts) if not isinstance(self.proj, nn.Identity) else facts
        pre_sum = gated_activations if explain_exclude_proj else (proj_contrib + gated_activations)
        output = pre_sum if explain_disable_norm else self.norm(pre_sum)

        if return_details:
            details = {
                "pre_rule_activations": rule_activations.detach(),
                "gated_activations": gated_activations.detach(),
                "gate_mask": gate_mask.detach(),
                "aggregator_weights": agg_weights.detach(),
                "selected_indices": topk_rule_idx.detach(),
                "fact_weights": mask.detach(),
                "pre_norm_sum": pre_sum.detach(),
                "proj_contrib": proj_contrib.detach(),
            }
            return output, details
        return output