from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxRuleLayer(nn.Module):
    """
    Learnable fact connections via softmax gating with per-rule top-k and aggregator mixing.
    Aggregators: AND, OR, k-of-n. Honest interventions supported.
    """
    def __init__(self, input_dim: int, num_rules: int, top_k_facts: int = 2, top_k_rules: int = 8) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_rules = num_rules
        self.top_k_facts = top_k_facts
        self.top_k_rules = top_k_rules
        
        self.fact_logits = nn.Parameter(torch.randn(num_rules, input_dim) * 0.01)
        self.num_aggregators = 3
        self.aggregator_logits = nn.Parameter(torch.zeros(num_rules, self.num_aggregators))
        self.rule_strength_raw = nn.Parameter(torch.zeros(num_rules))
        
        self.proj = nn.Linear(input_dim, num_rules, bias=False) if input_dim != num_rules else nn.Identity()
        self.norm = nn.LayerNorm(num_rules)
    
    def forward(
        self,
        facts: torch.Tensor,
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
        fact_logits_soft = F.softmax(self.fact_logits, dim=1)
        _, topk_fact_idx = torch.topk(fact_logits_soft, k=min(self.top_k_facts, self.input_dim), dim=1)
        mask = torch.zeros_like(fact_logits_soft)
        mask.scatter_(1, topk_fact_idx, 1.0)
        
        facts_expanded = facts.unsqueeze(1)
        mask_expanded = mask.unsqueeze(0)
        selected_facts = facts_expanded * mask_expanded
        
        and_agg = torch.prod(selected_facts + (1 - mask_expanded), dim=2)
        or_agg = 1 - torch.prod((1 - selected_facts) * mask_expanded + (1 - mask_expanded), dim=2)
        k_of_n_agg = torch.sum(selected_facts, dim=2) / (mask_expanded.sum(dim=2) + 1e-8)
        
        agg_weights = F.softmax(self.aggregator_logits, dim=1)
        aggregators = torch.stack([and_agg, or_agg, k_of_n_agg], dim=2)
        mixed_agg = torch.sum(aggregators * agg_weights.unsqueeze(0), dim=2)
        
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
                "pre_rule_activations": rule_activations.detach() if detach_details else rule_activations,
                "gated_activations": gated_activations.detach() if detach_details else gated_activations,
                "gate_mask": gate_mask.detach() if detach_details else gate_mask,
                "aggregator_weights": agg_weights.detach(),
                "selected_indices": topk_rule_idx.detach(),
                "facts_used": topk_fact_idx.detach(),
                "pre_norm_sum": pre_sum.detach() if detach_details else pre_sum,
                "proj_contrib": proj_contrib.detach() if detach_details else proj_contrib
            }
            return output, details
        return output