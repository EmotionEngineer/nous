from __future__ import annotations
import torch
import torch.nn as nn

class FixedPairRuleLayer(nn.Module):
    """
    Fixed random fact pairs with AND logic. Pair indices are distinct per rule.
    Output: rule_strength * (f_i * f_j)
    """
    def __init__(self, input_dim: int, num_rules: int) -> None:
        super().__init__()
        idx1 = torch.randint(0, input_dim, (num_rules,))
        offset = torch.randint(1, input_dim, (num_rules,))
        idx2 = (idx1 + offset) % input_dim
        idx = torch.stack([idx1, idx2], dim=1)
        self.register_buffer('idx', idx)
        self.weight = nn.Parameter(torch.ones(num_rules))

    def forward(self, facts: torch.Tensor) -> torch.Tensor:
        f1, f2 = facts[:, self.idx[:,0]], facts[:, self.idx[:,1]]
        rule_strength = torch.sigmoid(self.weight)
        return rule_strength * (f1 * f2)

    @torch.no_grad()
    def get_rules(self):
        return self.idx.cpu().numpy(), torch.sigmoid(self.weight).cpu().numpy()