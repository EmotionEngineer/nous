# nous/models.py
import torch
import torch.nn as nn
from typing import List, Literal, Tuple
from .layers import ExhaustiveAtomicFactLayer, SigmoidFactLayer, BetaFactLayer, LogicalRuleLayer

class NousBlock(nn.Module):
    """A single reasoning block in the Nous network with a residual connection."""
    def __init__(self, input_dim: int, num_rules: int, input_fact_names: List[str]):
        super().__init__()
        self.rule_layer = LogicalRuleLayer(input_dim, num_rules, input_fact_names)
        self.projection = nn.Linear(input_dim, num_rules) if input_dim != num_rules else nn.Identity()
        self.norm = nn.LayerNorm(num_rules)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        concepts, rule_activations = self.rule_layer(x)
        output = self.norm(self.projection(x) + concepts)
        return output, concepts, rule_activations
    
    @property
    def concept_names(self) -> List[str]:
        return self.rule_layer.concept_names

class NousNet(nn.Module):
    """
    The complete Nous neuro-symbolic network for regression and classification.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 feature_names: List[str],
                 fact_layer_type: Literal['beta', 'sigmoid', 'exhaustive'] = 'beta',
                 num_facts: int = 30,
                 num_rules_per_layer: List[int] = [10, 5]):
        super().__init__()
        self.feature_names = feature_names
        
        if fact_layer_type == 'beta':
            self.atomic_fact_layer = BetaFactLayer(input_dim, num_facts, feature_names)
        elif fact_layer_type == 'sigmoid':
            self.atomic_fact_layer = SigmoidFactLayer(input_dim, num_facts, feature_names)
        elif fact_layer_type == 'exhaustive':
            self.atomic_fact_layer = ExhaustiveAtomicFactLayer(input_dim, feature_names)
        else:
            raise ValueError("fact_layer_type must be 'beta', 'sigmoid', or 'exhaustive'")
            
        self.nous_blocks = nn.ModuleList()
        current_dim = self.atomic_fact_layer.output_dim
        
        for i, num_rules in enumerate(num_rules_per_layer):
            input_names = self.atomic_fact_layer.fact_names if i == 0 else self.nous_blocks[i-1].concept_names
            block = NousBlock(current_dim, num_rules, input_names)
            self.nous_blocks.append(block)
            current_dim = num_rules
            
        self.output_head = nn.Linear(current_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Returns logits for classification or direct values for regression.
        """
        h = self.atomic_fact_layer(x)
        for block in self.nous_blocks:
            h, _, _ = block(h)
        return self.output_head(h)
