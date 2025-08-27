# nous/layers.py
import torch
import torch.nn as nn
from typing import List, Tuple

# --- Fact Layers ---

class ExhaustiveAtomicFactLayer(nn.Module):
    """Generates atomic facts by exhaustively comparing all pairs of features."""
    def __init__(self, input_dim: int, feature_names: List[str]):
        super().__init__()
        if input_dim > 20:
            num_facts = input_dim * (input_dim - 1) // 2
            print(f"Warning: ExhaustiveAtomicFactLayer with {input_dim} features will create {num_facts} facts. This may be slow and memory-intensive.")
        self.indices = torch.combinations(torch.arange(input_dim), r=2)
        self.thresholds = nn.Parameter(torch.randn(self.indices.shape[0]) * 0.1)
        self.steepness = nn.Parameter(torch.ones(self.indices.shape[0]) * 5.0)
        self.fact_names = [f"({feature_names[i]} > {feature_names[j]})" for i, j in self.indices.numpy()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        steepness = torch.nn.functional.softplus(self.steepness) + 1e-4
        diffs = x[:, self.indices[:, 0]] - x[:, self.indices[:, 1]]
        return torch.sigmoid(steepness * (diffs - self.thresholds))

    @property
    def output_dim(self) -> int: return len(self.fact_names)

class LearnedAtomicFactLayer(nn.Module):
    """Base class for learnable fact layers (Sigmoid and Beta)."""
    def __init__(self, input_dim: int, num_facts: int, feature_names: List[str]):
        super().__init__()
        self.input_dim = input_dim
        self.num_facts = num_facts
        self._feature_names = feature_names
        self.projection_left = nn.Linear(input_dim, num_facts, bias=False)
        self.projection_right = nn.Linear(input_dim, num_facts, bias=False)
        self.thresholds = nn.Parameter(torch.randn(num_facts) * 0.1)

    @property
    def output_dim(self) -> int: return self.num_facts
        
    def get_base_diffs(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates the core difference term for all activation functions."""
        return (self.projection_left(x) - self.projection_right(x)) - self.thresholds

    def fact_names(self, prefix: str) -> List[str]:
        """Generates human-readable and unique names for facts."""
        names = []
        with torch.no_grad():
            w_left, w_right = self.projection_left.weight, self.projection_right.weight
            for i in range(self.output_dim):
                left_name = self._feature_names[w_left[i].abs().argmax().item()]
                right_name = self._feature_names[w_right[i].abs().argmax().item()]
                base_name = f"({left_name} vs {right_name})" if left_name != right_name else f"Thresh({left_name})"
                names.append(f"{prefix}-{i}{base_name}")
        return names

# --- Specialized Learnable Fact Layers ---

class SigmoidFactLayer(LearnedAtomicFactLayer):
    """A learnable fact layer using the standard sigmoid activation."""
    def __init__(self, input_dim: int, num_facts: int, feature_names: List[str]):
        super().__init__(input_dim, num_facts, feature_names)
        self.steepness = nn.Parameter(torch.ones(num_facts) * 5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diffs = self.get_base_diffs(x)
        steepness = torch.nn.functional.softplus(self.steepness) + 1e-4
        return torch.sigmoid(steepness * diffs)

    @property
    def fact_names(self) -> List[str]: return super().fact_names(prefix="Sigmoid")

class BetaFactLayer(LearnedAtomicFactLayer):
    """A learnable fact layer using a flexible, generalized logistic function."""
    def __init__(self, input_dim: int, num_facts: int, feature_names: List[str]):
        super().__init__(input_dim, num_facts, feature_names)
        self.k_raw = nn.Parameter(torch.ones(num_facts) * 0.5)
        self.nu_raw = nn.Parameter(torch.zeros(num_facts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diffs = self.get_base_diffs(x)
        k = torch.nn.functional.softplus(self.k_raw) + 1e-4
        nu = torch.nn.functional.softplus(self.nu_raw) + 1e-4
        return (1 + torch.exp(-k * diffs)) ** (-nu)
        
    @property
    def fact_names(self) -> List[str]: return super().fact_names(prefix="Beta")

# --- Rule/Concept Layers ---

class LogicalRuleLayer(nn.Module):
    """Forms logical rules (AND) from facts and outputs higher-level concepts."""
    def __init__(self, input_dim: int, num_rules: int, input_fact_names: List[str]):
        super().__init__()
        torch.manual_seed(input_dim + num_rules)
        
        if input_dim > 0 and num_rules > 0:
            self.register_buffer('rule_indices', torch.randint(0, input_dim, size=(num_rules, 2)))
            self.rule_names = [f"({input_fact_names[i]} AND {input_fact_names[j]})" for i, j in self.rule_indices]
        else:
            self.register_buffer('rule_indices', torch.empty(0, 2, dtype=torch.long))
            self.rule_names = []
            
        self.concept_generator = nn.Linear(num_rules, num_rules)
        self.concept_names = [f"Concept-{i}" for i in range(num_rules)]

    def forward(self, facts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rule_indices.shape[0] == 0:
            return torch.zeros(facts.shape[0], 0).to(facts.device), torch.zeros(facts.shape[0], 0).to(facts.device)
        fact1, fact2 = facts[:, self.rule_indices[:, 0]], facts[:, self.rule_indices[:, 1]]
        rule_activations = fact1 * fact2
        concepts = torch.sigmoid(self.concept_generator(rule_activations))
        return concepts, rule_activations

    @property
    def output_dim(self) -> int: return len(self.concept_names)
