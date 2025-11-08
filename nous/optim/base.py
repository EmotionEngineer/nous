from __future__ import annotations
import torch
from torch.optim import Optimizer
from typing import Dict, Iterable, Optional, Tuple

class BaseOptimizer(Optimizer):
    """Base class for Nous-specific optimizers with parameter grouping."""
    
    def __init__(self, params, defaults, model=None):
        super().__init__(params, defaults)
        self.gate_params = []
        self.nongate_params = []
        self.model = model
        self._categorize_parameters()
    
    def _categorize_parameters(self):
        """Categorize parameters into gate and non-gate groups."""
        self.gate_params = []
        self.nongate_params = []

        # Collect all parameters managed by this optimizer (across groups)
        managed_params = set()
        for group in self.param_groups:
            for p in group['params']:
                managed_params.add(p)

        if self.model is not None:
            # Use model's named parameters to categorize only the params handled by this optimizer
            for name, param in self.model.named_parameters():
                if param in managed_params and param.requires_grad:
                    if self._is_gate_param(name):
                        self.gate_params.append(param)
                    else:
                        self.nongate_params.append(param)
        else:
            # Fallback: check parameter names if available
            for p in managed_params:
                name = getattr(p, '_name', None) or getattr(p, 'name', None)
                if name and self._is_gate_param(name):
                    self.gate_params.append(p)
                else:
                    self.nongate_params.append(p)
    
    def _is_gate_param(self, name: str) -> bool:
        """Check if parameter is a gate parameter."""
        if not name:
            return False
        return ("fact_logits" in name) or ("aggregator_logits" in name) or ("rule_strength_raw" in name)

    def zero_grad(self, set_to_none: bool = False):
        # Clear grads for params in registered groups
        super().zero_grad(set_to_none=set_to_none)
        # Also clear grads for gate params that are not in param_groups
        for p in self.gate_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_()
                    p.grad.zero_()
    
    def step(self, closure=None):
        """Performs a single optimization step."""
        raise NotImplementedError("Subclasses must implement step()")