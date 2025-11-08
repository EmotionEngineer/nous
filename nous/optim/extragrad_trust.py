from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
import numpy as np
from typing import Dict, Iterable, Optional, Tuple
from .base import BaseOptimizer
from .mirror_radam import MirrorRAdam
from ..model import NousNet

def row_kl_div(p_new: torch.Tensor, p_old: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute KL divergence between two probability distributions row-wise."""
    p_new = torch.clamp(p_new, eps, 1.0)
    p_old = torch.clamp(p_old, eps, 1.0)
    return (p_new * (torch.log(p_new) - torch.log(p_old))).sum(dim=1)

def budget_push_direction(P: torch.Tensor, k_target: int, lam: float = 1e-3) -> torch.Tensor:
    """Compute direction to push towards target budget k_target."""
    B, R = P.shape
    over = P.sum(dim=1).mean() - float(k_target)
    return lam * over * torch.ones_like(P)

class ExtragradRAdamTrust(MirrorRAdam):
    """Extragradient RAdam optimizer with KL trust region for gate parameters.
    Combines extragradient steps with KL trust region constraints for stable
    optimization of rule selection gates.
    Args:
        params: Iterable of parameters to optimize
        model: The model being optimized (required for parameter categorization)
        lr: Learning rate for non-gate parameters (default: 1e-3)
        lr_gate: Learning rate for gate parameters (default: 1e-2)
        betas: Coefficients used for computing running averages of gradient and squared gradient (default: (0.9, 0.999))
        eps: Term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0)
        delta_kl: Maximum KL divergence allowed between updates (default: 1e-2)
        backtrack: Backtracking factor for KL constraint (default: 0.5)
        lam_budget: Budget constraint coefficient (default: 1e-3)
        k_target: Target number of active rules (default: 8)
        eta_pred: Prediction step size for extragradient (default: 5e-3)
    """
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        model: torch.nn.Module,
        lr: float = 1e-3,
        lr_gate: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        delta_kl: float = 1e-2,
        backtrack: float = 0.5,
        lam_budget: float = 1e-3,
        k_target: int = 8,
        eta_pred: float = 5e-3,
    ):
        super().__init__(params, model, lr, lr_gate, betas, eps, weight_decay)
        self.delta_kl = delta_kl
        self.backtrack = backtrack
        self.lam_budget = lam_budget
        self.k_target = k_target
        self.eta_pred = eta_pred

    def step(self, closure=None, model: Optional[NousNet] = None):
        """Performs a single optimization step with extragradient and trust region."""
        loss = None
        if closure is not None:
            loss = closure()
        if model is None:
            model = self.model

        # First, save current gate parameter values
        saved_gate_params = {}
        for name, p in model.named_parameters():
            if p.requires_grad and self._is_gate_param(name) and p.grad is not None and any(p is param for param in self.gate_params):
                saved_gate_params[name] = p.data.clone()

        # First pass: prediction step for extragradient
        for name, p in model.named_parameters():
            if p.requires_grad and self._is_gate_param(name) and p.grad is not None and any(p is param for param in self.gate_params):
                if p.dim() == 2:
                    P = F.softmax(p.data, dim=1)
                    P_tilde = P * torch.exp(-self.eta_pred * p.grad)
                    P_tilde = P_tilde / (P_tilde.sum(dim=1, keepdim=True) + 1e-8)
                    logits_tilde = torch.log(torch.clamp(P_tilde, 1e-8, 1.0))
                    logits_tilde -= logits_tilde.mean(dim=1, keepdim=True)
                    p.data.copy_(logits_tilde)
                else:
                    p.data.add_(-self.eta_pred * p.grad)

        # Second pass: compute gradients at predicted point
        if closure is not None:
            closure()

        # Restore original gate parameters
        for name, p in model.named_parameters():
            if name in saved_gate_params and any(p is param for param in self.gate_params):
                p.data.copy_(saved_gate_params[name])

        # Now perform the actual update with trust region constraints
        super().step(closure)

        # Apply KL trust region constraints to gate parameters
        # betas/eps from group0; lr_gate comes from defaults (gate params aren’t in param_groups)
        group0 = self.param_groups[0] if self.param_groups else {}
        beta1, beta2 = group0.get('betas', (0.9, 0.999))
        eps = group0.get('eps', 1e-8)
        lr_gate = self.defaults.get('lr_gate', group0.get('lr', 1e-2))

        for name, p in model.named_parameters():
            if p.grad is None or not any(p is param for param in self.gate_params) or not self._is_gate_param(name):
                continue

            # Optimizer state keyed by parameter tensor
            state = self.state[p]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
            state['step'] += 1
            step = state['step']
            grad = p.grad.data

            # Update moving averages
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # Bias correction
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            rt, rectified = self._radam_rectifier(step, beta2)

            # Compute step
            if rectified:
                step_size = lr_gate * rt * np.sqrt(bias_correction2) / bias_correction1
                denom = (exp_avg_sq.sqrt() / np.sqrt(bias_correction2)).add_(eps)
                step_vec = (exp_avg / denom) * step_size
            else:
                step_size = lr_gate / bias_correction1
                denom = (exp_avg_sq.sqrt() / np.sqrt(bias_correction2)).add_(eps)
                step_vec = (exp_avg / denom) * step_size

            # Apply trust region constraints
            if p.dim() == 2:
                P_old = F.softmax(p.data, dim=1)
                # Add budget correction
                step_vec = step_vec + budget_push_direction(P_old, k_target=self.k_target, lam=self.lam_budget)
                # Apply backtracking line search with KL constraint
                alpha = 1.0
                while alpha > 1e-4:
                    P_new = P_old * torch.exp(-alpha * step_vec)
                    P_new = P_new / (P_new.sum(dim=1, keepdim=True) + 1e-8)
                    kl = row_kl_div(P_new, P_old)
                    if torch.all(kl <= self.delta_kl):
                        logits_new = torch.log(torch.clamp(P_new, 1e-8, 1.0))
                        logits_new -= logits_new.mean(dim=1, keepdim=True)
                        p.data.copy_(logits_new)
                        break
                    alpha *= self.backtrack
                else:
                    # If backtracking fails, use a small step
                    P_new = P_old * torch.exp(-1e-4 * step_vec)
                    P_new = P_new / (P_new.sum(dim=1, keepdim=True) + 1e-8)
                    logits_new = torch.log(torch.clamp(P_new, 1e-8, 1.0))
                    logits_new -= logits_new.mean(dim=1, keepdim=True)
                    p.data.copy_(logits_new)
            else:
                p.data.add_(-step_vec)

            # Zero the gradient after update
            if p.grad is not None:
                p.grad.zero_()

        return loss