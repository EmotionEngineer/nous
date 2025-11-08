from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
from .base import BaseOptimizer
from ..model import NousNet

class SharpnessAwareGates(BaseOptimizer):
    """Sharpness-Aware Minimization (SAM) optimizer for rule gate parameters.
    Applies SAM perturbation only to gate logits in KL geometry (mirror step).
    Args:
        params: Iterable of parameters to optimize
        model: The model being optimized (required for parameter categorization)
        lr: Learning rate (default: 1e-3)
        rho: SAM perturbation radius (default: 0.05)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0)
        clip_grad: Maximum gradient norm for clipping (default: 0.5)
    """
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        model: torch.nn.Module,
        lr: float = 1e-3,
        rho: float = 0.05,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        clip_grad: float = 0.5,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if rho <= 0.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if clip_grad <= 0.0:
            raise ValueError(f"Invalid clip_grad value: {clip_grad}")
        defaults = dict(
            lr=lr,
            rho=rho,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            clip_grad=clip_grad,
            step=0
        )
        super().__init__(params, defaults, model=model)
        # Create parameter group for non-gate parameters
        self.param_groups = [
            {
                'params': self.nongate_params,
                'lr': lr,
                'betas': betas,
                'eps': eps,
                'weight_decay': weight_decay,
                'clip_grad': clip_grad
            }
        ]

    def step(self, closure=None, model: Optional[NousNet] = None):
        """Performs a single SAM optimization step for gate parameters."""
        if closure is None:
            raise ValueError("SharpnessAwareGates requires a closure")
        if model is None:
            model = self.model

        # Step 1: First forward-backward pass → collect grads
        loss = closure()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.defaults['clip_grad'])

        # Save original gate parameters AND gate gradients (critical!)
        original_params = {}
        gate_grads = {}
        for name, p in model.named_parameters():
            if p.requires_grad and self._is_gate_param(name) and any(p is param for param in self.gate_params):
                original_params[name] = p.data.clone()
                if p.grad is not None:
                    gate_grads[name] = p.grad.detach().clone()  # ← fix: cache before zeroing

        # Step 2: Compute SAM perturbation for gate parameters (using cached grads)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name not in original_params or name not in gate_grads:
                    continue
                rho = self.defaults['rho']
                grad = gate_grads[name]
                if p.dim() != 2:
                    # Non-softmax gates (treated as batch=1)
                    pv = p.data.unsqueeze(0)
                    gv = grad.unsqueeze(0)
                    P = F.softmax(pv, dim=1)
                    P_perturbed = P * torch.exp(rho * gv)
                    P_perturbed = P_perturbed / (P_perturbed.sum(dim=1, keepdim=True) + 1e-8)
                    logits = torch.log(torch.clamp(P_perturbed, 1e-8, 1.0))
                    p.data.copy_(logits.squeeze(0))
                else:
                    # Softmax gates (row-wise)
                    P = F.softmax(p.data, dim=1)
                    P_perturbed = P * torch.exp(rho * grad)
                    P_perturbed = P_perturbed / (P_perturbed.sum(dim=1, keepdim=True) + 1e-8)
                    logits = torch.log(torch.clamp(P_perturbed, 1e-8, 1.0))
                    logits -= logits.mean(dim=1, keepdim=True)
                    p.data.copy_(logits)

        # NOW zero grads (after perturbation is applied)
        model.zero_grad(set_to_none=True)

        # Step 3: Second forward-backward at perturbed point (fresh grads used for actual update)
        loss = closure()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.defaults['clip_grad'])

        # Step 4: Restore original gate parameters before update
        for name, p in model.named_parameters():
            if name in original_params and any(p is param for param in self.gate_params):
                p.data.copy_(original_params[name])

        # Update non-gate parameters with AdamW (unchanged)
        non_gate_group = self.param_groups[0]
        lr = non_gate_group['lr']
        weight_decay = non_gate_group['weight_decay']
        beta1, beta2 = non_gate_group['betas']
        eps = non_gate_group['eps']
        clip_grad = non_gate_group['clip_grad']
        step = non_gate_group.get('step', 0) + 1
        non_gate_group['step'] = step

        for p in non_gate_group['params']:
            if p.grad is None:
                continue
            grad = p.grad.data
            state = self.state[p]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
            state['step'] += 1
            step_t = state['step']
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

            # Apply weight decay
            if weight_decay != 0:
                grad = grad.add(p.data, alpha=weight_decay)

            # Mom update
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** step_t
            bias_correction2 = 1 - beta2 ** step_t
            step_size = lr * np.sqrt(bias_correction2) / bias_correction1
            denom = (exp_avg_sq.sqrt() / np.sqrt(bias_correction2)).add_(eps)
            p.data.addcdiv_(exp_avg, denom, value=-step_size)

        # Update gate parameters with mirror descent — now using lr (not rho!)
        for name, p in model.named_parameters():
            if p.grad is None or not any(p is param for param in self.gate_params) or not self._is_gate_param(name):
                continue

            grad = p.grad.data
            state = self.state[p]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
            state['step'] += 1
            step_t = state['step']
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']

            # Update moving averages
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # Bias correction
            bias_correction1 = 1 - beta1 ** step_t
            bias_correction2 = 1 - beta2 ** step_t

            # ← FIX: lr (learning rate), not rho (perturbation radius)
            step_size = lr / bias_correction1
            denom = (exp_avg_sq.sqrt() / np.sqrt(bias_correction2)).add_(eps)
            step_vec = (exp_avg / denom) * step_size

            # Mirror descent in KL geometry
            if p.dim() == 2:
                P_old = F.softmax(p.data, dim=1)
                P_new = P_old * torch.exp(-step_vec)
                P_new = P_new / (P_new.sum(dim=1, keepdim=True) + 1e-8)
                logits_new = torch.log(torch.clamp(P_new, 1e-8, 1.0))
                logits_new -= logits_new.mean(dim=1, keepdim=True)
                p.data.copy_(logits_new)
            else:
                p.data.add_(-step_vec)

            # Zero gradients for this gate param
            if p.grad is not None:
                p.grad.zero_()

        return loss