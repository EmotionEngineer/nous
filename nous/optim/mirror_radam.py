from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Dict, Iterable, Optional, Tuple
import numpy as np

from .base import BaseOptimizer

class MirrorRAdam(BaseOptimizer):
    """RAdam optimizer with mirror descent updates for gate parameters in KL geometry.
    
    This optimizer applies standard RAdam updates to non-gate parameters,
    while using a mirror descent step in the KL geometry for gate parameters.
    This approach stabilizes optimization of softmax-based gating mechanisms
    used in rule selection layers.
    
    Args:
        params: Iterable of parameters to optimize
        model: The model being optimized (required for parameter categorization)
        lr: Learning rate for non-gate parameters (default: 1e-3)
        lr_gate: Learning rate for gate parameters (default: 1e-2)
        betas: Coefficients used for computing running averages of gradient and squared gradient (default: (0.9, 0.999))
        eps: Term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0)
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
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if lr_gate <= 0.0:
            raise ValueError(f"Invalid gate learning rate: {lr_gate}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        
        defaults = dict(
            lr=lr,
            lr_gate=lr_gate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            step=0
        )
        super().__init__(params, defaults, model=model)
        
        # Create separate parameter groups for non-gate parameters
        self.param_groups = [
            {
                'params': self.nongate_params,
                'lr': lr,
                'weight_decay': weight_decay,
                'betas': betas,
                'eps': eps
            }
        ]
    
    def _radam_rectifier(self, t: int, beta2: float):
        """RAdam variance rectification term."""
        rho_inf = 2.0 / (1.0 - beta2) - 1.0
        rho_t = rho_inf - 2.0 * (beta2 ** t) / (1.0 - beta2 ** t)
        if rho_t > 4.0:
            rt = np.sqrt(((rho_t - 4.0) * (rho_t - 2.0) * rho_inf) / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
            return float(rt), True
        return 1.0, False
    
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        # Update step count
        for group in self.param_groups:
            group['step'] = group.get('step', 0) + 1
        
        # Get non-gate parameter group
        non_gate_group = self.param_groups[0]
        params = non_gate_group['params']
        lr = non_gate_group['lr']
        weight_decay = non_gate_group['weight_decay']
        beta1, beta2 = non_gate_group['betas']
        eps = non_gate_group['eps']
        step = non_gate_group['step']
        
        # Update non-gate parameters with RAdam
        for p in params:
            if p.grad is None:
                continue
            
            grad = p.grad.data
            state = self.state[p]
            
            # State initialization
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
            
            state['step'] += 1
            step = state['step']
            
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            
            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            # Compute bias-corrected moments
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            
            # Compute rectification term
            rt, rectified = self._radam_rectifier(step, beta2)
            
            if rectified:
                # Compute variance rectification
                step_size = lr * rt * np.sqrt(bias_correction2) / bias_correction1
                
                # Denominator with bias correction
                denom = exp_avg_sq.sqrt() / np.sqrt(bias_correction2) + eps
                
                # Update parameters
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay * lr)
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
            else:
                # Standard Adam update
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / np.sqrt(bias_correction2)).add_(eps)
                
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay * lr)
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        # Handle gate parameters with mirror descent in KL geometry
        gate_lr = self.defaults['lr_gate']
        gate_idx = 0
        for name, p in self.model.named_parameters():
            if p.grad is None or not any(p is param for param in self.gate_params):
                continue
            
            grad = p.grad.data
            if name not in self.state:
                self.state[name] = {}
                self.state[name]['exp_avg'] = torch.zeros_like(p.data)
                self.state[name]['exp_avg_sq'] = torch.zeros_like(p.data)
                self.state[name]['step'] = 0
            
            state = self.state[name]
            state['step'] += 1
            step = state['step']
            
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            
            # Update moving averages
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            # Bias correction
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            rt, rectified = self._radam_rectifier(step, beta2)
            
            # Compute step
            if rectified:
                step_size = gate_lr * rt * np.sqrt(bias_correction2) / bias_correction1
                denom = (exp_avg_sq.sqrt() / np.sqrt(bias_correction2)).add_(eps)
                step = (exp_avg / denom) * step_size
            else:
                step_size = gate_lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / np.sqrt(bias_correction2)).add_(eps)
                step = (exp_avg / denom) * step_size
            
            # Mirror descent step in KL geometry
            if p.dim() == 2:
                # For softmax gates (row-wise updates)
                P_old = F.softmax(p.data, dim=1)
                P_new = P_old * torch.exp(-step)
                P_new = P_new / (P_new.sum(dim=1, keepdim=True) + 1e-8)
                logits_new = torch.log(torch.clamp(P_new, 1e-8, 1.0))
                logits_new -= logits_new.mean(dim=1, keepdim=True)
                p.data.copy_(logits_new)
            else:
                # For scalar gates
                p.data.add_(-step)
            
            # Zero the gradient after update
            if p.grad is not None:
                p.grad.zero_()

            gate_idx += 1
        
        return loss