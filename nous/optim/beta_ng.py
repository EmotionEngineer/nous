from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import Optimizer
from typing import Dict, Iterable, Optional, Tuple

from .base import BaseOptimizer
from ..model import NousNet

class BetaNaturalGradient(BaseOptimizer):
    """Stabilized β-natural gradient preconditioner for BetaFactLayer parameters.
    
    Uses diagonal Fisher approximation with EMA smoothing and trust-ratio clipping
    to stabilize optimization of β-facts parameters.
    
    Args:
        params: Iterable of parameters to optimize
        model: The model being optimized (required for parameter categorization)
        lr: Learning rate (default: 1e-3)
        ema_beta: EMA decay for Fisher approximation (default: 0.9)
        eps: Small value for numerical stability (default: 1e-6)
        trust_clip: Maximum trust ratio for step size (default: 10.0)
        weight_decay: Weight decay coefficient (default: 0)
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        model: torch.nn.Module,
        lr: float = 1e-3,
        ema_beta: float = 0.9,
        eps: float = 1e-6,
        trust_clip: float = 10.0,
        weight_decay: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= ema_beta < 1.0:
            raise ValueError(f"Invalid EMA beta parameter: {ema_beta}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if trust_clip <= 0.0:
            raise ValueError(f"Invalid trust clip value: {trust_clip}")
        
        defaults = dict(
            lr=lr,
            ema_beta=ema_beta,
            eps=eps,
            trust_clip=trust_clip,
            weight_decay=weight_decay,
            step=0
        )
        super().__init__(params, defaults, model=model)
        
        # Fisher approximation state
        self.ema_state: Dict[str, torch.Tensor] = {}
        self.fact_param_names = [
            "fact.kraw", "fact.nuraw", "fact.th", 
            "fact.L.weight", "fact.R.weight"
        ]
    
    def _compute_fisher_approximations(self, model: NousNet, X_batch: torch.Tensor):
        """Compute diagonal Fisher approximations for BetaFactLayer parameters."""
        if not hasattr(model, "fact") or not hasattr(model.fact, "compute_diff_and_params"):
            return
        
        with torch.no_grad():
            if model.calibrators is not None:
                Xc = torch.stack([cal(xb) for xb, cal in zip(X_batch.t(), model.calibrators)], dim=1)
            else:
                Xc = X_batch
            
            diff, k_vec, nu_vec, net_w = model.fact.compute_diff_and_params(Xc)
            kd = k_vec.unsqueeze(0) * diff
            sig = torch.sigmoid(kd)
            q = 1.0 - sig
            log_sig = F.logsigmoid(kd)
            
            # Fisher for k
            Fk = ((nu_vec.unsqueeze(0) * diff * q) ** 2).mean(dim=0)
            
            # Fisher for nu
            Fnu = (log_sig ** 2).mean(dim=0)
            
            # Fisher for th
            Fth = (((nu_vec * k_vec).unsqueeze(0) * q) ** 2).mean(dim=0)
            
            # Fisher for L and R weights
            x2 = (Xc ** 2).mean(dim=0)
            si = (((nu_vec * k_vec).unsqueeze(0) * q) ** 2).mean(dim=0)
            FL = torch.ger(si, x2)
            FR = FL.clone()
            
            # Update EMA state
            self._update_ema("Fk", Fk)
            self._update_ema("Fnu", Fnu)
            self._update_ema("Fth", Fth)
            self._update_ema("FL", FL)
            self._update_ema("FR", FR)
    
    def _update_ema(self, key: str, new_value: torch.Tensor):
        """Update EMA state with new value."""
        if key not in self.ema_state:
            self.ema_state[key] = new_value.clone()
        else:
            beta = self.defaults['ema_beta']
            self.ema_state[key].mul_(beta).add_(new_value, alpha=1 - beta)
        
        # Add small epsilon for numerical stability
        self.ema_state[key] = self.ema_state[key] + self.defaults['eps']
    
    def _apply_preconditioning(self, model: NousNet):
        """Apply natural gradient preconditioning to gradients."""
        if not hasattr(model, "fact"):
            return
        
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            
            grad = p.grad.data
            
            # Match parameter names to Fisher approximations
            if name.endswith("fact.kraw") and grad.shape[-1] == self.ema_state["Fk"].shape[0]:
                ng = grad / self.ema_state["Fk"]
            elif name.endswith("fact.nuraw") and grad.shape[-1] == self.ema_state["Fnu"].shape[0]:
                ng = grad / self.ema_state["Fnu"]
            elif name.endswith("fact.th") and grad.shape[-1] == self.ema_state["Fth"].shape[0]:
                ng = grad / self.ema_state["Fth"]
            elif name.endswith("fact.L.weight") and grad.shape == self.ema_state["FL"].shape:
                ng = grad / self.ema_state["FL"]
            elif name.endswith("fact.R.weight") and grad.shape == self.ema_state["FR"].shape:
                ng = grad / self.ema_state["FR"]
            else:
                continue
            
            # Compute trust ratio to prevent exploding updates
            denom = (ng.norm() + 1e-9) / (grad.norm() + 1e-9)
            trust_ratio = torch.clamp(denom, max=self.defaults['trust_clip'])
            
            # Apply preconditioning
            p.grad.data = trust_ratio * ng
    
    def step(self, closure=None, model: Optional[NousNet] = None, X_batch: Optional[torch.Tensor] = None):
        """Performs a single optimization step with natural gradient preconditioning.
        
        Args:
            closure: Optional closure that reevaluates the model and returns the loss
            model: The NousNet model being optimized (required for preconditioning)
            X_batch: Input batch used for computing Fisher approximations
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        if model is None or X_batch is None:
            raise ValueError("BetaNaturalGradient requires model and X_batch arguments")
        
        # Compute Fisher approximations
        self._compute_fisher_approximations(model, X_batch)
        
        # Apply preconditioning to gradients
        self._apply_preconditioning(model)
        
        # Update non-gate parameters with AdamW
        for group in self.param_groups:
            for p in group['params']:
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
                beta1, beta2 = 0.9, 0.999
                lr = group['lr']
                weight_decay = group['weight_decay']
                eps = group['eps']
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute bias-corrected moments
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # Compute step size
                step_size = lr * np.sqrt(bias_correction2) / bias_correction1
                
                # Update parameters
                denom = (exp_avg_sq.sqrt() / np.sqrt(bias_correction2)).add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        # Zero gradients after update
        for p in self.gate_params + self.nongate_params:
            if p.grad is not None:
                p.grad.zero_()
        
        return loss