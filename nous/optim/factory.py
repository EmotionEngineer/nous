from __future__ import annotations
from typing import Any, Dict, Optional, Union
import torch

from .mirror_radam import MirrorRAdam
from .beta_ng import BetaNaturalGradient
from .extragrad_trust import ExtragradRAdamTrust
from .sam_gates import SharpnessAwareGates

def create_optimizer(
    model: torch.nn.Module, 
    optimizer_name: str,
    lr: float = 1e-3,
    **kwargs: Any
) -> Union[MirrorRAdam, BetaNaturalGradient, ExtragradRAdamTrust, SharpnessAwareGates]:
    """Factory function to create optimizers for Nous models.
    
    Args:
        model: The model to optimize
        optimizer_name: Name of the optimizer to create
            Options: "mirror_radam", "beta_ng", "extragrad_trust", "sharpness_aware_gates"
        lr: Learning rate
        **kwargs: Additional optimizer-specific arguments
    
    Returns:
        Optimizer instance
    """
    params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_name.lower() in ["mirror_radam", "radam_kl"]:
        return MirrorRAdam(
            params,
            model=model,
            lr=lr,
            lr_gate=kwargs.get("lr_gate", 1e-2),
            betas=kwargs.get("betas", (0.9, 0.999)),
            eps=kwargs.get("eps", 1e-8),
            weight_decay=kwargs.get("weight_decay", 0.0)
        )
    elif optimizer_name.lower() in ["beta_ng", "betangpp"]:
        return BetaNaturalGradient(
            params,
            model=model,
            lr=lr,
            ema_beta=kwargs.get("ema_beta", 0.9),
            eps=kwargs.get("eps", 1e-6),
            trust_clip=kwargs.get("trust_clip", 10.0),
            weight_decay=kwargs.get("weight_decay", 0.0)
        )
    elif optimizer_name.lower() in ["extragrad_trust", "radam_kl_trust_xtr"]:
        return ExtragradRAdamTrust(
            params,
            model=model,
            lr=lr,
            lr_gate=kwargs.get("lr_gate", 1e-2),
            betas=kwargs.get("betas", (0.9, 0.999)),
            eps=kwargs.get("eps", 1e-8),
            weight_decay=kwargs.get("weight_decay", 0.0),
            delta_kl=kwargs.get("delta_kl", 1e-2),
            backtrack=kwargs.get("backtrack", 0.5),
            lam_budget=kwargs.get("lam_budget", 1e-3),
            k_target=kwargs.get("k_target", 8),
            eta_pred=kwargs.get("eta_pred", 5e-3)
        )
    elif optimizer_name.lower() in ["sharpness_aware_gates", "klsam_gates"]:
        return SharpnessAwareGates(
            params,
            model=model,
            lr=lr,
            rho=kwargs.get("rho", 0.05),
            betas=kwargs.get("betas", (0.9, 0.999)),
            eps=kwargs.get("eps", 1e-8),
            weight_decay=kwargs.get("weight_decay", 0.0),
            clip_grad=kwargs.get("clip_grad", 0.5)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")