from __future__ import annotations
import numpy as np
import torch
from typing import Optional, Sequence, Dict, Any
from ..model import NousNet
from .mse import minimal_sufficient_explanation

def explanation_fidelity_metrics(
    model: NousNet, x_sample, feature_names: Sequence[str], class_names: Optional[Sequence[str]] = None,
    margin_tolerance: float = 0.0, prob_tolerance: float = 0.0, pred_tolerance_reg: float = 0.02,
    device=None
) -> Dict[str, float]:
    """
    Sufficiency and comprehensiveness metrics using MSE masks.
    """
    device = device or next(model.parameters()).device
    task = model.config['task_type']

    base_probas, base_logits, base_internals = model.forward_explain(x_sample, device=device)
    if task == "classification":
        pred_idx = int(np.argmax(base_probas))
        runner_up = int(np.argsort(base_logits)[-2]) if base_logits.size > 1 else pred_idx
        base_margin = float(base_logits[pred_idx] - base_logits[runner_up])
    else:
        base_pred = float(base_logits[0])

    mse = minimal_sufficient_explanation(
        model, x_sample, feature_names, class_names=class_names,
        margin_tolerance=margin_tolerance, prob_tolerance=prob_tolerance,
        pred_tolerance_reg=pred_tolerance_reg, freeze_non_active=True, device=device
    )
    kept_masks = mse["kept_masks"]

    prob_s, logit_s, _ = model.forward_explain(x_sample, restrict_masks=kept_masks, device=device)
    inv_masks = [(torch.ones_like(m) - m) for m in kept_masks]
    prob_c, logit_c, _ = model.forward_explain(x_sample, restrict_masks=inv_masks, device=device)

    if task == "classification":
        runner_up = int(np.argsort(base_logits)[-2]) if base_logits.size > 1 else pred_idx
        margin_s = float(logit_s[pred_idx] - logit_s[runner_up])
        margin_c = float(logit_c[pred_idx] - logit_c[runner_up])
        return {
            "base_margin": base_margin,
            "sufficiency_margin": margin_s,
            "comprehensiveness_margin": margin_c,
            "kept_size": float(sum(int(m.sum().item()) for m in kept_masks))
        }
    else:
        pred_s = float(logit_s[0])
        pred_c = float(logit_c[0])
        return {
            "base_pred": base_pred,
            "sufficiency_pred": pred_s,
            "comprehensiveness_pred": pred_c,
            "kept_size": float(sum(int(m.sum().item()) for m in kept_masks))
        }