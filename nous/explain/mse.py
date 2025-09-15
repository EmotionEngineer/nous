from __future__ import annotations
import numpy as np
import torch
from typing import Optional, Sequence, Dict, Any, List
from ..model import NousNet
from .loo import rule_impact_df

def minimal_sufficient_explanation(
    model: NousNet, x_sample, feature_names: Sequence[str], class_names: Optional[Sequence[str]] = None,
    margin_tolerance: float = 0.0, prob_tolerance: float = 0.0, pred_tolerance_reg: float = 0.02,
    freeze_non_active: bool = True, device=None
) -> Dict[str, Any]:
    """
    Greedy backward elimination among originally active rules.
    Preserves the prediction (class unchanged with margins within tolerance; regression within abs tol).
    """
    device = device or next(model.parameters()).device
    task = model.config['task_type']

    base_probas, base_logits, base_internals = model.forward_explain(x_sample, device=device)
    if task == "classification":
        pred_idx = int(np.argmax(base_probas))
        runner_up = int(np.argsort(base_logits)[-2]) if base_logits.size > 1 else pred_idx
        base_margin = float(base_logits[pred_idx] - base_logits[runner_up])
        base_conf = float(base_probas[pred_idx])
    else:
        base_pred = float(base_logits[0])

    # Extract active set (frozen)
    active_masks: List[torch.Tensor] = []
    block_keys = sorted([k for k in base_internals.keys() if k.startswith("block_")], key=lambda s: int(s.split("_")[1]))
    for key in block_keys:
        gm = base_internals[key]['gate_mask']
        if isinstance(gm, torch.Tensor):
            gm = gm.squeeze(0)
        active_masks.append((gm > 0).float())
    current_masks = [m.clone() for m in active_masks]

    # Rank removal candidates by (absolute) impact ascending
    imp = rule_impact_df(
        model, x_sample, feature_names, class_names=class_names,
        loo_mode='frozen', device=device
    )
    if imp.empty:
        return {
            "kept_masks": current_masks, "kept": [],
            "removed": [], "pred_preserved": True,
            "size": sum(int(m.sum().item()) for m in current_masks)
        }

    if task == "classification":
        margin_col = [c for c in imp.columns if c.startswith("Δmargin(")][0]
        imp = imp.assign(order=imp[margin_col].abs()).sort_values("order", ascending=True).drop(columns=["order"])
    else:
        imp = imp.assign(order=imp["Δprediction"].abs()).sort_values("order", ascending=True).drop(columns=["order"])

    removed = []
    for _, row in imp.iterrows():
        b = int(row["block"]) - 1
        r = int(row["rule"]) - 1
        if current_masks[b][r] == 0:
            continue

        proposal = [m.clone() for m in current_masks]
        proposal[b][r] = 0.0
        probas2, logits2, _ = model.forward_explain(x_sample, restrict_masks=proposal, device=device)

        if task == "classification":
            new_pred = int(np.argmax(probas2))
            if new_pred != pred_idx:
                continue
            new_margin = float(logits2[pred_idx] - logits2[runner_up])
            new_conf = float(probas2[pred_idx])
            if new_margin < base_margin - margin_tolerance:
                continue
            if prob_tolerance > 0.0 and (new_conf < base_conf - prob_tolerance):
                continue
        else:
            new_pred = float(logits2[0])
            if abs(new_pred - base_pred) > pred_tolerance_reg:
                continue

        current_masks = proposal
        removed.append((b+1, r+1))

    kept = [(i+1, int(idx.item())+1) for i, m in enumerate(current_masks) for idx in torch.where(m > 0)[0]]
    return {
        "kept_masks": current_masks,
        "kept": kept,
        "removed": removed,
        "pred_preserved": True,
        "size": sum(int(m.sum().item()) for m in current_masks)
    }