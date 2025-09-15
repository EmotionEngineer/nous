from __future__ import annotations
import numpy as np
from typing import Optional
from sklearn.metrics import mean_absolute_error
from ..model import NousNet

def select_pruning_threshold_global(
    model: NousNet, X_val, target_fidelity: float = 0.99, task_type: Optional[str] = None,
    metric_reg: str = "mae", tol_reg: float = 0.05, pointwise_tol: Optional[float] = None,
    max_samples: int = 500, device=None
) -> float:
    """
    Grid selection of activation threshold (post-gating) keeping fidelity (classification) or MAE (regression).
    """
    device = device or next(model.parameters()).device
    task = task_type or model.config['task_type']

    n = min(len(X_val), max_samples)
    Xv = X_val[:n]

    base_preds = []
    acts = []
    for i in range(n):
        _, logit_b, internals = model.forward_explain(Xv[i], device=device)
        base_preds.append(int(np.argmax(logit_b)) if task == "classification" else float(logit_b[0]))
        for key in [k for k in internals.keys() if k.startswith("block_")]:
            ga = internals[key]['gated_activations'].abs().cpu().numpy().ravel()
            acts.extend(list(ga))
    acts = np.array(acts)
    if acts.size == 0:
        return 0.0

    qs = np.linspace(0.5, 0.999, 25)
    candidates = np.unique(np.quantile(acts, qs))

    best_t = 0.0
    for t in candidates:
        if task == "classification":
            agree = 0
            for i in range(n):
                _, logit_p, _ = model.forward_explain(Xv[i], apply_pruning=True, pruning_threshold=float(t), device=device)
                agree += int(int(np.argmax(logit_p)) == base_preds[i])
            fidelity = agree / n
            if fidelity >= target_fidelity:
                best_t = float(t)
        else:
            preds_p, preds_b = [], []
            max_abs = 0.0
            for i in range(n):
                _, logit_p, _ = model.forward_explain(Xv[i], apply_pruning=True, pruning_threshold=float(t), device=device)
                _, logit_b, _ = model.forward_explain(Xv[i], device=device)
                pv, bv = float(logit_p[0]), float(logit_b[0])
                preds_p.append(pv); preds_b.append(bv)
                max_abs = max(max_abs, abs(pv - bv))
            mae_p = mean_absolute_error(preds_b, preds_p)
            ok_mae = (mae_p <= tol_reg)
            ok_point = True if pointwise_tol is None else (max_abs <= pointwise_tol)
            if ok_mae and ok_point:
                best_t = float(t)
    return best_t


def select_pruning_threshold_global_bs(
    model: NousNet, X_val, target_fidelity: float = 0.99, task_type: Optional[str] = None,
    metric_reg: str = "mae", tol_reg: float = 0.05, pointwise_tol: Optional[float] = None,
    max_samples: int = 500, device=None
) -> float:
    """
    Binary search selection of activation threshold (post-gating) with fidelity/MAE constraints.
    """
    device = device or next(model.parameters()).device
    task = task_type or model.config['task_type']
    n = min(len(X_val), max_samples)
    Xv = X_val[:n]

    base_preds = []
    acts = []
    for i in range(n):
        _, logit_b, internals = model.forward_explain(Xv[i], device=device)
        base_preds.append(int(np.argmax(logit_b)) if task == "classification" else float(logit_b[0]))
        for key in [k for k in internals.keys() if k.startswith("block_")]:
            ga = internals[key]['gated_activations'].abs().cpu().numpy().ravel()
            acts.extend(list(ga))
    acts = np.array(acts)
    if acts.size == 0:
        return 0.0

    lo, hi = 0.0, float(np.quantile(acts, 0.999))
    best_t = 0.0
    for _ in range(14):
        mid = (lo + hi) / 2.0
        if task == "classification":
            agree = 0
            for i in range(n):
                _, logit_p, _ = model.forward_explain(Xv[i], apply_pruning=True, pruning_threshold=float(mid), device=device)
                agree += int(int(np.argmax(logit_p)) == base_preds[i])
            fidelity = agree / n
            if fidelity >= target_fidelity:
                best_t = mid; lo = mid
            else:
                hi = mid
        else:
            preds_p, preds_b, max_abs = [], [], 0.0
            for i in range(n):
                _, logit_p, _ = model.forward_explain(Xv[i], apply_pruning=True, pruning_threshold=float(mid), device=device)
                _, logit_b, _ = model.forward_explain(Xv[i], device=device)
                pv, bv = float(logit_p[0]), float(logit_b[0])
                preds_p.append(pv); preds_b.append(bv)
                max_abs = max(max_abs, abs(pv - bv))
            mae_p = mean_absolute_error(preds_b, preds_p)
            ok_mae = (mae_p <= tol_reg)
            ok_point = True if pointwise_tol is None else (max_abs <= pointwise_tol)
            if ok_mae and ok_point:
                best_t = mid; lo = mid
            else:
                hi = mid
    return float(best_t)