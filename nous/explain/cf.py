from __future__ import annotations
import numpy as np
import torch
from typing import Optional, Sequence, List, Dict, Any
from ..model import NousNet
from .loo import rule_impact_df

def suggest_rule_counterfactuals(
    model: NousNet, x_sample, feature_names: Sequence[str], class_names: Optional[Sequence[str]] = None,
    target: str = "flip",              # 'flip' (classification), 'margin_drop', 'reg_delta'
    target_value: Optional[float] = None,  # for margin_drop/reg_delta
    y_scaler=None,
    k_rules: int = 3, fact_target_level: float = 0.1, max_features: int = 2,
    loo_mode: str = 'frozen', top_m_rules: int = 10, use_pre_norm: bool = False,
    alphas: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
    device=None
) -> List[Dict[str, Any]]:
    """
    Suggest counterfactual input deltas guided by influential rules using β-fact geometry.
    Verifies suggested deltas by forward_explain.
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

    imp = rule_impact_df(
        model, x_sample, feature_names, class_names=class_names,
        loo_mode=loo_mode, top_m_rules=top_m_rules, use_pre_norm=use_pre_norm
    )
    if imp.empty:
        return []

    if task == "classification":
        margin_col = [c for c in imp.columns if c.startswith("Δmargin(")][0]
        imp = imp.sort_values(by=margin_col, ascending=False)
    else:
        imp = imp.sort_values(by="Δprediction", ascending=False)
    imp = imp.head(k_rules)

    x = torch.tensor(x_sample, dtype=torch.float32, device=device).unsqueeze(0)
    if model.calibrators is not None:
        x_cal = torch.stack([calib(x[:, i]) for i, calib in enumerate(model.calibrators)], dim=1)
    else:
        x_cal = x
    diff, k_vec, nu_vec, net_w = model.fact.compute_diff_and_params(x_cal)  # [1,F], [F], [F], [F,D]
    facts_act = model.fact(x_cal).squeeze(0)

    suggestions = []
    for _, row in imp.iterrows():
        b = int(row["block"]) - 1
        r = int(row["rule"]) - 1
        details = base_internals[f'block_{b}']
        facts_used = details.get("facts_used", None)
        if isinstance(facts_used, torch.Tensor):
            facts_used = facts_used.cpu().numpy()
        if facts_used is None or facts_used.shape[0] <= r:
            continue
        used = facts_used[r]
        used = [int(used)] if np.ndim(used) == 0 else [int(u) for u in used.tolist()]

        used_sorted = sorted(used, key=lambda fid: float(facts_act[fid].item()), reverse=True)[:max(1, min(len(used), 2))]

        deltas: Dict[int, float] = {}
        for fid in used_sorted:
            y_now = float(facts_act[fid].item()) + 1e-12
            kf = float(k_vec[fid].item())
            nuf = float(nu_vec[fid].item())
            diff_now = float(diff[0, fid].item())
            w = net_w[fid].detach().clone()  # [D]

            y_target = float(fact_target_level)
            # Invert β: diff_target = (logit(y_target^(1/nu))) / k
            diff_target = float(torch.logit(torch.tensor(y_target, device=device).pow(1.0/max(nuf,1e-6))))
            diff_target = diff_target / max(kf, 1e-6)
            delta_diff = diff_target - diff_now

            w_np = w.cpu().numpy()
            idxs = np.argsort(-np.abs(w_np))[:max_features]
            w_sel = torch.zeros_like(w)
            w_sel[idxs] = w[idxs]
            denom = float(w_sel.pow(2).sum().item())
            if denom < 1e-12:
                continue
            delta_x_cal = (delta_diff / denom) * w_sel  # minimal L2 shift in x̃

            delta_x = delta_x_cal.clone()
            if model.calibrators is not None:
                for i in idxs:
                    xi = x[0, i]
                    slope_i = model.calibrators[i].local_slope(xi)
                    delta_x[i] = delta_x_cal[i] / slope_i

            for i in idxs:
                deltas[i] = deltas.get(i, 0.0) + float(delta_x[i].item())

        if not deltas:
            continue

        feat_deltas = sorted([(feature_names[i], d) for i, d in deltas.items()], key=lambda t: -abs(t[1]))
        success = False
        new_out = None
        for a in alphas:
            x_try = x.clone()
            for i, d in deltas.items():
                x_try[0, i] = x_try[0, i] + a * d
            prob2, logit2, _ = model.forward_explain(x_try.squeeze(0).cpu().numpy(), device=device)
            if task == "classification":
                new_pred = int(np.argmax(prob2))
                new_margin = float(logit2[pred_idx] - logit2[runner_up])
                if target == "flip" and new_pred != pred_idx:
                    success, new_out = True, {"pred": new_pred, "margin": new_margin}
                    break
                if target == "margin_drop" and target_value is not None and new_margin <= base_margin - float(target_value):
                    success, new_out = True, {"pred": new_pred, "margin": new_margin}
                    break
            else:
                new_pred = float(logit2[0])
                if target == "reg_delta" and target_value is not None:
                    if (new_pred - base_pred) <= float(target_value):
                        success, new_out = True, {"pred": new_pred}
                        break

        suggestions.append({
            "rule": (b+1, r+1),
            "facts": [f"F{fid+1}" for fid in used_sorted],
            "deltas": feat_deltas,
            "verified": success,
            "new_out": new_out
        })
    return suggestions