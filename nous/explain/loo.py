from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from typing import Optional, Sequence
from ..model import NousNet
from .facts_desc import render_fact_descriptions
from .aggregator import format_agg_mixture

def rule_impact_df(
    model: NousNet, x_sample, feature_names: Sequence[str], class_names: Optional[Sequence[str]] = None, vs_class: Optional[int] = None,
    loo_mode: str = 'replacement',  # 'replacement' | 'frozen' | 'active_only'
    top_m_rules: Optional[int] = None,
    use_pruning: bool = False, pruning_threshold: float = 0.0, use_pre_norm: bool = False,
    base_state=None, device=None
) -> pd.DataFrame:
    """
    Honest leave-one-out at the rule level with recomputed gating (before top-k),
    supporting multiple modes (replacement, frozen, active_only).
    """
    device = device or next(model.parameters()).device
    task = model.config['task_type']

    if base_state is None:
        base_probas, base_logits, base_internals = model.forward_explain(
            x_sample, apply_pruning=use_pruning, pruning_threshold=pruning_threshold, device=device
        )
    else:
        base_probas, base_logits, base_internals = base_state

    if task == "classification":
        base_logits = np.array(base_logits)
        pred_idx = int(np.argmax(base_probas))
        if vs_class is None:
            runner_up_idx = int(np.argsort(base_logits)[-2]) if base_logits.size > 1 else pred_idx
        else:
            runner_up_idx = int(vs_class)
        base_margin = float(base_logits[pred_idx] - base_logits[runner_up_idx])
    else:
        base_pred = float(base_logits[0])

    restrict_masks = None
    if loo_mode in ('frozen', 'active_only'):
        # Freeze current active set by extracting gate masks and reusing as restrict masks.
        restrict_masks = []
        block_keys = sorted([k for k in base_internals.keys() if k.startswith("block_")], key=lambda s: int(s.split("_")[1]))
        for key in block_keys:
            gm = base_internals[key]['gate_mask']
            if isinstance(gm, torch.Tensor):
                gm = gm.squeeze(0)
            restrict_masks.append((gm > 0).float())

    fact_desc = render_fact_descriptions(model, feature_names)

    rows = []
    block_keys = sorted([k for k in base_internals.keys() if k.startswith("block_")], key=lambda s: int(s.split("_")[1]))
    for b_idx, key in enumerate(block_keys):
        details = base_internals[key]
        metric_tensor = details['pre_norm_sum'] if (use_pre_norm and 'pre_norm_sum' in details) else details['gated_activations']
        ga_np = metric_tensor.squeeze(0).abs().cpu().numpy() if isinstance(metric_tensor, torch.Tensor) else np.abs(metric_tensor)
        active_rules = np.where(ga_np > 1e-12)[0]
        if len(active_rules) == 0:
            continue

        if top_m_rules is not None and len(active_rules) > top_m_rules:
            order = np.argsort(-ga_np[active_rules])
            active_rules = active_rules[order[:top_m_rules]]

        agg_w = details['aggregator_weights']
        if isinstance(agg_w, torch.Tensor):
            agg_w = agg_w.cpu()

        facts_used = details.get('facts_used', None)
        if isinstance(facts_used, torch.Tensor):
            facts_used = facts_used.cpu().numpy()

        for r in active_rules:
            drop_spec = (b_idx, int(r))
            restr = None if loo_mode == 'replacement' else restrict_masks

            drop_probas, drop_logits, _ = model.forward_explain(
                x_sample, drop_rule_spec=drop_spec, restrict_masks=restr,
                apply_pruning=use_pruning, pruning_threshold=pruning_threshold, device=device
            )

            if agg_w is not None:
                aggs = format_agg_mixture(agg_w[r])
            else:
                aggs = "AND (fixed)"

            facts_str = "—"
            if facts_used is not None and facts_used.shape[0] > r:
                used = facts_used[r]
                if np.ndim(used) == 0:
                    used = [int(used)]
                facts_str = ", ".join([f"F{int(fid)+1}" for fid in used])
                facts_str += "  →  " + " | ".join([f"[F{int(fid)+1}] {fact_desc[int(fid)]}" for fid in used])

            if task == "classification":
                drop_logits = np.array(drop_logits)
                dlogit = float(base_logits[pred_idx] - drop_logits[pred_idx])
                drop_margin = float(drop_logits[pred_idx] - drop_logits[runner_up_idx])
                dmargin = float(base_margin - drop_margin)
                critical = (np.argmax(drop_probas) != pred_idx)
                supports = dmargin > 0
                rows.append({
                    "block": b_idx+1, "rule": int(r+1),
                    "aggregators": aggs, "facts": facts_str,
                    "Δlogit(pred)": dlogit,
                    f"Δmargin(vs {class_names[runner_up_idx] if class_names else runner_up_idx})": dmargin,
                    "supports_pred": supports, "critical_flip": critical
                })
            else:
                drop_pred = float(drop_logits[0])
                d_pred = base_pred - drop_pred
                rows.append({
                    "block": b_idx+1, "rule": int(r+1),
                    "aggregators": aggs, "facts": facts_str,
                    "Δprediction": d_pred
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        if task == "classification":
            key = [c for c in df.columns if c.startswith("Δmargin(")][0]
            df['abs_impact'] = df[key].abs()
        else:
            df['abs_impact'] = df['Δprediction'].abs()
        df = df.sort_values(by='abs_impact', ascending=False).drop(columns=['abs_impact']).reset_index(drop=True)
    return df