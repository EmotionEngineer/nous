from __future__ import annotations
import numpy as np
from typing import Optional, Sequence
from ..model import NousNet
from .loo import rule_impact_df

def generate_enhanced_explanation(
    model: NousNet, x_sample, y_true, feature_names: Sequence[str], class_names: Optional[Sequence[str]] = None, y_scaler=None,
    loo_mode: str = 'replacement', top_m_rules: Optional[int] = None, use_pre_norm: bool = False,
    use_pruning: bool = False, pruning_threshold: float = 0.0, vs_class_idx: Optional[int] = None
) -> str:
    """
    Human-readable explanation text with top rule impacts for a single sample.
    """
    task = model.config['task_type']
    base_probas, base_logits, _ = model.forward_explain(
        x_sample, apply_pruning=use_pruning, pruning_threshold=pruning_threshold
    )

    if y_scaler is not None:
        y_true_unscaled = y_scaler.inverse_transform(np.array(y_true).reshape(-1, 1)).item()
        base_pred_unscaled = y_scaler.inverse_transform(base_logits.reshape(-1, 1)).item()
    else:
        y_true_unscaled = y_true
        base_pred_unscaled = base_logits[0] if task == 'regression' else None

    if task == "classification":
        pred_idx = int(np.argmax(base_probas))
        pred_name = class_names[pred_idx] if class_names else f"Class {pred_idx}"
        conf = float(base_probas[pred_idx])
        true_name = class_names[y_true] if class_names is not None else str(y_true)
    else:
        pred_name = f"Value: {base_pred_unscaled:.3f}"
        true_name = f"{y_true_unscaled:.3f}"

    lines = []
    model_tag = model.config['rule_selection_method'].upper()
    lines.append(f"MODEL: {model_tag} rules | TASK: {task.upper()}")
    lines.append(f"SAMPLE PREDICTION: {pred_name}")
    if task == "classification":
        lines.append(f"  - Confidence: {conf:.3f}")
        lines.append(f"  - Ground Truth: {true_name}")
    else:
        lines.append(f"  - Ground Truth: {true_name}")
    if use_pruning:
        lines.append(f"  - Pruning: |act| >= {pruning_threshold:.4f} (forward uses pruned activations)")
    lines.append("-"*60)

    imp = rule_impact_df(
        model, x_sample, feature_names, class_names=class_names, vs_class=vs_class_idx,
        loo_mode=loo_mode, top_m_rules=top_m_rules, use_pre_norm=use_pre_norm,
        use_pruning=use_pruning, pruning_threshold=pruning_threshold
    )
    if imp.empty:
        lines.append("No active rules above threshold.")
        return "\n".join(lines)

    if vs_class_idx is not None and task == "classification":
        lines.append(f"(Contrastive) Why '{pred_name}' vs '{class_names[vs_class_idx]}'?")
    lines.append("CAUSAL RULE IMPACT (Top 5):")

    for _, row in imp.head(5).iterrows():
        b, r = row['block'], row['rule']
        aggs, facts = row['aggregators'], row['facts']
        badge = ""
        if 'supports_pred' in row:
            badge = " [+]" if row['supports_pred'] else " [-]"
        if 'critical_flip' in row and row['critical_flip']:
            badge += " [CRITICAL]"

        if task == "classification":
            margin_col = [c for c in imp.columns if c.startswith("Δmargin(")][0]
            s = f"Δmargin={row[margin_col]:+.3f}{badge}"
        else:
            delta_pred = row['Δprediction']
            if y_scaler is not None:
                delta_pred_unscaled = delta_pred * y_scaler.scale_[0]
                s = f"Δprediction={delta_pred_unscaled:+.3f}"
            else:
                s = f"Δprediction={delta_pred:+.3f}"
        lines.append(f"  • B{b}/R{r}: {s} | {aggs}\n    {facts}")

    if len(imp) > 5:
        lines.append(f"  ... and {len(imp) - 5} more active rules.")

    return "\n".join(lines)