from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from ..zoo import (
    HierarchicalMoE,
    SegmentMoE,
    SoftLogitAND,
    SoftLogicInteraction,
    ThresholdFactBank,
    NousFamilies,
)


def _invert_standard_scaler(th_scaled: float, feat_idx: int, scaler: Any) -> float:
    # Works with sklearn.preprocessing.StandardScaler
    return float(th_scaled * float(scaler.scale_[feat_idx]) + float(scaler.mean_[feat_idx]))


def describe_threshold_fact(
    fact_bank: ThresholdFactBank,
    fact_index: int,
    feature_names: Sequence[str],
    scaler: Optional[Any] = None,
    negated: bool = False,
) -> str:
    """
    Render a single threshold fact into a readable condition.
    """
    feat = int(fact_bank.feat_idx[fact_index].item())
    th_scaled = float(fact_bank.th[fact_index].detach().cpu().item())
    if scaler is not None:
        th = _invert_standard_scaler(th_scaled, feat, scaler)
    else:
        th = th_scaled
    name = str(feature_names[feat])
    op = "<=" if negated else ">"
    return f"{name} {op} {th:.4g}"


def softlogitand_top_conditions(
    fact_bank: ThresholdFactBank,
    selector_probs: torch.Tensor,  # [R, Fin]
    rule_idx: int,
    feature_names: Sequence[str],
    scaler: Optional[Any] = None,
    top_k: int = 3,
) -> List[str]:
    """
    Pick top-k fact conditions by selector probability for a given rule.
    """
    p = selector_probs[int(rule_idx)].detach().cpu().numpy()
    fin = int(p.shape[0])
    F0 = int(fact_bank.num_facts)

    items: List[Tuple[float, str]] = []
    for j in range(fin):
        prob = float(p[j])
        if fin == 2 * F0:
            base = j % F0
            neg = (j >= F0)
            s = describe_threshold_fact(fact_bank, base, feature_names, scaler=scaler, negated=neg)
        else:
            s = describe_threshold_fact(fact_bank, j, feature_names, scaler=scaler, negated=False)
        items.append((prob, s))
    items.sort(key=lambda t: -t[0])
    return [f"{s} (p={prob:.3f})" for prob, s in items[: int(top_k)]]


def softlogitand_global_rules_df(
    model: SoftLogitAND,
    feature_names: Sequence[str],
    scaler: Optional[Any] = None,
    top_rules: int = 10,
    top_facts_per_rule: int = 3,
) -> pd.DataFrame:
    """
    Global view: rank rules by |head weight| and show their top conditions.
    """
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, model.facts.num_facts, device=next(model.parameters()).device)
        _, P = model.rules(dummy)  # [R, 2F]
        w = model.head.weight.view(-1).detach().cpu().numpy()

    idx = np.argsort(-np.abs(w))[: int(top_rules)]
    rows = []
    for r in idx:
        conds = softlogitand_top_conditions(model.facts, P, int(r), feature_names, scaler=scaler, top_k=top_facts_per_rule)
        rows.append({
            "rule": int(r),
            "weight": float(w[r]),
            "abs_weight": float(abs(w[r])),
            "rule_text": " AND ".join([c.split(" (p=")[0] for c in conds]),
            "top_conditions": " | ".join(conds),
        })
    return pd.DataFrame(rows)


def softlogitand_local_contrib_df(
    model: SoftLogitAND,
    x: np.ndarray,
    feature_names: Sequence[str],
    scaler: Optional[Any] = None,
    top_rules: int = 10,
    top_facts_per_rule: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Local view: compute per-rule contributions to the final logit: contrib = z_r * w_r.
    """
    model.eval()
    dev = next(model.parameters()).device
    xt = torch.tensor(np.asarray(x, dtype=np.float32)[None, :], device=dev)

    with torch.no_grad():
        f = model.facts(xt)
        z, P = model.rules(f)  # [1,R], [R,2F]
        w = model.head.weight.view(-1)
        b = model.head.bias.view(-1)[0] if model.head.bias is not None else torch.tensor(0.0, device=dev)
        contrib = (z.view(-1) * w).detach().cpu().numpy()
        z_np = z.view(-1).detach().cpu().numpy()
        w_np = w.detach().cpu().numpy()
        logit = float((z @ w.view(-1, 1) + b).view(-1).item())
        prob = float(torch.sigmoid(torch.tensor(logit)).item())

    idx = np.argsort(-np.abs(contrib))[: int(top_rules)]
    rows = []
    for r in idx:
        conds = softlogitand_top_conditions(model.facts, P, int(r), feature_names, scaler=scaler, top_k=top_facts_per_rule)
        rows.append({
            "rule": int(r),
            "activation": float(z_np[r]),
            "weight": float(w_np[r]),
            "contribution_to_logit": float(contrib[r]),
            "rule_text": " AND ".join([c.split(" (p=")[0] for c in conds]),
        })
    meta = {"logit": logit, "prob_pos": prob}
    return pd.DataFrame(rows), meta


@torch.no_grad()
def moe_gate_probs(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Get gate probabilities for SegmentMoE / HierarchicalMoE.
    """
    model.eval()
    dev = next(model.parameters()).device
    xt = torch.tensor(np.asarray(X, dtype=np.float32), device=dev)
    f = model.facts(xt)
    g = model.gate_probs_from_facts(f)
    return g.detach().cpu().numpy()


def moe_gate_summary_df(model: Any, X_ref: np.ndarray) -> pd.DataFrame:
    """
    Global view: average segment probabilities.
    """
    gp = moe_gate_probs(model, X_ref)
    avg = gp.mean(axis=0)
    return pd.DataFrame({"segment": np.arange(len(avg)), "avg_gate_prob": avg}).sort_values(
        "avg_gate_prob", ascending=False
    ).reset_index(drop=True)


def segmentmoe_local_explain_df(
    model: SegmentMoE,
    x: np.ndarray,
    feature_names: Sequence[str],
    scaler: Optional[Any] = None,
    top_segments: int = 3,
    top_facts: int = 8,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Local SegmentMoE explanation:
      - top segments by gate prob,
      - expert logit,
      - segment contribution to final logit,
      - top additive fact contributions inside the expert.
    """
    model.eval()
    dev = next(model.parameters()).device
    xt = torch.tensor(np.asarray(x, dtype=np.float32)[None, :], device=dev)

    with torch.no_grad():
        f = model.facts(xt)
        g = moe_gate_probs(model, np.asarray(x, dtype=np.float32)[None, :])[0]
        expert_logits = torch.stack([ex(f) for ex in model.experts], dim=1).view(-1).detach().cpu().numpy()
        logit = float((g * expert_logits).sum())
        prob = float(1.0 / (1.0 + np.exp(-logit)))

    seg_idx = np.argsort(-g)[: int(top_segments)]
    rows = []
    for s in seg_idx:
        ex = model.experts[int(s)]
        with torch.no_grad():
            w = ex.w.view(-1).detach().cpu().numpy()
            f_np = f.view(-1).detach().cpu().numpy()
            contrib = f_np * w

        top = np.argsort(-np.abs(contrib))[: int(top_facts)]
        items = []
        for fi in top:
            # SegmentMoE uses ThresholdFactBank facts (not negations)
            text = describe_threshold_fact(model.facts, int(fi), feature_names, scaler=scaler, negated=False)
            items.append(f"{text}: {contrib[fi]:+.3f}")

        rows.append({
            "segment": int(s),
            "gate_prob": float(g[s]),
            "expert_logit": float(expert_logits[s]),
            "segment_contribution_to_logit": float(g[s] * expert_logits[s]),
            "top_fact_contribs": " | ".join(items),
        })
    meta = {"logit": logit, "prob_pos": prob}
    return pd.DataFrame(rows), meta


def hiermoe_local_explain_df(
    model: HierarchicalMoE,
    x: np.ndarray,
    feature_names: Sequence[str],
    scaler: Optional[Any] = None,
    top_segments: int = 3,
    top_rules: int = 6,
    top_facts_per_rule: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Local HierarchicalMoE explanation:
      - top segments by gate prob,
      - expert logit + segment contribution,
      - inside each selected expert: top contributing rules (approx) with their top conditions.
    """
    model.eval()
    dev = next(model.parameters()).device
    xt = torch.tensor(np.asarray(x, dtype=np.float32)[None, :], device=dev)

    with torch.no_grad():
        f = model.facts(xt)
        g = moe_gate_probs(model, np.asarray(x, dtype=np.float32)[None, :])[0]
        expert_logits = torch.stack([ex(f) for ex in model.experts], dim=1).view(-1).detach().cpu().numpy()
        logit = float((g * expert_logits).sum())
        prob = float(1.0 / (1.0 + np.exp(-logit)))

    seg_idx = np.argsort(-g)[: int(top_segments)]
    rows = []
    for s in seg_idx:
        ex = model.experts[int(s)]
        with torch.no_grad():
            z, P = ex.rules(f)  # [1,R], [R,2F]
            w = ex.head.weight.view(-1).detach().cpu().numpy()
            z_np = z.view(-1).detach().cpu().numpy()
            contrib = z_np * w

        ridx = np.argsort(-np.abs(contrib))[: int(top_rules)]
        rule_texts = []
        for r in ridx:
            conds = softlogitand_top_conditions(model.facts, P, int(r), feature_names, scaler=scaler, top_k=top_facts_per_rule)
            rule_texts.append(f"[r{int(r)}] {contrib[r]:+.3f}: " + " AND ".join([c.split(" (p=")[0] for c in conds]))

        rows.append({
            "segment": int(s),
            "gate_prob": float(g[s]),
            "expert_logit": float(expert_logits[s]),
            "segment_contribution_to_logit": float(g[s] * expert_logits[s]),
            "top_expert_rules": " || ".join(rule_texts),
        })
    meta = {"logit": logit, "prob_pos": prob}
    return pd.DataFrame(rows), meta


def nousfamilies_global_summary_df(model: NousFamilies, top_families: int = 10) -> pd.DataFrame:
    """
    Global view for NousFamilies:
      - family weights,
      - "rule mass" assigned to each family (sum over rules of assignment probability).
    """
    model.eval()
    with torch.no_grad():
        W = model.head.W.view(-1).detach().cpu().numpy()
        A = model.head.assignment().detach().cpu().numpy()  # [R, F]
    fam_mass = A.sum(axis=0)

    df = pd.DataFrame({
        "family": np.arange(model.n_families),
        "weight_W": W,
        "abs_weight": np.abs(W),
        "assigned_rule_mass": fam_mass,
    })
    return df.sort_values("abs_weight", ascending=False).head(int(top_families)).reset_index(drop=True)


def nousfamilies_local_contrib_df(
    model: NousFamilies,
    x: np.ndarray,
    top_families: int = 8,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Local view for NousFamilies:
      - family activations,
      - family contributions to the final logit.
    """
    model.eval()
    dev = next(model.parameters()).device
    xt = torch.tensor(np.asarray(x, dtype=np.float32)[None, :], device=dev)

    with torch.no_grad():
        Z = model.extract_rule_matrix(xt, detach_internals=True)  # [1,R]
        A = model.head.assignment()                                # [R,F]
        denom = A.sum(dim=0, keepdim=True) + 1e-12
        fam = (Z @ A) / denom                                      # [1,F]
        W = model.head.W.view(-1)
        contrib = (fam.view(-1) * W).detach().cpu().numpy()

        logit = float(model(xt).view(-1).item())
        prob = float(1.0 / (1.0 + np.exp(-logit)))

    idx = np.argsort(-np.abs(contrib))[: int(top_families)]
    df = pd.DataFrame({
        "family": idx.astype(int),
        "family_activation": fam.view(-1).detach().cpu().numpy()[idx],
        "family_weight_W": W.detach().cpu().numpy()[idx],
        "family_contribution_to_logit": contrib[idx],
    }).reset_index(drop=True)

    meta = {"logit": logit, "prob_pos": prob}
    return df, meta