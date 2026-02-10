# nous/explain/zoo_v2.py
"""
Interpretation utilities for Nous zoo_v2 models.

Goals
-----
- Provide consistent global + local explanation APIs across model families.
- Render human-readable rule text with optional simplification.
- Support unscaling thresholds via sklearn-like scalers.

Notes
-----
These models are differentiable and often use soft selections; "global rules"
here are approximations derived from argmax selectors / top masked features.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from ..zoo import ThresholdFactBank
from ..zoo_v2.common import corner_product_z, safe_log, sparsemax
from ..zoo_v2.facts import ARFactBank, IntervalFactBank, MultiResAxisFactBank

Readability = Literal["exact", "pretty", "simplified", "clinical"]


# ===== Feature metadata utilities =====

@dataclass(frozen=True)
class Clause:
    base: str
    kind: Literal["numeric", "category", "binary"]
    op: Literal[">", "<", ">=", "<=", "==", "!="]
    value: Union[float, str, int]
    raw_feature: str
    feature_index: int


@dataclass
class SimplifiedCondition:
    base: str
    text: str
    kind: Literal["numeric_range", "category_set", "binary"]
    lower: Optional[float] = None
    upper: Optional[float] = None
    allowed: Optional[List[str]] = None
    banned: Optional[List[str]] = None


def _split_ohe(name: str) -> Optional[Tuple[str, str]]:
    """Split OHE feature name into base feature and category suffix.

    Heuristic: "Base_Suffix" and suffix not too long.
    """
    m = re.match(r"^(.+)_(.+)$", name)
    if not m or len(m.group(2)) > 30:
        return None
    return m.group(1), m.group(2)


def _compute_base_order(feature_names: List[str]) -> Dict[str, int]:
    """Compute canonical ordering of base features from OHE-expanded names."""
    order: Dict[str, int] = {}
    for i, fn in enumerate(feature_names):
        sp = _split_ohe(fn)
        base = sp[0] if sp else fn
        order.setdefault(base, i)
    return order


def _feature_is_binary(Xi: np.ndarray, max_rows: int = 6000) -> np.ndarray:
    """Detect binary features from (imputed) data matrix Xi."""
    Xs = np.asarray(Xi)[:max_rows]
    if Xs.ndim != 2:
        raise ValueError("Xi must be [N,D]")
    is_bin = np.zeros(Xs.shape[1], dtype=bool)
    for j in range(Xs.shape[1]):
        col = Xs[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        u = np.unique(np.round(col, 6))
        if len(u) <= 2 and set(u.tolist()).issubset({0.0, 1.0}):
            is_bin[j] = True
    return is_bin


def _is_binary_fallback_from_names(feature_names: Sequence[str]) -> np.ndarray:
    """Fallback binary detection when no reference data is available."""
    out = np.zeros(len(feature_names), dtype=bool)
    for j, n in enumerate(feature_names):
        out[j] = _split_ohe(n) is not None
    return out


def _build_ohe_groups(feature_names: List[str], is_bin: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Group OHE features by base feature name."""
    tmp: Dict[str, List[Tuple[int, str]]] = {}
    for j, name in enumerate(feature_names):
        sp = _split_ohe(name)
        if sp is None or not bool(is_bin[j]):
            continue
        tmp.setdefault(sp[0], []).append((j, sp[1]))

    out: Dict[str, Dict[str, Any]] = {}
    for base, pairs in tmp.items():
        if len(pairs) < 2:
            continue
        out[base] = {"sufs": sorted([s for _, s in pairs]), "by_col": {j: s for j, s in pairs}}
    return out


# ===== Threshold rendering utilities (dataset-specific defaults are optional) =====

FEATURE_ALIAS = {
    "Age": "Age (years)",
    "BP": "Resting blood pressure (mmHg)",
    "Cholesterol": "Cholesterol (mg/dL)",
    "Max HR": "Max heart rate (bpm)",
    "ST depression": "ST depression (Oldpeak)",
    "Number of vessels fluro": "Number of vessels (fluoroscopy)",
    "Sex": "Sex",
    "Exercise angina": "Exercise-induced angina",
    "FBS over 120": "Fasting blood sugar >120 mg/dL",
    "EKG results": "Resting ECG",
    "Thallium": "Thallium scan",
    "Slope of ST": "ST segment slope",
    "Chest pain type": "Chest pain type",
}

CATEGORY_VALUE_LABELS = {
    "Sex": {"0": "Female", "1": "Male"},
    "Exercise angina": {"0": "No", "1": "Yes"},
    "FBS over 120": {"0": "No", "1": "Yes"},
    "EKG results": {"0": "Normal", "1": "ST-T abnormal", "2": "LVH"},
    "Slope of ST": {"1": "Upsloping", "2": "Flat", "3": "Downsloping"},
    "Thallium": {"3": "Normal", "6": "Fixed defect", "7": "Reversible defect"},
}

CLINICAL_SNAP = {
    "Age": 1.0,
    "BP": 5.0,
    "Cholesterol": 10.0,
    "Max HR": 5.0,
    "ST depression": 0.1,
    "Number of vessels fluro": 1.0,
    "Chest pain type": 1.0,
}
INTEGER_BASES = {
    "Number of vessels fluro",
    "Chest pain type",
    "Slope of ST",
    "EKG results",
    "Thallium",
}


def _alias(base: str, mode: Readability) -> str:
    return base if mode == "exact" else FEATURE_ALIAS.get(base, base)


def _cat_label(base: str, val: str, mode: Readability) -> str:
    if mode in ("clinical", "pretty", "simplified"):
        m = CATEGORY_VALUE_LABELS.get(base)
        if m and val in m:
            return m[val]
    return val


def _op_sym(op: str) -> str:
    return {"<=": "≤", ">=": "≥", "!=": "≠", "==": "=", "<": "<", ">": ">"}.get(op, op)


def _snap_step(base: str, x: float, mode: Readability) -> float:
    if mode != "clinical" or not np.isfinite(x):
        return float(x)
    step = CLINICAL_SNAP.get(base)
    if not step:
        return float(x)
    return float(round(x / step) * step)


def _snap_integer_threshold(base: str, op: str, thr: float, mode: Readability) -> float:
    if not np.isfinite(thr):
        return float(thr)
    if mode != "clinical" or base not in INTEGER_BASES:
        return float(thr)
    if op in (">", ">="):
        return float(int(math.ceil(thr - 1e-9)))
    if op in ("<", "<="):
        return float(int(math.floor(thr + 1e-9)))
    return float(round(thr))


def _fmt_num(base: str, x: float, mode: Readability) -> str:
    if not np.isfinite(x):
        return "?"
    x = _snap_step(base, x, mode)
    if base in ("Age", "BP", "Cholesterol", "Max HR"):
        return f"{x:.0f}"
    if base in ("Number of vessels fluro", "Chest pain type"):
        if abs(x - round(x)) < 1e-6:
            return str(int(round(x)))
        return f"{x:.1f}"
    ax = abs(x)
    if ax >= 10:
        return f"{x:.2f}"
    if ax >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def clause_to_text(cl: Clause, mode: Readability) -> str:
    """Render a single clause as human-readable text."""
    base_disp = _alias(cl.base, mode)
    if cl.kind in ("category", "binary"):
        val = str(cl.value)
        if mode == "exact" and _split_ohe(cl.raw_feature) is not None:
            return f"{cl.raw_feature} {_op_sym(cl.op)} {val}"
        return f"{base_disp} {_op_sym(cl.op)} {_cat_label(cl.base, val, mode)}"

    v = float(cl.value)
    return f"{base_disp} {_op_sym(cl.op)} {_fmt_num(cl.base, v, mode)}"


def simplify_clauses(
    clauses: List[Clause],
    mode: Readability,
    *,
    ohe_groups: Dict[str, Any],
    base_order: Dict[str, int],
) -> List[SimplifiedCondition]:
    """Simplify multiple clauses on the same base feature into ranges/sets."""
    by_base: Dict[str, List[Clause]] = {}
    for cl in clauses:
        by_base.setdefault(cl.base, []).append(cl)

    out: List[SimplifiedCondition] = []
    for base, cls in by_base.items():
        nums = [c for c in cls if c.kind == "numeric"]
        cats = [c for c in cls if c.kind in ("category", "binary")]

        if mode == "exact":
            for c in cls:
                out.append(
                    SimplifiedCondition(
                        base=base,
                        text=clause_to_text(c, mode),
                        kind="numeric_range" if c.kind == "numeric" else "category_set",
                    )
                )
            continue

        # Numeric -> range
        if nums:
            lower = None
            upper = None
            for c in nums:
                v = float(c.value)
                if not np.isfinite(v):
                    out.append(SimplifiedCondition(base=base, text=clause_to_text(c, mode), kind="numeric_range"))
                    continue
                v = _snap_integer_threshold(base, c.op, v, mode)
                if c.op in (">", ">="):
                    lower = v if (lower is None or v > lower) else lower
                if c.op in ("<", "<="):
                    upper = v if (upper is None or v < upper) else upper

            base_disp = _alias(base, mode)
            if lower is not None and upper is not None and lower <= upper:
                out.append(
                    SimplifiedCondition(
                        base=base,
                        text=f"{base_disp}: ≥ {_fmt_num(base, lower, mode)} and ≤ {_fmt_num(base, upper, mode)}",
                        kind="numeric_range",
                        lower=lower,
                        upper=upper,
                    )
                )
            elif lower is not None:
                out.append(
                    SimplifiedCondition(
                        base=base,
                        text=f"{base_disp} ≥ {_fmt_num(base, lower, mode)}",
                        kind="numeric_range",
                        lower=lower,
                    )
                )
            elif upper is not None:
                out.append(
                    SimplifiedCondition(
                        base=base,
                        text=f"{base_disp} ≤ {_fmt_num(base, upper, mode)}",
                        kind="numeric_range",
                        upper=upper,
                    )
                )

        # Categorical -> sets
        if cats:
            eq_vals, ne_vals = [], []
            for c in cats:
                if c.op == "==":
                    eq_vals.append(str(c.value))
                if c.op == "!=":
                    ne_vals.append(str(c.value))

            base_disp = _alias(base, mode)
            g = ohe_groups.get(base)
            if eq_vals:
                allowed = sorted(set(eq_vals))
                if len(allowed) == 1:
                    out.append(
                        SimplifiedCondition(
                            base=base,
                            text=f"{base_disp} = {_cat_label(base, allowed[0], mode)}",
                            kind="category_set",
                            allowed=allowed,
                        )
                    )
                else:
                    vs = ", ".join(_cat_label(base, v, mode) for v in allowed)
                    out.append(
                        SimplifiedCondition(
                            base=base,
                            text=f"{base_disp} in {{{vs}}}",
                            kind="category_set",
                            allowed=allowed,
                        )
                    )
            elif ne_vals:
                banned = sorted(set(ne_vals))
                if g and len(g["sufs"]) == 2 and len(banned) == 1:
                    other = g["sufs"][0] if g["sufs"][1] == banned[0] else g["sufs"][1]
                    out.append(
                        SimplifiedCondition(
                            base=base,
                            text=f"{base_disp} = {_cat_label(base, other, mode)}",
                            kind="category_set",
                            allowed=[other],
                        )
                    )
                else:
                    if len(banned) == 1:
                        out.append(
                            SimplifiedCondition(
                                base=base,
                                text=f"{base_disp} ≠ {_cat_label(base, banned[0], mode)}",
                                kind="category_set",
                                banned=banned,
                            )
                        )
                    else:
                        vs = ", ".join(_cat_label(base, v, mode) for v in banned)
                        out.append(
                            SimplifiedCondition(
                                base=base,
                                text=f"{base_disp} not in {{{vs}}}",
                                kind="category_set",
                                banned=banned,
                            )
                        )

    out.sort(key=lambda sc: base_order.get(sc.base, 10_000))
    return out


def render_rule(
    clauses: List[Clause],
    mode: Readability,
    *,
    ohe_groups: Dict[str, Any],
    base_order: Dict[str, int],
) -> str:
    """Render a rule as text; optionally simplify."""
    if mode in ("simplified", "clinical"):
        conds = simplify_clauses(clauses, mode, ohe_groups=ohe_groups, base_order=base_order)
        return " AND ".join([c.text for c in conds])
    return " AND ".join([clause_to_text(c, mode) for c in clauses])


# ===== Unscaling =====

def _unscale_value(j: int, val_scaled: float, scaler: Optional[Any]) -> float:
    """Convert scaled threshold back to original feature space."""
    if scaler is None:
        return float(val_scaled)

    # Preferred: sklearn-like inverse_transform
    if hasattr(scaler, "inverse_transform"):
        try:
            D = getattr(scaler, "n_features_in_", None)
            if D is None:
                if hasattr(scaler, "mean_"):
                    D = len(scaler.mean_)
                elif hasattr(scaler, "scale_"):
                    D = len(scaler.scale_)
            if D is None:
                return float(val_scaled)
            tmp = np.zeros((1, int(D)), dtype=np.float32)
            tmp[0, int(j)] = float(val_scaled)
            inv = scaler.inverse_transform(tmp)
            return float(inv[0, int(j)])
        except Exception:
            pass

    # Fallback: StandardScaler-like
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        return float(scaler.mean_[j] + scaler.scale_[j] * val_scaled)

    return float(val_scaled)


# ===== Fact bank discovery + decoding =====

def _get_threshold_fact_bank(model: Any) -> Optional[Any]:
    """Extract a threshold-ish fact bank from a model if present."""
    if hasattr(model, "facts"):
        facts = getattr(model, "facts")
        if isinstance(facts, (ThresholdFactBank, ARFactBank, IntervalFactBank, MultiResAxisFactBank)):
            return facts
    if hasattr(model, "axis"):
        return getattr(model, "axis")
    if hasattr(model, "base_facts"):
        return getattr(model, "base_facts")
    return None


def _infer_n_thresh_per_feat(fb: Any, D: int) -> Optional[int]:
    """
    Infer thresholds-per-feature K for axis threshold fact banks.

    Supports:
    - fb.n_thresh_per_feat
    - fb.K
    - fb.num_facts / D
    - len(fb.th) / D when th is 1D
    """
    for attr in ("n_thresh_per_feat", "K"):
        if hasattr(fb, attr):
            try:
                K = int(getattr(fb, attr))
                if K > 0:
                    return K
            except Exception:
                pass

    if hasattr(fb, "num_facts"):
        try:
            F0 = int(getattr(fb, "num_facts"))
            if D > 0 and F0 % D == 0:
                K = int(F0 // D)
                if K > 0:
                    return K
        except Exception:
            pass

    if hasattr(fb, "th"):
        th = getattr(fb, "th")
        try:
            if isinstance(th, torch.Tensor) and th.ndim == 1:
                n = int(th.numel())
                if D > 0 and n % D == 0:
                    K = int(n // D)
                    if K > 0:
                        return K
            arr = np.asarray(th)
            if arr.ndim == 1:
                n = int(arr.size)
                if D > 0 and n % D == 0:
                    K = int(n // D)
                    if K > 0:
                        return K
        except Exception:
            pass

    return None


def _threshold_layout_from_factbank(fb: Any, D: int) -> Optional[Tuple[int, int]]:
    """Infer (D,K) for axis threshold banks where facts are D*K.

    Handles both:
    - th shaped [D,K]
    - th shaped [D*K] (flattened)
    """
    if hasattr(fb, "th"):
        th = fb.th
        shp = tuple(th.shape) if isinstance(th, torch.Tensor) else tuple(np.asarray(th).shape)

        if len(shp) == 2 and shp[0] == D:
            return D, int(shp[1])

        if len(shp) == 1:
            K = _infer_n_thresh_per_feat(fb, D)
            if K is not None:
                return D, int(K)

    if hasattr(fb, "num_facts"):
        F0 = int(getattr(fb, "num_facts"))
        if D > 0 and F0 % D == 0:
            return D, int(F0 // D)

    return None


def _axis_threshold_value(fb: Any, j: int, k: int, D: Optional[int] = None) -> float:
    """Get threshold value from axis threshold fact bank.

    Supports:
    - th as [D,K]
    - th as flattened [D*K] in feature-major contiguous order
    """
    th = fb.th

    def _infer_D() -> int:
        if D is not None and int(D) > 0:
            return int(D)
        for attr in ("input_dim", "D", "d_in", "n_features"):
            if hasattr(fb, attr):
                try:
                    v = int(getattr(fb, attr))
                    if v > 0:
                        return v
                except Exception:
                    pass
        return 0

    if isinstance(th, torch.Tensor):
        if th.ndim == 2:
            return float(th[int(j), int(k)].detach().cpu().item())
        if th.ndim == 1:
            D0 = _infer_D()
            D_guess = D0 if D0 > 0 else max(int(j) + 1, 1)
            K = _infer_n_thresh_per_feat(fb, D_guess) or 1
            idx = int(j) * int(K) + int(k)
            idx = max(0, min(idx, int(th.numel()) - 1))
            return float(th[idx].detach().cpu().item())
        return float("nan")

    arr = np.asarray(th)
    if arr.ndim == 2:
        return float(arr[int(j), int(k)])
    if arr.ndim == 1:
        D0 = _infer_D()
        D_guess = D0 if D0 > 0 else max(int(j) + 1, 1)
        K = _infer_n_thresh_per_feat(fb, D_guess) or 1
        idx = int(j) * int(K) + int(k)
        idx = max(0, min(idx, int(arr.size) - 1))
        return float(arr[idx])

    return float("nan")


def _extract_threshold_facts(
    model: Any,
    feature_names: Sequence[str],
    scaler: Optional[Any] = None,
    readability: Readability = "clinical",
    X_ref: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Extract human-readable threshold facts from a fact bank (global, not rules)."""
    facts = _get_threshold_fact_bank(model)
    if facts is None:
        return []

    if isinstance(facts, ARFactBank):
        facts = facts.axis

    D = len(feature_names)
    is_bin = _feature_is_binary(X_ref) if X_ref is not None else _is_binary_fallback_from_names(feature_names)

    out: List[Dict[str, Any]] = []

    if isinstance(facts, IntervalFactBank):
        a = facts.a.detach().cpu().numpy()
        w = np.exp(facts.log_width.detach().cpu().numpy())
        mid = a + 0.5 * w
        feat_idx = facts.feat_idx.detach().cpu().numpy()
        for i in range(len(mid)):
            j = int(feat_idx[i])
            if j >= D:
                continue
            raw_feat = feature_names[j]
            sp = _split_ohe(raw_feat)
            base = sp[0] if sp else raw_feat
            thr_u = _unscale_value(j, float(mid[i]), scaler)
            out.append(
                {
                    "fact_idx": int(i),
                    "clauses": [
                        Clause(
                            base=base,
                            kind="numeric",
                            op=">=",
                            value=float(thr_u),
                            raw_feature=raw_feat,
                            feature_index=j,
                        )
                    ],
                }
            )
        return out

    if isinstance(facts, MultiResAxisFactBank):
        K = int(facts.K)
        for j in range(D):
            for k in range(K):
                raw_feat = feature_names[j]
                sp = _split_ohe(raw_feat)
                base = sp[0] if sp else raw_feat
                thr_u = _unscale_value(j, float(facts.th[j, k].detach().cpu().item()), scaler)
                out.append(
                    {
                        "fact_idx": int(j * K + k),
                        "clauses": [
                            Clause(
                                base=base,
                                kind="numeric",
                                op=">=",
                                value=float(thr_u),
                                raw_feature=raw_feat,
                                feature_index=j,
                            )
                        ],
                    }
                )
        return out

    if hasattr(facts, "th"):
        layout = _threshold_layout_from_factbank(facts, D)
        if layout is None:
            return []
        _, K = layout
        for j in range(D):
            raw_feat = feature_names[j]
            sp = _split_ohe(raw_feat)
            base = sp[0] if sp else raw_feat
            for k in range(K):
                thr_u = _unscale_value(j, _axis_threshold_value(facts, j, k, D=D), scaler)
                if bool(is_bin[j]) and sp is not None:
                    out.append(
                        {
                            "fact_idx": int(j * K + k),
                            "clauses": [
                                Clause(
                                    base=base,
                                    kind="category",
                                    op="==",
                                    value=sp[1],
                                    raw_feature=raw_feat,
                                    feature_index=j,
                                )
                            ],
                        }
                    )
                else:
                    out.append(
                        {
                            "fact_idx": int(j * K + k),
                            "clauses": [
                                Clause(
                                    base=base,
                                    kind="numeric",
                                    op=">=",
                                    value=float(thr_u),
                                    raw_feature=raw_feat,
                                    feature_index=j,
                                )
                            ],
                        }
                    )
        return out

    return []


# ===== Model-family detection (lazy imports) =====

def _classes():
    """Import zoo_v2 model classes lazily to avoid hard import costs/cycles."""
    from ..zoo_v2.models import (
        # evidence
        EvidenceNet,
        MarginEvidenceNet,
        PerFeatureKappaEvidenceNet,
        LadderEvidenceNet,
        BiEvidenceNet,
        EvidenceKofNNet,
        # corner
        CornerNet,
        SoftMinCornerNet,
        KofNCornerNet,
        RingCornerNet,
        HybridCornerIntervalNet,
        # forests
        PredicateForest,
        ObliviousForest,
        LeafLinearForest,
        AttentiveForest,
        MultiResForest,
        GroupFirstForest,
        BudgetedForest,
        # group evidence
        GroupEvidenceKofNNet,
        SoftGroupEvidenceKofNNet,
        GroupSoftMinNet,
        GroupContrastNet,
        GroupRingNet,
        # regime
        RegimeRulesNet,
        # scorecard
        ScorecardWithRules,
    )
    return locals()


# ===== Global extraction: Evidence-family =====

def _extract_evidence_rules(
    model: Any,
    feature_names: Sequence[str],
    scaler: Optional[Any] = None,
    readability: Readability = "clinical",
    top_feats: int = 4,
    X_ref: Optional[np.ndarray] = None,
    class_idx: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Extract approximate rules from EvidenceNet-like models."""
    D = len(feature_names)
    is_bin = _feature_is_binary(X_ref) if X_ref is not None else _is_binary_fallback_from_names(feature_names)

    th = getattr(model, "th", None)
    if th is None or not isinstance(th, torch.Tensor):
        return []
    th_np = th.detach().cpu().numpy()  # [R,Dm]
    if th_np.ndim != 2:
        return []
    R, Dm = th_np.shape
    D_use = min(D, Dm)

    # ineq sign
    if hasattr(model, "ineq_sign_param"):
        ineq = np.tanh(model.ineq_sign_param.detach().cpu().numpy())
    elif hasattr(model, "ineq"):
        ineq = np.tanh(model.ineq.detach().cpu().numpy())
    else:
        ineq = np.ones_like(th_np)

    # evidence sign
    if hasattr(model, "e_sign_param"):
        es = np.tanh(model.e_sign_param.detach().cpu().numpy())
    elif hasattr(model, "esign"):
        es = np.tanh(model.esign.detach().cpu().numpy())
    else:
        es = np.ones_like(th_np)

    # mask
    if hasattr(model, "mask_logit"):
        mask = torch.sigmoid(model.mask_logit).detach().cpu().numpy()
    elif hasattr(model, "mask"):
        mask = torch.sigmoid(model.mask).detach().cpu().numpy()
    else:
        mask = np.ones_like(th_np)

    if not hasattr(model, "head") or not hasattr(model.head, "weight"):
        return []
    W = model.head.weight.detach().cpu().numpy()
    if W.ndim == 1:
        w = W
    else:
        c = int(class_idx) if class_idx is not None else 0
        c = max(0, min(c, W.shape[0] - 1))
        w = W[c]

    rules: List[Dict[str, Any]] = []
    for r in range(R):
        score = mask[r, :D_use] * (0.6 + 0.4 * np.abs(es[r, :D_use]))
        fidx = np.argsort(-score)[: int(top_feats)]

        clauses: List[Clause] = []
        for jj in fidx:
            j = int(jj)
            raw_feat = feature_names[j]
            sp = _split_ohe(raw_feat)
            base = sp[0] if sp else raw_feat

            op_base = ">=" if ineq[r, j] >= 0 else "<="
            op = op_base if es[r, j] >= 0 else ("<=" if op_base == ">=" else ">=")

            thr_u = _unscale_value(j, float(th_np[r, j]), scaler)

            if bool(is_bin[j]) and sp is not None:
                want_one = op in (">", ">=")
                clauses.append(
                    Clause(
                        base=base,
                        kind="category",
                        op=("==" if want_one else "!="),
                        value=sp[1],
                        raw_feature=raw_feat,
                        feature_index=j,
                    )
                )
            else:
                clauses.append(
                    Clause(
                        base=base,
                        kind="numeric",
                        op=op,
                        value=float(thr_u),
                        raw_feature=raw_feat,
                        feature_index=j,
                    )
                )

        rules.append(
            {
                "rule_idx": int(r),
                "weight": float(w[r]) if r < len(w) else 0.0,
                "clauses": clauses,
                "direction": "increases" if (r < len(w) and w[r] > 0) else "decreases",
            }
        )

    rules.sort(key=lambda rr: abs(rr["weight"]), reverse=True)
    return rules


# ===== Global extraction: Corner-family =====

def _extract_corner_rules(
    model: Any,
    feature_names: Sequence[str],
    scaler: Optional[Any] = None,
    readability: Readability = "clinical",
    top_feats: int = 4,
    X_ref: Optional[np.ndarray] = None,
    class_idx: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Extract approximate conjunction rules from CornerNet-like models."""
    D = len(feature_names)
    is_bin = _feature_is_binary(X_ref) if X_ref is not None else _is_binary_fallback_from_names(feature_names)

    if not hasattr(model, "th") or not hasattr(model, "head"):
        return []

    th = model.th.detach().cpu().numpy()  # [R,D]
    if th.ndim != 2:
        return []
    R, Dm = th.shape
    D_use = min(D, Dm)

    if hasattr(model, "sign_param"):
        ineq = np.tanh(model.sign_param.detach().cpu().numpy())
    elif hasattr(model, "sign_c"):
        ineq = np.tanh(model.sign_c.detach().cpu().numpy())
    else:
        ineq = np.ones_like(th)

    if hasattr(model, "mask_logit"):
        mask = torch.sigmoid(model.mask_logit).detach().cpu().numpy()
    elif hasattr(model, "mask"):
        mask = torch.sigmoid(model.mask).detach().cpu().numpy()
    else:
        mask = np.ones_like(th)

    W = model.head.weight.detach().cpu().numpy()
    if W.ndim == 1:
        w = W
    else:
        c = int(class_idx) if class_idx is not None else 0
        c = max(0, min(c, W.shape[0] - 1))
        w = W[c]

    rules: List[Dict[str, Any]] = []
    for r in range(R):
        score = mask[r, :D_use] * (0.6 + 0.4 * np.abs(ineq[r, :D_use]))
        fidx = np.argsort(-score)[: int(top_feats)]

        clauses: List[Clause] = []
        for jj in fidx:
            j = int(jj)
            raw_feat = feature_names[j]
            sp = _split_ohe(raw_feat)
            base = sp[0] if sp else raw_feat

            op = ">=" if ineq[r, j] >= 0 else "<="
            thr_u = _unscale_value(j, float(th[r, j]), scaler)

            if bool(is_bin[j]) and sp is not None:
                want_one = op in (">", ">=")
                clauses.append(
                    Clause(
                        base=base,
                        kind="category",
                        op=("==" if want_one else "!="),
                        value=sp[1],
                        raw_feature=raw_feat,
                        feature_index=j,
                    )
                )
            else:
                clauses.append(
                    Clause(
                        base=base,
                        kind="numeric",
                        op=op,
                        value=float(thr_u),
                        raw_feature=raw_feat,
                        feature_index=j,
                    )
                )

        rules.append(
            {
                "rule_idx": int(r),
                "weight": float(w[r]) if r < len(w) else 0.0,
                "clauses": clauses,
                "direction": "increases" if (r < len(w) and w[r] > 0) else "decreases",
            }
        )

    rules.sort(key=lambda rr: abs(rr["weight"]), reverse=True)
    return rules


# ===== Global extraction: Forest-family =====

def _sel_probs(logits: torch.Tensor, selector: str, tau: float = 0.7, dim: int = -1) -> torch.Tensor:
    if selector == "sparsemax":
        return sparsemax(logits, dim=dim)
    return torch.softmax(logits / max(float(tau), 1e-6), dim=dim)


def _decode_fact_index_to_jk(
    facts: Any,
    D: int,
    base_idx: int,
) -> Tuple[int, int, Optional[float]]:
    """Decode a base fact index (0..F0-1) into (feature j, thresh k, thr_value)."""
    if isinstance(facts, MultiResAxisFactBank):
        K = int(facts.K)
        tmp = int(base_idx // K)
        j = int(tmp % D)
        k = int(base_idx % K)
        thr = float(facts.th[j, k].detach().cpu().item())
        return j, k, thr

    layout = _threshold_layout_from_factbank(facts, D)
    if layout is None:
        j = int(base_idx % D)
        return j, 0, None
    _, K = layout
    j = int(base_idx // K)
    k = int(base_idx % K)
    thr = _axis_threshold_value(facts, j, k, D=D)
    return j, k, float(thr)


def _extract_forest_rules(
    model: Any,
    feature_names: Sequence[str],
    scaler: Optional[Any] = None,
    readability: Readability = "clinical",
    n_trees: int = 6,
    X_ref: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Extract crisp (approximate) rules from forest models."""
    if not hasattr(model, "sel_logits") or not hasattr(model, "leaf_value") or not hasattr(model, "facts"):
        return []

    D = len(feature_names)
    is_bin = _feature_is_binary(X_ref) if X_ref is not None else _is_binary_fallback_from_names(feature_names)

    facts = model.facts
    sel_logits = model.sel_logits.detach()
    leaf = model.leaf_value.detach()

    T = int(getattr(model, "T", 0)) or int(getattr(model, "n_trees", 0)) or None
    depth = int(getattr(model, "depth", 0)) or None
    selector = str(getattr(model, "selector", "sparsemax"))
    tau = float(getattr(model, "tau_select", 0.7))

    if T is None or depth is None or T <= 0 or depth <= 0:
        depth = depth or 4
        T = sel_logits.shape[0] // depth

    Fin = int(sel_logits.shape[1])
    F0 = Fin // 2

    sel = _sel_probs(sel_logits, selector=selector, tau=tau, dim=1).cpu().numpy()
    sel = sel.reshape(T, depth, Fin)

    leaf_np = leaf.cpu().numpy()
    leaf_score = leaf_np[:, :, 0] if leaf_np.ndim == 3 else leaf_np

    rules: List[Dict[str, Any]] = []
    for t in range(min(int(n_trees), int(T))):
        sel_t = sel[t]
        topf = [int(np.argmax(sel_t[d])) for d in range(depth)]

        li = int(np.argmax(np.abs(leaf_score[t])))
        lv = float(leaf_score[t, li])

        clauses: List[Clause] = []
        for d in range(depth):
            bit = (li >> (depth - 1 - d)) & 1

            f_idx = int(topf[d])
            is_neg = f_idx >= F0
            base_idx = f_idx - F0 if is_neg else f_idx

            j, _k, thr_scaled = _decode_fact_index_to_jk(facts, D, base_idx)
            if j < 0 or j >= D:
                continue

            raw_feat = feature_names[j]
            sp = _split_ohe(raw_feat)
            base = sp[0] if sp else raw_feat

            want_literal_true = (bit == 1)
            if not is_neg:
                op = ">=" if want_literal_true else "<="
            else:
                op = "<=" if want_literal_true else ">="

            if bool(is_bin[j]) and sp is not None:
                want_one = op in (">", ">=")
                clauses.append(
                    Clause(
                        base=base,
                        kind="category",
                        op=("==" if want_one else "!="),
                        value=sp[1],
                        raw_feature=raw_feat,
                        feature_index=j,
                    )
                )
            else:
                thr_u = float("nan") if thr_scaled is None else _unscale_value(j, float(thr_scaled), scaler)
                clauses.append(
                    Clause(
                        base=base,
                        kind="numeric",
                        op=op,
                        value=float(thr_u),
                        raw_feature=raw_feat,
                        feature_index=j,
                    )
                )

        rules.append(
            {
                "tree_idx": int(t),
                "leaf_idx": int(li),
                "leaf_score": float(lv),
                "clauses": clauses,
                "direction": "increases" if lv > 0 else "decreases",
            }
        )

    rules.sort(key=lambda rr: abs(rr["leaf_score"]), reverse=True)
    return rules


# ===== Local contributions =====

def _device_of(model: Any) -> torch.device:
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


def _sigmoid_np(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(x))))


def _predict_logits(model: Any, x_scaled_2d: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x_scaled_2d, dtype=torch.float32, device=_device_of(model))
        y = model(xt)
        y = y.detach().cpu().numpy()
    return np.asarray(y).reshape(-1)


def _compute_meta_from_logits(logits: np.ndarray) -> Dict[str, Any]:
    if logits.size <= 1:
        logit = float(logits.ravel()[0])
        prob = _sigmoid_np(logit)
        pred = int(logit > 0)
        return {"logit": logit, "prob": prob, "prediction": pred, "class_idx": 0}

    c = int(np.argmax(logits))
    z = logits - np.max(logits)
    p = np.exp(z) / (np.exp(z).sum() + 1e-12)
    return {"logit": float(np.max(logits)), "prob": float(p[c]), "prediction": c, "class_idx": c}


def _to_R_D_param(p: Any, R: int, D: int, like: torch.Tensor, name: str = "param") -> torch.Tensor:
    """
    Convert/broadcast p to shape [R,D] on like.device/like.dtype.

    Accepts:
      - scalar
      - [R]
      - [D]
      - [R,D]
      - [R,1] or [1,D]
      - [R*D] (reshaped)
    """
    if not isinstance(p, torch.Tensor):
        p = like.new_tensor(p)

    p = p.to(device=like.device, dtype=like.dtype)

    if p.ndim == 0:
        return p.expand(R, D)

    if p.ndim == 1:
        n = int(p.numel())
        if n == R:
            return p[:, None].expand(R, D)
        if n == D:
            return p[None, :].expand(R, D)
        if n == R * D:
            return p.reshape(R, D)
        return p.reshape(-1)[0].expand(R, D)

    if p.ndim == 2:
        if tuple(p.shape) == (R, D):
            return p
        if tuple(p.shape) == (R, 1):
            return p.expand(R, D)
        if tuple(p.shape) == (1, D):
            return p.expand(R, D)
        if int(p.numel()) == R * D:
            return p.reshape(R, D)
        return p.reshape(-1)[0].expand(R, D)

    return p.reshape(-1)[0].expand(R, D)


def _get_rule_threshold_t(model: Any, R: int, like: torch.Tensor) -> torch.Tensor:
    """
    Fetch per-rule threshold t:[R] for evidence-style gates.

    Some models use: t, t_e, t_rule, t_group.
    Some store scalar t; we broadcast to [R].
    """
    for attr in ("t", "t_e", "t_rule", "t_group"):
        if hasattr(model, attr):
            t = getattr(model, attr)
            if isinstance(t, torch.Tensor):
                tt = t.detach().to(device=like.device, dtype=like.dtype)
                if tt.numel() == 1:
                    return tt.reshape(1).repeat(int(R))
                return tt.reshape(-1)[: int(R)]
            try:
                tt = like.new_tensor(t).reshape(-1)
                if tt.numel() == 1:
                    return tt.repeat(int(R))
                return tt[: int(R)]
            except Exception:
                pass
    return like.new_zeros(int(R))


def _generic_evidence_like_rule_activations(model: Any, xt: torch.Tensor) -> torch.Tensor:
    """
    Best-effort rule activations z:[R] for "evidence-ish" models.

    Used for Group* evidence models and any future models that expose a similar
    parameterization (th + ineq + esign + mask + optional kofN gating).
    """
    if not hasattr(model, "th"):
        raise TypeError("Model has no .th; cannot compute evidence-like activations.")

    th = model.th
    if not isinstance(th, torch.Tensor) or th.ndim != 2:
        raise TypeError("Expected model.th to be a 2D torch.Tensor [R,D].")

    R, D = int(th.shape[0]), int(th.shape[1])

    # kappa can be scalar / [R] / [D] / [R,D]
    if hasattr(model, "log_kappa"):
        kappa = torch.exp(model.log_kappa).clamp(0.5, 50.0)
        kappa_RD = _to_R_D_param(kappa, R, D, like=th, name="kappa")
    else:
        kappa_RD = th.new_full((R, D), 6.0)

    # inequality direction
    if hasattr(model, "ineq_sign_param"):
        ineq_raw = torch.tanh(model.ineq_sign_param)
    elif hasattr(model, "ineq"):
        ineq_raw = torch.tanh(model.ineq)
    else:
        ineq_raw = th.new_ones(R, D)
    ineq_RD = _to_R_D_param(ineq_raw, R, D, like=th, name="ineq")

    # evidence sign
    if hasattr(model, "e_sign_param"):
        es_raw = torch.tanh(model.e_sign_param)
    elif hasattr(model, "esign"):
        es_raw = torch.tanh(model.esign)
    else:
        es_raw = th.new_ones(R, D)
    es_RD = _to_R_D_param(es_raw, R, D, like=th, name="esign")

    # mask
    if hasattr(model, "mask_logit"):
        m_raw = torch.sigmoid(model.mask_logit)
    elif hasattr(model, "mask"):
        m_raw = torch.sigmoid(model.mask)
    else:
        m_raw = th.new_ones(R, D)
    m_RD = _to_R_D_param(m_raw, R, D, like=th, name="mask")

    # optional margin
    margin = None
    if hasattr(model, "margin_param"):
        try:
            margin = torch.nn.functional.softplus(model.margin_param)
        except Exception:
            margin = None
    if margin is not None and isinstance(margin, torch.Tensor):
        margin_RD = _to_R_D_param(margin, R, D, like=th, name="margin")
    else:
        margin_RD = None

    expr = ineq_RD[None, :, :] * (xt[:, None, :] - th[None, :, :])
    if margin_RD is not None:
        expr = expr - margin_RD[None, :, :]

    c = torch.sigmoid(kappa_RD[None, :, :] * expr)  # [1,R,D]
    evidence = (m_RD[None, :, :] * es_RD[None, :, :] * (2.0 * c - 1.0)).sum(dim=2)  # [1,R]
    count = (m_RD[None, :, :] * c).sum(dim=2)  # [1,R]

    # K-of-N gating if present
    if hasattr(model, "k_frac_param"):
        t_e = _get_rule_threshold_t(model, R, like=th)
        msum = (m_RD.sum(dim=1)[None, :] + 1e-6)
        k = torch.sigmoid(model.k_frac_param)[None, :] * msum
        alpha = float(getattr(model, "alpha", getattr(model, "beta_group", 6.0)))
        beta_k = float(getattr(model, "beta_k", getattr(model, "beta", 8.0)))
        z = torch.sigmoid(alpha * (evidence - t_e[None, :])) * torch.sigmoid(beta_k * (count - k))
        return z[0].clamp(0.0, 1.0)

    # Plain evidence threshold
    t = _get_rule_threshold_t(model, R, like=th)
    beta = float(getattr(model, "beta", getattr(model, "beta_rules", 6.0)))
    z = torch.sigmoid(beta * (evidence - t[None, :]))
    return z[0].clamp(0.0, 1.0)


def _evidence_like_rule_activations(model: Any, xt: torch.Tensor) -> torch.Tensor:
    """Return z: [R] for EvidenceNet-like models (output_dim independent)."""
    # If it looks K-of-N-ish, generic is safest (covers GroupEvidenceKofNNet variants)
    if hasattr(model, "th") and hasattr(model, "k_frac_param"):
        return _generic_evidence_like_rule_activations(model, xt)

    # EvidenceNet / MarginEvidenceNet-ish
    if hasattr(model, "th") and hasattr(model, "ineq_sign_param") and hasattr(model, "e_sign_param") and hasattr(model, "mask_logit"):
        kappa = torch.exp(model.log_kappa).clamp(0.5, 50.0) if hasattr(model, "log_kappa") else xt.new_tensor(6.0)
        ineq = torch.tanh(model.ineq_sign_param)
        if hasattr(model, "margin_param"):
            margin = torch.nn.functional.softplus(model.margin_param)
            c = torch.sigmoid(kappa * (ineq[None, :, :] * (xt[:, None, :] - model.th[None, :, :]) - margin[None, :, :]))
        else:
            c = torch.sigmoid(kappa * ineq[None, :, :] * (xt[:, None, :] - model.th[None, :, :]))
        m = torch.sigmoid(model.mask_logit)[None, :, :]
        es = torch.tanh(model.e_sign_param)[None, :, :]
        evidence = (m * es * (2.0 * c - 1.0)).sum(dim=2)  # [1,R]

        beta = float(getattr(model, "beta", 6.0))
        R = int(model.th.shape[0])
        t = _get_rule_threshold_t(model, R, like=model.th)
        z = torch.sigmoid(beta * (evidence - t[None, :]))
        return z[0]

    # PerFeatureKappaEvidenceNet (and GroupContrastNet-like attribute pattern)
    if hasattr(model, "th") and hasattr(model, "ineq") and hasattr(model, "esign") and hasattr(model, "mask") and hasattr(model, "log_kappa"):
        th = model.th
        if not isinstance(th, torch.Tensor) or th.ndim != 2:
            return _generic_evidence_like_rule_activations(model, xt)

        R, D = int(th.shape[0]), int(th.shape[1])

        kappa = torch.exp(model.log_kappa).clamp(0.5, 50.0)
        kappa_RD = _to_R_D_param(kappa, R, D, like=th, name="kappa")

        ineq_RD = _to_R_D_param(torch.tanh(model.ineq), R, D, like=th, name="ineq")
        es_RD = _to_R_D_param(torch.tanh(model.esign), R, D, like=th, name="esign")
        m_RD = _to_R_D_param(torch.sigmoid(model.mask), R, D, like=th, name="mask")

        expr = kappa_RD[None, :, :] * ineq_RD[None, :, :] * (xt[:, None, :] - th[None, :, :])
        c = torch.sigmoid(expr)  # [1,R,D]
        evidence = (m_RD[None, :, :] * es_RD[None, :, :] * (2.0 * c - 1.0)).sum(dim=2)  # [1,R]

        beta = float(getattr(model, "beta", 6.0))
        t = _get_rule_threshold_t(model, R, like=th)
        z = torch.sigmoid(beta * (evidence - t[None, :]))
        return z[0]

    # LadderEvidenceNet
    if hasattr(model, "th0") and hasattr(model, "delta") and hasattr(model, "ineq") and hasattr(model, "esign") and hasattr(model, "mask"):
        inc = torch.nn.functional.softplus(model.delta)
        th = torch.cat([model.th0[:, :, None], model.th0[:, :, None] + torch.cumsum(inc, dim=2)], dim=2)  # [R,D,L]
        kappa = torch.exp(model.log_kappa).clamp(0.5, 50.0)
        ineq = torch.tanh(model.ineq)
        c = torch.sigmoid(kappa * ineq[None, :, :, None] * (xt[:, None, :, None] - th[None, :, :, :]))  # [1,R,D,L]
        w = torch.nn.functional.softplus(model.level_w_param) + 1e-6
        sev = (c * w[None, None, None, :]).sum(dim=3) / w.sum()  # [1,R,D]
        m = torch.sigmoid(model.mask)[None, :, :]
        es = torch.tanh(model.esign)[None, :, :]
        evidence = (m * es * (2.0 * sev - 1.0)).sum(dim=2)  # [1,R]

        beta = float(getattr(model, "beta", 6.0))
        R = int(model.th0.shape[0])
        t = _get_rule_threshold_t(model, R, like=model.th0)
        z = torch.sigmoid(beta * (evidence - t[None, :]))
        return z[0]

    # BiEvidenceNet
    if hasattr(model, "center") and hasattr(model, "log_width") and hasattr(model, "e_low") and hasattr(model, "e_high") and hasattr(model, "mask"):
        width = torch.exp(model.log_width).clamp(1e-3, 50.0)
        t_low = model.center - 0.5 * width
        t_high = model.center + 0.5 * width
        kappa = torch.exp(model.log_kappa).clamp(0.5, 50.0)
        low = torch.sigmoid(kappa * (t_low[None, :, :] - xt[:, None, :]))
        high = torch.sigmoid(kappa * (xt[:, None, :] - t_high[None, :, :]))
        m = torch.sigmoid(model.mask)[None, :, :]
        el = torch.tanh(model.e_low)[None, :, :]
        eh = torch.tanh(model.e_high)[None, :, :]
        evidence = (m * (el * (2.0 * low - 1.0) + eh * (2.0 * high - 1.0))).sum(dim=2)  # [1,R]

        beta = float(getattr(model, "beta", 6.0))
        R = int(model.center.shape[0])
        t = _get_rule_threshold_t(model, R, like=model.center)
        z = torch.sigmoid(beta * (evidence - t[None, :]))
        return z[0]

    # EvidenceKofNNet (classic)
    if hasattr(model, "th") and hasattr(model, "k_frac_param") and hasattr(model, "t_e") and hasattr(model, "head"):
        # (Usually already caught by the k_frac early return)
        return _generic_evidence_like_rule_activations(model, xt)

    # Generic fallback (covers GroupSoftMinNet, GroupRingNet, etc)
    if hasattr(model, "th"):
        return _generic_evidence_like_rule_activations(model, xt)

    raise TypeError("Unsupported evidence-like model for local rule activations.")


def _corner_like_rule_activations(model: Any, xt: torch.Tensor) -> torch.Tensor:
    """Return z: [R] for CornerNet-like models."""
    if hasattr(model, "th") and hasattr(model, "sign_param") and hasattr(model, "mask_logit") and hasattr(model, "log_kappa"):
        z, _, _ = corner_product_z(xt, model.th, model.sign_param, model.mask_logit, model.log_kappa)
        return z[0]

    if hasattr(model, "th_o") and hasattr(model, "sign_o") and hasattr(model, "mask_o") and hasattr(model, "th_i"):
        zo, _, _ = corner_product_z(xt, model.th_o, model.sign_o, model.mask_o, model.log_kappa)
        zi, _, _ = corner_product_z(xt, model.th_i, model.sign_i, model.mask_i, model.log_kappa)
        return (zo * (1.0 - zi))[0]

    if hasattr(model, "th_c") and hasattr(model, "sign_c") and hasattr(model, "center") and hasattr(model, "log_width"):
        kappa = torch.exp(model.log_kappa).clamp(0.5, 50.0)
        sgn = torch.tanh(model.sign_c)
        c_corner = torch.sigmoid(kappa * sgn[None, :, :] * (xt[:, None, :] - model.th_c[None, :, :]))

        width = torch.exp(model.log_width).clamp(1e-3, 50.0)
        a = model.center - 0.5 * width
        b = model.center + 0.5 * width
        c_int = torch.sigmoid(kappa * (xt[:, None, :] - a[None, :, :])) * torch.sigmoid(kappa * (b[None, :, :] - xt[:, None, :]))

        t = torch.softmax(model.type_logits, dim=2)  # [R,D,2]
        c = t[None, :, :, 0] * c_corner + t[None, :, :, 1] * c_int
        m = torch.sigmoid(model.mask_logit)
        term = (1.0 - m[None, :, :]) + m[None, :, :] * c
        z = torch.exp(safe_log(term).sum(dim=2)).clamp(0.0, 1.0)
        return z[0]

    raise TypeError("Unsupported corner-like model for local rule activations.")


def _forest_tree_outputs(model: Any, xt: torch.Tensor, class_idx: int = 0) -> torch.Tensor:
    """Compute per-tree outputs for forest family (for a single sample). Returns [T]."""
    if not hasattr(model, "facts") or not hasattr(model, "sel_logits") or not hasattr(model, "leaf_value"):
        raise TypeError("Model missing forest attributes.")

    f = model.facts(xt)  # [1,F0] or [1,num_facts]
    f_aug = torch.cat([f, 1.0 - f], dim=1)

    selector = str(getattr(model, "selector", "sparsemax"))
    tau = float(getattr(model, "tau_select", 0.7))
    depth = int(getattr(model, "depth", 4))
    T = int(getattr(model, "T", model.sel_logits.shape[0] // depth))

    sel = _sel_probs(model.sel_logits, selector=selector, tau=tau, dim=1).view(T, depth, -1)
    p = torch.einsum("bf,tdf->btd", f_aug, sel).clamp(1e-6, 1 - 1e-6)  # [1,T,depth]

    probs = xt.new_ones(1, T, 1)
    for d in range(depth):
        pd = p[:, :, d].unsqueeze(-1)
        probs = torch.cat([probs * (1.0 - pd), probs * pd], dim=2)  # [1,T,L]

    leaf = model.leaf_value
    if leaf.ndim == 2:
        y_t = (probs * leaf[None, :, :]).sum(dim=2)  # [1,T]
        return y_t[0]

    c = int(class_idx)
    c = max(0, min(c, leaf.shape[-1] - 1))
    y_t = torch.einsum("btl,tl->bt", probs, leaf[:, :, c])  # [1,T]
    return y_t[0]


def _make_signed_evidence_adapter(rules_layer: Any, weight_vec: torch.Tensor) -> Any:
    """
    Create a lightweight object compatible with _extract_evidence_rules()
    from a SignedEvidenceRuleLayer and a chosen head weight vector [R].
    """
    adapter = type("SignedEvidenceAdapter", (), {})()
    adapter.th = rules_layer.th
    adapter.ineq_sign_param = rules_layer.ineq_sign_param
    adapter.e_sign_param = rules_layer.e_sign_param
    adapter.mask_logit = rules_layer.mask_logit
    adapter.log_kappa = rules_layer.log_kappa
    adapter.t = rules_layer.t
    adapter.beta = float(getattr(rules_layer, "beta", 6.0))

    head = type("Head", (), {})()
    head.weight = weight_vec.detach().reshape(1, -1)
    adapter.head = head
    return adapter


# ===== Public API =====

def global_rules_df(
    model: Any,
    feature_names: Sequence[str],
    scaler: Optional[Any] = None,
    top_rules: int = 10,
    readability: Readability = "clinical",
    X_ref: Optional[np.ndarray] = None,
    class_idx: Optional[int] = None,
    **kwargs: Any,
):
    """Extract global rules from zoo_v2 models ranked by importance."""
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required for global_rules_df()")

    C = _classes()
    base_order = _compute_base_order(list(feature_names))

    is_bin = _feature_is_binary(X_ref) if X_ref is not None else _is_binary_fallback_from_names(feature_names)
    ohe_groups = _build_ohe_groups(list(feature_names), is_bin)

    # Evidence family (+ Group* evidence models treated as evidence-like)
    if isinstance(
        model,
        (
            C["EvidenceNet"],
            C["MarginEvidenceNet"],
            C["PerFeatureKappaEvidenceNet"],
            C["LadderEvidenceNet"],
            C["BiEvidenceNet"],
            C["EvidenceKofNNet"],
            C["GroupEvidenceKofNNet"],
            C["SoftGroupEvidenceKofNNet"],
            C["GroupSoftMinNet"],
            C["GroupContrastNet"],
            C["GroupRingNet"],
        ),
    ):
        rules = _extract_evidence_rules(
            model,
            feature_names,
            scaler=scaler,
            readability=readability,
            top_feats=int(kwargs.get("top_feats", 4)),
            X_ref=X_ref,
            class_idx=class_idx,
        )
        if rules:
            rows = []
            for r in rules[: int(top_rules)]:
                rows.append(
                    {
                        "rule_idx": r["rule_idx"],
                        "weight": r["weight"],
                        "rule_text": render_rule(r["clauses"], readability, ohe_groups=ohe_groups, base_order=base_order),
                        "direction": r["direction"],
                    }
                )
            return pd.DataFrame(rows)
        # else fall through

    # Corner family
    if isinstance(
        model,
        (
            C["CornerNet"],
            C["SoftMinCornerNet"],
            C["KofNCornerNet"],
            C["RingCornerNet"],
            C["HybridCornerIntervalNet"],
        ),
    ):
        rules = _extract_corner_rules(
            model,
            feature_names,
            scaler=scaler,
            readability=readability,
            top_feats=int(kwargs.get("top_feats", 4)),
            X_ref=X_ref,
            class_idx=class_idx,
        )
        rows = []
        for r in rules[: int(top_rules)]:
            rows.append(
                {
                    "rule_idx": r["rule_idx"],
                    "weight": r["weight"],
                    "rule_text": render_rule(r["clauses"], readability, ohe_groups=ohe_groups, base_order=base_order),
                    "direction": r["direction"],
                }
            )
        return pd.DataFrame(rows)

    # Forest family
    if isinstance(
        model,
        (
            C["PredicateForest"],
            C["ObliviousForest"],
            C["LeafLinearForest"],
            C["AttentiveForest"],
            C["MultiResForest"],
            C["GroupFirstForest"],
            C["BudgetedForest"],
        ),
    ):
        rules = _extract_forest_rules(
            model,
            feature_names,
            scaler=scaler,
            readability=readability,
            n_trees=int(kwargs.get("n_trees", min(6, int(top_rules)))),
            X_ref=X_ref,
        )
        rows = []
        for r in rules[: int(top_rules)]:
            rows.append(
                {
                    "rule_idx": r.get("tree_idx", -1),
                    "weight": r.get("leaf_score", 0.0),
                    "rule_text": render_rule(r["clauses"], readability, ohe_groups=ohe_groups, base_order=base_order),
                    "direction": r.get("direction", "unknown"),
                }
            )
        return pd.DataFrame(rows)

    # RegimeRulesNet: show regime gate conditions ranked by prior
    if isinstance(model, (C["RegimeRulesNet"],)) and hasattr(model, "th_r") and hasattr(model, "regime_prior"):
        try:
            prior = model.regime_prior.detach()
            pi0 = torch.softmax(prior, dim=0).detach().cpu().numpy()

            th_r = model.th_r.detach().cpu().numpy()  # [K,D]
            ineq_r = np.tanh(model.ineq_r.detach().cpu().numpy())  # [K,D]
            es_r = np.tanh(model.esign_r.detach().cpu().numpy())  # [K,D]
            mask_r = torch.sigmoid(model.mask_r).detach().cpu().numpy()  # [K,D]

            top_feats_regime = int(kwargs.get("top_feats_regime", kwargs.get("top_feats", 5)))
            order = np.argsort(-pi0)[: int(top_rules)]
            rows = []
            for k in order:
                score = mask_r[k] * (0.6 + 0.4 * np.abs(es_r[k]))
                fidx = np.argsort(-score)[:top_feats_regime]

                clauses: List[Clause] = []
                for j0 in fidx:
                    j = int(j0)
                    raw_feat = feature_names[j]
                    sp = _split_ohe(raw_feat)
                    base = sp[0] if sp else raw_feat

                    op_base = ">=" if ineq_r[k, j] >= 0 else "<="
                    op = op_base if es_r[k, j] >= 0 else ("<=" if op_base == ">=" else ">=")
                    thr_u = _unscale_value(j, float(th_r[k, j]), scaler)

                    if bool(is_bin[j]) and sp is not None:
                        want_one = op in (">", ">=")
                        clauses.append(
                            Clause(
                                base=base,
                                kind="category",
                                op=("==" if want_one else "!="),
                                value=sp[1],
                                raw_feature=raw_feat,
                                feature_index=j,
                            )
                        )
                    else:
                        clauses.append(
                            Clause(
                                base=base,
                                kind="numeric",
                                op=op,
                                value=float(thr_u),
                                raw_feature=raw_feat,
                                feature_index=j,
                            )
                        )

                rows.append(
                    {
                        "rule_idx": int(k),
                        "weight": float(pi0[int(k)]),
                        "rule_text": render_rule(clauses, readability, ohe_groups=ohe_groups, base_order=base_order),
                        "direction": "unknown",
                    }
                )
            return pd.DataFrame(rows)
        except Exception:
            pass

    # ScorecardWithRules: expose correction rules globally (unweighted library)
    if isinstance(model, C["ScorecardWithRules"]) and hasattr(model, "rules"):
        rules_layer = model.rules
        adapter = type("EvidenceAdapter", (), {})()
        adapter.th = rules_layer.th
        adapter.ineq_sign_param = rules_layer.ineq_sign_param
        adapter.e_sign_param = rules_layer.e_sign_param
        adapter.mask_logit = rules_layer.mask_logit
        adapter.log_kappa = rules_layer.log_kappa
        adapter.t = rules_layer.t
        adapter.beta = float(getattr(rules_layer, "beta", 6.0))

        R = int(rules_layer.th.shape[0])
        head = type("Head", (), {})()
        head.weight = torch.zeros(1, R, device=rules_layer.th.device, dtype=rules_layer.th.dtype)
        adapter.head = head

        facts = _extract_evidence_rules(
            adapter,
            feature_names,
            scaler=scaler,
            readability=readability,
            top_feats=int(kwargs.get("top_feats", 4)),
            X_ref=X_ref,
            class_idx=0,
        )
        rows = []
        for rr in facts[: int(top_rules)]:
            rows.append(
                {
                    "rule_idx": rr["rule_idx"],
                    "weight": 0.0,
                    "rule_text": render_rule(rr["clauses"], readability, ohe_groups=ohe_groups, base_order=base_order),
                    "direction": "unknown",
                }
            )
        return pd.DataFrame(rows)

    # Fallback: threshold facts
    facts = _extract_threshold_facts(model, feature_names, scaler=scaler, readability=readability, X_ref=X_ref)
    if facts:
        rows = []
        for f in facts[: int(top_rules)]:
            rows.append(
                {
                    "rule_idx": f.get("fact_idx", -1),
                    "weight": 1.0,
                    "rule_text": render_rule(f["clauses"], readability, ohe_groups=ohe_groups, base_order=base_order),
                    "direction": "unknown",
                }
            )
        return pd.DataFrame(rows)

    return pd.DataFrame(columns=["rule_idx", "weight", "rule_text", "direction"])


def local_contrib_df(
    model: Any,
    x: np.ndarray,
    feature_names: Sequence[str],
    scaler: Optional[Any] = None,
    top_rules: int = 10,
    readability: Readability = "clinical",
    X_ref: Optional[np.ndarray] = None,
    **kwargs: Any,
):
    """Compute local rule/tree contributions for a single sample."""
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required for local_contrib_df()")

    x = np.asarray(x, dtype=np.float32).reshape(1, -1)
    if scaler is not None:
        x_scaled = np.asarray(scaler.transform(x), dtype=np.float32)
    else:
        x_scaled = x

    logits = _predict_logits(model, x_scaled)
    meta = _compute_meta_from_logits(logits)
    class_idx = int(meta["class_idx"])

    base_order = _compute_base_order(list(feature_names))
    is_bin = _feature_is_binary(X_ref) if X_ref is not None else _is_binary_fallback_from_names(feature_names)
    ohe_groups = _build_ohe_groups(list(feature_names), is_bin)

    C = _classes()
    device = _device_of(model)
    xt = torch.tensor(x_scaled, dtype=torch.float32, device=device)

    # Evidence family (+ Group* evidence models)
    if isinstance(
        model,
        (
            C["EvidenceNet"],
            C["MarginEvidenceNet"],
            C["PerFeatureKappaEvidenceNet"],
            C["LadderEvidenceNet"],
            C["BiEvidenceNet"],
            C["EvidenceKofNNet"],
            C["GroupEvidenceKofNNet"],
            C["SoftGroupEvidenceKofNNet"],
            C["GroupSoftMinNet"],
            C["GroupContrastNet"],
            C["GroupRingNet"],
        ),
    ):
        with torch.no_grad():
            z = _evidence_like_rule_activations(model, xt)  # [R]
            if hasattr(model, "head") and hasattr(model.head, "weight"):
                W = model.head.weight
                if W.ndim == 2:
                    w = W[class_idx]
                else:
                    w = W
            else:
                # last resort if some model variant has no head
                w = z.new_ones(z.shape[0])
            contrib = (z * w).detach().cpu().numpy()

        global_rules = _extract_evidence_rules(
            model,
            feature_names,
            scaler=scaler,
            readability=readability,
            top_feats=int(kwargs.get("top_feats", 4)),
            X_ref=X_ref,
            class_idx=class_idx,
        )
        rule_map = {int(rr["rule_idx"]): rr for rr in global_rules}

        idxs = np.argsort(-np.abs(contrib))[: int(top_rules)]
        rows = []
        for i in idxs:
            r = int(i)
            rr = rule_map.get(r)
            if rr is None:
                continue
            rows.append(
                {
                    "rule_idx": r,
                    "contribution": float(contrib[r]),
                    "activation": float(z[r].detach().cpu().item()),
                    "weight": float(w[r].detach().cpu().item()) if isinstance(w, torch.Tensor) else float(w[r]),
                    "rule_text": render_rule(rr["clauses"], readability, ohe_groups=ohe_groups, base_order=base_order),
                    "direction": "increases" if contrib[r] > 0 else "decreases",
                }
            )
        return pd.DataFrame(rows), meta

    # Corner family
    if isinstance(
        model,
        (
            C["CornerNet"],
            C["SoftMinCornerNet"],
            C["KofNCornerNet"],
            C["RingCornerNet"],
            C["HybridCornerIntervalNet"],
        ),
    ):
        with torch.no_grad():
            z = _corner_like_rule_activations(model, xt)  # [R]
            W = model.head.weight
            if W.ndim == 2:
                w = W[class_idx]
            else:
                w = W
            contrib = (z * w).detach().cpu().numpy()

        global_rules = _extract_corner_rules(
            model,
            feature_names,
            scaler=scaler,
            readability=readability,
            top_feats=int(kwargs.get("top_feats", 4)),
            X_ref=X_ref,
            class_idx=class_idx,
        )
        rule_map = {int(rr["rule_idx"]): rr for rr in global_rules}

        idxs = np.argsort(-np.abs(contrib))[: int(top_rules)]
        rows = []
        for i in idxs:
            r = int(i)
            rr = rule_map.get(r)
            if rr is None:
                continue
            rows.append(
                {
                    "rule_idx": r,
                    "contribution": float(contrib[r]),
                    "activation": float(z[r].detach().cpu().item()),
                    "weight": float(w[r].detach().cpu().item()),
                    "rule_text": render_rule(rr["clauses"], readability, ohe_groups=ohe_groups, base_order=base_order),
                    "direction": "increases" if contrib[r] > 0 else "decreases",
                }
            )
        return pd.DataFrame(rows), meta

    # RegimeRulesNet (pi-weighted rule contributions)
    if isinstance(model, (C["RegimeRulesNet"],)):
        if not (hasattr(model, "_regime_group_evidence") and hasattr(model, "regime_gate") and hasattr(model, "rules")):
            df = global_rules_df(
                model,
                feature_names,
                scaler=scaler,
                top_rules=top_rules,
                readability=readability,
                X_ref=X_ref,
                **kwargs,
            )
            df = df.copy()
            df["contribution"] = 0.0
            df["activation"] = 0.0
            df = df[["rule_idx", "contribution", "activation", "weight", "rule_text", "direction"]]
            return df.head(int(top_rules)), meta

        with torch.no_grad():
            eg = model._regime_group_evidence(xt)  # [1,K,G]
            z_reg = model.regime_gate(eg).clamp_min(1e-6)  # [1,K]
            logits_k = model.regime_prior[None, :] + torch.log(z_reg)  # [1,K]
            pi = torch.softmax(logits_k, dim=1)[0]  # [K]

            z_rules = model.rules(xt)[0]  # [R]

            out_dim = int(getattr(model, "output_dim", 1) or 1)
            if out_dim == 1:
                Wkr = model.W  # [K,R]
                w_vec = (pi[:, None] * Wkr).sum(dim=0)  # [R]
            else:
                c = int(class_idx)
                Wkrc = model.W[:, :, c]  # [K,R]
                w_vec = (pi[:, None] * Wkrc).sum(dim=0)  # [R]

            contrib = (z_rules * w_vec).detach().cpu().numpy()

        adapter = _make_signed_evidence_adapter(model.rules, w_vec)
        global_rules = _extract_evidence_rules(
            adapter,
            feature_names,
            scaler=scaler,
            readability=readability,
            top_feats=int(kwargs.get("top_feats", 4)),
            X_ref=X_ref,
            class_idx=0,
        )
        rule_map = {int(rr["rule_idx"]): rr for rr in global_rules}

        idxs = np.argsort(-np.abs(contrib))[: int(top_rules)]
        rows = []
        for i in idxs:
            r = int(i)
            rr = rule_map.get(r)
            if rr is None:
                continue
            rows.append(
                {
                    "rule_idx": r,
                    "contribution": float(contrib[r]),
                    "activation": float(z_rules[r].detach().cpu().item()),
                    "weight": float(w_vec[r].detach().cpu().item()),
                    "rule_text": render_rule(rr["clauses"], readability, ohe_groups=ohe_groups, base_order=base_order),
                    "direction": "increases" if contrib[r] > 0 else "decreases",
                }
            )

        try:
            pi_np = pi.detach().cpu().numpy()
            topk = np.argsort(-pi_np)[: min(5, len(pi_np))]
            meta = dict(meta)
            meta["regime_pi_top"] = [(int(k), float(pi_np[k])) for k in topk]
        except Exception:
            pass

        return pd.DataFrame(rows), meta

    # Forest family (tree contributions)
    if isinstance(
        model,
        (
            C["PredicateForest"],
            C["ObliviousForest"],
            C["LeafLinearForest"],
            C["AttentiveForest"],
            C["MultiResForest"],
            C["GroupFirstForest"],
            C["BudgetedForest"],
        ),
    ):
        with torch.no_grad():
            y_t = _forest_tree_outputs(model, xt, class_idx=class_idx)  # [T]

            if isinstance(model, C["AttentiveForest"]):
                f = model.facts(xt)
                f_aug = torch.cat([f, 1.0 - f], dim=1)
                a_logits = model.att(f_aug)  # [1,T]
                selector = str(getattr(model, "selector", "sparsemax"))
                a = _sel_probs(a_logits, selector=selector, tau=float(getattr(model, "tau_select", 0.7)), dim=1)[0]
                contrib_t = (a * y_t).detach().cpu().numpy()
            else:
                contrib_t = y_t.detach().cpu().numpy()

        global_rules = _extract_forest_rules(
            model,
            feature_names,
            scaler=scaler,
            readability=readability,
            n_trees=int(min(int(top_rules), int(getattr(model, "T", len(contrib_t))))),
            X_ref=X_ref,
        )

        idxs = np.argsort(-np.abs(contrib_t))[: int(top_rules)]
        rows = []
        for i in idxs:
            t = int(i)
            rr = next((g for g in global_rules if int(g.get("tree_idx", -1)) == t), None)
            rule_text = ""
            leaf_score = float("nan")
            direction = "unknown"
            if rr is not None:
                rule_text = render_rule(rr["clauses"], readability, ohe_groups=ohe_groups, base_order=base_order)
                leaf_score = float(rr.get("leaf_score", float("nan")))
                direction = rr.get("direction", "unknown")

            rows.append(
                {
                    "tree_idx": t,
                    "contribution": float(contrib_t[t]),
                    "tree_output": float(y_t[t].detach().cpu().item()),
                    "leaf_score": leaf_score,
                    "rule_text": rule_text,
                    "direction": direction,
                }
            )
        return pd.DataFrame(rows), meta

    # Fallback: global rules with zero contributions
    df = global_rules_df(
        model,
        feature_names,
        scaler=scaler,
        top_rules=top_rules,
        readability=readability,
        X_ref=X_ref,
        **kwargs,
    )
    if len(df) == 0:
        return (
            pd.DataFrame(columns=["rule_idx", "contribution", "activation", "weight", "rule_text", "direction"]),
            meta,
        )
    df = df.copy()
    df["contribution"] = 0.0
    df["activation"] = 0.0
    df = df[["rule_idx", "contribution", "activation", "weight", "rule_text", "direction"]]
    return df.head(int(top_rules)), meta


def explain_prediction(
    model: Any,
    x: np.ndarray,
    feature_names: Sequence[str],
    scaler: Optional[Any] = None,
    readability: Readability = "clinical",
    top_rules: int = 5,
    X_ref: Optional[np.ndarray] = None,
) -> str:
    """Generate a human-readable explanation for a single prediction."""
    contrib_df, meta = local_contrib_df(
        model,
        x,
        feature_names,
        scaler=scaler,
        top_rules=top_rules,
        readability=readability,
        X_ref=X_ref,
    )

    lines: List[str] = []
    lines.append(f"Prediction: {meta['prediction']} (probability: {meta['prob']:.3f})")
    lines.append(f"Logit: {meta['logit']:+.3f}")
    lines.append("-" * 60)

    if "rule_idx" in contrib_df.columns:
        lines.append(f"Top {min(int(top_rules), len(contrib_df))} contributing rules:")
        for _, row in contrib_df.head(int(top_rules)).iterrows():
            direction = "increases" if row.get("direction", "unknown") == "increases" else "decreases"
            lines.append(f"\n{direction} ({float(row.get('contribution', 0.0)):+.3f}):")
            lines.append(f"  IF {row.get('rule_text', '')}")
    else:
        lines.append(f"Top {min(int(top_rules), len(contrib_df))} contributing trees:")
        for _, row in contrib_df.head(int(top_rules)).iterrows():
            direction = row.get("direction", "unknown")
            lines.append(f"\n{direction} ({float(row.get('contribution', 0.0)):+.3f}):")
            lines.append(f"  IF {row.get('rule_text', '')}")

    return "\n".join(lines)


def export_global_rules(
    model: Any,
    feature_names: Sequence[str],
    path: str,
    scaler: Optional[Any] = None,
    top_rules: int = 20,
    readability: Readability = "clinical",
    format: Literal["txt", "json", "csv"] = "txt",
    X_ref: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> None:
    """Export global rules to file."""
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required for export_global_rules()")

    df = global_rules_df(
        model,
        feature_names,
        scaler=scaler,
        top_rules=top_rules,
        readability=readability,
        X_ref=X_ref,
        **kwargs,
    )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if format == "txt":
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Global rules for {type(model).__name__}\n")
            f.write("=" * 60 + "\n\n")
            for _, row in df.iterrows():
                f.write(f"Rule {int(row['rule_idx'])} ({row.get('direction','unknown')} prediction):\n")
                f.write(f"  Weight: {float(row['weight']):+,.6f}\n")
                f.write(f"  IF {row['rule_text']}\n\n")

    elif format == "json":
        import json

        rules = []
        for _, row in df.iterrows():
            rules.append(
                {
                    "rule_idx": int(row["rule_idx"]),
                    "weight": float(row["weight"]),
                    "rule_text": str(row["rule_text"]),
                    "direction": str(row.get("direction", "unknown")),
                }
            )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"model": type(model).__name__, "readability": readability, "rules": rules},
                f,
                indent=2,
            )

    elif format == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unknown format: {format}")


__all__ = [
    "Clause",
    "SimplifiedCondition",
    "global_rules_df",
    "local_contrib_df",
    "explain_prediction",
    "export_global_rules",
]
