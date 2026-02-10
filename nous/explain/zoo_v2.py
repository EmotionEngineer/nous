# nous/explain/zoo_v2.py
"""
Interpretation utilities for Nous zoo_v2 models.
Provides consistent global/local explanation APIs across model families.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ..zoo_v2 import (
    # Evidence family
    EvidenceNet, MarginEvidenceNet, PerFeatureKappaEvidenceNet, BiEvidenceNet,
    LadderEvidenceNet, EvidenceKofNNet,
    # Group evidence family
    GroupEvidenceKofNNet, SoftGroupEvidenceKofNNet, GroupSoftMinNet,
    GroupContrastNet, GroupRingNet,
    # Corner family
    CornerNet, SoftMinCornerNet, KofNCornerNet, RingCornerNet, HybridCornerIntervalNet,
    # Regime models
    RegimeRulesNet,
    # Forest family
    PredicateForest, ObliviousForest, LeafLinearForest, AttentiveForest,
    MultiResForest, GroupFirstForest, BudgetedForest,
    # Rule trees
    RuleTree, SparseRuleTree, RuleDiagram,
    # Other models
    TemplateNet, RuleListNet, FactDiagram, PriorityMixtureNet,
    ClauseNet, ARLogitAND, Scorecard, ScorecardWithRules, NALogicNet,
    IntervalLogitAND, RelationalLogitAND, BoxNet,
    # Fact banks
    ThresholdFactBank, IntervalFactBank, RelationalFactBank, ARFactBank,
    MultiResAxisFactBank,
)
from ..zoo import ThresholdFactBank as LegacyThresholdFactBank
from .facts_desc import render_fact_descriptions as _render_beta_facts

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
    """Split OHE feature name into base feature and category value."""
    m = re.match(r"^(.+)_(.+)$", name)
    if not m or len(m.group(2)) > 30:  # avoid splitting non-OHE names
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
    """Detect binary features from imputed data."""
    Xs = Xi[:max_rows]
    is_bin = np.zeros(Xs.shape[1], dtype=bool)
    for j in range(Xs.shape[1]):
        u = np.unique(np.round(Xs[:, j], 6))
        if len(u) <= 2 and set(u.tolist()).issubset({0.0, 1.0}):
            is_bin[j] = True
    return is_bin

def _build_ohe_groups(feature_names: List[str], is_bin: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Group OHE features by base feature name."""
    tmp: Dict[str, List[Tuple[int, str]]] = {}
    for j, name in enumerate(feature_names):
        sp = _split_ohe(name)
        if sp is None or not is_bin[j]:
            continue
        tmp.setdefault(sp[0], []).append((j, sp[1]))
    
    out: Dict[str, Dict[str, Any]] = {}
    for base, pairs in tmp.items():
        if len(pairs) < 2:
            continue
        out[base] = {"sufs": sorted([s for _, s in pairs]), "by_col": {j: s for j, s in pairs}}
    return out

# ===== Threshold rendering utilities =====
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
    "Age": 1.0, "BP": 5.0, "Cholesterol": 10.0, "Max HR": 5.0,
    "ST depression": 0.1, "Number of vessels fluro": 1.0, "Chest pain type": 1.0
}
INTEGER_BASES = {
    "Number of vessels fluro", "Chest pain type", "Slope of ST",
    "EKG results", "Thallium"
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
    base_order: Dict[str, int]
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
                out.append(SimplifiedCondition(
                    base=base,
                    text=clause_to_text(c, mode),
                    kind="numeric_range" if c.kind == "numeric" else "category_set"
                ))
            continue
        
        # Numeric -> range simplification
        if nums:
            lower = None
            upper = None
            for c in nums:
                v = float(c.value)
                if not np.isfinite(v):
                    out.append(SimplifiedCondition(
                        base=base,
                        text=clause_to_text(c, mode),
                        kind="numeric_range"
                    ))
                    continue
                v = _snap_integer_threshold(base, c.op, v, mode)
                if c.op in (">", ">="):
                    lower = v if (lower is None or v > lower) else lower
                if c.op in ("<", "<="):
                    upper = v if (upper is None or v < upper) else upper
            
            base_disp = _alias(base, mode)
            if lower is not None and upper is not None and lower <= upper:
                out.append(SimplifiedCondition(
                    base=base,
                    text=f"{base_disp}: ≥ {_fmt_num(base, lower, mode)} and ≤ {_fmt_num(base, upper, mode)}",
                    kind="numeric_range",
                    lower=lower,
                    upper=upper
                ))
            elif lower is not None:
                out.append(SimplifiedCondition(
                    base=base,
                    text=f"{base_disp} ≥ {_fmt_num(base, lower, mode)}",
                    kind="numeric_range",
                    lower=lower
                ))
            elif upper is not None:
                out.append(SimplifiedCondition(
                    base=base,
                    text=f"{base_disp} ≤ {_fmt_num(base, upper, mode)}",
                    kind="numeric_range",
                    upper=upper
                ))
        
        # Categorical -> set simplification
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
                    out.append(SimplifiedCondition(
                        base=base,
                        text=f"{base_disp} = {_cat_label(base, allowed[0], mode)}",
                        kind="category_set",
                        allowed=allowed
                    ))
                else:
                    vs = ", ".join(_cat_label(base, v, mode) for v in allowed)
                    out.append(SimplifiedCondition(
                        base=base,
                        text=f"{base_disp} in {{{vs}}}",
                        kind="category_set",
                        allowed=allowed
                    ))
            elif ne_vals:
                banned = sorted(set(ne_vals))
                if g and len(g["sufs"]) == 2 and len(banned) == 1:
                    other = g["sufs"][0] if g["sufs"][1] == banned[0] else g["sufs"][1]
                    out.append(SimplifiedCondition(
                        base=base,
                        text=f"{base_disp} = {_cat_label(base, other, mode)}",
                        kind="category_set",
                        allowed=[other]
                    ))
                else:
                    if len(banned) == 1:
                        out.append(SimplifiedCondition(
                            base=base,
                            text=f"{base_disp} ≠ {_cat_label(base, banned[0], mode)}",
                            kind="category_set",
                            banned=banned
                        ))
                    else:
                        vs = ", ".join(_cat_label(base, v, mode) for v in banned)
                        out.append(SimplifiedCondition(
                            base=base,
                            text=f"{base_disp} not in {{{vs}}}",
                            kind="category_set",
                            banned=banned
                        ))
    
    out.sort(key=lambda sc: base_order.get(sc.base, 10_000))
    return out

def render_rule(
    clauses: List[Clause],
    mode: Readability,
    *,
    ohe_groups: Dict[str, Any],
    base_order: Dict[str, int]
) -> str:
    """Render a rule as human-readable text with optional simplification."""
    if mode in ("simplified", "clinical"):
        conds = simplify_clauses(clauses, mode, ohe_groups=ohe_groups, base_order=base_order)
        return " AND ".join([c.text for c in conds])
    return " AND ".join([clause_to_text(c, mode) for c in clauses])

# ===== Model-specific interpretation utilities =====
def _unscale_value(model: Any, j: int, val_scaled: float, scaler: Optional[Any]) -> float:
    """Convert scaled threshold back to original feature space."""
    if scaler is None or not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_'):
        return float(val_scaled)
    return float(scaler.mean_[j] + scaler.scale_[j] * val_scaled)

def _get_threshold_fact_bank(model: Any) -> Optional[Union[ThresholdFactBank, LegacyThresholdFactBank, ARFactBank]]:
    """Extract threshold fact bank from model if present."""
    if hasattr(model, 'facts'):
        facts = model.facts
        if isinstance(facts, (ThresholdFactBank, LegacyThresholdFactBank, ARFactBank)):
            return facts
    if hasattr(model, 'axis'):
        return model.axis
    if hasattr(model, 'base_facts'):
        return model.base_facts
    return None

def _extract_threshold_facts(
    model: Any,
    feature_names: Sequence[str],
    scaler: Optional[Any] = None,
    readability: Readability = "clinical"
) -> List[Dict]:
    """Extract human-readable threshold facts from fact bank."""
    facts = _get_threshold_fact_bank(model)
    if facts is None:
        return []
    
    # Handle different fact bank types
    if isinstance(facts, ARFactBank):
        # Extract axis facts only (relational facts are harder to interpret)
        facts = facts.axis
    
    D = len(feature_names)
    is_bin = _feature_is_binary(np.zeros((1, D)))  # placeholder; real usage needs data
    
    # Get thresholds and parameters
    if hasattr(facts, 'th'):
        th = facts.th.detach().cpu().numpy()
        feat_idx = facts.feat_idx.detach().cpu().numpy() if hasattr(facts, 'feat_idx') else np.arange(len(th))
    elif hasattr(facts, 'a') and hasattr(facts, 'log_width'):
        # Interval facts
        a = facts.a.detach().cpu().numpy()
        b = a + np.exp(facts.log_width.detach().cpu().numpy())
        # We'll use midpoints for simplicity in global view
        th = (a + b) / 2
        feat_idx = facts.feat_idx.detach().cpu().numpy() if hasattr(facts, 'feat_idx') else np.arange(len(th))
    else:
        return []
    
    # Build clauses for each fact
    clauses_list = []
    for i in range(len(th)):
        j = int(feat_idx[i]) if i < len(feat_idx) else i % D
        if j >= D:
            continue
        
        raw_feat = feature_names[j]
        sp = _split_ohe(raw_feat)
        base = sp[0] if sp else raw_feat
        
        # Determine clause properties
        if is_bin[j] and sp is not None:
            # Binary/OHE feature
            clauses_list.append({
                'fact_idx': i,
                'clauses': [Clause(
                    base=base,
                    kind="category",
                    op="==",
                    value=sp[1] if sp else "1",
                    raw_feature=raw_feat,
                    feature_index=j
                )]
            })
        else:
            # Numeric feature - use midpoint threshold
            thr_u = _unscale_value(model, j, float(th[i]), scaler)
            clauses_list.append({
                'fact_idx': i,
                'clauses': [Clause(
                    base=base,
                    kind="numeric",
                    op=">=",
                    value=float(thr_u),
                    raw_feature=raw_feat,
                    feature_index=j
                )]
            })
    
    return clauses_list

def _extract_evidence_rules(
    model: Any,
    feature_names: Sequence[str],
    scaler: Optional[Any] = None,
    readability: Readability = "clinical",
    top_feats: int = 4
) -> List[Dict]:
    """Extract rules from EvidenceNet family models."""
    D = len(feature_names)
    is_bin = _feature_is_binary(np.zeros((1, D)))  # placeholder
    
    # Get model parameters
    if not hasattr(model, 'th') or not hasattr(model, 'head'):
        return []
    
    th = model.th.detach().cpu().numpy()
    R, D_model = th.shape
    
    # Get evidence signs and masks
    if hasattr(model, 'e_sign_param'):
        es = np.tanh(model.e_sign_param.detach().cpu().numpy())
        ineq = np.tanh(model.ineq_sign_param.detach().cpu().numpy())
        mask = torch.sigmoid(model.mask_logit).detach().cpu().numpy() if hasattr(model, 'mask_logit') else np.ones_like(th)
    elif hasattr(model, 'esign'):
        es = np.tanh(model.esign.detach().cpu().numpy())
        ineq = np.tanh(model.ineq.detach().cpu().numpy())
        mask = torch.sigmoid(model.mask).detach().cpu().numpy() if hasattr(model, 'mask') else np.ones_like(th)
    else:
        return []
    
    # Get head weights
    w = model.head.weight.detach().cpu().numpy().flatten()
    
    rules = []
    for r in range(min(R, 100)):  # limit to 100 rules for performance
        # Compute feature importance scores for this rule
        score = mask[r] * (0.6 + 0.4 * np.abs(es[r]))
        fidx = np.argsort(-score)[:top_feats]
        
        clauses = []
        for j in fidx:
            j = int(j)
            if j >= D:
                continue
            
            raw_feat = feature_names[j]
            sp = _split_ohe(raw_feat)
            base = sp[0] if sp else raw_feat
            
            # Determine operator direction
            op_base = ">=" if ineq[r, j] >= 0 else "<="
            op = op_base if es[r, j] >= 0 else ("<=" if op_base == ">=" else ">=")
            
            # Get unscaled threshold
            thr_u = _unscale_value(model, j, float(th[r, j]), scaler)
            
            # Create clause
            if is_bin[j] and sp is not None:
                want_one = op in (">", ">=")
                clauses.append(Clause(
                    base=base,
                    kind="category",
                    op=("==" if want_one else "!="),
                    value=sp[1] if sp else str(int(want_one)),
                    raw_feature=raw_feat,
                    feature_index=j
                ))
            else:
                clauses.append(Clause(
                    base=base,
                    kind="numeric",
                    op=op,
                    value=float(thr_u),
                    raw_feature=raw_feat,
                    feature_index=j
                ))
        
        rules.append({
            'rule_idx': r,
            'weight': float(w[r]) if r < len(w) else 0.0,
            'clauses': clauses,
            'direction': 'increases' if w[r] > 0 else 'decreases' if r < len(w) else 'unknown'
        })
    
    # Sort by absolute weight
    rules.sort(key=lambda r: abs(r['weight']), reverse=True)
    return rules

def _extract_forest_rules(
    model: Any,
    feature_names: Sequence[str],
    scaler: Optional[Any] = None,
    readability: Readability = "clinical",
    n_trees: int = 6
) -> List[Dict]:
    """Extract crisp rules from forest models."""
    if not hasattr(model, 'sel_logits') or not hasattr(model, 'leaf_value'):
        return []
    
    D = len(feature_names)
    is_bin = _feature_is_binary(np.zeros((1, D)))  # placeholder
    
    sel = torch.softmax(model.sel_logits / max(0.7, 1e-6), dim=1).detach().cpu().numpy()
    leaf = model.leaf_value.detach().cpu().numpy()
    
    T = int(model.T) if hasattr(model, 'T') else sel.shape[0] // (int(model.depth) if hasattr(model, 'depth') else 4)
    depth = int(model.depth) if hasattr(model, 'depth') else 4
    Fin = sel.shape[1]
    F0 = Fin // 2
    
    # Get threshold grid if available
    th_grid = None
    if hasattr(model, 'facts') and hasattr(model.facts, 'th'):
        th_grid = model.facts.th.detach().cpu().numpy()
    elif hasattr(model, 'th'):
        th_grid = model.th.detach().cpu().numpy()
    
    rules = []
    for t in range(min(n_trees, T)):
        # Get top feature selector per depth
        sel_t = sel[t*depth:(t+1)*depth] if sel.shape[0] > T else sel.reshape(T, depth, Fin)[t]
        topf = [int(np.argmax(sel_t[d])) for d in range(depth)]
        
        # Get leaf index with highest value
        leaf_idx = int(np.argmax(leaf[t]) if leaf.ndim == 2 else np.argmax(leaf))
        leaf_score = float(leaf[t, leaf_idx] if leaf.ndim == 2 else leaf[leaf_idx])
        
        clauses = []
        for d in range(depth):
            bit = (leaf_idx >> (depth - 1 - d)) & 1  # right=1
            f_idx = topf[d]
            is_neg = (f_idx >= F0)
            base_idx = f_idx - F0 if is_neg else f_idx
            j = int(base_idx // (Fin // (2 * D))) if Fin > 2 * D else base_idx % D
            
            if j >= D:
                continue
            
            raw_feat = feature_names[j]
            sp = _split_ohe(raw_feat)
            base = sp[0] if sp else raw_feat
            
            # Determine operator
            negate = (bit == 0)
            want_one = not (is_neg ^ negate)
            
            if is_bin[j] and sp is not None:
                clauses.append(Clause(
                    base=base,
                    kind="category",
                    op=("==" if want_one else "!="),
                    value=sp[1] if sp else str(int(want_one)),
                    raw_feature=raw_feat,
                    feature_index=j
                ))
            else:
                # Get threshold value
                thr_u = float("nan")
                if th_grid is not None:
                    if th_grid.ndim == 2:
                        thr_u = _unscale_value(model, j, float(th_grid[j, base_idx % (Fin // (2 * D))]), scaler)
                    else:
                        thr_u = _unscale_value(model, j, float(th_grid[t, j]), scaler)
                
                op = "<=" if is_neg ^ negate else ">="
                clauses.append(Clause(
                    base=base,
                    kind="numeric",
                    op=op,
                    value=float(thr_u),
                    raw_feature=raw_feat,
                    feature_index=j
                ))
        
        rules.append({
            'tree_idx': t,
            'leaf_idx': leaf_idx,
            'leaf_score': leaf_score,
            'clauses': clauses,
            'direction': 'increases' if leaf_score > 0 else 'decreases'
        })
    
    # Sort by absolute leaf score
    rules.sort(key=lambda r: abs(r['leaf_score']), reverse=True)
    return rules

# ===== Public API =====
def global_rules_df(
    model: Any,
    feature_names: Sequence[str],
    scaler: Optional[Any] = None,
    top_rules: int = 10,
    readability: Readability = "clinical",
    **kwargs
) -> pd.DataFrame:
    """
    Extract global rules from zoo_v2 models ranked by importance.
    
    Parameters
    ----------
    model : Any
        Trained Nous zoo_v2 model
    feature_names : Sequence[str]
        Original feature names (before OHE expansion)
    scaler : Optional[Any]
        Scaler used during training (e.g., StandardScaler) to unscale thresholds
    top_rules : int
        Number of top rules to return
    readability : Readability
        Readability mode: "exact", "pretty", "simplified", or "clinical"
    **kwargs : dict
        Model-specific parameters (e.g., top_feats for evidence models)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: rule_idx, weight, rule_text, direction
    """
    # Determine model family and dispatch to appropriate extractor
    model_type = type(model).__name__
    
    # Evidence family
    if isinstance(model, (
        EvidenceNet, MarginEvidenceNet, PerFeatureKappaEvidenceNet,
        BiEvidenceNet, LadderEvidenceNet, EvidenceKofNNet
    )):
        rules = _extract_evidence_rules(
            model, feature_names, scaler, readability,
            top_feats=kwargs.get('top_feats', 4)
        )
    
    # Forest family
    elif isinstance(model, (
        PredicateForest, ObliviousForest, LeafLinearForest, AttentiveForest,
        MultiResForest, GroupFirstForest, BudgetedForest
    )):
        rules = _extract_forest_rules(
            model, feature_names, scaler, readability,
            n_trees=kwargs.get('n_trees', min(6, top_rules))
        )
    
    # Corner family (simplified - treat as evidence-like)
    elif isinstance(model, (
        CornerNet, SoftMinCornerNet, KofNCornerNet, RingCornerNet, HybridCornerIntervalNet
    )):
        # Reuse evidence extractor with adapted parameters
        rules = _extract_evidence_rules(
            model, feature_names, scaler, readability,
            top_feats=kwargs.get('top_feats', 4)
        )
    
    # Group evidence models (simplified extraction)
    elif isinstance(model, (
        GroupEvidenceKofNNet, SoftGroupEvidenceKofNNet, GroupSoftMinNet,
        GroupContrastNet, GroupRingNet
    )):
        # Extract as evidence rules but note group structure in metadata
        rules = _extract_evidence_rules(
            model, feature_names, scaler, readability,
            top_feats=kwargs.get('top_feats', 4)
        )
        # Add group metadata if available
        if hasattr(model, 'gi') and hasattr(model.gi, 'groups'):
            for r in rules:
                r['groups'] = model.gi.groups
    
    # RegimeRulesNet (extract regime conditions + top rules)
    elif isinstance(model, RegimeRulesNet):
        # Extract regime conditions
        if hasattr(model, 'th_r'):
            regimes = []
            th_r = model.th_r.detach().cpu().numpy()
            K, D = th_r.shape
            
            for k in range(min(K, top_rules)):
                # Simplified regime condition extraction
                regimes.append({
                    'regime_idx': k,
                    'condition': f"Regime {k} (prior={float(torch.softmax(model.regime_prior, dim=0)[k]):.3f})",
                    'weight': float(torch.softmax(model.regime_prior, dim=0)[k])
                })
            return pd.DataFrame(regimes)
        return pd.DataFrame()
    
    # Fallback: threshold facts only
    else:
        facts = _extract_threshold_facts(model, feature_names, scaler, readability)
        if facts:
            return pd.DataFrame([{
                'fact_idx': f['fact_idx'],
                'rule_text': render_rule(
                    f['clauses'],
                    readability,
                    ohe_groups={},
                    base_order=_compute_base_order(list(feature_names))
                ),
                'weight': 1.0,  # placeholder
                'direction': 'unknown'
            } for f in facts[:top_rules]])
        return pd.DataFrame(columns=['rule_idx', 'weight', 'rule_text', 'direction'])
    
    # Convert rules to DataFrame
    if not rules:
        return pd.DataFrame(columns=['rule_idx', 'weight', 'rule_text', 'direction'])
    
    # Build rule texts with proper metadata
    base_order = _compute_base_order(list(feature_names))
    is_bin = _feature_is_binary(np.zeros((1, len(feature_names))))  # placeholder
    ohe_groups = _build_ohe_groups(list(feature_names), is_bin)
    
    rows = []
    for r in rules[:top_rules]:
        rule_text = render_rule(
            r['clauses'],
            readability,
            ohe_groups=ohe_groups,
            base_order=base_order
        )
        rows.append({
            'rule_idx': r.get('rule_idx', r.get('tree_idx', -1)),
            'weight': r.get('weight', r.get('leaf_score', 0.0)),
            'rule_text': rule_text,
            'direction': r.get('direction', 'unknown')
        })
    
    return pd.DataFrame(rows)

def local_contrib_df(
    model: Any,
    x: np.ndarray,
    feature_names: Sequence[str],
    scaler: Optional[Any] = None,
    top_rules: int = 10,
    readability: Readability = "clinical",
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compute local rule contributions for a single sample.
    
    Parameters
    ----------
    model : Any
        Trained Nous zoo_v2 model
    x : np.ndarray
        Single sample (1D array) in original feature space
    feature_names : Sequence[str]
        Original feature names
    scaler : Optional[Any]
        Scaler used during training
    top_rules : int
        Number of top contributing rules to return
    readability : Readability
        Readability mode for rule rendering
    **kwargs : dict
        Additional parameters for model-specific interpretation
    
    Returns
    -------
    pd.DataFrame
        DataFrame with rule contributions
    Dict[str, float]
        Metadata including prediction probability/logit
    """
    # Ensure x is 2D for model inference
    x_2d = x.reshape(1, -1).astype(np.float32)
    
    # Preprocess with scaler if provided
    if scaler is not None:
        x_scaled = scaler.transform(x_2d).astype(np.float32)
    else:
        x_scaled = x_2d
    
    # Get model prediction
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_scaled, device=next(model.parameters()).device)
        logits = model(x_tensor).cpu().numpy().ravel()
    
    # For classification, compute probability
    if len(logits) == 1 or (hasattr(model, 'output_dim') and model.output_dim == 1):
        prob = 1.0 / (1.0 + np.exp(-logits[0]))
    else:
        # Multiclass
        logits_stable = logits - np.max(logits)
        exp_logits = np.exp(logits_stable)
        prob = exp_logits / np.sum(exp_logits)
        prob = prob[np.argmax(prob)]  # Probability of predicted class
    
    meta = {
        'logit': float(logits[0]) if len(logits) == 1 else float(np.max(logits)),
        'prob': float(prob),
        'prediction': int(np.argmax(logits)) if len(logits) > 1 else int(logits[0] > 0)
    }
    
    # Model-specific contribution extraction
    model_type = type(model).__name__
    
    # Evidence family - reuse notebook logic adapted for library
    if isinstance(model, (
        EvidenceNet, MarginEvidenceNet, PerFeatureKappaEvidenceNet, BiEvidenceNet
    )):
        # Simplified version of notebook's rule_contrib_for_row_evidence_like
        with torch.no_grad():
            x_tensor = torch.tensor(x_scaled, device=next(model.parameters()).device)
            w = model.head.weight[0]
            
            if isinstance(model, BiEvidenceNet):
                # BiEvidenceNet-specific computation
                width = torch.exp(model.log_kappa).clamp(1e-3, 50.0)
                t_low = model.center - 0.5 * width
                t_high = model.center + 0.5 * width
                kappa = torch.exp(model.log_kappa).clamp(0.5, 50.0)
                low = torch.sigmoid(kappa * (t_low[None, :, :] - x_tensor[:, None, :]))
                high = torch.sigmoid(kappa * (x_tensor[:, None, :] - t_high[None, :, :]))
                mask = torch.sigmoid(model.mask)[None, :, :]
                el = torch.tanh(model.e_low)[None, :, :]
                eh = torch.tanh(model.e_high)[None, :, :]
                evidence = (mask * (el * (2.0 * low - 1.0) + eh * (2.0 * high - 1.0))).sum(dim=2)
                z = torch.sigmoid(model.beta * (evidence - model.t[None, :]))[0]
                contrib = (z * w).detach().cpu().numpy()
            else:
                # Standard evidence models
                kappa = torch.exp(model.log_kappa).clamp(0.5, 50.0)
                ineq = torch.tanh(model.ineq_sign_param)
                c = torch.sigmoid(kappa * ineq[None, :, :] * (x_tensor[:, None, :] - model.th[None, :, :]))
                mask = torch.sigmoid(model.mask_logit)[None, :, :]
                es = torch.tanh(model.e_sign_param)[None, :, :]
                evidence = (mask * es * (2.0 * c - 1.0)).sum(dim=2)
                z = torch.sigmoid(model.beta * (evidence - model.t[None, :]))[0]
                contrib = (z * w).detach().cpu().numpy()
        
        # Get global rules for rendering
        global_rules = _extract_evidence_rules(model, feature_names, scaler, readability, top_feats=4)
        
        # Build contribution DataFrame
        rows = []
        for i in np.argsort(-np.abs(contrib))[:top_rules]:
            r = int(i)
            if r >= len(global_rules):
                continue
            
            rule_info = global_rules[r]
            rows.append({
                'rule_idx': r,
                'contribution': float(contrib[r]),
                'activation': float(z[r].item() if 'z' in locals() else 0.0),
                'weight': float(w[r].item() if hasattr(w, 'item') else w[r]),
                'rule_text': render_rule(
                    rule_info['clauses'],
                    readability,
                    ohe_groups=_build_ohe_groups(list(feature_names), _feature_is_binary(x_2d)),
                    base_order=_compute_base_order(list(feature_names))
                ),
                'direction': 'increases' if contrib[r] > 0 else 'decreases'
            })
        
        return pd.DataFrame(rows), meta
    
    # Forest models - simplified tree contribution
    elif isinstance(model, (
        PredicateForest, ObliviousForest, LeafLinearForest, AttentiveForest,
        MultiResForest, GroupFirstForest, BudgetedForest
    )):
        # Get tree contributions (simplified)
        with torch.no_grad():
            x_tensor = torch.tensor(x_scaled, device=next(model.parameters()).device)
            f = model.facts(x_tensor) if hasattr(model, 'facts') else x_tensor
            f_aug = torch.cat([f, 1.0 - f], dim=1) if hasattr(model, 'facts') else f
            
            # Get tree outputs
            if hasattr(model, 'leaf_value'):
                # Simplified tree contribution via leaf values
                tree_out = model.leaf_value.detach().cpu().numpy().sum(axis=1)
                tree_contrib = tree_out[:top_rules]
                
                # Get global rules for rendering
                global_rules = _extract_forest_rules(model, feature_names, scaler, readability, n_trees=top_rules)
                
                rows = []
                for i, r in enumerate(np.argsort(-np.abs(tree_contrib))[:top_rules]):
                    if r >= len(global_rules):
                        continue
                    
                    rule_info = global_rules[r]
                    rows.append({
                        'tree_idx': r,
                        'contribution': float(tree_contrib[r]),
                        'tree_score': float(rule_info['leaf_score']),
                        'rule_text': render_rule(
                            rule_info['clauses'],
                            readability,
                            ohe_groups=_build_ohe_groups(list(feature_names), _feature_is_binary(x_2d)),
                            base_order=_compute_base_order(list(feature_names))
                        ),
                        'direction': 'increases' if tree_contrib[r] > 0 else 'decreases'
                    })
                
                return pd.DataFrame(rows), meta
        
        # Fallback for forests without direct tree output access
        global_rules = _extract_forest_rules(model, feature_names, scaler, readability, n_trees=top_rules)
        rows = [{
            'tree_idx': i,
            'contribution': 0.0,  # placeholder
            'tree_score': r['leaf_score'],
            'rule_text': render_rule(
                r['clauses'],
                readability,
                ohe_groups=_build_ohe_groups(list(feature_names), _feature_is_binary(x_2d)),
                base_order=_compute_base_order(list(feature_names))
            ),
            'direction': r['direction']
        } for i, r in enumerate(global_rules[:top_rules])]
        
        return pd.DataFrame(rows), meta
    
    # Fallback: global rules with zero contributions
    global_rules = global_rules_df(
        model, feature_names, scaler, top_rules, readability, **kwargs
    )
    
    if len(global_rules) == 0:
        return pd.DataFrame(columns=[
            'rule_idx', 'contribution', 'activation', 'weight', 'rule_text', 'direction'
        ]), meta
    
    # Add contribution column with zeros as placeholder
    global_rules['contribution'] = 0.0
    global_rules['activation'] = 0.0
    global_rules = global_rules[[
        'rule_idx', 'contribution', 'activation', 'weight', 'rule_text', 'direction'
    ]]
    
    return global_rules.head(top_rules), meta

# ===== Convenience functions =====
def explain_prediction(
    model: Any,
    x: np.ndarray,
    feature_names: Sequence[str],
    scaler: Optional[Any] = None,
    readability: Readability = "clinical",
    top_rules: int = 5
) -> str:
    """
    Generate human-readable explanation for a single prediction.
    
    Parameters
    ----------
    model : Any
        Trained Nous zoo_v2 model
    x : np.ndarray
        Single sample (1D array)
    feature_names : Sequence[str]
        Feature names
    scaler : Optional[Any]
        Feature scaler used during training
    readability : Readability
        Readability mode
    top_rules : int
        Number of top rules to include in explanation
    
    Returns
    -------
    str
        Human-readable explanation text
    """
    contrib_df, meta = local_contrib_df(
        model, x, feature_names, scaler, top_rules, readability
    )
    
    lines = []
    lines.append(f"Prediction: {meta['prediction']} (probability: {meta['prob']:.3f})")
    lines.append(f"Logit: {meta['logit']:+.3f}")
    lines.append("-" * 60)
    lines.append(f"Top {min(top_rules, len(contrib_df))} contributing rules:")
    
    for idx, row in contrib_df.head(top_rules).iterrows():
        direction = "↑ increases" if row['direction'] == 'increases' else "↓ decreases"
        contrib = row['contribution']
        rule_text = row['rule_text']
        lines.append(f"\n{direction} ({contrib:+.3f}):")
        lines.append(f"  IF {rule_text}")
    
    return "\n".join(lines)

def export_global_rules(
    model: Any,
    feature_names: Sequence[str],
    path: str,
    scaler: Optional[Any] = None,
    top_rules: int = 20,
    readability: Readability = "clinical",
    format: Literal["txt", "json", "csv"] = "txt"
) -> None:
    """
    Export global rules to file in specified format.
    
    Parameters
    ----------
    model : Any
        Trained Nous zoo_v2 model
    feature_names : Sequence[str]
        Feature names
    path : str
        Output file path
    scaler : Optional[Any]
        Feature scaler
    top_rules : int
        Number of top rules to export
    readability : Readability
        Readability mode for rule text
    format : Literal["txt", "json", "csv"]
        Output format
    """
    df = global_rules_df(model, feature_names, scaler, top_rules, readability)
    
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    
    if format == "txt":
        with open(path, "w") as f:
            f.write(f"Global rules for {type(model).__name__}\n")
            f.write("=" * 60 + "\n\n")
            for idx, row in df.iterrows():
                f.write(f"Rule {row['rule_idx']} ({row['direction']} prediction):\n")
                f.write(f"  Weight: {row['weight']:+.3f}\n")
                f.write(f"  IF {row['rule_text']}\n\n")
    
    elif format == "json":
        import json
        rules = []
        for idx, row in df.iterrows():
            rules.append({
                "rule_idx": int(row["rule_idx"]),
                "weight": float(row["weight"]),
                "rule_text": row["rule_text"],
                "direction": row["direction"]
            })
        
        with open(path, "w") as f:
            json.dump({
                "model": type(model).__name__,
                "readability": readability,
                "rules": rules
            }, f, indent=2)
    
    elif format == "csv":
        df.to_csv(path, index=False)
