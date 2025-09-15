from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Sequence
from ..model import NousNet
from .loo import rule_impact_df

def global_rulebook(
    model: NousNet, X, feature_names: Sequence[str], class_names: Optional[Sequence[str]] = None,
    pruning_threshold: Optional[float] = None, use_pruning: bool = False, allow_replacement: bool = True, freeze_non_active: bool = False,
    max_samples: int = 1000, device=None
) -> pd.DataFrame:
    """
    Global rulebook aggregation across samples via honest LOO impacts.
    """
    from tqdm.auto import tqdm  # optional dep in extras

    device = device or next(model.parameters()).device
    n = min(len(X), max_samples)
    totals = {}

    loo_mode = 'replacement'
    if not allow_replacement and freeze_non_active:
        loo_mode = 'frozen'

    for i in tqdm(range(n), desc="Analyzing samples"):
        df = rule_impact_df(
            model, X[i], feature_names, class_names=class_names,
            loo_mode=loo_mode, use_pruning=use_pruning, pruning_threshold=(pruning_threshold or 0.0), device=device
        )
        if df.empty:
            continue
        for _, row in df.iterrows():
            key = (int(row["block"]), int(row["rule"]), row["aggregators"])
            d = totals.setdefault(key, {"count": 0, "sum_abs_impact": 0.0, "critical": 0})
            d["count"] += 1
            if "critical_flip" in row and row["critical_flip"]:
                d["critical"] += 1
            metric_col = [c for c in row.index if c.startswith("Δmargin(")]
            val = abs(float(row[metric_col[0]])) if metric_col else abs(float(row.get("Δprediction", 0.0)))
            d["sum_abs_impact"] += val

    rows = []
    for (b, r, agg), v in totals.items():
        count = v["count"]
        rows.append({
            "block": b, "rule": r, "aggregators": agg,
            "activation_freq": count / n,
            "mean_abs_impact": v["sum_abs_impact"] / count if count else 0.0,
            "critical_rate": v["critical"] / count if count else 0.0
        })
    return pd.DataFrame(rows).sort_values("mean_abs_impact", ascending=False).reset_index(drop=True)