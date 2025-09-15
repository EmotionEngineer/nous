from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List
from ..model import NousNet

AGG_NAMES = ['AND', 'OR', 'k-of-n', 'NOT']

def format_agg_mixture(weights) -> str:
    parts = []
    for i in range(weights.shape[0]):
        w = float(weights[i])
        if w > 1e-6:
            parts.append(f"{w:.2f} {AGG_NAMES[i]}")
    return " + ".join(parts) if parts else "∅"

def aggregator_mixture_report(model: NousNet, X, max_samples: int = 1000, device=None) -> pd.DataFrame:
    device = device or next(model.parameters()).device
    n = min(len(X), max_samples)
    acc = []
    for i in range(n):
        _, _, internals = model.forward_explain(X[i], device=device)
        for key in [k for k in internals.keys() if k.startswith("block_")]:
            aw = internals[key]['aggregator_weights']
            if aw is None:
                continue
            acc.append(aw.cpu().numpy())
    if not acc:
        return pd.DataFrame(columns=["AND", "OR", "k-of-n", "NOT", "entropy"])
    A = np.concatenate(acc, axis=0)
    mean = A.mean(axis=0)
    ent = (-A * np.clip(np.log(A + 1e-12), -50, 50)).sum(axis=1).mean()
    cols = AGG_NAMES[:A.shape[1]]
    return pd.DataFrame([dict(**{c: float(v) for c, v in zip(cols, mean)}, entropy=float(ent))])
