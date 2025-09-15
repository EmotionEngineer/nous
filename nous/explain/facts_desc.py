from __future__ import annotations
from typing import Dict, Sequence
from ..model import NousNet

def render_fact_descriptions(model: NousNet, feature_names: Sequence[str], top_k_feats: int = 4, eps: float = 0.03) -> Dict[int, str]:
    """
    Create human-readable descriptions of base β-facts using (L-R) weights.
    """
    L, R, th, k, nu = model.fact.get_rule_parameters()
    desc = {}
    for fid in range(L.shape[0]):
        net = L[fid] - R[fid]
        pos = [(feature_names[i], net[i]) for i in range(len(net)) if net[i] > eps]
        neg = [(feature_names[i], -net[i]) for i in range(len(net)) if net[i] < -eps]
        pos = sorted(pos, key=lambda t: -abs(t[1]))[:top_k_feats]
        neg = sorted(neg, key=lambda t: -abs(t[1]))[:top_k_feats]
        pos_str = " + ".join([f"{w:.2f}·{n}" for n, w in pos]) if pos else "0"
        neg_str = " + ".join([f"{w:.2f}·{n}" for n, w in neg]) if neg else "0"
        base = f"β( [L−R](x̃) = ({pos_str}) − ({neg_str}) > {th[fid]:.2f}; k={k[fid]:.2f}, ν={nu[fid]:.2f} )"
        if model.calibrators is not None:
            base += "  where x̃ are calibrated features"
        desc[fid] = base
    return desc