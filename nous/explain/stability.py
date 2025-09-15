from __future__ import annotations
import numpy as np
import torch
from typing import Optional, Sequence, Dict, Any
from ..model import NousNet
from .loo import rule_impact_df

def explanation_stability(
    model: NousNet, x_sample, feature_names: Sequence[str], class_names: Optional[Sequence[str]] = None,
    k_top: int = 5, sigma: float = 0.05, trials: int = 20, loo_mode: str = 'frozen',
    use_pruning: bool = False, pruning_threshold: float = 0.0, device=None
) -> Dict[str, Any]:
    """
    Stability of top-k rule explanations under small input perturbations.
    Returns mean/std Jaccard overlap vs base top-k selection.
    """
    device = device or next(model.parameters()).device
    base_imp = rule_impact_df(
        model, x_sample, feature_names, class_names=class_names,
        loo_mode=loo_mode, use_pruning=use_pruning, pruning_threshold=pruning_threshold
    )
    base_set = set((int(r.block), int(r.rule)) for _, r in base_imp.head(k_top).iterrows()) if not base_imp.empty else set()

    overlaps = []
    for _ in range(trials):
        x = torch.tensor(x_sample, dtype=torch.float32, device=device)
        noise = torch.randn_like(x) * sigma
        x_noisy = (x + noise).cpu().numpy()

        imp = rule_impact_df(
            model, x_noisy, feature_names, class_names=class_names,
            loo_mode=loo_mode, use_pruning=use_pruning, pruning_threshold=pruning_threshold
        )
        cur_set = set((int(r.block), int(r.rule)) for _, r in imp.head(k_top).iterrows()) if not imp.empty else set()
        inter = len(base_set & cur_set)
        union = len(base_set | cur_set) if (base_set | cur_set) else 1
        overlaps.append(inter / union)
    return {
        "mean_jaccard": float(np.mean(overlaps)),
        "std_jaccard": float(np.std(overlaps)),
        "base_top_rules": list(base_set)
    }