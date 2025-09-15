from __future__ import annotations
from ..rules.sparse import SparseRuleLayer

def make_sparse_regression_hook(
    base_lambda: float = 3e-4, warmup: int = 40, ramp: int = 80,
    temp_start: float = 0.60, temp_end: float = 0.25, temp_epochs: int = 200,
    disable_topk: bool = True
):
    """
    Return an after-epoch scheduler hook for sparse regression:
      - L0 penalty ramp,
      - Hard-Concrete temperature schedule,
      - Disable top-k gating to preserve gradients.
    """
    def hook(model, epoch: int):
        # 1) L0 schedule
        l0_factor = 0.0 if epoch < warmup else min(1.0, (epoch - warmup) / max(1, ramp))
        for blk in model.blocks:
            if isinstance(blk, SparseRuleLayer):
                blk.l0_lambda = base_lambda * l0_factor

        # 2) HC temperature schedule
        alpha = min(1.0, epoch / max(1, temp_epochs))
        t = temp_start * (1 - alpha) + temp_end * alpha
        for blk in model.blocks:
            if isinstance(blk, SparseRuleLayer):
                blk.hard_concrete.temperature = t

        # 3) Disable top-k (for regression)
        if disable_topk:
            for blk in model.blocks:
                if hasattr(blk, "top_k_rules"):
                    blk.top_k_rules = blk.num_rules
    return hook