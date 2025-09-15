from __future__ import annotations
import torch

def extract_active_masks(internals) -> list[torch.Tensor]:
    """
    Extract per-block active gate masks (0/1) from internals produced by forward_explain.
    """
    masks = []
    keys = sorted([k for k in internals.keys() if k.startswith("block_")], key=lambda s: int(s.split("_")[1]))
    for key in keys:
        gate_mask = internals[key]['gate_mask']
        if isinstance(gate_mask, torch.Tensor):
            gate_mask = gate_mask.squeeze(0)
        masks.append((gate_mask > 0).float())
    return masks