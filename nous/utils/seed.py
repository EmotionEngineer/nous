from __future__ import annotations
import random
import numpy as np
import torch

def set_global_seed(seed: int = 42) -> None:
    """
    Set seeds for Python, NumPy, and PyTorch (CPU/CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)