import numpy as np
from nous.utils import make_quantile_calibrators

def test_make_quantile_calibrators():
    X = np.random.randn(100, 3)
    calibrators = make_quantile_calibrators(X, num_bins=4)
    assert len(calibrators) == 3
    assert all(hasattr(c, 'forward') for c in calibrators)