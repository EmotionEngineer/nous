import numpy as np
import tempfile
import os
import pytest

from nous import NousNet
from nous.export import export_numpy_inference, load_numpy_module, validate_numpy_vs_torch  # type: ignore

GATERS = [
    ("soft_rank", {"tau": 0.5}),
    ("logistic_threshold", {"tau": 0.5, "iters": 25}),
    ("capped_simplex", {}),
    ("sparsemax_k", {"tau": 0.7}),
    ("hard_concrete_budget", {"tau": 0.5, "l0_lambda": 1e-3, "mu_budget": 1e-2, "target_k": 6}),
]

@pytest.mark.parametrize("gname,gkw", GATERS)
def test_export_numpy_softfact_with_gater(gname, gkw):
    # Small soft-fact model with a single block using the selected gater
    model = NousNet(
        input_dim=8,
        num_outputs=3,
        task_type="classification",
        num_facts=12,
        rules_per_layer=(6,),
        rule_selection_method="soft_fact",
        use_calibrators=False,
        use_prototypes=False,
        rule_gater_name=gname,
        rule_gater_kwargs=gkw,
    )

    X = np.random.randn(64, 8).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "infer.py")
        code = export_numpy_inference(model, file_path=path)
        assert isinstance(code, str) and len(code) > 100
        mod = load_numpy_module(path)
        res = validate_numpy_vs_torch(model, mod, X, "classification", n=32)
        assert "pass" in res
        assert res["pass"], f"Failed parity for gater={gname}: {res}"