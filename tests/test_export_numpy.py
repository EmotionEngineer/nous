import numpy as np
import tempfile
import os

from nous import NousNet
from nous.export import export_numpy_inference, load_numpy_module, validate_numpy_vs_torch  # type: ignore

def test_export_and_validate_smoke():
    model = NousNet(
        input_dim=8,
        num_outputs=3,
        task_type="classification",
        num_facts=10,
        rules_per_layer=(6, ),
        rule_selection_method="fixed"
    )
    X = np.random.randn(64, 8).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "infer.py")
        code = export_numpy_inference(model, file_path=path)
        assert isinstance(code, str) and len(code) > 100
        mod = load_numpy_module(path)
        res = validate_numpy_vs_torch(model, mod, X, "classification", n=32)
        assert "pass" in res