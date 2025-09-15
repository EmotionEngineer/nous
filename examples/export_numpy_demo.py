"""
Export NumPy-only inference module and validate parity against PyTorch.
"""
import os
import numpy as np
from nous import NousNet
from nous.export import export_numpy_inference, load_numpy_module, validate_numpy_vs_torch

def main():
    model = NousNet(
        input_dim=10,
        num_outputs=4,
        task_type="classification",
        num_facts=16,
        rules_per_layer=(8, ),
        rule_selection_method="fixed",
        use_calibrators=True
    )
    X = np.random.randn(256, 10).astype(np.float32)
    os.makedirs("exports", exist_ok=True)
    path = "exports/nous_numpy_infer_demo.py"
    export_numpy_inference(model, file_path=path)
    mod = load_numpy_module(path)
    res = validate_numpy_vs_torch(model, mod, X, "classification", n=128)
    print(res)

if __name__ == "__main__":
    main()