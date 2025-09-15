"""
California Housing regression demo for NousNet with fidelity-driven pruning and honest explanations.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nous import (
    NousNet,
    SparseRuleLayer,                  # needed for 'NOT' suppression
    get_california_housing_data,
    train_model,
    evaluate_regression,
    select_pruning_threshold_global_bs,
    generate_enhanced_explanation,
    make_sparse_regression_hook,      # scheduler logic
)

# === Global training hyperparameters ===
EPOCHS = 1000
PATIENCE = 200
BATCH_SIZE = 64
LR = 1e-3  # will be reduced to 1e-4 for the sparse model variant


def main():
    print("\n\n" + "=" * 100)
    print("CALIFORNIA HOUSING REGRESSION BENCHMARK")
    print("=" * 100)

    # Load data with target scaling (y)
    X_train, X_val, X_test, y_train_sc, y_val_sc, y_test_orig, feature_names, _, task_type, y_scaler = \
        get_california_housing_data(scale_y=True)
    input_dim = X_train.shape[1]
    num_outputs = 1

    # Scale y_test for dataloaders
    y_test_sc = y_scaler.transform(y_test_orig.reshape(-1, 1)).ravel()

    # Tensors and loaders
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_sc, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_sc, dtype=torch.float32)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE)

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_sc, dtype=torch.float32)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE)

    # Configurations
    regression_configurations = [
        {
            "name": "Simple (Fixed Rules)",
            "params": {"rule_selection_method": "fixed", "use_calibrators": False},
        },
        {
            "name": "Advanced (Softmax Rules)",
            "params": {"rule_selection_method": "softmax", "use_calibrators": True},
        },
        {
            "name": "Sparse (Hard Concrete Rules)",
            "params": {
                "rule_selection_method": "sparse",
                "use_calibrators": True,
                "l0_lambda": 0.05,
                "hc_temperature": 0.2,
            },
        },
    ]

    results_housing = {}
    sample_idx = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for config in regression_configurations:
        name, params = config["name"], config["params"]
        print(f"\n{'-'*80}\nTraining {name} for California Housing\n{'-'*80}")

        model = NousNet(
            input_dim=input_dim,
            num_outputs=num_outputs,
            task_type=task_type,
            feature_names=feature_names,
            num_facts=64,
            rules_per_layer=(32, 16),
            **params,
        )

        # Suppress NOT aggregator for sparse model
        if name == "Sparse (Hard Concrete Rules)":
            print("Suppressing 'NOT' aggregator for sparse regression model.")
            for blk in model.blocks:
                if isinstance(blk, SparseRuleLayer):
                    with torch.no_grad():
                        if blk.aggregator_logits.shape[1] == 4:
                            blk.aggregator_logits[:, 3].fill_(-3.0)

        print("Model Summary:", model.model_summary())

        # Optimizer and loss (Huber/SmoothL1)
        criterion = nn.SmoothL1Loss(beta=1.0)
        lr = LR
        if name == "Sparse (Hard Concrete Rules)":
            lr = 1e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        # Training schedulers (L0, temperature, and disable top-k)
        after_hook = None
        if name == "Sparse (Hard Concrete Rules)":
            print("Applying training schedulers (L0, Temp, Top-K) for Sparse Regression model.")
            after_hook = make_sparse_regression_hook(
                base_lambda=params["l0_lambda"],  # 0.05
                warmup=60,
                ramp=120,
                temp_start=params["hc_temperature"],  # 0.2
                temp_end=0.25,
                temp_epochs=700,
                disable_topk=True,
            )

        start_time = time.time()
        best_val_loss = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            epochs=EPOCHS,
            patience=PATIENCE,
            device=device,
            after_epoch_hook=after_hook,
            # verbose=1, log_every=50, use_tqdm=True
        )
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds.")

        rmse, mae, r2, _, _ = evaluate_regression(model, test_loader, device, y_scaler=y_scaler)
        results_housing[name] = {
            "model": model,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "train_time": train_time,
            "val_loss": best_val_loss,
        }
        print(f"Metrics: RMSE={rmse:.4f} MAE={mae:.4f} R²={r2:.4f}")

        # Fidelity-driven pruning threshold (binary search; scaled space)
        t_prune = select_pruning_threshold_global_bs(
            model,
            X_val,
            task_type="regression",
            tol_reg=0.05 * y_scaler.scale_[0],
            max_samples=200,
            device=device,
        )
        print(f"Selected fidelity-driven pruning threshold: {t_prune:.4f}")

        # Honest explanation (reports unscaled values)
        print(f"\n HONEST EXPLANATION (no pruning) — {name} (Sample #{sample_idx})")
        print("-" * 80)
        print(
            generate_enhanced_explanation(
                model,
                X_test[sample_idx],
                y_test_sc[sample_idx],
                feature_names,
                y_scaler=y_scaler,
                use_pruning=False,
            )
        )

        # Pruned explanation
        print(f"\n PRUNED EXPLANATION (apply threshold) — {name} (Sample #{sample_idx})")
        print("-" * 80)
        print(
            generate_enhanced_explanation(
                model,
                X_test[sample_idx],
                y_test_sc[sample_idx],
                feature_names,
                y_scaler=y_scaler,
                use_pruning=True,
                pruning_threshold=t_prune,
            )
        )


if __name__ == "__main__":
    main()