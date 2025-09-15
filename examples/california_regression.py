"""
California Housing regression demo for NousNet with pruning and explanations.
"""
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from nous import (
    NousNet, get_california_housing_data, train_model, evaluate_regression,
    select_pruning_threshold_global_bs, generate_enhanced_explanation, make_sparse_regression_hook
)

def main():
    X_train, X_val, X_test, y_train_scaled, y_val_scaled, y_test_orig, feature_names, _, task_type, y_scaler = get_california_housing_data(scale_y=True)
    y_test_scaled = y_scaler.transform(y_test_orig.reshape(-1, 1)).ravel()

    model = NousNet(
        input_dim=X_train.shape[1], num_outputs=1, task_type="regression",
        feature_names=feature_names, num_facts=64, rules_per_layer=(32, 16),
        rule_selection_method='sparse', use_calibrators=True, use_prototypes=False,
        l0_lambda=0.05, hc_temperature=0.2
    )

    # Suppress NOT aggregator by biasing logits if present (optional)
    for blk in model.blocks:
        if hasattr(blk, "aggregator_logits") and blk.aggregator_logits.shape[1] == 4:
            with torch.no_grad():
                blk.aggregator_logits[:, 3].fill_(-3.0)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train_scaled, dtype=torch.float32)), batch_size=64, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_val,   dtype=torch.float32), torch.tensor(y_val_scaled,   dtype=torch.float32)), batch_size=64)
    test_loader  = DataLoader(TensorDataset(torch.tensor(X_test,  dtype=torch.float32), torch.tensor(y_test_scaled,  dtype=torch.float32)), batch_size=64)

    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    after_hook = make_sparse_regression_hook(
        base_lambda=0.05, warmup=60, ramp=120,
        temp_start=0.2, temp_end=0.25, temp_epochs=700,
        disable_topk=True
    )

    start = time.time()
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=500, patience=80, device=device, after_epoch_hook=after_hook)
    print(f"Training time: {time.time() - start:.2f}s")

    rmse, mae, r2, _, _ = evaluate_regression(model, test_loader, device, y_scaler=y_scaler)
    print(f"RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")

    t_prune = select_pruning_threshold_global_bs(model, X_val, task_type="regression", tol_reg=0.05 * y_scaler.scale_[0], max_samples=200, device=device)
    print(f"Selected pruning threshold: {t_prune:.4f}")

    sample_idx = 0
    print("HONEST EXPLANATION (no pruning)")
    print(generate_enhanced_explanation(model, X_test[sample_idx], y_test_scaled[sample_idx], feature_names, y_scaler=y_scaler, use_pruning=False))

    print("\nPRUNED EXPLANATION")
    print(generate_enhanced_explanation(model, X_test[sample_idx], y_test_scaled[sample_idx], feature_names, y_scaler=y_scaler, use_pruning=True, pruning_threshold=t_prune))

if __name__ == "__main__":
    main()