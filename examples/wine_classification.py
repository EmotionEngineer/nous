"""
Wine classification demo for NousNet with honest explanations.
"""
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from nous import (
    NousNet, get_wine_data, train_model, evaluate_classification,
    select_pruning_threshold_global_bs, generate_enhanced_explanation
)

def main():
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, class_names, task_type, _ = get_wine_data()

    model = NousNet(
        input_dim=X_train.shape[1], num_outputs=len(class_names), task_type="classification",
        feature_names=feature_names, num_facts=32, rules_per_layer=(16, 8),
        rule_selection_method='softmax', use_calibrators=True, use_prototypes=True
    )

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=64, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_val,   dtype=torch.float32), torch.tensor(y_val,   dtype=torch.long)), batch_size=64)
    test_loader  = DataLoader(TensorDataset(torch.tensor(X_test,  dtype=torch.float32), torch.tensor(y_test,  dtype=torch.long)), batch_size=64)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start = time.time()
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=300, patience=50, device=device)
    print(f"Training time: {time.time() - start:.2f}s")

    acc, auc, _, _ = evaluate_classification(model, test_loader, device)
    print(f"Accuracy={acc:.4f} AUC={auc:.4f}")

    t_prune = select_pruning_threshold_global_bs(model, X_val, target_fidelity=0.99, task_type="classification", max_samples=200, device=device)
    print(f"Selected pruning threshold: {t_prune:.4f}")

    sample_idx = 0
    print("HONEST EXPLANATION (no pruning)")
    print(generate_enhanced_explanation(model, X_test[sample_idx], y_test[sample_idx], feature_names, class_names, use_pruning=False))

    print("\nPRUNED EXPLANATION")
    print(generate_enhanced_explanation(model, X_test[sample_idx], y_test[sample_idx], feature_names, class_names, use_pruning=True, pruning_threshold=t_prune))

if __name__ == "__main__":
    main()