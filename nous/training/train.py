from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, Optional

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    patience: int,
    device,
    after_epoch_hook: Optional[Callable[[nn.Module, int], None]] = None
) -> float:
    """
    Train with early stopping. Adds L0 loss (if model exposes compute_total_l0_loss) and gradient clipping.
    """
    model.to(device)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch if outputs.ndim == y_batch.ndim else y_batch.float())
            l0_loss = getattr(model, "compute_total_l0_loss", lambda: torch.tensor(0.0, device=device))()
            total_loss = loss + l0_loss
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch if outputs.ndim == y_batch.ndim else y_batch.float())
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        if after_epoch_hook is not None:
            after_epoch_hook(model, epoch)

        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
        model.to(device)
    return best_val_loss