from __future__ import annotations

import inspect
from typing import Callable, Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    patience: int,
    device,
    after_epoch_hook: Optional[Callable[..., None]] = None,
    # Progress controls
    verbose: int = 1,
    log_every: int = 10,
    use_tqdm: bool = False,
    print_l0: bool = True,
    # NEW: training behavior controls (to match notebook loops)
    clip_grad_max_norm: float | None = 0.5,
    zero_grad_set_to_none: bool = False,
    loss_average: str = "batch",  # "batch" (legacy) | "sample" (notebook-style)
) -> float:
    """
    Train with early stopping. Adds L0 loss (if model exposes compute_total_l0_loss) and optional gradient clipping.

    Key behavior options
    --------------------
    clip_grad_max_norm:
        If not None, applies nn.utils.clip_grad_norm_ with this max norm.
        Default remains 0.5 for backward compatibility with earlier Nous behavior.
        To match your notebook loop, use clip_grad_max_norm=1.0.

    zero_grad_set_to_none:
        If True, calls optimizer.zero_grad(set_to_none=True) (notebook-style).
        Default False for backward compatibility.

    loss_average:
        - "batch": averages train/val loss by number of batches (legacy behavior).
        - "sample": averages train/val loss by number of samples (notebook-style).
        This affects early stopping selection and logging, not gradients (criterion still returns mean loss).

    Progress
    --------
    - verbose >= 1 prints epoch-level logs every `log_every` epochs, on improvement, and on first/last epoch.
    - use_tqdm shows a progress bar over epochs with train/val (and L0) in the postfix.
    - after_epoch_hook can be:
        (model, epoch)                      -- legacy signature
        (model, epoch, metrics_dict)        -- extended signature
      where metrics_dict contains:
        {epoch, train_loss, val_loss, l0_loss, improved}.
    """
    if loss_average not in ("batch", "sample"):
        raise ValueError("loss_average must be 'batch' or 'sample'")

    model.to(device)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    # Epoch iterator (optionally tqdm)
    if use_tqdm:
        try:
            from tqdm.auto import tqdm as _tqdm
            epoch_iter = _tqdm(range(epochs), leave=False, desc="Training")
        except Exception:
            epoch_iter = range(epochs)
    else:
        epoch_iter = range(epochs)

    for epoch in epoch_iter:
        # -------------------------
        # Train
        # -------------------------
        model.train()
        total_train_loss_sum = 0.0
        total_train_weight = 0.0

        total_l0_loss = 0.0
        steps = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # zero_grad (optionally set_to_none)
            try:
                optimizer.zero_grad(set_to_none=zero_grad_set_to_none)
            except TypeError:
                optimizer.zero_grad()

            outputs = model(X_batch)
            if isinstance(criterion, nn.CrossEntropyLoss) or (outputs.ndim == 2 and outputs.size(-1) > 1):
                target = y_batch.long()    # classification
            else:
                target = y_batch.float()   # regression / BCE-logits
            loss = criterion(outputs, target)

            l0_loss = getattr(model, "compute_total_l0_loss", lambda: torch.tensor(0.0, device=device))()
            total_loss = loss + l0_loss

            total_loss.backward()

            if clip_grad_max_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_grad_max_norm))

            optimizer.step()

            # Bookkeeping: average loss either by batches or by samples
            bs = int(X_batch.shape[0])
            if loss_average == "sample":
                total_train_loss_sum += float(loss.item()) * bs
                total_train_weight += bs
            else:
                total_train_loss_sum += float(loss.item())
                total_train_weight += 1.0

            total_l0_loss += float(l0_loss.item()) if isinstance(l0_loss, torch.Tensor) else float(l0_loss)
            steps += 1

        avg_train_loss = total_train_loss_sum / max(1.0, total_train_weight)
        avg_l0_loss = total_l0_loss / max(1, steps)

        # -------------------------
        # Validate
        # -------------------------
        model.eval()
        total_val_loss_sum = 0.0
        total_val_weight = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                if isinstance(criterion, nn.CrossEntropyLoss) or (outputs.ndim == 2 and outputs.size(-1) > 1):
                    target = y_batch.long()
                else:
                    target = y_batch.float()

                vloss = criterion(outputs, target)
                bs = int(X_batch.shape[0])

                if loss_average == "sample":
                    total_val_loss_sum += float(vloss.item()) * bs
                    total_val_weight += bs
                else:
                    total_val_loss_sum += float(vloss.item())
                    total_val_weight += 1.0

        avg_val_loss = total_val_loss_sum / max(1.0, total_val_weight)

        improved = avg_val_loss < (best_val_loss - 1e-6)

        # -------------------------
        # Progress reporting
        # -------------------------
        if use_tqdm and hasattr(epoch_iter, "set_postfix"):
            postfix: Dict[str, str] = {"train": f"{avg_train_loss:.4f}", "val": f"{avg_val_loss:.4f}"}
            if print_l0:
                postfix["l0"] = f"{avg_l0_loss:.4f}"
            try:
                epoch_iter.set_postfix(postfix)  # type: ignore[attr-defined]
            except Exception:
                pass

        if verbose >= 1:
            should_log = (
                epoch == 0
                or (epoch + 1 == epochs)
                or improved
                or ((epoch + 1) % max(1, log_every) == 0)
            )
            if should_log:
                msg = f"Epoch [{epoch+1}/{epochs}] train={avg_train_loss:.4f} val={avg_val_loss:.4f}"
                if print_l0:
                    msg += f" l0={avg_l0_loss:.4f}"
                if improved:
                    msg += " (*)"
                print(msg)

        # -------------------------
        # Hook (backward compatible)
        # -------------------------
        if after_epoch_hook is not None:
            metrics: Dict[str, float | int | bool] = dict(
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                l0_loss=avg_l0_loss,
                improved=improved,
            )
            try:
                sig = inspect.signature(after_epoch_hook)
                if len(sig.parameters) >= 3:
                    after_epoch_hook(model, epoch, metrics)  # extended
                else:
                    after_epoch_hook(model, epoch)           # legacy
            except Exception:
                # Fallback to legacy on any inspection/runtime error
                try:
                    after_epoch_hook(model, epoch)
                except Exception:
                    pass

        # -------------------------
        # Early stopping
        # -------------------------
        if improved:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose >= 1:
                    print(f"Early stopping at epoch {epoch+1} (best val={best_val_loss:.4f})")
                break

    # Restore best
    if best_model_state:
        model.load_state_dict(best_model_state)
        model.to(device)
        if verbose >= 1:
            print(f"Restored best model (val={best_val_loss:.4f})")

    return best_val_loss