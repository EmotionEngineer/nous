import numpy as np
import torch
import pytest
from torch.utils.data import TensorDataset, DataLoader

from nous import SoftLogitAND
from nous.training import train_model


def _make_small_binary_dataset(n=512, d=20, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)
    w = rng.randn(d).astype(np.float32)
    logits = (X @ w) * 0.2
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.rand(n) < p).astype(np.float32)
    return X, y


def _make_loaders(X, y, batch_size=64):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    return dl


def test_train_model_supports_new_options_sample_average_and_set_to_none():
    X, y = _make_small_binary_dataset(n=512, d=25, seed=1)

    # SoftLogitAND uses init_from_data; feed scaled-like data directly for a smoke test
    model = SoftLogitAND(input_dim=X.shape[1], n_rules=32, n_thresh_per_feat=4, tau=0.7, use_negations=True)
    model.init_from_data(X)

    train_loader = _make_loaders(X[:400], y[:400], batch_size=64)
    val_loader = _make_loaders(X[400:], y[400:], batch_size=64)

    device = torch.device("cpu")
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = torch.nn.BCEWithLogitsLoss()

    best = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=crit,
        optimizer=opt,
        epochs=3,
        patience=2,
        device=device,
        verbose=0,
        print_l0=False,
        clip_grad_max_norm=1.0,
        zero_grad_set_to_none=True,
        loss_average="sample",
    )

    assert isinstance(best, float)
    assert np.isfinite(best)


def test_train_model_rejects_invalid_loss_average():
    X, y = _make_small_binary_dataset(n=128, d=10, seed=2)
    model = SoftLogitAND(input_dim=X.shape[1], n_rules=16, n_thresh_per_feat=4, tau=0.7, use_negations=True)
    model.init_from_data(X)

    dl = _make_loaders(X, y, batch_size=32)
    device = torch.device("cpu")
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = torch.nn.BCEWithLogitsLoss()

    with pytest.raises(ValueError):
        train_model(
            model=model,
            train_loader=dl,
            val_loader=dl,
            criterion=crit,
            optimizer=opt,
            epochs=2,
            patience=1,
            device=device,
            verbose=0,
            print_l0=False,
            loss_average="not_a_mode",
        )