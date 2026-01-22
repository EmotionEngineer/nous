#!/usr/bin/env python3
"""
examples/softlogit_synthetic.py

Minimal SoftLogitAND example:
- synthetic many-feature binary classification with hidden ground-truth rules
- train SoftLogitAND on *scaled* features
- eval train/val/test
- post-hoc explain (local markdown + a few global CSVs)

Install (typical):
  pip install numpy pandas scikit-learn torch
  pip install nous
Run:
  python examples/softlogit_synthetic.py
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

from nous import set_global_seed, SoftLogitAND, SoftLogitANDPosthocExplainer
from nous.training import train_model
from nous.explain.softlogitand_posthoc import predict_logits_softlogitand


# ----------------------------
# Config (edit here)
# ----------------------------
SEED = 42
N = 30_000
D = 200
LABEL_FLIP = 0.03
CORR_STRENGTH = 0.90

# SoftLogitAND hyperparams
N_RULES = 256
N_THRESH_PER_FEAT = 4
TAU = 0.7
USE_NEGATIONS = True

# Training
BATCH_SIZE = 512
LR = 2e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 1500
PATIENCE = 150

# Outputs / reporting
OUT_DIR = "reports"
SAMPLE_ID = 3  # local explanation for this test sample index


# ----------------------------
# Data (synthetic + truth rules)
# ----------------------------
def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def make_truth_rules() -> List[dict]:
    # Each rule: weight + list of (feature_id, op, threshold), op in {">", "<="}
    return [
        {"name": "R1_pos", "weight": +2.4, "conds": [(3, ">", +0.7), (17, "<=", -0.2)]},
        {"name": "R2_pos", "weight": +2.0, "conds": [(55, ">", 0.0), (120, "<=", 0.0), (7, ">", 0.0)]},
        {"name": "R3_neg", "weight": -2.0, "conds": [(10, "<=", -1.2), (11, "<=", -0.8)]},
        {"name": "R4_pos", "weight": +1.2, "conds": [(80, ">", +0.3), (81, ">", +0.3)]},
    ]


def eval_truth_rule(X: np.ndarray, rule: dict) -> np.ndarray:
    m = np.ones((X.shape[0],), dtype=bool)
    for fid, op, thr in rule["conds"]:
        if op == ">":
            m &= (X[:, fid] > thr)
        elif op == "<=":
            m &= (X[:, fid] <= thr)
        else:
            raise ValueError(f"Unknown op: {op}")
    return m


def make_rule_dataset(
    n: int,
    d: int,
    seed: int,
    label_flip: float,
    corr_strength: float,
) -> Tuple[np.ndarray, np.ndarray, List[dict], List[str]]:
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)

    # Add correlated "decoy" features for some truth features
    def make_correlated(dst: int, src: int) -> None:
        eps = rng.normal(size=(n,)).astype(np.float32)
        X[:, dst] = corr_strength * X[:, src] + np.sqrt(max(1e-6, 1 - corr_strength**2)) * eps

    make_correlated(4, 3)
    make_correlated(18, 17)
    make_correlated(12, 10)
    make_correlated(13, 11)

    truth = make_truth_rules()
    logits = np.zeros((n,), dtype=np.float64) - 0.3  # bias

    for r in truth:
        sat = eval_truth_rule(X, r).astype(np.float64)
        logits += r["weight"] * sat

    p = sigmoid_np(logits)
    y = rng.binomial(1, p).astype(np.float32)

    # Label noise
    flip = rng.rand(n) < label_flip
    y[flip] = 1.0 - y[flip]

    feature_names = [f"x{i:03d}" for i in range(d)]
    return X, y, truth, feature_names


# ----------------------------
# Training & eval helpers
# ----------------------------
def make_loaders(
    Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    tr = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32))
    va = TensorDataset(torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32))
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(va, batch_size=batch_size, shuffle=False, drop_last=False),
    )


def eval_binary_from_logits(logits: np.ndarray, y_true01: np.ndarray) -> Dict[str, float]:
    logits = np.asarray(logits, dtype=np.float64).ravel()
    prob = sigmoid_np(logits)
    prob = np.clip(prob, 1e-7, 1.0 - 1e-7)
    y_int = np.asarray(y_true01, dtype=np.int64).ravel()
    pred = (prob >= 0.5).astype(np.int64)
    return {
        "auc": float(roc_auc_score(y_int, prob)),
        "acc": float(accuracy_score(y_int, pred)),
        "logloss": float(log_loss(y_int, prob, labels=[0, 1])),
    }


@torch.no_grad()
def eval_softlogitand(model: SoftLogitAND, X_scaled: np.ndarray, y_true01: np.ndarray, batch_size: int = 4096) -> Dict[str, float]:
    logits = predict_logits_softlogitand(model, X_scaled, batch_size=batch_size)
    return eval_binary_from_logits(logits, y_true01)


def main() -> None:
    # Repro + device
    set_global_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    X_raw, y_raw, truth_rules, feature_names = make_rule_dataset(
        n=N, d=D, seed=SEED, label_flip=LABEL_FLIP, corr_strength=CORR_STRENGTH
    )

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=SEED, stratify=y_raw
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, random_state=SEED, stratify=y_trainval
    )  # train 70%, val 10%, test 20%

    # Scaling (SoftLogitAND expects scaled)
    x_scaler = StandardScaler().fit(X_train)
    X_train_s = x_scaler.transform(X_train).astype(np.float32)
    X_val_s = x_scaler.transform(X_val).astype(np.float32)
    X_test_s = x_scaler.transform(X_test).astype(np.float32)

    # Model
    teacher = SoftLogitAND(
        input_dim=X_train_s.shape[1],
        n_rules=N_RULES,
        n_thresh_per_feat=N_THRESH_PER_FEAT,
        tau=TAU,
        use_negations=USE_NEGATIONS,
    )
    teacher.init_from_data(X_train_s)
    teacher.to(device)

    tr_loader, va_loader = make_loaders(X_train_s, y_train, X_val_s, y_val, batch_size=BATCH_SIZE)

    _ = train_model(
        model=teacher,
        train_loader=tr_loader,
        val_loader=va_loader,
        criterion=nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW(teacher.parameters(), lr=LR, weight_decay=WEIGHT_DECAY),
        epochs=EPOCHS,
        patience=PATIENCE,
        device=device,
        clip_grad_max_norm=1.0,
        zero_grad_set_to_none=True,
        loss_average="sample",
        print_l0=False,
    )

    # Metrics
    df_metrics = pd.DataFrame(
        [
            {"split": "train", **eval_softlogitand(teacher, X_train_s, y_train)},
            {"split": "val", **eval_softlogitand(teacher, X_val_s, y_val)},
            {"split": "test", **eval_softlogitand(teacher, X_test_s, y_test)},
        ]
    )
    print("\n=== SoftLogitAND metrics ===")
    print(df_metrics.to_string(index=False))

    # Post-hoc explainer
    expl = SoftLogitANDPosthocExplainer(
        model=teacher,
        feature_names=feature_names,
        x_scaler=x_scaler,
        # reporting config
        k_rules=10,
        k_literals=4,
        summary_top_features=8,
        z_threshold=0.5,
        # selection / diversity
        topL=12,
        lambda_div=0.40,
        candidate_pool=140,
        cap_per_feature=2,
        # themes
        n_clusters=16,
        # surrogate
        use_surrogate=True,
        surrogate_alpha=1e-3,
    )
    expl.fit_posthoc(X_ref_scaled=X_train_s)
    expl.fit_rule_stats(X_train_scaled=X_train_s, y_train01=y_train, X_test_scaled=X_test_s, y_test01=y_test)
    expl.add_mean_abs_contrib_to_rule_stats(X_train_scaled=X_train_s, X_test_scaled=X_test_s)

    print("\n=== Cluster summary (top-8) ===")
    print(expl.cluster_summary.head(8).to_string(index=False))

    # Local markdown report for one test sample
    os.makedirs(OUT_DIR, exist_ok=True)
    sample_id = int(np.clip(SAMPLE_ID, 0, len(X_test) - 1))
    x_sample_raw = X_test[sample_id]
    y_sample = float(y_test[sample_id])

    rep = expl.report(x_sample_raw, y_true01=y_sample, x_is_scaled=False, top_clusters=8)
    local_path = os.path.join(OUT_DIR, f"softlogitand_local_report_sample_{sample_id}.md")
    expl.save_markdown(rep, local_path)
    print(f"\nSaved local report: {local_path}")

    # Some global CSVs (handy for CI checks / quick review)
    expl.global_rule_importance(split="train", topn=20, sort_by="mean_abs_contrib").to_csv(
        os.path.join(OUT_DIR, "global_rules_train_top20.csv"), index=False
    )
    expl.global_feature_importance_mass_weighted(split="train", topn=30).to_csv(
        os.path.join(OUT_DIR, "global_features_train_top30.csv"), index=False
    )
    expl.global_cluster_importance(split="train").head(16).to_csv(
        os.path.join(OUT_DIR, "global_clusters_train_top16.csv"), index=False
    )
    print(f"Saved global CSVs to: {OUT_DIR}")

    # Optional sanity print: which truth rules fire for the explained sample?
    X1 = x_sample_raw.reshape(1, -1)
    truth_rows = []
    for r in truth_rules:
        truth_rows.append(
            {
                "truth_rule": r["name"],
                "weight": float(r["weight"]),
                "satisfied": bool(eval_truth_rule(X1, r)[0]),
                "conds": str(r["conds"]),
            }
        )
    print("\n=== Truth rules for sample ===")
    print(pd.DataFrame(truth_rows).to_string(index=False))


if __name__ == "__main__":
    main()
