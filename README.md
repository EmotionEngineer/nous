# Nous: A Neuro-Symbolic Library for Interpretable AI

[![PyPI](https://img.shields.io/pypi/v/nous.svg)](https://pypi.org/project/nous/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python ≥3.10](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![PyTorch ≥2.1](https://img.shields.io/badge/PyTorch-2.1%2B-orange)](https://pytorch.org/)

**Nous** (Greek: νοῦς, “mind”) is a neuro-symbolic library for **interpretable learning** in PyTorch. Nous models are built from **facts** and **rules**, and explanations are derived from the model’s computation via **intervention-style forward evaluation** (not post-hoc gradients).

> **Design:** features → facts → rules → prediction  
> **Explanations:** recompute forward passes under controlled rule/fact interventions

---

## Installation

```bash
pip install nous
```

Optional extras:

```bash
pip install "nous[examples]"
pip install "nous[dev]"
```

---

## Core Concepts (short)

| Concept | Meaning |
|---|---|
| **Facts** | Differentiable feature transforms (e.g., thresholds) producing values in \[0,1] |
| **Rules** | Soft logical compositions (AND/OR/k-of-n/NOT) over facts |
| **Gating** | Sparse rule selection / activation |
| **Heads** | Prediction layer (binary, regression, multiclass) |

---

## Quick Start (SoftLogitAND)

SoftLogitAND is a strong default for tabular tasks when you want a clean rules-first structure.

```python
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from nous import SoftLogitAND
from nous.training import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# X_train, y_train, X_val, y_val are numpy arrays
scaler = StandardScaler().fit(X_train)
Xtr = scaler.transform(X_train).astype("float32")
Xva = scaler.transform(X_val).astype("float32")

model = SoftLogitAND(
    input_dim=Xtr.shape[1],
    n_rules=256,
    n_thresh_per_feat=4,
    tau=0.7,
    use_negations=True,
)
model.init_from_data(Xtr)
model.to(device)

train_loader = DataLoader(
    TensorDataset(torch.tensor(Xtr), torch.tensor(y_train, dtype=torch.float32)),
    batch_size=512, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(torch.tensor(Xva), torch.tensor(y_val, dtype=torch.float32)),
    batch_size=512, shuffle=False
)

train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=nn.BCEWithLogitsLoss(),
    optimizer=torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4),
    epochs=200,
    patience=25,
    device=device,
)
```

---

## Model Zoo (high level)

| Family | Examples |
|---|---|
| Evidence-style | `EvidenceNet`, `MarginEvidenceNet`, `PerFeatureKappaEvidenceNet` |
| Grouped logic | `GroupEvidenceKofNNet`, `GroupSoftMinNet`, `GroupContrastNet` |
| Regimes | `RegimeRulesNet` |
| Differentiable forests | `PredicateForest`, `ObliviousForest`, `GroupFirstForest`, `BudgetedForest` |

---

## Model Zoo: minimal interpretation example (global + local)

This is the “notebook-style” workflow, but **kept small**: train a zoo_v2 model, predict, then extract:

- **global rules** (`global_rules_df`)
- **local contributions** (`local_contrib_df`)
- **text explanation** (`explain_prediction`)
- optional export (`export_global_rules`)

```python
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from nous.zoo_v2 import EvidenceNet
from nous.explain.zoo_v2 import global_rules_df, local_contrib_df, explain_prediction, export_global_rules

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assume:
#   X_train_raw: (N, D) float numpy
#   y_train: (N,) in {0,1}
#   feature_names: list[str] length D

# 1) Scale for training (common for zoo_v2)
scaler = StandardScaler().fit(X_train_raw)
X_train = scaler.transform(X_train_raw).astype("float32")

# 2) Train a zoo_v2 model (binary classification => output_dim=1)
model = EvidenceNet(input_dim=X_train.shape[1], n_rules=128, init_kappa=6.0, beta=6.0, output_dim=1).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

Xt = torch.tensor(X_train, device=device)
yt = torch.tensor(y_train, dtype=torch.float32, device=device)

model.train()
for _ in range(100):
    opt.zero_grad(set_to_none=True)
    logits = model(Xt).view(-1)
    loss = loss_fn(logits, yt)
    loss.backward()
    opt.step()

# 3) Predict (probability)
model.eval()
with torch.no_grad():
    p0 = torch.sigmoid(model(Xt[:1]).view(-1)).item()
print("pred_proba(x0):", p0)

# 4) Interpret
# For explain helpers: pass X_ref and x in the *pre-scaler* space if you provide scaler=...
X_ref = X_train_raw[:5000]
x0_raw = X_train_raw[0]

df_g = global_rules_df(
    model,
    feature_names,
    scaler=scaler,          # makes thresholds readable in the raw/original feature space
    X_ref=X_ref,
    readability="clinical",
    top_rules=12,
    top_feats=4,
    n_trees=6,              # used by forest-style models; safe to keep for others
)
print(df_g.head())

df_l, meta = local_contrib_df(
    model,
    x0_raw,                 # raw x (pre-scaler) because scaler=... is provided
    feature_names,
    scaler=scaler,
    X_ref=X_ref,
    readability="clinical",
    top_rules=12,
)
print("local meta:", meta)
print(df_l.head())

print(explain_prediction(
    model, x0_raw, feature_names,
    scaler=scaler, X_ref=X_ref,
    readability="clinical", top_rules=8
))

# 5) Optional: export global rules
export_global_rules(model, feature_names, path="rules.txt",  format="txt",  scaler=scaler, X_ref=X_ref, readability="clinical", top_rules=30, top_feats=4, n_trees=6)
export_global_rules(model, feature_names, path="rules.json", format="json", scaler=scaler, X_ref=X_ref, readability="clinical", top_rules=30, top_feats=4, n_trees=6)
```

---

## Real‑world Benchmarks (5‑Fold CV)

These are small reference runs (mean ± std across folds) comparing selected Nous (torch) models with **XGBoost** and **EBM**. They are provided as sanity/reference points.

### QSAR Fish Toxicity (Regression)

| model | kind | time_sec_mean | RMSE (mean±std) | MAE (mean±std) | R² (mean±std) |
|---|---|---:|---:|---:|---:|
| XGB(depth=6) *(best XGB)* | sklearn_xgb | 0.662 | **0.904±0.038** | **0.648±0.029** | **0.612±0.031** |
| EBM(interactions=30) *(best EBM)* | sklearn_ebm | 4.783 | 0.912±0.041 | 0.656±0.039 | 0.605±0.028 |
| BudgetedForest(k=3,trees=32,depth=4) | torch | 8.861 | 0.927±0.022 | 0.673±0.016 | 0.591±0.042 |
| EvidenceNet | torch | 7.219 | 0.942±0.034 | 0.683±0.018 | 0.575±0.059 |
| GroupContrastNet | torch | 9.783 | 0.945±0.016 | 0.687±0.032 | 0.574±0.040 |
| GroupEvidenceKofNNet | torch | 12.349 | 0.956±0.033 | 0.689±0.023 | 0.565±0.042 |
| GroupFirstForest(trees=32,depth=4) | torch | 9.208 | 0.929±0.029 | 0.671±0.025 | 0.589±0.044 |
| GroupSoftMinNet | torch | 9.982 | 0.921±0.035 | 0.672±0.033 | 0.595±0.048 |
| MarginEvidenceNet | torch | 6.123 | 0.938±0.025 | 0.681±0.022 | 0.579±0.057 |
| ObliviousForest(trees=32,depth=4) | torch | 11.227 | 0.965±0.048 | 0.690±0.018 | 0.557±0.046 |
| PerFeatureKappaEvidenceNet | torch | 7.534 | 0.952±0.031 | 0.687±0.015 | 0.566±0.065 |
| PredicateForest(trees=32,depth=4) | torch | 11.297 | 0.965±0.048 | 0.690±0.018 | 0.557±0.046 |
| RegimeRulesNet | torch | 8.914 | 0.930±0.040 | 0.680±0.028 | 0.587±0.051 |

### Concrete Compressive Strength (Regression)

| model | kind | time_sec_mean | RMSE (mean±std) | MAE (mean±std) | R² (mean±std) |
|---|---|---:|---:|---:|---:|
| EBM(interactions=30) *(best EBM)* | sklearn_ebm | 20.506 | **3.966±0.375** | 2.779±0.228 | **0.943±0.011** |
| XGB(depth=6) *(best XGB)* | sklearn_xgb | 3.419 | 4.194±0.366 | 2.681±0.186 | 0.936±0.013 |
| BudgetedForest(k=3,trees=32,depth=4) | torch | 30.789 | 6.198±0.271 | 4.689±0.156 | 0.861±0.016 |
| EvidenceNet | torch | 16.476 | 4.153±0.335 | 2.757±0.227 | 0.938±0.009 |
| GroupContrastNet | torch | 21.682 | 4.176±0.359 | 2.781±0.226 | 0.937±0.010 |
| GroupEvidenceKofNNet | torch | 18.880 | 4.176±0.382 | 2.809±0.276 | 0.937±0.010 |
| GroupFirstForest(trees=32,depth=4) | torch | 28.413 | 10.565±3.227 | 8.399±2.821 | 0.567±0.268 |
| GroupSoftMinNet | torch | 29.670 | 4.275±0.442 | 2.886±0.322 | 0.934±0.014 |
| MarginEvidenceNet | torch | 24.249 | 4.052±0.368 | **2.681±0.222** | 0.941±0.009 |
| ObliviousForest(trees=32,depth=4) | torch | 31.236 | 7.752±1.612 | 6.060±1.349 | 0.774±0.094 |
| PerFeatureKappaEvidenceNet | torch | 16.744 | 4.195±0.465 | 2.727±0.308 | 0.936±0.014 |
| PredicateForest(trees=32,depth=4) | torch | 31.166 | 7.752±1.612 | 6.060±1.349 | 0.774±0.094 |
| RegimeRulesNet | torch | 33.734 | 4.469±0.474 | 3.006±0.374 | 0.928±0.015 |

### Myocardial Infarction Complications (Multiclass)

| model | kind | time_sec_mean | logloss (mean±std) | acc (mean±std) | f1_macro (mean±std) | auc_ovr (mean±std) |
|---|---|---:|---:|---:|---:|---:|
| MarginEvidenceNet *(best Nous here)* | torch | 3.369 | **0.525±0.025** | 0.864±0.006 | 0.184±0.007 | **0.833±0.028** |
| XGB(depth=4) *(best XGB)* | sklearn_xgb | 1.534 | 0.545±0.025 | **0.865±0.006** | 0.183±0.009 | 0.815±0.043 |
| EBM(interactions=30) *(best EBM)* | sklearn_ebm | 80.716 | 0.547±0.027 | 0.863±0.004 | 0.184±0.010 | 0.809±0.040 |
| EvidenceNet | torch | 2.974 | 0.536±0.028 | 0.861±0.008 | 0.180±0.011 | 0.819±0.035 |
| GroupContrastNet | torch | 15.010 | 0.536±0.028 | 0.864±0.010 | 0.184±0.009 | 0.823±0.032 |
| GroupEvidenceKofNNet | torch | 13.212 | 0.534±0.027 | 0.862±0.007 | 0.181±0.009 | 0.826±0.033 |
| GroupFirstForest(trees=32,depth=4) | torch | 6.020 | 0.540±0.034 | 0.865±0.007 | 0.188±0.011 | 0.821±0.036 |
| GroupSoftMinNet | torch | 27.423 | 0.596±0.100 | 0.855±0.014 | 0.157±0.039 | 0.827±0.033 |
| BudgetedForest(k=3,trees=32,depth=4) | torch | 5.528 | 0.571±0.012 | 0.861±0.011 | 0.191±0.032 | 0.775±0.024 |
| ObliviousForest(trees=32,depth=4) | torch | 5.927 | 0.543±0.025 | 0.862±0.010 | 0.192±0.021 | 0.825±0.031 |
| PerFeatureKappaEvidenceNet | torch | 2.967 | 0.535±0.027 | 0.861±0.007 | 0.179±0.009 | 0.820±0.036 |
| PredicateForest(trees=32,depth=4) | torch | 5.923 | 0.543±0.025 | 0.862±0.010 | 0.192±0.021 | 0.825±0.031 |
| RegimeRulesNet | torch | 4.407 | 0.547±0.026 | 0.858±0.008 | 0.179±0.009 | 0.819±0.029 |

---

## Citation

```bibtex
@software{tlupov2025nous,
  author = {Tlupov, Islam},
  title = {Nous: A Neuro-Symbolic Library for Interpretable AI},
  url = {https://github.com/EmotionEngineer/nous},
  year = {2025}
}
```

**License:** MIT
