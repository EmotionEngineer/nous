# Nous — A Neuro-Symbolic Library for Interpretable AI

[![PyPI](https://img.shields.io/pypi/v/nous.svg)](https://pypi.org/project/nous/)
[![Python](https://img.shields.io/pypi/pyversions/nous.svg)](https://pypi.org/project/nous/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Make tabular models you can read.  
Nous learns compact logical rules and optional case‑based prototypes inside one differentiable model — so prediction and explanation come from the same place.

- 🧩 One white‑box → two styles: rules and/or prototypes
- 🔀 Learned AND / OR / k‑of‑n mixtures capture interactions without bloat
- ✂️ Minimal, faithful stories: pruning + sufficiency/comprehensiveness checks
- 🚀 Practical: competitive accuracy, NumPy export, unit‑tested toolkit

## Key Features

- Intrinsic interpretability (not post‑hoc): explanations are part of the forward pass
- Switchable style: enable/disable prototypes; choose rule selection (fixed / softmax / sparse); add calibrators
- Fidelity diagnostics: pruned‑forward inference, minimal‑sufficient explanations, stability tools
- Ready to ship: pure‑NumPy export for inference without PyTorch

## Installation

```bash
# Stable from PyPI
pip install nous

# With example extras (plots, progress, UCI fetchers)
pip install "nous[examples]"

# Dev setup (tests, linters, type checks)
pip install "nous[dev]"
```

Requirements (core):
- Python 3.9+
- torch>=2.1
- numpy>=1.22
- pandas>=1.5
- scikit-learn>=1.2

Extras:
- examples: matplotlib>=3.6, seaborn>=0.12, tqdm>=4.65, ucimlrepo>=0.0.5
- dev: pytest>=7.0, pytest-cov>=4.0, mypy>=1.5, ruff>=0.5, black>=23.0, matplotlib>=3.6, seaborn>=0.12, tqdm>=4.65, ucimlrepo>=0.0.5

## Recommended Configurations

| Profile | Rule selection | Calibrators | Prototypes | Use when | Speed |
|--------|-----------------|-------------|------------|----------|-------|
| Fast baseline | fixed | off | off | quick sweeps, ablations | ⚡⚡⚡ |
| Default rules | softmax | on  | off | general use, strong accuracy | ⚡⚡ |
| Explain‑everything | softmax | on  | on  | rich case‑based narratives | ⚡ |

Tips:
- Train with prototypes off for speed; enable them only on the final model if you need case‑based stories.
- 300 epochs with patience≈50 works well on common tabular datasets.

## Bench Snapshot (5‑fold CV, typical)

| Dataset | Metric | Nous (rules) | Nous (+proto) | EBM | XGBoost |
|--------|--------|--------------|---------------|-----|---------|
| HELOC (cls) | AUC | ~0.791 | ~0.792 | ~0.799 | ~0.796 |
| Adult (cls) | AUC | ~0.913 | ~0.914 | ~0.926 | ~0.929 |
| Breast Cancer (cls) | Acc | ~0.975 | ~0.983 | ~0.970 | ~0.965 |
| California (reg) | RMSE | ~0.514 | ~0.505 | ~0.562 | ~0.439 |

Numbers vary with seed/HPO. See examples/benchmark.ipynb for reproducible runs.

## What makes Nous different?

- The explanation is the model: rules and prototypes live in the forward pass
- Interactions without clutter: AND/OR/k‑of‑n mixtures keep explanations short
- Verified stories: minimal‑sufficient explanations + pruned‑forward confidence checks
- Lightweight deployment: NumPy export (no torch at inference)

## Repository Layout

- examples/
  - benchmark.ipynb — end‑to‑end comparison on classic tabular data
  - wine_classification.py, california_regression.py — minimal scripts
  - export_numpy_demo.py — deploy without torch
- nous/
  - model.py (NousNet), facts.py (calibrated L−R facts)
  - rules/* (fixed/softmax/sparse), explain/* (pruning, fidelity, traces, prototypes)
  - training/* (loop, schedulers), export/* (NumPy), utils/*
- tests/ — unit tests for forward, rules, facts, prototypes, explanations, export

## License

MIT — see LICENSE.
