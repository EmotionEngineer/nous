# Nous: A Neuro-Symbolic Library for Interpretable AI

Nous is a rule-based neural network with honest, fidelity-preserving interpretability for classification and regression. It combines:
- Beta fact activations over calibrated inputs,
- Rule layers with explicit aggregator mixtures (AND, OR, k-of-n, NOT),
- Honest leave-one-out explanations (pre-gating recomputation),
- Fidelity-driven pruning,
- Minimal Sufficient Explanations (MSE) via greedy elimination,
- Prototype head (optional) and end-to-end tracing: prototype → rule → β-facts,
- Sparse Hard-Concrete connections with L0 scheduling,
- NumPy-only export with probability-first validation.

This package provides production-ready modules, tests, and examples.

## Installation
```bash
pip install nous
```

## Quickstart

```python
import torch
from nous import NousNet

# Build a small classifier
model = NousNet(
    input_dim=20,
    num_outputs=3,
    task_type="classification",
    num_facts=32,
    rules_per_layer=(16, 8),
    rule_selection_method="softmax",
    use_calibrators=True,
    use_prototypes=False
)

x = torch.randn(4, 20)
logits = model(x)  # [4, 3]

# Honest explain for a single sample
probas, logits1, internals = model.forward_explain(x[0])
```

## Key features
- Honest LOO: rule removal before top-k, full top-k recomputation; frozen modes supported.
- Clean explanations: disable LayerNorm and exclude residual projection when desired.
- Fidelity-driven pruning: grid and binary search selection with fidelity or MAE constraints.
- MSE: greedy backward elimination that preserves the prediction under tolerances.
- Rule-based counterfactuals via β-fact geometry (with calibrators).
- Aggregator mixtures reported as soft mixtures (not argmax).
- Sparse L0: Hard-Concrete gates, safe sampling, schedulers for regression.
- Prototypes: global report, per-sample contributions, tracing to β-facts.
- NumPy-only export and validator (probability-first).

## Examples
See the `examples/` directory for:
- Wine classification,
- California housing regression,
- NumPy export and validation.

## Contributing
See CONTRIBUTING.md

## License
MIT License. See LICENSE.