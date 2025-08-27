# Nous: A Neuro-Symbolic Library for Interpretable AI

[![PyPI version](https://badge.fury.io/py/nous.svg)](https://badge.fury.io/py/nous)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Nous** is a PyTorch-based library for building "white-box" machine learning models for both **regression** and **classification**. It enables models that don't just predict, but also **reason** and **explain** their decisions in human-understandable terms.

## Key Features

-   **Deeply Interpretable**: Generate a complete, step-by-step logical trace (`fact -> rule -> concept -> prediction`) for any decision.
-   **Supports Regression & Classification**: A unified API for predicting both continuous values and class probabilities.
-   **Causal by Design**: Natively supports counterfactual analysis ("What if?") to provide actionable recommendations for both regression and classification tasks.
-   **High Performance**: Achieves accuracy competitive with traditional black-box models.
-   **Scalable & Flexible**: Choose between a high-performance `beta` activation, a robust `sigmoid`, or a maximally transparent `exhaustive` fact layer.

## Installation

```bash
pip install nous
```

## Quickstart: A 5-Minute Example (Regression)

Let's predict a house price and understand the model's reasoning.

```python
import torch
import pandas as pd
from sklearn.datasets import make_regression
from nous.models import NousNet
from nous.interpret import trace_decision_graph, explain_fact
from nous.causal import find_counterfactual

# 1. Prepare Data
X_raw, y = make_regression(n_samples=1000, n_features=5, n_informative=3, noise=20, random_state=42)
feature_names = ['area_sqft', 'num_bedrooms', 'dist_to_center', 'age_years', 'renovation_quality']
X = torch.tensor(X_raw, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# 2. Define and Train a NousNet for Regression
model = NousNet(
    input_dim=5,
    output_dim=1, # Single output for regression
    feature_names=feature_names,
    fact_layer_type='beta'
)
# Training: Use a regression loss like nn.MSELoss
# loss_fn = torch.nn.MSELoss()
# ... (standard training loop omitted)
model.eval()

# 3. Analyze a specific house
x_sample = X[50]
predicted_price = model(x_sample).item()
print(f"Model's predicted price for house #50: ${predicted_price:,.2f}")

# 4. Get the Step-by-Step Reasoning
graph = trace_decision_graph(model, x_sample)
top_facts = sorted(graph['trace']['Atomic Facts'].items(), key=lambda i: i[1]['value'], reverse=True)
fact_to_analyze = top_facts[0][0]
print(f"\nTop activated fact influencing the price: '{fact_to_analyze}'")

# 5. Decode the Learned Fact
details_df = explain_fact(model, fact_name=fact_to_analyze)
print(f"\nDecoding '{fact_to_analyze}':")
display(details_df.head())

# 6. Get an Actionable Recommendation
# What's the smallest change to increase the predicted price to $150,000?
recommendation = find_counterfactual(
    model,
    x_sample,
    target_output=150.0, # Target value for regression
    task='regression'
)
print("\nRecommendation to increase value to $150k:")
for feature, old_val, new_val in recommendation['changes']:
    print(f"- Change '{feature}' from {old_val:.2f} to {new_val:.2f}")

```

## Choosing a `fact_layer_type`

-   `'beta'` (**Default, Recommended**): Best performance and flexibility.
-   `'sigmoid'`: A robust and reliable alternative.
-   `'exhaustive'`: Maximum transparency. Best for low-dimensional problems (<15 features).

## License

This project is licensed under the MIT License.