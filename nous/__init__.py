from .version import __version__
from .model import NousNet
from .facts import BetaFactLayer, PiecewiseLinearCalibrator
from .prototypes import ScaledPrototypeLayer
from .rules import FixedPairRuleLayer, SoftmaxRuleLayer, SparseRuleLayer, SimpleNousBlock
from .explain import (
    rule_impact_df,
    minimal_sufficient_explanation,
    select_pruning_threshold_global,
    select_pruning_threshold_global_bs,
    global_rulebook,
    generate_enhanced_explanation,
    explanation_fidelity_metrics,
    explanation_stability,
    aggregator_mixture_report,
    suggest_rule_counterfactuals,
    render_fact_descriptions,
    AGG_NAMES,
)
from .export import export_numpy_inference, validate_numpy_vs_torch, export_and_validate
from .training import train_model, evaluate_classification, evaluate_regression, make_sparse_regression_hook

__all__ = [
    "__version__",
    "NousNet",
    "BetaFactLayer",
    "PiecewiseLinearCalibrator",
    "ScaledPrototypeLayer",
    "FixedPairRuleLayer",
    "SoftmaxRuleLayer",
    "SparseRuleLayer",
    "SimpleNousBlock",
    "rule_impact_df",
    "minimal_sufficient_explanation",
    "select_pruning_threshold_global",
    "select_pruning_threshold_global_bs",
    "global_rulebook",
    "generate_enhanced_explanation",
    "explanation_fidelity_metrics",
    "explanation_stability",
    "aggregator_mixture_report",
    "suggest_rule_counterfactuals",
    "render_fact_descriptions",
    "AGG_NAMES",
    "export_numpy_inference",
    "validate_numpy_vs_torch",
    "export_and_validate",
    "train_model",
    "evaluate_classification",
    "evaluate_regression",
    "make_sparse_regression_hook",
]