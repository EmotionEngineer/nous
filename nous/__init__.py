from .version import __version__
from .model import NousNet
from .facts import BetaFactLayer, PiecewiseLinearCalibrator, PiecewiseLinearCalibratorQuantile
from .prototypes import ScaledPrototypeLayer
from .rules import FixedPairRuleLayer, SoftmaxRuleLayer, SparseRuleLayer, SoftFactRuleLayer, SimpleNousBlock
from .rules import (
    BaseRuleGater,
    GateSoftRank,
    GateLogisticThresholdExactK,
    GateCappedSimplex,
    GateSparsemaxK,
    GateHardConcreteBudget,
    make_rule_gater,
)
# Explainability (core API)
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
from .explain.aggregator import format_agg_mixture
# Prototype tracing utilities
from .explain.traces import (
    describe_prototype,
    prototype_report_global,
    prototype_contribution_df,
    prototype_top_rules,
    trace_rule_to_base_facts,
    get_last_block_static_metadata,
)
# Export utilities
from .export import (
    export_numpy_inference,
    validate_numpy_vs_torch,
    export_and_validate,
    load_numpy_module,
)
# Training and evaluation
from .training import (
    train_model,
    evaluate_classification,
    evaluate_regression,
    make_sparse_regression_hook,
)
# Dataset helpers (used in examples)
from .data import get_wine_data, get_california_housing_data
# Utilities
from .utils import set_global_seed, make_quantile_calibrators

__all__ = [
    "__version__",
    # Core model and components
    "NousNet",
    "BetaFactLayer",
    "PiecewiseLinearCalibrator",
    "PiecewiseLinearCalibratorQuantile",
    "ScaledPrototypeLayer",
    "FixedPairRuleLayer",
    "SoftmaxRuleLayer",
    "SparseRuleLayer",
    "SoftFactRuleLayer",
    "SimpleNousBlock",
    # Differentiable rule gaters
    "BaseRuleGater",
    "GateSoftRank",
    "GateLogisticThresholdExactK",
    "GateCappedSimplex",
    "GateSparsemaxK",
    "GateHardConcreteBudget",
    "make_rule_gater",
    # Explainability (core)
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
    "format_agg_mixture",
    # Prototype tracing utilities
    "describe_prototype",
    "prototype_report_global",
    "prototype_contribution_df",
    "prototype_top_rules",
    "trace_rule_to_base_facts",
    "get_last_block_static_metadata",
    # Export utilities
    "export_numpy_inference",
    "validate_numpy_vs_torch",
    "export_and_validate",
    "load_numpy_module",
    # Training and evaluation
    "train_model",
    "evaluate_classification",
    "evaluate_regression",
    "make_sparse_regression_hook",
    # Dataset helpers
    "get_wine_data",
    "get_california_housing_data",
    # Utilities
    "set_global_seed",
    "make_quantile_calibrators",
]