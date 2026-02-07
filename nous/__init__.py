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

# Model zoo (business-facing models)
from .zoo import (
    SoftLogitAND,
    SoftLogicInteraction,
    SegmentMoE,
    HierarchicalMoE,
    NousFamilies,
)

# Model zoo v2
from .zoo_v2 import (
    MLP, BoxNet,
    IntervalFactBank, RelationalFactBank, RelationalDiffFactBank, ARFactBank, MultiResAxisFactBank,
    IntervalLogitAND, RelationalLogitAND, TemplateNet, RuleListNet, FactDiagram,
    NALogicNet,
    CornerNet, SoftMinCornerNet, KofNCornerNet, RingCornerNet, HybridCornerIntervalNet,
    PriorityMixtureNet, EvidenceNet, MarginEvidenceNet, PerFeatureKappaEvidenceNet,
    LadderEvidenceNet, BiEvidenceNet, EvidenceKofNNet,
    GroupEvidenceKofNNet, SoftGroupEvidenceKofNNet, GroupSoftMinNet, GroupContrastNet, GroupRingNet,
    RegimeRulesNet,
    PredicateForest, ObliviousForest, LeafLinearForest, AttentiveForest,
    RuleTree, SparseRuleTree, RuleDiagram,
    ClauseNet, ARLogitAND, MultiResForest, GroupFirstForest,
    Scorecard, ScorecardWithRules, BudgetedForest,
)

# Explainability for model zoo
from .explain import (
    describe_threshold_fact,
    softlogitand_global_rules_df,
    softlogitand_local_contrib_df,
    moe_gate_summary_df,
    segmentmoe_local_explain_df,
    hiermoe_local_explain_df,
    nousfamilies_global_summary_df,
    nousfamilies_local_contrib_df,
)

# NEW: SoftLogitAND post-hoc (cluster themes + selection strategies)
from .explain.softlogitand_posthoc import (
    SoftLogitANDPosthocReport,
    SoftLogitANDPosthocExplainer,
    mmr_select_rules,
    hard_cap_topk_rules,
    mmr_with_hardcap,
)

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
    # Model zoo
    "SoftLogitAND",
    "SoftLogicInteraction",
    "SegmentMoE",
    "HierarchicalMoE",
    "NousFamilies",
    # Model zoo v2
    "MLP",
    "BoxNet",
    "IntervalFactBank",
    "RelationalFactBank",
    "RelationalDiffFactBank",
    "ARFactBank",
    "MultiResAxisFactBank",
    "IntervalLogitAND",
    "RelationalLogitAND",
    "TemplateNet",
    "RuleListNet",
    "FactDiagram",
    "NALogicNet",
    "CornerNet",
    "SoftMinCornerNet",
    "KofNCornerNet",
    "RingCornerNet",
    "HybridCornerIntervalNet",
    "PriorityMixtureNet",
    "EvidenceNet",
    "MarginEvidenceNet",
    "PerFeatureKappaEvidenceNet",
    "LadderEvidenceNet",
    "BiEvidenceNet",
    "EvidenceKofNNet",
    "GroupEvidenceKofNNet",
    "SoftGroupEvidenceKofNNet",
    "GroupSoftMinNet",
    "GroupContrastNet",
    "GroupRingNet",
    "RegimeRulesNet",
    "PredicateForest",
    "ObliviousForest",
    "LeafLinearForest",
    "AttentiveForest",
    "RuleTree",
    "SparseRuleTree",
    "RuleDiagram",
    "ClauseNet",
    "ARLogitAND",
    "MultiResForest",
    "GroupFirstForest",
    "Scorecard",
    "ScorecardWithRules",
    "BudgetedForest",
    # Explainability for model zoo
    "describe_threshold_fact",
    "softlogitand_global_rules_df",
    "softlogitand_local_contrib_df",
    "moe_gate_summary_df",
    "segmentmoe_local_explain_df",
    "hiermoe_local_explain_df",
    "nousfamilies_global_summary_df",
    "nousfamilies_local_contrib_df",
    # SoftLogitAND post-hoc
    "SoftLogitANDPosthocReport",
    "SoftLogitANDPosthocExplainer",
    "mmr_select_rules",
    "hard_cap_topk_rules",
    "mmr_with_hardcap",
]