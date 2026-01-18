from .aggregator import AGG_NAMES, aggregator_mixture_report, format_agg_mixture
from .facts_desc import render_fact_descriptions
from .loo import rule_impact_df
from .mse import minimal_sufficient_explanation
from .pruning import select_pruning_threshold_global, select_pruning_threshold_global_bs
from .global_book import global_rulebook
from .generate import generate_enhanced_explanation
from .fidelity import explanation_fidelity_metrics
from .stability import explanation_stability
from .cf import suggest_rule_counterfactuals

# Explainability for model zoo
from .zoo import (
    describe_threshold_fact,
    softlogitand_global_rules_df,
    softlogitand_local_contrib_df,
    moe_gate_summary_df,
    segmentmoe_local_explain_df,
    hiermoe_local_explain_df,
    nousfamilies_global_summary_df,
    nousfamilies_local_contrib_df,
)

__all__ = [
    "AGG_NAMES",
    "aggregator_mixture_report",
    "format_agg_mixture",
    "render_fact_descriptions",
    "rule_impact_df",
    "minimal_sufficient_explanation",
    "select_pruning_threshold_global",
    "select_pruning_threshold_global_bs",
    "global_rulebook",
    "generate_enhanced_explanation",
    "explanation_fidelity_metrics",
    "explanation_stability",
    "suggest_rule_counterfactuals",
    # Zoo explainability
    "describe_threshold_fact",
    "softlogitand_global_rules_df",
    "softlogitand_local_contrib_df",
    "moe_gate_summary_df",
    "segmentmoe_local_explain_df",
    "hiermoe_local_explain_df",
    "nousfamilies_global_summary_df",
    "nousfamilies_local_contrib_df",
]