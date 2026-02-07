# nous/zoo_v2/__init__.py
"""
zoo_v2: Production-ready interpretable models with clean naming.
All models follow the pattern: init_from_data(X) for proper initialization.
"""

# Simple baselines
from .models import MLP, BoxNet

# Fact banks (can be used standalone or composed)
from .facts import (
    IntervalFactBank,
    RelationalFactBank,
    RelationalDiffFactBank,
    ARFactBank,
    MultiResAxisFactBank,
)

# Rule-based models
from .models import (
    IntervalLogitAND,
    RelationalLogitAND,
    TemplateNet,
    RuleListNet,
    FactDiagram,
    NALogicNet,
    CornerNet,
    SoftMinCornerNet,
    KofNCornerNet,
    RingCornerNet,
    HybridCornerIntervalNet,
    PriorityMixtureNet,
    EvidenceNet,
    MarginEvidenceNet,
    PerFeatureKappaEvidenceNet,
    LadderEvidenceNet,
    BiEvidenceNet,
    EvidenceKofNNet,
    GroupEvidenceKofNNet,
    SoftGroupEvidenceKofNNet,
    GroupSoftMinNet,
    GroupContrastNet,
    GroupRingNet,
    RegimeRulesNet,
)

# Forest/router family
from .models import (
    PredicateForest,
    ObliviousForest,
    LeafLinearForest,
    AttentiveForest,
    RuleTree,
    SparseRuleTree,
    RuleDiagram,
    ClauseNet,
    ARLogitAND,
    MultiResForest,
    GroupFirstForest,
    Scorecard,
    ScorecardWithRules,
    BudgetedForest,
)

__all__ = [
    # Simple baselines
    "MLP",
    "BoxNet",

    # Fact banks
    "IntervalFactBank",
    "RelationalFactBank",
    "RelationalDiffFactBank",
    "ARFactBank",
    "MultiResAxisFactBank",

    # Rule-based models
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

    # Forest/router family
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
]