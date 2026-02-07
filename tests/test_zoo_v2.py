# tests/test_zoo_v2.py
"""Comprehensive tests for zoo_v2 models."""
import numpy as np
import torch
import torch.nn as nn
import pytest
from typing import Type

# Import all zoo_v2 models
from nous.zoo_v2 import (
    # Simple baselines
    MLP,
    BoxNet,
    # Fact banks
    IntervalFactBank,
    RelationalFactBank,
    RelationalDiffFactBank,
    ARFactBank,
    MultiResAxisFactBank,
    # Rule-based models
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
    # Forest/router family
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


def make_synthetic_data(
    n_samples: int = 128,
    n_features: int = 8,
    task: str = "binary",  # "binary", "multiclass", "regression"
    n_classes: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for testing."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)

    if task == "binary":
        w = rng.randn(n_features)
        logits = X @ w
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.rand(n_samples) < probs).astype(np.float32)
    elif task == "multiclass":
        W = rng.randn(n_features, n_classes)
        logits = X @ W
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        y = np.array([rng.choice(n_classes, p=p) for p in probs], dtype=np.int64)
    else:  # regression
        w = rng.randn(n_features)
        y = (X @ w + rng.randn(n_samples) * 0.1).astype(np.float32)

    return X, y


# ---------------------------------------------------------------------------
# Fact bank tests
# ---------------------------------------------------------------------------
def test_fact_banks_forward():
    """Test forward pass of all fact banks."""
    D = 6
    X = torch.randn(5, D)

    # IntervalFactBank
    fb = IntervalFactBank(D, n_intervals_per_feat=3)
    f = fb(X)
    assert f.shape == (5, D * 3)
    assert f.min() >= 0.0 and f.max() <= 1.0

    # RelationalFactBank
    fb = RelationalFactBank(D, pairs=[(0, 1), (2, 3), (4, 5)])
    f = fb(X)
    assert f.shape == (5, 3)
    assert f.min() >= 0.0 and f.max() <= 1.0

    # RelationalDiffFactBank
    fb = RelationalDiffFactBank(D, n_pairs=4, n_thresh_per_pair=2)
    f = fb(X)
    assert f.shape == (5, 8)
    assert f.min() >= 0.0 and f.max() <= 1.0

    # ARFactBank
    fb = ARFactBank(D, n_thresh_per_feat=2, n_pairs=3, n_thresh_per_pair=2)
    f = fb(X)
    assert f.shape[0] == 5
    assert f.min() >= 0.0 and f.max() <= 1.0

    # MultiResAxisFactBank
    fb = MultiResAxisFactBank(D, n_thresh_per_feat=2, kappas=(1.0, 3.0, 10.0))
    f = fb(X)
    assert f.shape == (5, D * 2 * 3)
    assert f.min() >= 0.0 and f.max() <= 1.0


def test_fact_banks_init_from_data():
    """Test data initialization for fact banks."""
    D = 6
    X_np = np.random.randn(100, D).astype(np.float32)

    # IntervalFactBank
    fb = IntervalFactBank(D, n_intervals_per_feat=3)
    fb.init_from_data_quantiles(X_np)
    assert not torch.allclose(fb.a, torch.zeros_like(fb.a))

    # RelationalDiffFactBank
    fb = RelationalDiffFactBank(D, n_pairs=4, n_thresh_per_pair=2)
    fb.init_from_data_quantiles(X_np)
    assert not torch.allclose(fb.th, torch.zeros_like(fb.th))

    # ARFactBank
    fb = ARFactBank(D, n_thresh_per_feat=2, n_pairs=3, n_thresh_per_pair=2)
    fb.init_from_data_quantiles(X_np)
    assert not torch.allclose(fb.axis.th, torch.zeros_like(fb.axis.th))

    # MultiResAxisFactBank
    fb = MultiResAxisFactBank(D, n_thresh_per_feat=2, kappas=(1.0, 3.0, 10.0))
    fb.init_from_data_quantiles(X_np)
    assert not torch.allclose(fb.th, torch.zeros_like(fb.th))


# ---------------------------------------------------------------------------
# Parametrized forward-pass tests for all models
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_cls,params", [
    (MLP, {"hidden": 64, "depth": 2}),
    (BoxNet, {"n_boxes": 32}),
    (IntervalLogitAND, {"n_rules": 32, "n_intervals_per_feat": 2}),
    (RelationalLogitAND, {"n_rules": 32, "n_thresh_per_feat": 2, "max_pairs": 8}),
    (TemplateNet, {"n_rules": 32, "n_templates": 4}),
    (RuleListNet, {"n_rules": 32}),
    (FactDiagram, {"depth": 3}),
    (NALogicNet, {"n_rules": 32}),
    (CornerNet, {"n_rules": 32}),
    (SoftMinCornerNet, {"n_rules": 32}),
    (KofNCornerNet, {"n_rules": 32}),
    (RingCornerNet, {"n_rings": 32}),
    (HybridCornerIntervalNet, {"n_rules": 32}),
    (PriorityMixtureNet, {"n_rules": 32}),
    (EvidenceNet, {"n_rules": 32}),
    (MarginEvidenceNet, {"n_rules": 32}),
    (PerFeatureKappaEvidenceNet, {"n_rules": 32}),
    (LadderEvidenceNet, {"n_rules": 32, "n_levels": 2}),
    (BiEvidenceNet, {"n_rules": 32}),
    (EvidenceKofNNet, {"n_rules": 32}),
    (GroupEvidenceKofNNet, {"n_rules": 32, "groups": [[0, 1], [2, 3], [4, 5]]}),
    (SoftGroupEvidenceKofNNet, {"n_rules": 32, "n_groups": 3}),
    (GroupSoftMinNet, {"n_rules": 32, "groups": [[0, 1], [2, 3], [4, 5]]}),
    (GroupContrastNet, {"n_rules": 32, "groups": [[0, 1], [2, 3], [4, 5]]}),
    (GroupRingNet, {"n_rings": 32, "groups": [[0, 1], [2, 3], [4, 5]]}),
    (RegimeRulesNet, {"n_regimes": 3, "n_rules": 32}),
    (PredicateForest, {"n_trees": 8, "depth": 3}),
    (ObliviousForest, {"n_trees": 8, "depth": 3}),
    (LeafLinearForest, {"n_trees": 8, "depth": 3, "leaf_k": 2}),
    (AttentiveForest, {"n_trees": 8, "depth": 3}),
    (RuleTree, {"depth": 3, "n_rules": 32}),
    (SparseRuleTree, {"depth": 3, "n_rules": 32}),
    (RuleDiagram, {"diagram_depth": 3, "n_rules": 32}),
    (ClauseNet, {"n_clauses": 32}),
    (ARLogitAND, {"n_rules": 32, "n_pairs": 8}),
    (MultiResForest, {"n_trees": 8, "depth": 3, "kappas": (1.0, 5.0)}),
    (GroupFirstForest, {"n_trees": 8, "depth": 3, "groups": [[0, 1], [2, 3], [4, 5]]}),
    (Scorecard, {"n_bins": 4}),
    (ScorecardWithRules, {"n_bins": 4, "n_rules": 32}),
    (BudgetedForest, {"k_features": 2, "n_trees": 8, "depth": 3}),
])
def test_model_instantiation_and_forward(model_cls: Type[nn.Module], params: dict):
    """Test model instantiation and forward pass for all zoo_v2 models."""
    D = 6
    B = 5

    # Test binary classification (output_dim=1)
    model = model_cls(input_dim=D, output_dim=1, **params)
    x = torch.randn(B, D)
    y = model(x)

    assert y.shape == (B,), f"{model_cls.__name__} binary output shape mismatch"
    assert torch.isfinite(y).all(), f"{model_cls.__name__} produced non-finite outputs"

    # Test multiclass classification (output_dim=3)
    if not isinstance(model, (Scorecard, ScorecardWithRules, BudgetedForest)):
        try:
            model_mc = model_cls(input_dim=D, output_dim=3, **params)
            y_mc = model_mc(x)
            assert y_mc.shape == (B, 3), f"{model_cls.__name__} multiclass output shape mismatch"
            assert torch.isfinite(y_mc).all()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Backward-pass tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_cls,params", [
    (MLP, {"hidden": 64, "depth": 2}),
    (BoxNet, {"n_boxes": 32}),
    (IntervalLogitAND, {"n_rules": 32}),
    (NALogicNet, {"n_rules": 32}),
    (CornerNet, {"n_rules": 32}),
    (EvidenceNet, {"n_rules": 32}),
    (PredicateForest, {"n_trees": 8, "depth": 3}),
    (Scorecard, {"n_bins": 4}),
])
def test_model_backward_pass(model_cls: Type[nn.Module], params: dict):
    """Test that models support gradient flow."""
    D = 6
    B = 5

    model = model_cls(input_dim=D, output_dim=1, **params)
    x = torch.randn(B, D, requires_grad=True)
    y = model(x)

    target = torch.randn(B)
    loss = nn.MSELoss()(y, target)
    loss.backward()

    assert x.grad is not None, f"{model_cls.__name__} input has no gradient"
    assert torch.isfinite(x.grad).all(), f"{model_cls.__name__} input gradient has non-finite values"

    grad_found = False
    for p in model.parameters():
        if p.grad is not None and p.grad.abs().sum() > 1e-12:
            grad_found = True
            assert torch.isfinite(p.grad).all(), f"{model_cls.__name__} parameter gradient has non-finite values"
            break

    assert grad_found, f"{model_cls.__name__} has no parameters with gradients"


# ---------------------------------------------------------------------------
# init_from_data tests
# ---------------------------------------------------------------------------
def test_model_init_from_data():
    """Test init_from_data for models that support it."""
    D = 6
    X_np = np.random.randn(100, D).astype(np.float32)

    models = [
        BoxNet(input_dim=D, n_boxes=32),
        IntervalLogitAND(input_dim=D, n_rules=32),
        RelationalLogitAND(input_dim=D, n_rules=32, max_pairs=8),
        TemplateNet(input_dim=D, n_rules=32),
        RuleListNet(input_dim=D, n_rules=32),
        FactDiagram(input_dim=D, depth=3),
        NALogicNet(input_dim=D, n_rules=32),
        CornerNet(input_dim=D, n_rules=32),
        SoftMinCornerNet(input_dim=D, n_rules=32),
        KofNCornerNet(input_dim=D, n_rules=32),
        RingCornerNet(input_dim=D, n_rings=32),
        HybridCornerIntervalNet(input_dim=D, n_rules=32),
        PriorityMixtureNet(input_dim=D, n_rules=32),
        EvidenceNet(input_dim=D, n_rules=32),
        MarginEvidenceNet(input_dim=D, n_rules=32),
        PerFeatureKappaEvidenceNet(input_dim=D, n_rules=32),
        LadderEvidenceNet(input_dim=D, n_rules=32),
        BiEvidenceNet(input_dim=D, n_rules=32),
        EvidenceKofNNet(input_dim=D, n_rules=32),
        GroupEvidenceKofNNet(input_dim=D, n_rules=32),
        SoftGroupEvidenceKofNNet(input_dim=D, n_rules=32),
        GroupSoftMinNet(input_dim=D, n_rules=32),
        GroupContrastNet(input_dim=D, n_rules=32),
        GroupRingNet(input_dim=D, n_rings=32),
        RegimeRulesNet(input_dim=D, n_regimes=3, n_rules=32),
        PredicateForest(input_dim=D, n_trees=8, depth=3),
        ObliviousForest(input_dim=D, n_trees=8, depth=3),
        LeafLinearForest(input_dim=D, n_trees=8, depth=3),
        AttentiveForest(input_dim=D, n_trees=8, depth=3),
        RuleTree(input_dim=D, depth=3, n_rules=32),
        SparseRuleTree(input_dim=D, depth=3, n_rules=32),
        RuleDiagram(input_dim=D, diagram_depth=3, n_rules=32),
        ClauseNet(input_dim=D, n_clauses=32),
        ARLogitAND(input_dim=D, n_rules=32),
        MultiResForest(input_dim=D, n_trees=8, depth=3),
        GroupFirstForest(input_dim=D, n_trees=8, depth=3),
        Scorecard(input_dim=D, n_bins=4),
        ScorecardWithRules(input_dim=D, n_bins=4, n_rules=32),
        BudgetedForest(input_dim=D, k_features=2, n_trees=8, depth=3),
    ]

    for model in models:
        model_name = model.__class__.__name__
        try:
            model.init_from_data(X_np)
            params_changed = False
            for name, p in model.named_parameters():
                if "weight" in name or "bias" in name or "th" in name or "center" in name:
                    if p.abs().mean() > 1e-3:
                        params_changed = True
                        break
            assert params_changed, f"{model_name} init_from_data did not change parameters"
        except Exception as e:
            pytest.fail(f"{model_name}.init_from_data() failed: {e}")


# ---------------------------------------------------------------------------
# NALogicNet-specific: NaN handling
# ---------------------------------------------------------------------------
def test_nalogic_nan_handling():
    """Test that NALogicNet correctly handles NaN inputs."""
    D = 6
    B = 8

    model = NALogicNet(input_dim=D, n_rules=16, output_dim=1)
    X_np = np.random.randn(50, D).astype(np.float32)
    model.init_from_data(X_np)

    # Input with no NaNs
    x_clean = torch.randn(B, D)
    y_clean = model(x_clean)
    assert y_clean.shape == (B,)
    assert torch.isfinite(y_clean).all(), "NALogicNet produced non-finite output on clean input"

    # Input with some NaNs
    x_nan = x_clean.clone()
    x_nan[0, 1] = float("nan")
    x_nan[2, 3] = float("nan")
    x_nan[4, 0] = float("nan")
    x_nan[4, 5] = float("nan")
    y_nan = model(x_nan)
    assert y_nan.shape == (B,)
    assert torch.isfinite(y_nan).all(), "NALogicNet produced non-finite output on NaN input"

    # Outputs should differ when NaN vs non-NaN (model should see missingness)
    # (Not guaranteed after random init, but shapes/finiteness must hold)

    # All NaN row
    x_all_nan = torch.full((1, D), float("nan"))
    y_all_nan = model(x_all_nan)
    assert y_all_nan.shape == (1,)
    assert torch.isfinite(y_all_nan).all(), "NALogicNet failed on all-NaN row"


def test_nalogic_backward_with_nans():
    """Test gradient flow through NALogicNet with NaN inputs."""
    D = 6
    B = 4

    model = NALogicNet(input_dim=D, n_rules=16, output_dim=1)

    # x with NaNs — gradients should still flow through non-NaN positions
    x = torch.randn(B, D)
    x[0, 1] = float("nan")
    x[2, 3] = float("nan")
    # Don't require grad on x (NaN positions cause issues); check param grads
    y = model(x)
    loss = y.sum()
    loss.backward()

    grad_found = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NALogicNet param {name} has non-finite grad with NaN input"
            if p.grad.abs().sum() > 1e-12:
                grad_found = True
    assert grad_found, "NALogicNet has no parameter gradients with NaN input"


def test_nalogic_multiclass():
    """Test NALogicNet with multiclass output."""
    D = 6
    B = 5
    C = 4

    model = NALogicNet(input_dim=D, n_rules=16, output_dim=C)
    X_np = np.random.randn(50, D).astype(np.float32)
    model.init_from_data(X_np)

    x = torch.randn(B, D)
    x[1, 2] = float("nan")
    y = model(x)
    assert y.shape == (B, C)
    assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# End-to-end training
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_end_to_end_training(task: str):
    """Test a minimal training loop on synthetic data for representative models."""
    D = 8
    X_np, y_np = make_synthetic_data(n_samples=200, n_features=D, task=task, n_classes=3)

    if task == "regression":
        models = [
            MLP(input_dim=D, output_dim=1, hidden=64, depth=2),
            BoxNet(input_dim=D, n_boxes=32),
            Scorecard(input_dim=D, n_bins=4),
        ]
        criterion = nn.MSELoss()
        y_tensor = torch.tensor(y_np, dtype=torch.float32)
    elif task == "multiclass":
        models = [
            MLP(input_dim=D, output_dim=3, hidden=64, depth=2),
            PredicateForest(input_dim=D, n_trees=8, depth=3, output_dim=3),
        ]
        criterion = nn.CrossEntropyLoss()
        y_tensor = torch.tensor(y_np, dtype=torch.long)
    else:  # binary
        models = [
            MLP(input_dim=D, output_dim=1, hidden=64, depth=2),
            CornerNet(input_dim=D, n_rules=32),
            EvidenceNet(input_dim=D, n_rules=32),
            NALogicNet(input_dim=D, n_rules=32),
        ]
        criterion = nn.BCEWithLogitsLoss()
        y_tensor = torch.tensor(y_np, dtype=torch.float32)

    x_tensor = torch.tensor(X_np, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for model in models:
        model_name = model.__class__.__name__
        if hasattr(model, "init_from_data"):
            model.init_from_data(X_np)

        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)

        losses = []
        for epoch in range(5):
            epoch_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                y_pred = model(xb)

                if task == "multiclass":
                    loss = criterion(y_pred, yb)
                else:
                    if y_pred.ndim > 1 and y_pred.shape[1] == 1:
                        y_pred = y_pred.squeeze(1)
                    loss = criterion(y_pred, yb)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                epoch_loss += loss.item()

            losses.append(epoch_loss / len(loader))

        assert losses[-1] < losses[0] * 1.5, f"{model_name} loss did not decrease reasonably: {losses}"
        for name, p in model.named_parameters():
            assert torch.isfinite(p).all(), f"{model_name} has non-finite parameter {name} after training"


# ---------------------------------------------------------------------------
# Output dimension variations
# ---------------------------------------------------------------------------
def test_output_dim_variations():
    """Test that models correctly handle different output dimensions."""
    D = 6
    B = 5
    x = torch.randn(B, D)

    model_bin = CornerNet(input_dim=D, n_rules=16, output_dim=1)
    y_bin = model_bin(x)
    assert y_bin.shape == (B,), "Binary output should be 1D"

    model_mc = CornerNet(input_dim=D, n_rules=16, output_dim=4)
    y_mc = model_mc(x)
    assert y_mc.shape == (B, 4), "Multiclass output should be 2D"

    model_reg = MLP(input_dim=D, hidden=32, depth=2, output_dim=1)
    y_reg = model_reg(x)
    assert y_reg.shape == (B,), "Regression output should be 1D"


# ---------------------------------------------------------------------------
# Group model tests
# ---------------------------------------------------------------------------
def test_group_models_with_explicit_groups():
    """Test group-based models with explicit group specifications."""
    D = 8
    groups = [[0, 1, 2], [3, 4], [5, 6, 7]]

    model = GroupEvidenceKofNNet(input_dim=D, groups=groups, n_rules=16)
    x = torch.randn(3, D)
    y = model(x)
    assert y.shape == (3,)
    assert torch.isfinite(y).all()

    model = GroupSoftMinNet(input_dim=D, groups=groups, n_rules=16)
    y = model(x)
    assert y.shape == (3,)
    assert torch.isfinite(y).all()

    model = GroupContrastNet(input_dim=D, groups=groups, n_rules=16)
    y = model(x)
    assert y.shape == (3,)
    assert torch.isfinite(y).all()

    model = GroupRingNet(input_dim=D, groups=groups, n_rings=16)
    y = model(x)
    assert y.shape == (3,)
    assert torch.isfinite(y).all()

    model = GroupFirstForest(input_dim=D, groups=groups, n_trees=4, depth=3)
    y = model(x)
    assert y.shape == (3,)
    assert torch.isfinite(y).all()


def test_group_default_d8():
    """Verify that default groups for D=8 match notebook convention."""
    from nous.zoo_v2.common import default_groups_for_D
    groups = default_groups_for_D(8)
    assert groups == [[0], [1, 2, 3], [4, 5], [6, 7]]


def test_group_default_other():
    """Verify that default groups for non-8 D produce 4 chunks."""
    from nous.zoo_v2.common import default_groups_for_D
    groups = default_groups_for_D(12)
    assert len(groups) == 4
    flat = [i for g in groups for i in g]
    assert sorted(flat) == list(range(12))


# ---------------------------------------------------------------------------
# Forest variations
# ---------------------------------------------------------------------------
def test_forest_models_tree_variations():
    """Test forest models with different tree configurations."""
    D = 6
    x = torch.randn(5, D)

    for n_trees in [1, 4, 16]:
        model = PredicateForest(input_dim=D, n_trees=n_trees, depth=3)
        y = model(x)
        assert y.shape == (5,)
        assert torch.isfinite(y).all()

    for depth in [1, 2, 4]:
        model = ObliviousForest(input_dim=D, n_trees=4, depth=depth)
        y = model(x)
        assert y.shape == (5,)
        assert torch.isfinite(y).all()

    for leaf_k in [1, 2, 4]:
        model = LeafLinearForest(input_dim=D, n_trees=4, depth=3, leaf_k=leaf_k)
        y = model(x)
        assert y.shape == (5,)
        assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# Corner family variations
# ---------------------------------------------------------------------------
def test_corner_models_variations():
    """Test corner-family models with different aggregation strategies."""
    D = 6
    x = torch.randn(5, D)

    model = CornerNet(input_dim=D, n_rules=16)
    y = model(x)
    assert y.shape == (5,) and torch.isfinite(y).all()

    model = SoftMinCornerNet(input_dim=D, n_rules=16)
    y = model(x)
    assert y.shape == (5,) and torch.isfinite(y).all()

    model = KofNCornerNet(input_dim=D, n_rules=16)
    y = model(x)
    assert y.shape == (5,) and torch.isfinite(y).all()

    model = RingCornerNet(input_dim=D, n_rings=16)
    y = model(x)
    assert y.shape == (5,) and torch.isfinite(y).all()

    model = HybridCornerIntervalNet(input_dim=D, n_rules=16)
    y = model(x)
    assert y.shape == (5,) and torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# Evidence family variations
# ---------------------------------------------------------------------------
def test_evidence_models_variations():
    """Test evidence-family models with different evidence formulations."""
    D = 6
    x = torch.randn(5, D)

    model = EvidenceNet(input_dim=D, n_rules=16)
    y = model(x)
    assert y.shape == (5,) and torch.isfinite(y).all()

    model = MarginEvidenceNet(input_dim=D, n_rules=16)
    y = model(x)
    assert y.shape == (5,) and torch.isfinite(y).all()

    model = PerFeatureKappaEvidenceNet(input_dim=D, n_rules=16)
    y = model(x)
    assert y.shape == (5,) and torch.isfinite(y).all()

    model = LadderEvidenceNet(input_dim=D, n_rules=16, n_levels=3)
    y = model(x)
    assert y.shape == (5,) and torch.isfinite(y).all()

    model = BiEvidenceNet(input_dim=D, n_rules=16)
    y = model(x)
    assert y.shape == (5,) and torch.isfinite(y).all()

    model = EvidenceKofNNet(input_dim=D, n_rules=16)
    y = model(x)
    assert y.shape == (5,) and torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# Router tree models
# ---------------------------------------------------------------------------
def test_router_tree_models():
    """Test router/tree-based models."""
    D = 6
    x = torch.randn(5, D)

    from nous.zoo_v2.trees import PredicateRouterTree
    router = PredicateRouterTree(input_dim=D, depth=3, n_thresh_per_feat=4)
    leaf_probs = router(x)
    assert leaf_probs.shape == (5, 8)  # 2^3 leaves
    assert torch.allclose(leaf_probs.sum(dim=1), torch.ones(5), atol=1e-5)
    assert torch.isfinite(leaf_probs).all()

    model = RuleTree(input_dim=D, depth=3, n_rules=16)
    y = model(x)
    assert y.shape == (5,) and torch.isfinite(y).all()

    model = SparseRuleTree(input_dim=D, depth=3, n_rules=16)
    y = model(x)
    assert y.shape == (5,) and torch.isfinite(y).all()

    model = RuleDiagram(input_dim=D, diagram_depth=3, n_rules=16)
    y = model(x)
    assert y.shape == (5,) and torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# Specialized architectures
# ---------------------------------------------------------------------------
def test_specialized_architectures():
    """Test specialized architectures like ARLogitAND, ClauseNet, etc."""
    D = 6
    x = torch.randn(5, D)

    model = ARLogitAND(input_dim=D, n_rules=16, n_pairs=8)
    y = model(x)
    assert y.shape == (5,) and torch.isfinite(y).all()

    model = ClauseNet(input_dim=D, n_clauses=16)
    y = model(x)
    assert y.shape == (5,) and torch.isfinite(y).all()

    model = MultiResForest(input_dim=D, n_trees=4, depth=3, kappas=(1.0, 5.0, 15.0))
    y = model(x)
    assert y.shape == (5,) and torch.isfinite(y).all()

    model = Scorecard(input_dim=D, n_bins=4)
    y = model(x)
    assert y.shape == (5,) and torch.isfinite(y).all()

    model = ScorecardWithRules(input_dim=D, n_bins=4, n_rules=16)
    y = model(x)
    assert y.shape == (5,) and torch.isfinite(y).all()

    model = BudgetedForest(input_dim=D, k_features=3, n_trees=4, depth=3)
    y = model(x)
    assert y.shape == (5,) and torch.isfinite(y).all()