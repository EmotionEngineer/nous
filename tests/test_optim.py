import torch
import pytest
import numpy as np
from nous import NousNet
from nous.optim import (
    MirrorRAdam,
    BetaNaturalGradient,
    ExtragradRAdamTrust,
    SharpnessAwareGates,
    create_optimizer
)

def _create_simple_model():
    """Create a simple model for testing optimizers."""
    return NousNet(
        input_dim=10,
        num_outputs=3,
        task_type="classification",
        num_facts=16,
        rules_per_layer=(8, 4),
        rule_selection_method="soft_fact",
        use_calibrators=False
    )

def _create_test_data(batch_size=32):
    """Create test data for optimization steps."""
    X = torch.randn(batch_size, 10)
    y = torch.randint(0, 3, (batch_size,))
    return X, y

def test_mirror_radam_optimizer():
    """Test MirrorRAdam optimizer basic functionality."""
    model = _create_simple_model()
    X, y = _create_test_data()
    
    optimizer = MirrorRAdam(
        [p for p in model.parameters() if p.requires_grad],
        model=model,
        lr=1e-3,
        lr_gate=1e-2
    )
    
    # Forward pass
    logits = model(X)
    loss = torch.nn.CrossEntropyLoss()(logits, y)
    
    # Backward pass and optimization step
    loss.backward()
    optimizer.step()
    
    # Check that parameters were updated
    param_sum = sum(p.data.sum().item() for p in model.parameters())
    assert not np.isnan(param_sum) and not np.isinf(param_sum)

def test_beta_natural_gradient():
    """Test BetaNaturalGradient optimizer functionality."""
    model = _create_simple_model()
    X, y = _create_test_data()
    
    optimizer = BetaNaturalGradient(
        [p for p in model.parameters() if p.requires_grad],
        model=model,
        lr=1e-3
    )
    
    # Forward pass
    logits = model(X)
    loss = torch.nn.CrossEntropyLoss()(logits, y)
    
    # Backward pass
    loss.backward()
    
    # Optimization step with required model and X_batch arguments
    optimizer.step(model=model, X_batch=X)
    
    # Check that parameters were updated
    param_sum = sum(p.data.sum().item() for p in model.parameters())
    assert not np.isnan(param_sum) and not np.isinf(param_sum)

def test_extragrad_radam_trust():
    """Test ExtragradRAdamTrust optimizer functionality."""
    model = _create_simple_model()
    X, y = _create_test_data()
    
    optimizer = ExtragradRAdamTrust(
        [p for p in model.parameters() if p.requires_grad],
        model=model,
        lr=1e-3
    )
    
    # Define closure for SAM-style optimization
    def closure():
        optimizer.zero_grad()
        logits = model(X)
        loss = torch.nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        return loss
    
    # Optimization step
    optimizer.step(closure=closure, model=model)
    
    # Check that parameters were updated
    param_sum = sum(p.data.sum().item() for p in model.parameters())
    assert not np.isnan(param_sum) and not np.isinf(param_sum)

def test_sharpness_aware_gates():
    """Test SharpnessAwareGates optimizer functionality."""
    model = _create_simple_model()
    X, y = _create_test_data()
    
    optimizer = SharpnessAwareGates(
        [p for p in model.parameters() if p.requires_grad],
        model=model,
        lr=1e-3
    )
    
    # Define closure for SAM-style optimization
    def closure():
        optimizer.zero_grad()
        logits = model(X)
        loss = torch.nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        return loss
    
    # Optimization step
    optimizer.step(closure=closure, model=model)
    
    # Check that parameters were updated
    param_sum = sum(p.data.sum().item() for p in model.parameters())
    assert not np.isnan(param_sum) and not np.isinf(param_sum)

@pytest.mark.parametrize("optimizer_name", [
    "mirror_radam",
    "beta_ng",
    "extragrad_trust",
    "sharpness_aware_gates"
])
def test_optimizer_factory(optimizer_name):
    """Test optimizer factory function."""
    model = _create_simple_model()
    
    optimizer = create_optimizer(
        model,
        optimizer_name,
        lr=1e-3
    )
    
    assert optimizer is not None
    assert hasattr(optimizer, 'step')
    assert hasattr(optimizer, 'zero_grad')

def test_optimizer_parameter_grouping():
    """Test that optimizers correctly group gate and non-gate parameters."""
    model = _create_simple_model()
    
    optimizer = MirrorRAdam(
        [p for p in model.parameters() if p.requires_grad],
        model=model,
        lr=1e-3
    )
    
    # Check that we have both gate and non-gate parameters
    assert len(optimizer.gate_params) > 0
    assert len(optimizer.nongate_params) > 0
    
    # Check parameter categorization
    gate_param_names = []
    nongate_param_names = []
    
    for name, param in model.named_parameters():
        if any(p is param for p in optimizer.gate_params):
            gate_param_names.append(name)
        if any(p is param for p in optimizer.nongate_params):
            nongate_param_names.append(name)
    
    for name in gate_param_names:
        assert ("fact_logits" in name) or ("aggregator_logits" in name) or ("rule_strength_raw" in name)
    
    for name in nongate_param_names:
        assert not (("fact_logits" in name) or ("aggregator_logits" in name) or ("rule_strength_raw" in name))