# tests/test_layers.py

import torch
import pytest
from nous.layers import (
    ExhaustiveAtomicFactLayer,
    SigmoidFactLayer,
    BetaFactLayer,
    LogicalRuleLayer,
)

# --- Test Constants ---
BATCH_SIZE = 4
INPUT_DIM = 5
NUM_FACTS_LEARNED = 10
NUM_RULES = 8
FEATURE_NAMES = [f'feat_{i}' for i in range(INPUT_DIM)]

# --- Fixtures ---
@pytest.fixture
def sample_data():
    return torch.randn(BATCH_SIZE, INPUT_DIM)

# --- Layer Tests ---

def test_exhaustive_atomic_fact_layer(sample_data):
    layer = ExhaustiveAtomicFactLayer(input_dim=INPUT_DIM, feature_names=FEATURE_NAMES)
    
    # Check output dimension: C(5, 2) = 10
    expected_dim = INPUT_DIM * (INPUT_DIM - 1) // 2
    assert layer.output_dim == expected_dim
    assert len(layer.fact_names) == expected_dim

    # Check forward pass
    output = layer(sample_data)
    assert output.shape == (BATCH_SIZE, expected_dim)
    assert not torch.isnan(output).any()
    assert (output >= 0).all() and (output <= 1).all()

    # Check gradient flow
    output.sum().backward()
    assert layer.thresholds.grad is not None
    assert layer.steepness.grad is not None

@pytest.mark.parametrize("layer_class", [SigmoidFactLayer, BetaFactLayer])
def test_learned_atomic_fact_layers(sample_data, layer_class):
    layer = layer_class(input_dim=INPUT_DIM, num_facts=NUM_FACTS_LEARNED, feature_names=FEATURE_NAMES)
    
    # Check output dimension
    assert layer.output_dim == NUM_FACTS_LEARNED
    assert len(layer.fact_names) == NUM_FACTS_LEARNED

    # Check forward pass
    output = layer(sample_data)
    assert output.shape == (BATCH_SIZE, NUM_FACTS_LEARNED)
    assert not torch.isnan(output).any()
    assert (output >= 0).all() and (output <= 1).all()

    # Check gradient flow
    output.sum().backward()
    assert layer.projection_left.weight.grad is not None
    assert layer.projection_right.weight.grad is not None
    assert layer.thresholds.grad is not None
    # Check specific params
    if isinstance(layer, SigmoidFactLayer):
        assert layer.steepness.grad is not None
    if isinstance(layer, BetaFactLayer):
        assert layer.k_raw.grad is not None
        assert layer.nu_raw.grad is not None

def test_logical_rule_layer():
    input_facts = torch.rand(BATCH_SIZE, NUM_FACTS_LEARNED)
    fact_names = [f'fact_{i}' for i in range(NUM_FACTS_LEARNED)]
    
    layer = LogicalRuleLayer(input_dim=NUM_FACTS_LEARNED, num_rules=NUM_RULES, input_fact_names=fact_names)

    # Check output dimension
    assert layer.output_dim == NUM_RULES
    
    # Check forward pass
    concepts, rule_activations = layer(input_facts)
    assert concepts.shape == (BATCH_SIZE, NUM_RULES)
    assert rule_activations.shape == (BATCH_SIZE, NUM_RULES)
    assert not torch.isnan(concepts).any()
    
    # Check gradient flow
    concepts.sum().backward()
    assert layer.concept_generator.weight.grad is not None
    
def test_logical_rule_layer_empty_input():
    """Test that the layer handles zero input facts gracefully."""
    input_facts = torch.rand(BATCH_SIZE, 0)
    layer = LogicalRuleLayer(input_dim=0, num_rules=0, input_fact_names=[])
    
    concepts, rule_activations = layer(input_facts)
    assert concepts.shape == (BATCH_SIZE, 0)
    assert rule_activations.shape == (BATCH_SIZE, 0)
