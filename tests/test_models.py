# tests/test_models.py

import torch
import pytest
from nous.models import NousNet, NousBlock

# --- Test Constants ---
BATCH_SIZE = 4
INPUT_DIM = 8
OUTPUT_DIM_BINARY = 1
OUTPUT_DIM_MULTI = 3
FEATURE_NAMES = [f'feat_{i}' for i in range(INPUT_DIM)]

# --- Fixtures ---
@pytest.fixture
def sample_data():
    return torch.randn(BATCH_SIZE, INPUT_DIM)

@pytest.fixture
def nous_block_input():
    return torch.rand(BATCH_SIZE, 10) # 10 input facts/concepts

# --- Model Tests ---

def test_nous_block(nous_block_input):
    block = NousBlock(input_dim=10, num_rules=5, input_fact_names=[f'f_{i}' for i in range(10)])
    
    # Check forward pass
    output, concepts, rules = block(nous_block_input)
    assert output.shape == (BATCH_SIZE, 5)
    assert concepts.shape == (BATCH_SIZE, 5)
    assert rules.shape == (BATCH_SIZE, 5)
    
    # Check gradient flow
    output.sum().backward()
    assert all(p.grad is not None for p in block.parameters())

@pytest.mark.parametrize("fact_type", ['beta', 'sigmoid', 'exhaustive'])
def test_nous_net_binary_classification(sample_data, fact_type):
    model = NousNet(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM_BINARY,
        feature_names=FEATURE_NAMES,
        fact_layer_type=fact_type,
        num_facts=15,
        num_rules_per_layer=[10, 5]
    )
    
    # Check forward pass
    output = model(sample_data)
    assert output.shape == (BATCH_SIZE, OUTPUT_DIM_BINARY)
    assert not torch.isnan(output).any()

    # Check gradient flow
    output.sum().backward()
    assert all(p.grad is not None for p in model.parameters())

@pytest.mark.parametrize("fact_type", ['beta', 'sigmoid', 'exhaustive'])
def test_nous_net_multiclass_classification(sample_data, fact_type):
    model = NousNet(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM_MULTI,
        feature_names=FEATURE_NAMES,
        fact_layer_type=fact_type,
        num_facts=15,
        num_rules_per_layer=[10, 5]
    )
    
    # Check forward pass
    output = model(sample_data)
    assert output.shape == (BATCH_SIZE, OUTPUT_DIM_MULTI)

@pytest.mark.parametrize("fact_type", ['beta', 'sigmoid', 'exhaustive'])
def test_nous_net_regression(sample_data, fact_type):
    model = NousNet(
        input_dim=INPUT_DIM,
        output_dim=1, # Regression output
        feature_names=FEATURE_NAMES,
        fact_layer_type=fact_type
    )
    
    # Check forward pass
    output = model(sample_data)
    assert output.shape == (BATCH_SIZE, 1)

def test_nous_net_invalid_fact_layer():
    with pytest.raises(ValueError):
        NousNet(
            input_dim=INPUT_DIM,
            output_dim=1,
            feature_names=FEATURE_NAMES,
            fact_layer_type='invalid_type'
        )
