# tests/test_interpret_causal.py
import torch
import pytest
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nous.models import NousNet
from nous.interpret import trace_decision_graph, explain_fact, plot_fact_activation_function, plot_final_layer_contributions, plot_logic_graph
from nous.causal import find_counterfactual

INPUT_DIM = 5
FEATURE_NAMES = [f'feat_{i}' for i in range(INPUT_DIM)]

@pytest.fixture
def beta_model():
    return NousNet(input_dim=INPUT_DIM, output_dim=1, feature_names=FEATURE_NAMES,
                   fact_layer_type='beta', num_facts=10, num_rules_per_layer=[5])

@pytest.fixture
def exhaustive_model():
    return NousNet(input_dim=INPUT_DIM, output_dim=1, feature_names=FEATURE_NAMES, fact_layer_type='exhaustive')

@pytest.fixture
def multiclass_model():
    return NousNet(input_dim=INPUT_DIM, output_dim=3, feature_names=FEATURE_NAMES, fact_layer_type='sigmoid')

@pytest.fixture
def x_sample(): return torch.randn(INPUT_DIM)

def test_trace_decision_graph(beta_model, x_sample):
    graph = trace_decision_graph(beta_model, x_sample)
    assert isinstance(graph, dict)
    assert 'trace' in graph
    assert len(graph['trace']['Atomic Facts']) == beta_model.atomic_fact_layer.output_dim
    assert len(graph['trace']['Concepts L0']) == beta_model.nous_blocks[0].rule_layer.output_dim

def test_explain_fact(beta_model):
    fact_name = beta_model.atomic_fact_layer.fact_names[0]
    df = explain_fact(beta_model, fact_name=fact_name)
    assert isinstance(df, pd.DataFrame)
    assert 'net_effect' in df.columns

def test_explain_fact_errors(exhaustive_model, beta_model):
    with pytest.raises(TypeError): explain_fact(exhaustive_model, fact_name="any")
    with pytest.raises(ValueError): explain_fact(beta_model, fact_name="non_existent_fact")

def test_plot_fact_activation_function(beta_model):
    fact_name = beta_model.atomic_fact_layer.fact_names[0]
    plot_fact_activation_function(beta_model, fact_name=fact_name)
    plt.close()

def test_plot_final_layer_contributions(beta_model, multiclass_model, x_sample):
    plot_final_layer_contributions(beta_model, x_sample)
    plt.close()
    plot_final_layer_contributions(multiclass_model, x_sample)
    plt.close()

def test_plot_logic_graph(beta_model, x_sample):
    graph = trace_decision_graph(beta_model, x_sample)
    plot_logic_graph(graph)

def test_find_counterfactual(beta_model, x_sample):
    # Classification
    result_class = find_counterfactual(beta_model, x_sample, target_output=0.8, task='classification')
    assert 'counterfactual_x' in result_class
    
    # Regression
    result_reg = find_counterfactual(beta_model, x_sample, target_output=150.0, task='regression')
    assert 'changes' in result_reg

def test_find_counterfactual_errors(multiclass_model, beta_model, x_sample):
    with pytest.raises(NotImplementedError):
        find_counterfactual(multiclass_model, x_sample, target_output=0.8)
    with pytest.raises(ValueError):
        find_counterfactual(beta_model, x_sample, target_output=1.1, task='classification')
