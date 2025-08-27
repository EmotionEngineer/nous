# nous/interpret.py
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from .models import NousNet
from .layers import LearnedAtomicFactLayer, BetaFactLayer

def trace_decision_graph(model: NousNet, x_sample: torch.Tensor) -> dict:
    """Traces the full reasoning path for a single sample."""
    if x_sample.dim() == 1: x_sample = x_sample.unsqueeze(0)
    model.eval()
    graph_data = {"trace": {}}
    with torch.no_grad():
        facts = model.atomic_fact_layer(x_sample).squeeze(0)
        graph_data['trace']['Atomic Facts'] = {name: {"value": facts[i].item()} for i, name in enumerate(model.atomic_fact_layer.fact_names)}
        h = facts.unsqueeze(0)
        for i, block in enumerate(model.nous_blocks):
            h, concepts, rule_activations = block(h)
            concepts, rule_activations = concepts.squeeze(0), rule_activations.squeeze(0)
            graph_data['trace'][f'Rules L{i}'] = {name: {"value": rule_activations[j].item()} for j, name in enumerate(block.rule_layer.rule_names)}
            graph_data['trace'][f'Concepts L{i}'] = {name: {"value": concepts[j].item()} for j, name in enumerate(block.concept_names)}
    return graph_data

def explain_fact(model: NousNet, fact_name: str) -> pd.DataFrame:
    """Provides a detailed breakdown of a single learned fact, showing feature weights."""
    fact_layer = model.atomic_fact_layer
    if not isinstance(fact_layer, LearnedAtomicFactLayer):
        raise TypeError("explain_fact is only applicable to 'beta' or 'sigmoid' fact layers.")
    try:
        fact_index = fact_layer.fact_names.index(fact_name)
    except ValueError:
        raise ValueError(f"Fact '{fact_name}' not found.")
    
    with torch.no_grad():
        w_left, w_right = fact_layer.projection_left.weight[fact_index], fact_layer.projection_right.weight[fact_index]
        threshold = fact_layer.thresholds[fact_index]
    df = pd.DataFrame({
        "feature": model.feature_names,
        "left_weight": w_left.cpu().detach().numpy(),
        "right_weight": w_right.cpu().detach().numpy(),
    })
    df['net_effect'] = df['left_weight'] - df['right_weight']
    print(f"Explanation for fact '{fact_name}':")
    print(f"Fact is TRUE when: (Sum(left_weight * feat) - Sum(right_weight * feat)) > {threshold.item():.3f}")
    return df.sort_values(by='net_effect', key=abs, ascending=False)

def plot_fact_activation_function(model: NousNet, fact_name: str, x_range=(-3, 3), n_points=200):
    """Visualizes the learned activation function for a single learned fact."""
    fact_layer = model.atomic_fact_layer
    if not isinstance(fact_layer, LearnedAtomicFactLayer):
        raise TypeError("This function is only for 'beta' or 'sigmoid' fact layers.")
    try:
        fact_index = fact_layer.fact_names.index(fact_name)
    except ValueError:
        raise ValueError(f"Fact '{fact_name}' not found.")
        
    diff_range = torch.linspace(x_range[0], x_range[1], n_points)
    
    with torch.no_grad():
        if isinstance(fact_layer, BetaFactLayer):
            k = torch.nn.functional.softplus(fact_layer.k_raw[fact_index]) + 1e-4
            nu = torch.nn.functional.softplus(fact_layer.nu_raw[fact_index]) + 1e-4
            activations = (1 + torch.exp(-k * diff_range))**(-nu)
            label = f'Learned Beta-like (k={k:.2f}, ν={nu:.2f})'
        else: # SigmoidFactLayer
            steepness = torch.nn.functional.softplus(fact_layer.steepness[fact_index]) + 1e-4
            activations = torch.sigmoid(steepness * diff_range)
            label = f'Learned Sigmoid (steepness={steepness:.2f})'
            
    plt.figure(figsize=(8, 5))
    plt.plot(diff_range.numpy(), activations.numpy(), label=label, linewidth=2.5)
    plt.plot(diff_range.numpy(), torch.sigmoid(diff_range).numpy(), label='Standard Sigmoid', linestyle='--', color='gray')
    plt.title(f"Activation Function for Fact:\n'{fact_name}'")
    plt.xlabel("Difference Value (Left Projection - Right Projection - Threshold)")
    plt.ylabel("Fact Activation (Truth Value)")
    plt.legend(); plt.grid(True, linestyle=':'); plt.ylim(-0.05, 1.05); plt.show()

def plot_final_layer_contributions(model: NousNet, x_sample: torch.Tensor):
    """Calculates and plots which high-level concepts most influenced the final prediction."""
    if x_sample.dim() == 1: x_sample = x_sample.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        h = model.atomic_fact_layer(x_sample)
        for block in model.nous_blocks: h, _, _ = block(h)
        final_activations = h.squeeze(0)
        
        output_dim = model.output_head.out_features
        weights = model.output_head.weight.squeeze(0)
        title = "Top Final Layer Concept Contributions"
        
        if output_dim > 1:
            predicted_class = model.output_head(h).argmax().item()
            weights = model.output_head.weight[predicted_class]
            title += f" for Predicted Class {predicted_class}"
            
    contributions = final_activations * weights
    concept_names = model.nous_blocks[-1].concept_names
    
    df = pd.DataFrame({'concept': concept_names, 'contribution': contributions.cpu().detach().numpy()})
    df = df.sort_values('contribution', key=abs, ascending=False).head(15)
    
    plt.figure(figsize=(10, 6)); 
    colors = ['#5fba7d' if c > 0 else '#d65f5f' for c in df['contribution']]
    sns.barplot(x='contribution', y='concept', data=df, palette=colors, dodge=False)
    plt.title(title); plt.xlabel("Contribution (Activation * Weight)"); plt.ylabel("Final Layer Concept"); plt.show()

def plot_logic_graph(*args, **kwargs):
    print("Graph visualization is planned for a future release.")
    pass
