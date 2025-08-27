# nous/__init__.py

__version__ = "0.1.0"

from .models import NousNet
from .interpret import (
    trace_decision_graph,
    explain_fact,
    plot_logic_graph,
    plot_fact_activation_function,
    plot_final_layer_contributions,
)
from .causal import find_counterfactual

__all__ = [
    # Main Model
    "NousNet",
    # Interpretation Suite
    "trace_decision_graph",
    "explain_fact",
    "plot_logic_graph",
    "plot_fact_activation_function",
    "plot_final_layer_contributions",
    # Causal Analysis
    "find_counterfactual",
]
