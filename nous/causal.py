# nous/causal.py
import torch
import torch.nn as nn
from typing import Literal
from .models import NousNet

def find_counterfactual(
    model: NousNet,
    x_sample: torch.Tensor,
    target_output: float,
    task: Literal['regression', 'classification'] = 'classification',
    lr: float = 0.01,
    steps: int = 200,
    l1_lambda: float = 0.5
) -> dict:
    """
    Finds a minimal change to the input to achieve a target output.
    
    Args:
        model (NousNet): The trained model.
        x_sample (torch.Tensor): The original input tensor.
        target_output (float): For classification, the desired probability (e.g., 0.8). 
                               For regression, the desired absolute value (e.g., 150.0).
        task (Literal['regression', 'classification']): The type of task.
        lr, steps, l1_lambda: Optimization parameters.

    Returns:
        dict: A dictionary containing the counterfactual sample and a list of changes.
    """
    if task == 'classification':
        if not (0 < target_output < 1): 
            raise ValueError("Target for classification must be a probability between 0 and 1.")
        if model.output_head.out_features > 1: 
            raise NotImplementedError("Counterfactual analysis for multi-class models is not yet supported.")
        # Calculate target in logit space for numerical stability
        target = torch.log(torch.tensor(target_output) / (1 - torch.tensor(target_output)))
    else: # regression
        target = torch.tensor(target_output, dtype=torch.float32)

    x_sample = x_sample.clone().detach()
    delta = torch.zeros_like(x_sample, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)
    
    for _ in range(steps):
        optimizer.zero_grad()
        x_perturbed = x_sample + delta
        prediction = model(x_perturbed.unsqueeze(0)).squeeze()
        
        target_loss = (prediction - target)**2
        l1_loss = torch.norm(delta, p=1)
        total_loss = target_loss + l1_lambda * l1_loss
        
        # This will now only be called for scalar losses, as multi-class is caught above
        total_loss.backward()
        optimizer.step()

    final_x = x_sample + delta.detach()
    changes = []
    for i, name in enumerate(model.feature_names):
        if not torch.isclose(x_sample[i], final_x[i], atol=1e-3):
            changes.append((name, x_sample[i].item(), final_x[i].item()))
            
    return {"counterfactual_x": final_x, "changes": changes}
