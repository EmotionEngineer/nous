from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class ScaledPrototypeLayer(nn.Module):
    """
    Prototype-based head for classification with temperature scaling on distances.
    Uses L2-normalized features/prototypes and exp(-tau * distance).
    """
    def __init__(self, input_dim: int, num_prototypes: int, num_classes: int) -> None:
        super().__init__()
        self.num_prototypes = num_prototypes
        self.num_classes = num_classes
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, input_dim) * 0.1)
        self.prototype_class = nn.Parameter(torch.randn(num_prototypes, num_classes))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_norm = F.normalize(h, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)
        distances = torch.cdist(h_norm, prototypes_norm)
        activations = torch.exp(-F.softplus(self.temperature) * distances)
        class_logits = torch.matmul(activations, self.prototype_class)
        return class_logits

    @torch.no_grad()
    def get_params(self) -> Dict[str, torch.Tensor]:
        P = self.prototypes.detach()
        Pn = F.normalize(P, p=2, dim=1)
        W = self.prototype_class.detach()
        Wsm = F.softmax(W, dim=1)
        tau = F.softplus(self.temperature).detach()
        return dict(prototypes=P, prototypes_norm=Pn, class_weights=W, class_probs=Wsm, temperature=tau)

    @torch.no_grad()
    def compute_dist_act(self, h: torch.Tensor):
        h_norm = F.normalize(h, p=2, dim=1)
        Pn = F.normalize(self.prototypes, p=2, dim=1)
        d = torch.cdist(h_norm, Pn)
        act = torch.exp(-F.softplus(self.temperature) * d)
        return d, act