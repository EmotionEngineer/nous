from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .facts import BetaFactLayer, PiecewiseLinearCalibrator
from .prototypes import ScaledPrototypeLayer
from .rules.blocks import SimpleNousBlock
from .rules.softmax import SoftmaxRuleLayer
from .rules.sparse import SparseRuleLayer

class NousNet(nn.Module):
    """
    NousNet: rule-based neural network with honest interpretability for classification and regression.
    """
    def __init__(
        self,
        input_dim: int,
        num_outputs: int,
        task_type: str = "classification",
        feature_names: Optional[Sequence[str]] = None,
        num_facts: int = 48,
        rules_per_layer: Sequence[int] = (24, 12),
        use_calibrators: bool = False,
        rule_selection_method: str = "fixed",
        use_prototypes: bool = False,
        l0_lambda: float = 1e-3,
        hc_temperature: float = 0.1,
        custom_calibrators: Optional[nn.ModuleList] = None,
    ) -> None:
        super().__init__()
        self.config = {
            'input_dim': input_dim, 'num_outputs': num_outputs, 'task_type': task_type,
            'feature_names': list(feature_names) if feature_names is not None else [f"Feature_{i}" for i in range(input_dim)],
            'num_facts': num_facts, 'rules_per_layer': tuple(rules_per_layer),
            'use_calibrators': use_calibrators, 'rule_selection_method': rule_selection_method,
            'use_prototypes': bool(use_prototypes and task_type == "classification"),
            'l0_lambda': l0_lambda, 'hc_temperature': hc_temperature
        }
        if custom_calibrators is not None:
            self.calibrators = custom_calibrators
        elif use_calibrators:
            self.calibrators = nn.ModuleList([PiecewiseLinearCalibrator() for _ in range(input_dim)])
        else:
            self.calibrators = None

        self.fact = BetaFactLayer(input_dim, num_facts)
        blocks: List[nn.Module] = []
        cur = num_facts
        for r in rules_per_layer:
            if rule_selection_method == 'fixed':
                blocks.append(SimpleNousBlock(cur, r))
            elif rule_selection_method == 'softmax':
                blocks.append(SoftmaxRuleLayer(cur, r))
            elif rule_selection_method == 'sparse':
                blocks.append(SparseRuleLayer(cur, r, l0_lambda=l0_lambda, hc_temperature=hc_temperature))
            elif rule_selection_method == 'soft_fact':
                blocks.append(SoftFactRuleLayer(cur, r))
            else:
                raise ValueError(f"Unknown rule_selection_method: {rule_selection_method}")
            cur = r
        self.blocks = nn.ModuleList(blocks)
        if self.config['use_prototypes']:
            self.head = ScaledPrototypeLayer(cur, num_prototypes=10, num_classes=num_outputs)
        else:
            self.head = nn.Linear(cur, num_outputs)
        if self.config['task_type'] == "regression" and self.config['num_outputs'] != 1:
            self.config['num_outputs'] = 1
            if isinstance(self.head, nn.Linear):
                self.head = nn.Linear(cur, 1)

    def forward(self, x: torch.Tensor, return_internals: bool = False):
        internals: Dict[str, torch.Tensor] = {}
        if self.calibrators is not None:
            x = torch.stack([calib(x[:, i]) for i, calib in enumerate(self.calibrators)], dim=1)

        h_facts = self.fact(x)
        h = h_facts
        if return_internals:
            internals['facts'] = h_facts.detach()

        for i, blk in enumerate(self.blocks):
            if return_internals:
                h, details = blk(h, return_details=True)
                internals[f'block_{i}'] = details
            else:
                h = blk(h)

        logits = self.head(h)
        if self.config['task_type'] == "regression":
            logits = logits.squeeze(-1)

        if return_internals:
            return logits, internals
        return logits

    @torch.no_grad()
    def forward_explain(
        self,
        x: Union[np.ndarray, torch.Tensor],
        drop_rule_spec: Optional[Tuple[int, int]] = None,
        restrict_masks: Optional[List[torch.Tensor]] = None,
        apply_pruning: bool = False,
        pruning_threshold: float = 0.0,
        device: Optional[torch.device] = None,
        explain_disable_norm: bool = False,
        explain_exclude_proj: bool = False
    ):
        """
        Honest forward for explanations with interventions/gating recompute.

        Returns:
          - classification: (probas, logits, internals)
          - regression: (pred, pred, internals)
        """
        self.eval()
        device = device or next(self.parameters()).device

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        elif isinstance(x, torch.Tensor) and x.dim() == 1:
            x = x.unsqueeze(0).to(device)
        else:
            x = x.to(device)

        if self.calibrators is not None:
            x_cal = torch.stack([calib(x[:, i]) for i, calib in enumerate(self.calibrators)], dim=1)
        else:
            x_cal = x

        h = self.fact(x_cal)
        internals: Dict[str, torch.Tensor] = {'facts': h.detach()}

        for i, blk in enumerate(self.blocks):
            drop_idx = None
            if drop_rule_spec is not None and drop_rule_spec[0] == i:
                drop_idx = int(drop_rule_spec[1])
            restrict = None
            if restrict_masks is not None and i < len(restrict_masks) and restrict_masks[i] is not None:
                restrict = restrict_masks[i].to(device)
            prune = pruning_threshold if apply_pruning else None

            h, details = blk(
                h,
                return_details=True,
                drop_rule_idx=drop_idx,
                restrict_mask=restrict,
                prune_below=prune,
                explain_disable_norm=explain_disable_norm,
                explain_exclude_proj=explain_exclude_proj
            )
            internals[f'block_{i}'] = details

        logits = self.head(h)
        if self.config['task_type'] == "classification":
            probas = F.softmax(logits, dim=-1)
            return probas.squeeze(0).cpu().numpy(), logits.squeeze(0).cpu().numpy(), internals
        else:
            pred = logits.squeeze(-1).squeeze(0).cpu().numpy()
            return np.array([pred]), np.array([pred]), internals

    def compute_total_l0_loss(self) -> torch.Tensor:
        if self.config['rule_selection_method'] != 'sparse':
            return torch.tensor(0.0, device=next(self.parameters()).device)
        total_l0 = 0.0
        for blk in self.blocks:
            if hasattr(blk, 'compute_l0_loss'):
                total_l0 += blk.compute_l0_loss()
        return total_l0

    def model_summary(self) -> Dict[str, object]:
        summary = {
            "Task": self.config['task_type'],
            "Rule Selection": self.config['rule_selection_method'],
            "Use Calibrators": self.config['use_calibrators'],
            "Use Prototypes": self.config['use_prototypes'],
            "Num Facts": self.config['num_facts'],
            "Rules per Layer": self.config['rules_per_layer'],
            "Total Parameters": sum(p.numel() for p in self.parameters())
        }
        if self.config['rule_selection_method'] == 'sparse':
            summary["L0 Lambda"] = self.config['l0_lambda']
        return summary

    @torch.no_grad()
    def encode(
        self,
        X: Union[np.ndarray, torch.Tensor, Sequence[Sequence[float]]],
        device: Optional[torch.device] = None,
        batch_size: int = 2048,
        explain_disable_norm: bool = False,
        explain_exclude_proj: bool = False
    ) -> torch.Tensor:
        """
        Return H [N, D_last] — representations at the head input.

        Flags allow clean representations (without LayerNorm / without residual projection).
        """
        self.eval()
        device = device or next(self.parameters()).device

        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        elif isinstance(X, torch.Tensor):
            X_tensor = X.detach().cpu().float()
        else:
            X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32)

        H_list = []
        for i in range(0, len(X_tensor), batch_size):
            xb = X_tensor[i:i+batch_size].to(device)

            if self.calibrators is not None:
                xb_cal = torch.stack([calib(xb[:, j]) for j, calib in enumerate(self.calibrators)], dim=1)
            else:
                xb_cal = xb

            h = self.fact(xb_cal)
            for blk in self.blocks:
                h, _ = blk(h, return_details=True,
                           explain_disable_norm=explain_disable_norm,
                           explain_exclude_proj=explain_exclude_proj)
            H_list.append(h.detach().cpu())
        H = torch.cat(H_list, dim=0)
        return H