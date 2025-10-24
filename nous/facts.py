from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class BetaFactLayer(nn.Module):
    """
    Beta-like fact activation layer.

    Computes β = exp(nu * log(sigmoid(k * (Lx - Rx - th)))) with numerical clamps.
    """
    def __init__(self, input_dim: int, num_facts: int) -> None:
        super().__init__()
        self.L = nn.Linear(input_dim, num_facts, bias=False)
        self.R = nn.Linear(input_dim, num_facts, bias=False)
        self.th   = nn.Parameter(torch.randn(num_facts) * 0.1)
        self.kraw = nn.Parameter(torch.ones(num_facts) * 0.5)
        self.nuraw= nn.Parameter(torch.zeros(num_facts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = (self.L(x) - self.R(x)) - self.th
        k  = F.softplus(self.kraw)  + 1e-4
        nu = F.softplus(self.nuraw) + 1e-4
        log_beta = torch.clamp(nu * F.logsigmoid(k * diff), min=-80.0)
        return torch.exp(log_beta)

    @torch.no_grad()
    def get_rule_parameters(self):
        k = F.softplus(self.kraw).cpu().numpy() + 1e-4
        nu = F.softplus(self.nuraw).cpu().numpy() + 1e-4
        L_weights = self.L.weight.detach().cpu().numpy()
        R_weights = self.R.weight.detach().cpu().numpy()
        thresholds = self.th.detach().cpu().numpy()
        return L_weights, R_weights, thresholds, k, nu

    @torch.no_grad()
    def compute_diff_and_params(self, x_cal: torch.Tensor):
        """
        Returns
        -------
        diff : torch.Tensor
            (Lx - Rx - th) of shape [B, F].
        k : torch.Tensor
            Softplus(kraw) + eps of shape [F].
        nu : torch.Tensor
            Softplus(nuraw) + eps of shape [F].
        net_w : torch.Tensor
            L.weight - R.weight of shape [F, D].
        """
        diff = (self.L(x_cal) - self.R(x_cal)) - self.th
        k  = F.softplus(self.kraw)  + 1e-4
        nu = F.softplus(self.nuraw) + 1e-4
        net_w = (self.L.weight - self.R.weight)
        return diff, k, nu, net_w


class PiecewiseLinearCalibrator(nn.Module):
    """
    Monotonic piecewise-linear per-feature calibrator using cumulative positive deltas.
    """
    def __init__(self, num_bins: int = 8, input_range=(-3.0, 3.0)) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.input_min, self.input_max = input_range
        self.register_buffer('bin_edges', torch.linspace(self.input_min, self.input_max, num_bins + 1))
        self.deltas = nn.Parameter(torch.ones(num_bins) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normalized = (x - self.input_min) / (self.input_max - self.input_min + 1e-8)
        x_normalized = torch.clamp(x_normalized, 0.0, 1.0)
        bin_idx = torch.floor(x_normalized * self.num_bins).long().clamp(0, self.num_bins - 1)
        cum_deltas = torch.cumsum(F.softplus(self.deltas), dim=0)
        cum_deltas = torch.cat([torch.zeros(1, device=x.device), cum_deltas])
        left_vals = self.bias + cum_deltas[bin_idx]
        right_vals = self.bias + cum_deltas[bin_idx + 1]
        t = (x_normalized * self.num_bins) - bin_idx.float()
        y = left_vals + t * (right_vals - left_vals)
        return y

    def local_slope(self, x: torch.Tensor) -> torch.Tensor:
        x_normalized = (x - self.input_min) / (self.input_max - self.input_min + 1e-8)
        x_normalized = torch.clamp(x_normalized, 0.0, 1.0)
        bin_idx = torch.floor(x_normalized * self.num_bins).long().clamp(0, self.num_bins - 1)
        deltas_sp = F.softplus(self.deltas)
        cum = torch.cumsum(deltas_sp, dim=0)
        cum = torch.cat([torch.zeros(1, device=x.device), cum])
        left_vals = self.bias + cum[bin_idx]
        right_vals = self.bias + cum[bin_idx + 1]
        slope_y_vs_xnorm = (right_vals - left_vals)
        slope = slope_y_vs_xnorm * (self.num_bins / (self.input_max - self.input_min + 1e-8))
        return torch.clamp(slope, min=1e-6)

    @torch.no_grad()
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        device = y.device
        deltas_sp = F.softplus(self.deltas)
        cum = torch.cumsum(deltas_sp, dim=0)
        cum = torch.cat([torch.zeros(1, device=device), cum])
        vals = self.bias + cum

        y_flat = y.view(-1)
        idx = torch.searchsorted(vals, y_flat, right=True) - 1
        idx = torch.clamp(idx, 0, self.num_bins - 1)

        y_left = vals[idx]
        y_right = vals[idx + 1]
        t = (y_flat - y_left) / torch.clamp((y_right - y_left), min=1e-6)
        x_norm = (idx.float() + t) / self.num_bins
        x = x_norm * (self.input_max - self.input_min) + self.input_min
        return x.view_as(y)

class PiecewiseLinearCalibratorQuantile(nn.Module):
    """
    Monotonic piecewise-linear calibrator with bin edges defined by empirical quantiles.
    Edges are fixed at construction time (non-learnable); only slopes and bias are learned.
    """
    def __init__(self, edges: torch.Tensor) -> None:
        super().__init__()
        if edges.ndim != 1 or len(edges) < 2:
            raise ValueError("edges must be a 1D tensor with at least 2 elements")
        self.register_buffer('edges', edges.float())
        self.num_bins = edges.numel() - 1
        self.deltas = nn.Parameter(torch.ones(self.num_bins) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bin_idx = torch.searchsorted(self.edges, x, right=True) - 1
        bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1)
        cum_deltas = torch.cumsum(F.softplus(self.deltas), dim=0)
        cum_deltas = torch.cat([torch.zeros(1, device=x.device), cum_deltas])
        left_vals = self.bias + cum_deltas[bin_idx]
        right_vals = self.bias + cum_deltas[bin_idx + 1]
        left_edge = self.edges[bin_idx]
        right_edge = self.edges[bin_idx + 1]
        t = (x - left_edge) / (right_edge - left_edge + 1e-8)
        t = torch.clamp(t, 0.0, 1.0)
        return left_vals + t * (right_vals - left_vals)

    def local_slope(self, x: torch.Tensor) -> torch.Tensor:
        bin_idx = torch.searchsorted(self.edges, x, right=True) - 1
        bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1)
        deltas_sp = F.softplus(self.deltas)
        slope = deltas_sp[bin_idx] / (self.edges[bin_idx + 1] - self.edges[bin_idx] + 1e-8)
        return torch.clamp(slope, min=1e-6)

    @torch.no_grad()
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        device = y.device
        deltas_sp = F.softplus(self.deltas)
        cum = torch.cumsum(deltas_sp, dim=0)
        cum = torch.cat([torch.zeros(1, device=device), cum])
        vals = self.bias + cum
        y_flat = y.view(-1)
        idx = torch.searchsorted(vals, y_flat, right=True) - 1
        idx = torch.clamp(idx, 0, self.num_bins - 1)
        y_left = vals[idx]
        y_right = vals[idx + 1]
        t = (y_flat - y_left) / torch.clamp((y_right - y_left), min=1e-6)
        x = self.edges[idx] + t * (self.edges[idx + 1] - self.edges[idx])
        return x.view_as(y)