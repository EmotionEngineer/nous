import torch
import numpy as np
import pytest

from nous.rules.gaters import (
    GateSoftRank,
    GateLogisticThresholdExactK,
    GateCappedSimplex,
    GateSparsemaxK,
    GateHardConcreteBudget,
)
from nous.rules.soft_fact import SoftFactRuleLayer
from nous import NousNet

def _rand_scores(B=3, R=10, seed=123):
    torch.manual_seed(seed)
    return torch.randn(B, R)

@pytest.mark.parametrize("gater_cls,kwargs,exact_sum", [
    (GateSoftRank, dict(tau=0.5), False),
    (GateLogisticThresholdExactK, dict(tau=0.5, iters=20), True),
    (GateCappedSimplex, dict(), True),
    (GateSparsemaxK, dict(tau=0.7), False),
    (GateHardConcreteBudget, dict(R=10, tau=0.5), False),
])
def test_gaters_shapes_and_bounds(gater_cls, kwargs, exact_sum):
    B, R, k = 4, 12, 5
    s = _rand_scores(B=B, R=R)
    # Ensure HardConcrete budget uses correct R
    kw = dict(kwargs)
    if 'R' in kw:
        kw['R'] = R
    gater = gater_cls(**kw)
    g = gater(s, k=k, mask=None)
    assert g.shape == (B, R)
    assert torch.all(g >= 0) and torch.all(g <= 1)
    if exact_sum:
        sums = g.sum(dim=1)
        assert torch.allclose(sums, torch.full_like(sums, float(min(k, R))), atol=1e-2)

def test_soft_fact_with_gater_forward_details():
    torch.manual_seed(0)
    layer = SoftFactRuleLayer(input_dim=10, num_rules=6, rule_gater=GateSoftRank(tau=0.5), top_k_facts=3, top_k_rules=4)
    x = torch.rand(2, 10)
    y, details = layer(x, return_details=True)
    assert y.shape == (2, 6)
    gm = details["gate_mask"]
    assert gm.shape == (2, 6)
    assert torch.all(gm >= 0) and torch.all(gm <= 1)
    assert "facts_used" in details
    assert "aggregator_weights" in details

def test_nousnet_with_rule_gater_name_smoke():
    model = NousNet(
        input_dim=8,
        num_outputs=3,
        task_type="classification",
        num_facts=12,
        rules_per_layer=(8, ),
        rule_selection_method="soft_fact",
        rule_gater_name="soft_rank",
        use_calibrators=False,
        use_prototypes=False,
    )
    x = torch.randn(5, 8)
    out = model(x)
    assert out.shape == (5, 3)