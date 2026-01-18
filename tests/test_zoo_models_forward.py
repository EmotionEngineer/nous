import numpy as np
import torch

from nous import SoftLogitAND, SoftLogicInteraction, SegmentMoE, HierarchicalMoE, NousFamilies


def test_zoo_models_forward_and_backward_smoke():
    torch.manual_seed(0)
    np.random.seed(0)

    D = 6
    B = 8
    feature_names = ["mean a", "mean b", "worst c", "x error", "other1", "other2"]

    X = np.random.randn(64, D).astype(np.float32)
    xb = torch.tensor(np.random.randn(B, D).astype(np.float32))
    yb = torch.randint(0, 2, (B,), dtype=torch.float32)

    models = [
        SoftLogitAND(D, n_rules=16, n_thresh_per_feat=4),
        SoftLogicInteraction(D, n_rules=16, n_thresh_per_feat=4, n_inter=8),
        SegmentMoE(D, feature_names=feature_names, n_segments=4, n_thresh_per_feat=4),
        HierarchicalMoE(D, feature_names=feature_names, n_segments=4, n_rules_expert=8, n_thresh_per_feat=4),
        NousFamilies(D, feature_names=feature_names, num_facts=16, rules_per_layer=(8, 6), n_families=5),
    ]

    for m in models:
        if hasattr(m, "init_from_data"):
            m.init_from_data(X)

        out = m(xb)
        assert out.shape == (B,)

        loss = torch.nn.BCEWithLogitsLoss()(out, yb)
        loss.backward()