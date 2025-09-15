import torch
from nous import NousNet

def test_classification_forward_shapes():
    model = NousNet(
        input_dim=12,
        num_outputs=3,
        task_type="classification",
        num_facts=16,
        rules_per_layer=(8, 6),
        rule_selection_method="softmax",
        use_calibrators=True,
        use_prototypes=False
    )
    x = torch.randn(5, 12)
    logits = model(x)
    assert logits.shape == (5, 3)

def test_regression_forward_shapes():
    model = NousNet(
        input_dim=10,
        num_outputs=1,
        task_type="regression",
        num_facts=16,
        rules_per_layer=(8, ),
        rule_selection_method="sparse",
        use_calibrators=False,
        use_prototypes=False
    )
    x = torch.randn(7, 10)
    y = model(x)
    assert y.shape == (7,)