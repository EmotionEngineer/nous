import numpy as np
from nous import NousNet

def _make_clf():
    return NousNet(
        input_dim=10,
        num_outputs=4,
        task_type="classification",
        num_facts=12,
        rules_per_layer=(8, 6),
        rule_selection_method="softmax",
        use_calibrators=False
    )

def test_forward_explain_smoke():
    model = _make_clf()
    x = np.random.randn(10).astype(np.float32)
    probas, logits, internals = model.forward_explain(x)
    assert probas.shape == (4,)
    assert logits.shape == (4,)
    assert 'facts' in internals
    assert any(k.startswith('block_') for k in internals.keys())