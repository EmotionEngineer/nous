import numpy as np
from nous import NousNet
from nous.explain.traces import prototype_contribution_df

def test_prototype_head_contrib_df():
    model = NousNet(
        input_dim=10,
        num_outputs=3,
        task_type="classification",
        num_facts=12,
        rules_per_layer=(8,),
        rule_selection_method="softmax",
        use_prototypes=True
    )
    x = np.random.randn(10).astype(np.float32)
    df = prototype_contribution_df(model, x, class_names=["A","B","C"], top_k=3)
    assert df is not None