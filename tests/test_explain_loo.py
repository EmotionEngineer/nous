import numpy as np
from nous import NousNet, rule_impact_df

def test_rule_impact_df_nonempty():
    model = NousNet(
        input_dim=10,
        num_outputs=3,
        task_type="classification",
        num_facts=12,
        rules_per_layer=(8, 6),
        rule_selection_method="softmax"
    )
    feature_names = [f"f{i}" for i in range(10)]
    x = np.random.randn(10).astype(np.float32)
    df = rule_impact_df(model, x, feature_names)
    assert df is not None