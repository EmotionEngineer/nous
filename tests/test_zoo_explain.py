import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from nous import SoftLogitAND, SegmentMoE, HierarchicalMoE, NousFamilies
from nous.explain import (
    softlogitand_global_rules_df,
    softlogitand_local_contrib_df,
    moe_gate_summary_df,
    segmentmoe_local_explain_df,
    hiermoe_local_explain_df,
    nousfamilies_global_summary_df,
    nousfamilies_local_contrib_df,
)


def test_zoo_explain_outputs_nonempty():
    np.random.seed(0)

    D = 6
    feature_names = ["mean a", "mean b", "worst c", "x error", "other1", "other2"]

    X = np.random.randn(128, D).astype(np.float32)
    x1 = X[0]

    scaler = StandardScaler().fit(X)

    # SoftLogitAND
    m1 = SoftLogitAND(D, n_rules=16, n_thresh_per_feat=4)
    m1.init_from_data(X)
    df_g = softlogitand_global_rules_df(m1, feature_names, scaler=scaler, top_rules=5, top_facts_per_rule=2)
    assert isinstance(df_g, pd.DataFrame) and len(df_g) > 0

    df_l, meta = softlogitand_local_contrib_df(m1, x1, feature_names, scaler=scaler, top_rules=5, top_facts_per_rule=2)
    assert isinstance(df_l, pd.DataFrame) and len(df_l) > 0
    assert "prob_pos" in meta

    # SegmentMoE
    m2 = SegmentMoE(D, feature_names=feature_names, n_segments=4, n_thresh_per_feat=4)
    m2.init_from_data(X)
    df_gate = moe_gate_summary_df(m2, X[:32])
    assert isinstance(df_gate, pd.DataFrame) and len(df_gate) > 0

    df_seg, meta2 = segmentmoe_local_explain_df(m2, x1, feature_names, scaler=scaler, top_segments=2, top_facts=4)
    assert isinstance(df_seg, pd.DataFrame) and len(df_seg) > 0
    assert "prob_pos" in meta2

    # HierarchicalMoE
    m3 = HierarchicalMoE(D, feature_names=feature_names, n_segments=4, n_rules_expert=8, n_thresh_per_feat=4)
    m3.init_from_data(X)
    df_h, meta3 = hiermoe_local_explain_df(
        m3, x1, feature_names, scaler=scaler, top_segments=2, top_rules=3, top_facts_per_rule=2
    )
    assert isinstance(df_h, pd.DataFrame) and len(df_h) > 0
    assert "prob_pos" in meta3

    # NousFamilies
    m4 = NousFamilies(D, feature_names=feature_names, num_facts=16, rules_per_layer=(8, 6), n_families=5)
    m4.init_from_data(X)
    df_fg = nousfamilies_global_summary_df(m4, top_families=3)
    assert isinstance(df_fg, pd.DataFrame) and len(df_fg) > 0

    df_fl, meta4 = nousfamilies_local_contrib_df(m4, x1, top_families=3)
    assert isinstance(df_fl, pd.DataFrame) and len(df_fl) > 0
    assert "prob_pos" in meta4