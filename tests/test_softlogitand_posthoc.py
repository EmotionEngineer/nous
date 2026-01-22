import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from nous import SoftLogitAND
from nous.explain.softlogitand_posthoc import (
    SoftLogitANDPosthocExplainer,
    hard_cap_topk_rules,
)


def _make_data(n=512, d=20, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)
    # Simple synthetic labels: noisy linear rule
    w = rng.randn(d).astype(np.float32)
    logits = (X @ w) * 0.3
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.rand(n) < p).astype(np.float32)
    return X, y


def test_softlogitand_posthoc_smoke_and_strategies():
    X_raw, y = _make_data(n=800, d=30, seed=1)
    feature_names = [f"x{i:03d}" for i in range(X_raw.shape[1])]

    # Scale for SoftLogitAND training/inference
    scaler = StandardScaler().fit(X_raw[:600])
    X_scaled = scaler.transform(X_raw).astype(np.float32)

    # Model (no training needed for smoke; init_from_data requires X)
    model = SoftLogitAND(input_dim=X_scaled.shape[1], n_rules=64, n_thresh_per_feat=4, tau=0.7, use_negations=True)
    model.init_from_data(X_scaled[:600])

    expl = SoftLogitANDPosthocExplainer(
        model=model,
        feature_names=feature_names,
        x_scaler=scaler,
        k_rules=10,
        k_literals=3,
        n_clusters=8,
        use_surrogate=True,
        surrogate_alpha=1e-3,
        z_threshold=0.5,
    )

    # Fit posthoc on reference distribution
    expl.fit_posthoc(X_ref_scaled=X_scaled[:600])

    # Fit rule stats + add mean abs contrib
    expl.fit_rule_stats(
        X_train_scaled=X_scaled[:600],
        y_train01=y[:600],
        X_test_scaled=X_scaled[600:],
        y_test01=y[600:],
    )
    expl.add_mean_abs_contrib_to_rule_stats(X_train_scaled=X_scaled[:600], X_test_scaled=X_scaled[600:])

    # Local report
    x0 = X_raw[0]
    rep = expl.report(x0, y_true01=float(y[0]), x_is_scaled=False, top_clusters=6)
    assert isinstance(rep.markdown, str) and len(rep.markdown) > 10
    assert isinstance(rep.tables, dict) and len(rep.tables) > 0

    # Ensure strategy tables exist (surrogate may exist because use_surrogate=True)
    expected_keys = {
        "mmr_plus_hardcap_rules",
        "vanilla_rules",
        "mmr_topL_rules",
        "mmr_feature_mass_rules",
        "hardcap_rules",
        "cluster_themes",
        "surrogate_top_rules",
    }
    # Surrogate may fail to fit in some extreme cases; allow missing surrogate key
    missing = expected_keys - set(rep.tables.keys())
    assert missing.issubset({"surrogate_top_rules"}), f"Missing unexpected tables: {missing}"

    # Cluster themes table sanity
    df_ct = rep.tables["cluster_themes"]
    assert isinstance(df_ct, pd.DataFrame)
    assert len(df_ct) > 0
    assert "cluster" in df_ct.columns
    assert "cluster_contrib" in df_ct.columns

    # Fidelity table sanity
    df_fid = expl.fidelity_table_for_x(x0, x_is_scaled=False)
    assert isinstance(df_fid, pd.DataFrame)
    assert len(df_fid) > 0
    assert "coverage_abs_contrib" in df_fid.columns

    # Global tables
    df_rules = expl.global_rule_importance(split="train", topn=10, sort_by="mean_abs_contrib")
    assert isinstance(df_rules, pd.DataFrame) and len(df_rules) > 0

    df_clusters = expl.global_cluster_importance(split="train")
    assert isinstance(df_clusters, pd.DataFrame) and len(df_clusters) > 0

    df_feats = expl.global_feature_importance_mass_weighted(split="train", topn=10)
    assert isinstance(df_feats, pd.DataFrame) and len(df_feats) > 0


def test_hardcap_respects_cap_per_feature():
    rng = np.random.RandomState(0)
    R = 50
    contrib_abs = np.abs(rng.randn(R))
    top_feature_ids = rng.randint(0, 5, size=R)  # 5 dominant features
    k = 20
    cap = 2

    chosen = hard_cap_topk_rules(contrib_abs, top_feature_ids, k=k, cap_per_feature=cap)
    assert len(chosen) <= k

    # Verify cap is respected
    counts = {}
    for r in chosen:
        f = int(top_feature_ids[r])
        counts[f] = counts.get(f, 0) + 1
    assert all(v <= cap for v in counts.values())