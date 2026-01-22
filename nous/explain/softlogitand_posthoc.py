from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import Lasso

from ..zoo import SoftLogitAND
from .zoo import describe_threshold_fact


# ----------------------------
# Numerics / small utilities
# ----------------------------

def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def _logit_np(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / ((np.linalg.norm(a) + eps) * (np.linalg.norm(b) + eps)))


def topL_normalized(p: np.ndarray, L: int = 12, eps: float = 1e-12) -> np.ndarray:
    """
    Keep only top-L entries of vector p (by value) and renormalize to sum=1.
    """
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("topL_normalized expects a 1D vector.")
    q = np.zeros_like(p, dtype=np.float64)
    if p.size == 0:
        return q
    L_eff = int(min(max(1, L), p.size))
    idx = np.argsort(-p)[:L_eff]
    q[idx] = p[idx]
    s = float(q.sum())
    if s > 0:
        q /= (s + eps)
    return q


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """
    Weighted quantile with non-negative weights. If total weight is zero, falls back to median.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    order = np.argsort(values)
    v = values[order]
    w = np.clip(weights[order], 0.0, None)
    cw = np.cumsum(w)
    tot = float(cw[-1]) if cw.size else 0.0
    if tot <= 0:
        return float(np.median(values))
    t = float(q) * tot
    idx = int(np.searchsorted(cw, t, side="left"))
    idx = min(max(idx, 0), len(v) - 1)
    return float(v[idx])


# ----------------------------
# SoftLogitAND forward parts
# ----------------------------

@torch.no_grad()
def forward_parts_softlogitand(model: SoftLogitAND, x_scaled_1d: np.ndarray) -> Dict[str, Any]:
    """
    Forward pass returning components needed for post-hoc explanation.

    Returns dict with:
      f: base facts [F0]
      z: rule activations [R]
      P: selector probs [R, Fin]
      w: head weights [R]
      b: head bias (float)
      pred_logit, pred_prob
    """
    model.eval()
    dev = next(model.parameters()).device
    xt = torch.tensor(np.asarray(x_scaled_1d, dtype=np.float32)[None, :], device=dev)

    f = model.facts(xt)     # [1, F0]
    z, P = model.rules(f)   # z: [1, R], P: [R, Fin]
    w = model.head.weight.view(-1)
    b = model.head.bias.view(-1)[0] if model.head.bias is not None else torch.tensor(0.0, device=dev)

    logit = float((z @ w.view(-1, 1) + b).view(-1).item())
    prob = float(_sigmoid_np(np.array([logit], dtype=np.float64))[0])

    return dict(
        f=f.detach().cpu().numpy().squeeze(0),
        z=z.detach().cpu().numpy().squeeze(0),
        P=P.detach().cpu().numpy(),
        w=w.detach().cpu().numpy(),
        b=float(b.detach().cpu().item()),
        pred_logit=logit,
        pred_prob=prob,
    )


@torch.no_grad()
def get_global_P_and_w(model: SoftLogitAND) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return global selector probabilities and head weights:
      P: [R, Fin]
      w: [R]
    """
    model.eval()
    P = model.rules.selector_probs().detach().cpu().numpy()
    w = model.head.weight.view(-1).detach().cpu().numpy()
    return P.astype(np.float64), w.astype(np.float64)


def literal_to_fact_and_neg(model: SoftLogitAND, lit_idx: int) -> Tuple[int, bool]:
    """
    Map literal index in augmented facts to base fact index and negation flag.
    If use_negations=True, Fin = 2*F0, where j>=F0 means negated literal.
    """
    F0 = int(model.facts.num_facts)
    fin = int(model.rules.fin)
    if fin == 2 * F0:
        base = int(lit_idx % F0)
        neg = bool(lit_idx >= F0)
        return base, neg
    return int(lit_idx), False


def literal_text(
    model: SoftLogitAND,
    lit_idx: int,
    feature_names: Sequence[str],
    scaler: Optional[Any] = None,
) -> str:
    """
    Render a literal as a readable threshold condition.
    """
    base, neg = literal_to_fact_and_neg(model, lit_idx)
    return describe_threshold_fact(model.facts, base, feature_names, scaler=scaler, negated=neg)


def literal_contribs_for_rule(model: SoftLogitAND, f: np.ndarray, P_r: np.ndarray) -> np.ndarray:
    """
    Heuristic local literal score:
      c_j(x) = P_r[j] * logit(fact_aug_j(x))
    This is used only for ranking "top literals" inside a rule.
    """
    F0 = int(model.facts.num_facts)
    fin = int(P_r.shape[0])
    f = np.asarray(f, dtype=np.float64).ravel()

    if fin == 2 * F0:
        f_aug = np.concatenate([f, 1.0 - f], axis=0)
    else:
        f_aug = f

    return np.asarray(P_r, dtype=np.float64) * _logit_np(f_aug)


def rule_feature_mass(model: SoftLogitAND, P_r: np.ndarray, D: int) -> np.ndarray:
    """
    Aggregate selector probability mass from literals to base input features.

    Returns mass vector [D] (normalized to sum=1 if total mass > 0).
    """
    feat_idx = model.facts.feat_idx.detach().cpu().numpy()  # [F0]
    fin = int(P_r.shape[0])
    mass = np.zeros((D,), dtype=np.float64)
    for j in range(fin):
        base, _neg = literal_to_fact_and_neg(model, j)
        feat = int(feat_idx[base])
        if 0 <= feat < D:
            mass[feat] += float(P_r[j])
    s = float(mass.sum())
    if s > 0:
        mass /= s
    return mass


def top_feature_of_rule(model: SoftLogitAND, P_r: np.ndarray, D: int) -> int:
    """
    Dominant feature id of a rule: argmax of feature-mass.
    """
    return int(np.argmax(rule_feature_mass(model, P_r, D)))


def selector_feature_summary(
    model: SoftLogitAND,
    P_vec: np.ndarray,
    feature_names: Sequence[str],
    scaler: Optional[Any] = None,
    top_features: int = 8,
    qlo: float = 0.2,
    qhi: float = 0.8,
) -> List[str]:
    """
    Produce a compact textual summary of where selector mass goes:
      - group by (feature, direction) where direction is "pos" (x > th) or "neg" (x <= th)
      - compute mass and weighted quantile range of thresholds

    Notes:
    - Thresholds are stored in the same space as model inputs (often standardized).
    - If scaler is provided (StandardScaler), thresholds are converted back to raw units.
    """
    feat_idx = model.facts.feat_idx.detach().cpu().numpy()  # [F0]
    th_scaled = model.facts.th.detach().cpu().numpy()       # [F0]
    fin = int(P_vec.shape[0])

    buckets: Dict[Tuple[int, str], Dict[str, list]] = {}
    for j in range(fin):
        p = float(P_vec[j])
        if p <= 0:
            continue

        base, neg = literal_to_fact_and_neg(model, j)
        feat = int(feat_idx[base])
        direction = "neg" if neg else "pos"

        th = float(th_scaled[base])
        if scaler is not None:
            # Invert StandardScaler: raw = scaled * scale + mean
            th = float(th * float(scaler.scale_[feat]) + float(scaler.mean_[feat]))

        key = (feat, direction)
        buckets.setdefault(key, {"p": [], "th": []})
        buckets[key]["p"].append(p)
        buckets[key]["th"].append(th)

    items = []
    for (feat, direction), d in buckets.items():
        p = np.array(d["p"], dtype=np.float64)
        t = np.array(d["th"], dtype=np.float64)
        mass = float(p.sum())
        th_lo = weighted_quantile(t, p, qlo)
        th_hi = weighted_quantile(t, p, qhi)
        th_mid = weighted_quantile(t, p, 0.5)
        items.append((mass, feat, direction, th_lo, th_mid, th_hi))

    items.sort(key=lambda x: -x[0])
    out = []
    for mass, feat, direction, th_lo, th_mid, th_hi in items[: int(top_features)]:
        name = str(feature_names[feat]) if 0 <= feat < len(feature_names) else f"f{feat}"
        if direction == "pos":
            out.append(f"{name} > ~{th_mid:.3g} (range {th_lo:.3g}..{th_hi:.3g}, mass={mass:.2f})")
        else:
            out.append(f"{name} <= ~{th_mid:.3g} (range {th_lo:.3g}..{th_hi:.3g}, mass={mass:.2f})")
    return out


# ----------------------------
# Rule selection strategies
# ----------------------------

def mmr_select_rules(
    contrib_abs: np.ndarray,
    Rep: np.ndarray,
    k: int = 10,
    lambda_div: float = 0.40,
    candidate_pool: int = 140,
) -> List[int]:
    """
    Maximal Marginal Relevance selection:
      score(r) = imp(r) - lambda_div * max_{s in chosen} sim(r, s)
    where imp(r) = |contrib(r)|, sim = cosine similarity in Rep space.
    """
    contrib_abs = np.asarray(contrib_abs, dtype=np.float64).ravel()
    Rep = np.asarray(Rep, dtype=np.float64)
    R = int(contrib_abs.shape[0])
    if R == 0:
        return []

    pool = np.argsort(-contrib_abs)[: min(int(candidate_pool), R)].astype(int).tolist()
    chosen: List[int] = []

    def sim_rs(r: int, s: int) -> float:
        return cosine_sim(Rep[r], Rep[s])

    while len(chosen) < min(int(k), len(pool)):
        best_r = None
        best_score = -1e18
        for r in pool:
            if r in chosen:
                continue
            imp = float(contrib_abs[r])
            if not chosen:
                score = imp
            else:
                max_sim = max(sim_rs(r, s) for s in chosen)
                score = imp - float(lambda_div) * max_sim
            if score > best_score:
                best_score = score
                best_r = r
        if best_r is None:
            break
        chosen.append(int(best_r))
    return chosen


def hard_cap_topk_rules(
    contrib_abs: np.ndarray,
    top_feature_ids: np.ndarray,
    k: int = 10,
    cap_per_feature: int = 2,
) -> List[int]:
    """
    Hardcap strategy:
      - sort rules by importance desc
      - add rule if its dominant feature hasn't exceeded cap_per_feature
    """
    contrib_abs = np.asarray(contrib_abs, dtype=np.float64).ravel()
    top_feature_ids = np.asarray(top_feature_ids, dtype=int).ravel()

    order = np.argsort(-contrib_abs).astype(int).tolist()
    counts: Dict[int, int] = {}
    chosen: List[int] = []
    for r in order:
        f = int(top_feature_ids[r]) if 0 <= r < top_feature_ids.size else -1
        if counts.get(f, 0) >= int(cap_per_feature):
            continue
        chosen.append(int(r))
        counts[f] = counts.get(f, 0) + 1
        if len(chosen) >= int(k):
            break
    return chosen


def mmr_with_hardcap(
    contrib_abs: np.ndarray,
    Rep: np.ndarray,
    top_feature_ids: np.ndarray,
    k: int = 10,
    lambda_div: float = 0.40,
    candidate_pool: int = 140,
    cap_per_feature: int = 2,
) -> List[int]:
    """
    MMR selection with a hard constraint on max rules per dominant feature.
    """
    contrib_abs = np.asarray(contrib_abs, dtype=np.float64).ravel()
    Rep = np.asarray(Rep, dtype=np.float64)
    top_feature_ids = np.asarray(top_feature_ids, dtype=int).ravel()

    R = int(contrib_abs.shape[0])
    if R == 0:
        return []

    pool = np.argsort(-contrib_abs)[: min(int(candidate_pool), R)].astype(int).tolist()
    chosen: List[int] = []
    counts: Dict[int, int] = {}

    def sim_rs(r: int, s: int) -> float:
        return cosine_sim(Rep[r], Rep[s])

    while len(chosen) < min(int(k), len(pool)):
        best_r = None
        best_score = -1e18
        for r in pool:
            if r in chosen:
                continue
            f = int(top_feature_ids[r]) if 0 <= r < top_feature_ids.size else -1
            if counts.get(f, 0) >= int(cap_per_feature):
                continue
            imp = float(contrib_abs[r])
            if not chosen:
                score = imp
            else:
                max_sim = max(sim_rs(r, s) for s in chosen)
                score = imp - float(lambda_div) * max_sim
            if score > best_score:
                best_score = score
                best_r = r

        if best_r is None:
            break
        chosen.append(int(best_r))
        f = int(top_feature_ids[int(best_r)]) if 0 <= int(best_r) < top_feature_ids.size else -1
        counts[f] = counts.get(f, 0) + 1

    return chosen


# ----------------------------
# Markdown rendering (no tabulate dependency)
# ----------------------------

def sanitize_markdown_ascii(s: str) -> str:
    """
    Make markdown output more robust across environments by replacing some unicode dashes/spaces.
    """
    if s is None:
        return ""
    repl = {
        "\u2011": "-", "\u2010": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
        "\u00a0": " ",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def _md_escape_cell(x: Any) -> str:
    s = sanitize_markdown_ascii(str(x))
    s = s.replace("\t", " ").replace("\n", " ")
    # Robust pipe escaping inside markdown tables
    s = s.replace("|", "&#124;")
    return s


def df_to_markdown_ascii(df: pd.DataFrame, max_rows: int = 12) -> str:
    """
    Simple markdown table renderer without requiring `tabulate`.
    Escapes pipe characters inside cells.
    """
    if df is None or len(df) == 0:
        return "_(empty)_"

    dfx = df.head(int(max_rows)).copy()
    cols = [str(c) for c in dfx.columns.tolist()]

    header = "| " + " | ".join(_md_escape_cell(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"

    rows = []
    for _, r in dfx.iterrows():
        rows.append("| " + " | ".join(_md_escape_cell(r[c]) for c in cols) + " |")

    return sanitize_markdown_ascii("\n".join([header, sep] + rows))


# ----------------------------
# Report objects
# ----------------------------

@dataclass
class SoftLogitANDPosthocReport:
    meta: Dict[str, Any]
    tables: Dict[str, pd.DataFrame]
    markdown: str


class SoftLogitANDPosthocExplainer:
    """
    Post-hoc explainer for SoftLogitAND including:
      - multiple rule selection strategies (vanilla/MMR/hardcap/surrogate)
      - cluster themes (rule clustering by selector profile)
      - per-rule dataset statistics (train/test)
      - global importance tables (rules/clusters/features)
      - local markdown report generation

    This class is designed to be fit once on a reference distribution (e.g. train),
    then used for many local explanations.
    """

    def __init__(
        self,
        model: SoftLogitAND,
        feature_names: Sequence[str],
        x_scaler: Optional[Any] = None,
        # Selection / report settings
        k_rules: int = 10,
        k_literals: int = 4,
        summary_top_features: int = 8,
        z_threshold: float = 0.5,
        # MMR settings
        topL: int = 12,
        lambda_div: float = 0.40,
        candidate_pool: int = 140,
        # Hardcap settings
        cap_per_feature: int = 2,
        # Clustering
        n_clusters: int = 16,
        # Surrogate
        use_surrogate: bool = True,
        surrogate_alpha: float = 1e-3,
        # Determinism
        random_state: int = 42,
    ):
        self.model = model
        self.feature_names = list(feature_names)
        self.x_scaler = x_scaler

        self.k_rules = int(k_rules)
        self.k_literals = int(k_literals)
        self.summary_top_features = int(summary_top_features)
        self.z_threshold = float(z_threshold)

        self.topL = int(topL)
        self.lambda_div = float(lambda_div)
        self.candidate_pool = int(candidate_pool)

        self.cap_per_feature = int(cap_per_feature)

        self.n_clusters = int(n_clusters)
        self.use_surrogate = bool(use_surrogate)
        self.surrogate_alpha = float(surrogate_alpha)
        self.random_state = int(random_state)

        # Cached global structures (after fit_posthoc)
        self.P_global: Optional[np.ndarray] = None          # [R, Fin]
        self.w_global: Optional[np.ndarray] = None          # [R]
        self.Rep_topL: Optional[np.ndarray] = None          # [R, Fin]
        self.Rep_featmass: Optional[np.ndarray] = None      # [R, D]
        self.top_feature_ids: Optional[np.ndarray] = None   # [R]

        # Clustering (after fit_posthoc)
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_summary: Optional[pd.DataFrame] = None
        self.cluster_Pbar: Optional[Dict[int, np.ndarray]] = None

        # Surrogate (after fit_posthoc)
        self.surrogate: Optional[Lasso] = None
        self.surrogate_coefs: Optional[np.ndarray] = None

        # Dataset stats (after fit_rule_stats)
        self.rule_stats: Optional[pd.DataFrame] = None

    @torch.no_grad()
    def compute_Z_matrix(self, X_scaled: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """
        Compute rule activation matrix Z of shape [N, R] on scaled inputs.
        """
        self.model.eval()
        dev = next(self.model.parameters()).device
        out = []
        X_scaled = np.asarray(X_scaled, dtype=np.float32)

        for i in range(0, len(X_scaled), int(batch_size)):
            xb = torch.tensor(X_scaled[i:i + int(batch_size)], device=dev)
            f = self.model.facts(xb)
            z, _ = self.model.rules(f)
            out.append(z.detach().cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def fit_posthoc(self, X_ref_scaled: np.ndarray) -> "SoftLogitANDPosthocExplainer":
        """
        Fit global post-hoc structures:
          - selector representations
          - clustering (themes)
          - optional surrogate model
        """
        P, w = get_global_P_and_w(self.model)
        self.P_global = P
        self.w_global = w

        R = int(P.shape[0])
        D = int(len(self.feature_names))

        # Build selector representations
        self.Rep_topL = np.stack([topL_normalized(P[r], L=self.topL) for r in range(R)], axis=0)
        self.Rep_featmass = np.stack([rule_feature_mass(self.model, P[r], D) for r in range(R)], axis=0)
        self.top_feature_ids = np.array([top_feature_of_rule(self.model, P[r], D) for r in range(R)], dtype=int)

        # Cluster rules using cosine distance in top-L selector space
        labels = self._fit_clusters(P=P, w=w, rep=self.Rep_topL)
        self.cluster_labels = labels

        # Fit surrogate: Lasso on Z -> model logits
        if self.use_surrogate:
            Z = self.compute_Z_matrix(X_ref_scaled, batch_size=4096)
            y_teacher_logit = predict_logits_softlogitand(self.model, X_ref_scaled)
            reg = Lasso(alpha=self.surrogate_alpha, max_iter=20000, random_state=self.random_state)
            reg.fit(Z, y_teacher_logit)
            self.surrogate = reg
            self.surrogate_coefs = np.asarray(reg.coef_, dtype=np.float64).ravel()

        return self

    def _fit_clusters(self, P: np.ndarray, w: np.ndarray, rep: np.ndarray) -> np.ndarray:
        """
        Internal clustering routine + cluster summary creation.
        """
        R = int(P.shape[0])
        n_clusters = int(min(max(2, self.n_clusters), max(2, R)))

        # Compute cosine distance matrix in a vectorized way: D = 1 - (X @ X^T)
        X = np.asarray(rep, dtype=np.float64)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        Xn = X / norms
        sim = Xn @ Xn.T
        Dm = 1.0 - sim
        np.fill_diagonal(Dm, 0.0)

        # sklearn compatibility: "metric" (new) vs "affinity" (old)
        try:
            cl = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
        except TypeError:
            cl = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage="average")

        labels = cl.fit_predict(Dm).astype(int)

        # Build cluster summaries with Pbar weighted by |w|
        self.cluster_Pbar = {}
        rows = []
        for g in sorted(np.unique(labels).tolist()):
            idx = np.where(labels == g)[0].astype(int)
            if idx.size == 0:
                continue
            ww = np.abs(w[idx]) + 1e-12
            Pbar = (P[idx] * ww[:, None]).sum(axis=0)
            Pbar = Pbar / (float(Pbar.sum()) + 1e-12)
            self.cluster_Pbar[int(g)] = Pbar

            rows.append({
                "cluster": int(g),
                "n_rules": int(idx.size),
                "sum_abs_w": float(np.abs(w[idx]).sum()),
                "top_feature_ranges": " || ".join(selector_feature_summary(
                    self.model, Pbar, self.feature_names, scaler=self.x_scaler, top_features=6
                )),
            })

        self.cluster_summary = pd.DataFrame(rows).sort_values("sum_abs_w", ascending=False).reset_index(drop=True)
        return labels

    def fit_rule_stats(
        self,
        X_train_scaled: np.ndarray,
        y_train01: np.ndarray,
        X_test_scaled: np.ndarray,
        y_test01: np.ndarray,
        batch_size: int = 4096,
    ) -> "SoftLogitANDPosthocExplainer":
        """
        Compute per-rule dataset stats for train/test:
          - mean z
          - fraction z > threshold
          - support counts
          - mean y in/out of z>thr
          - delta y (in - out)
          - mean(w*z)
        """
        df_tr = self._compute_rule_stats_on_dataset(
            X_scaled=X_train_scaled,
            y01=y_train01,
            split_name="train",
            z_threshold=self.z_threshold,
            batch_size=batch_size,
        )
        df_te = self._compute_rule_stats_on_dataset(
            X_scaled=X_test_scaled,
            y01=y_test01,
            split_name="test",
            z_threshold=self.z_threshold,
            batch_size=batch_size,
        )
        self.rule_stats = df_tr.merge(df_te, on="rule", how="inner").set_index("rule")
        return self

    @torch.no_grad()
    def add_mean_abs_contrib_to_rule_stats(
        self,
        X_train_scaled: np.ndarray,
        X_test_scaled: np.ndarray,
        batch_size: int = 4096,
    ) -> "SoftLogitANDPosthocExplainer":
        """
        Add mean(abs(w*z)) to rule_stats for train and test splits.
        """
        if self.rule_stats is None:
            raise RuntimeError("Call fit_rule_stats(...) before add_mean_abs_contrib_to_rule_stats(...).")
        self.rule_stats["train_mean_abs_contrib_abs_wz"] = self._mean_abs_contrib_on_dataset(X_train_scaled, batch_size=batch_size)
        self.rule_stats["test_mean_abs_contrib_abs_wz"] = self._mean_abs_contrib_on_dataset(X_test_scaled, batch_size=batch_size)
        return self

    @torch.no_grad()
    def _mean_abs_contrib_on_dataset(self, X_scaled: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        if self.w_global is None:
            raise RuntimeError("Call fit_posthoc(...) first.")
        w = np.asarray(self.w_global, dtype=np.float64)  # [R]
        R = int(w.shape[0])

        self.model.eval()
        dev = next(self.model.parameters()).device
        X_scaled = np.asarray(X_scaled, dtype=np.float32)

        sum_abs = np.zeros((R,), dtype=np.float64)
        n = int(len(X_scaled))
        for i in range(0, n, int(batch_size)):
            xb = torch.tensor(X_scaled[i:i + int(batch_size)], device=dev)
            f = self.model.facts(xb)
            z, _ = self.model.rules(f)
            z = z.detach().cpu().numpy().astype(np.float64)  # [B, R]
            sum_abs += np.abs(z * w[None, :]).sum(axis=0)

        return (sum_abs / max(1, n)).astype(np.float64)

    @torch.no_grad()
    def _compute_rule_stats_on_dataset(
        self,
        X_scaled: np.ndarray,
        y01: np.ndarray,
        split_name: str,
        z_threshold: float,
        batch_size: int = 4096,
    ) -> pd.DataFrame:
        if self.w_global is None:
            raise RuntimeError("Call fit_posthoc(...) first.")
        w = np.asarray(self.w_global, dtype=np.float64)
        R = int(w.shape[0])
        thr = float(z_threshold)

        self.model.eval()
        dev = next(self.model.parameters()).device

        sum_z = np.zeros((R,), dtype=np.float64)
        sum_contrib = np.zeros((R,), dtype=np.float64)

        sum_in = np.zeros((R,), dtype=np.float64)
        sum_y_in = np.zeros((R,), dtype=np.float64)

        y01 = np.asarray(y01, dtype=np.float64).ravel()
        y_total_sum = float(y01.sum())
        X_scaled = np.asarray(X_scaled, dtype=np.float32)
        n = int(len(X_scaled))

        for i in range(0, n, int(batch_size)):
            xb = torch.tensor(X_scaled[i:i + int(batch_size)], device=dev)
            yb = y01[i:i + int(batch_size)]

            f = self.model.facts(xb)
            z, _ = self.model.rules(f)  # [B, R]
            z = z.detach().cpu().numpy().astype(np.float64)

            sum_z += z.sum(axis=0)
            sum_contrib += (z * w[None, :]).sum(axis=0)

            mask = (z > thr).astype(np.float64)
            sum_in += mask.sum(axis=0)
            sum_y_in += mask.T @ yb

        mean_z = sum_z / max(1, n)
        mean_contrib = sum_contrib / max(1, n)
        frac_in = sum_in / max(1, n)

        eps = 1e-12
        mean_y_in = np.full((R,), np.nan, dtype=np.float64)
        mean_y_out = np.full((R,), np.nan, dtype=np.float64)

        in_mask = sum_in > 0
        out_mask = (n - sum_in) > 0

        mean_y_in[in_mask] = sum_y_in[in_mask] / (sum_in[in_mask] + eps)

        sum_out = n - sum_in
        sum_y_out = y_total_sum - sum_y_in
        mean_y_out[out_mask] = sum_y_out[out_mask] / (sum_out[out_mask] + eps)

        df = pd.DataFrame({
            "rule": np.arange(R, dtype=int),
            f"{split_name}_mean_z": mean_z,
            f"{split_name}_frac_z_gt_{thr:.2f}": frac_in,
            f"{split_name}_support_in": sum_in,
            f"{split_name}_mean_y_in_z_gt_thr": mean_y_in,
            f"{split_name}_mean_y_out_z_le_thr": mean_y_out,
            f"{split_name}_delta_y_in_out": mean_y_in - mean_y_out,
            f"{split_name}_mean_contrib_wz": mean_contrib,
        })
        return df

    # ----------------------------
    # Local report tables
    # ----------------------------

    def select_rules_for_x(self, x_raw: np.ndarray, x_is_scaled: bool = False) -> Dict[str, List[int]]:
        """
        Return rule indices for each selection strategy.
        """
        x_s = self._to_scaled_1d(x_raw, x_is_scaled=x_is_scaled)
        parts = forward_parts_softlogitand(self.model, x_s)
        contrib_abs = np.abs(parts["z"] * parts["w"]).astype(np.float64)

        if self.Rep_topL is None or self.Rep_featmass is None or self.top_feature_ids is None:
            raise RuntimeError("Call fit_posthoc(X_ref_scaled) first.")

        vanilla = np.argsort(-contrib_abs)[: self.k_rules].astype(int).tolist()
        mmr_topL = mmr_select_rules(
            contrib_abs=contrib_abs,
            Rep=self.Rep_topL,
            k=self.k_rules,
            lambda_div=self.lambda_div,
            candidate_pool=self.candidate_pool,
        )
        mmr_featmass = mmr_select_rules(
            contrib_abs=contrib_abs,
            Rep=self.Rep_featmass,
            k=self.k_rules,
            lambda_div=self.lambda_div,
            candidate_pool=self.candidate_pool,
        )
        hardcap = hard_cap_topk_rules(
            contrib_abs=contrib_abs,
            top_feature_ids=self.top_feature_ids,
            k=self.k_rules,
            cap_per_feature=self.cap_per_feature,
        )
        mmr_hardcap = mmr_with_hardcap(
            contrib_abs=contrib_abs,
            Rep=self.Rep_topL,
            top_feature_ids=self.top_feature_ids,
            k=self.k_rules,
            lambda_div=self.lambda_div,
            candidate_pool=self.candidate_pool,
            cap_per_feature=self.cap_per_feature,
        )

        out: Dict[str, List[int]] = {
            "vanilla_top_abs_wz": vanilla,
            "mmr_topL": mmr_topL,
            "mmr_feature_mass": mmr_featmass,
            "hardcap_only": hardcap,
            "mmr_plus_hardcap": mmr_hardcap,
        }

        if self.use_surrogate and self.surrogate_coefs is not None:
            sc = np.asarray(self.surrogate_coefs, dtype=np.float64)
            sur_contrib = sc * np.asarray(parts["z"], dtype=np.float64)
            sur_idx = np.argsort(-np.abs(sur_contrib))[: self.k_rules].astype(int).tolist()
            out["surrogate_top_abs_beta_z"] = sur_idx

        return out

    def _rule_table(self, x_scaled_1d: np.ndarray, selected_rules: List[int], title: str = "") -> pd.DataFrame:
        """
        Build a local per-rule table for a given selection list.
        """
        if self.P_global is None or self.w_global is None:
            raise RuntimeError("Call fit_posthoc(...) first.")

        parts = forward_parts_softlogitand(self.model, x_scaled_1d)
        f, z, P_local = parts["f"], parts["z"], parts["P"]
        w = np.asarray(parts["w"], dtype=np.float64)
        contrib = (np.asarray(z, dtype=np.float64) * w).astype(np.float64)

        rows: List[Dict[str, Any]] = []
        for r in selected_rules:
            r = int(r)
            if r < 0 or r >= len(w):
                continue

            P_r = P_local[r]
            summary = " ; ".join(selector_feature_summary(
                self.model, P_r, self.feature_names, scaler=self.x_scaler, top_features=self.summary_top_features
            ))

            # Top literals by heuristic local literal contribution
            c = literal_contribs_for_rule(self.model, f, P_r)
            lit_idx = np.argsort(-np.abs(c))[: self.k_literals].astype(int).tolist()
            lits = [
                f"{literal_text(self.model, j, self.feature_names, scaler=self.x_scaler)} (p={float(P_r[j]):.3f}, c={float(c[j]):+.3f})"
                for j in lit_idx
            ]

            top_feat_id = top_feature_of_rule(self.model, P_r, len(self.feature_names))
            top_feat = self.feature_names[top_feat_id] if 0 <= top_feat_id < len(self.feature_names) else f"f{top_feat_id}"

            row: Dict[str, Any] = {
                "rule": r,
                "top_feature": top_feat,
                "z": float(z[r]),
                "w": float(w[r]),
                "contrib_wz": float(contrib[r]),
                f"z>{self.z_threshold:.2f}?": bool(float(z[r]) > self.z_threshold),
                "rule_summary": summary,
                "top_literals": " | ".join(lits),
            }

            if self.rule_stats is not None and r in self.rule_stats.index:
                st = self.rule_stats.loc[r].to_dict()
                for k, v in st.items():
                    # Keep numeric columns numeric; strings are not expected here
                    try:
                        row[k] = float(v)
                    except Exception:
                        row[k] = v

            rows.append(row)

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df["abs_contrib_tmp"] = df["contrib_wz"].abs()
            df = df.sort_values("abs_contrib_tmp", ascending=False).drop(columns=["abs_contrib_tmp"]).reset_index(drop=True)
        if title:
            df.attrs["title"] = str(title)
        return df

    def _cluster_table(self, x_scaled_1d: np.ndarray, top_clusters: int = 8) -> pd.DataFrame:
        """
        Local cluster theme table with exact contributions (sum of rule contribs in the cluster).
        """
        if self.cluster_labels is None or self.cluster_Pbar is None:
            raise RuntimeError("Call fit_posthoc(...) first (clustering not available).")

        parts = forward_parts_softlogitand(self.model, x_scaled_1d)
        z = np.asarray(parts["z"], dtype=np.float64)
        w = np.asarray(parts["w"], dtype=np.float64)
        P_local = parts["P"]
        contrib = z * w

        rows: List[Dict[str, Any]] = []
        for g in sorted(np.unique(self.cluster_labels).tolist()):
            idx = np.where(self.cluster_labels == g)[0].astype(int)
            if idx.size == 0:
                continue

            cluster_contrib = float(contrib[idx].sum())
            cluster_abs_contrib = float(abs(cluster_contrib))

            top_rule = int(idx[np.argmax(np.abs(contrib[idx]))])

            # Two representative rules by |contrib|
            rep = idx[np.argsort(-np.abs(contrib[idx]))[:2]]
            rep_lines = []
            for r in rep:
                rep_lines.append(
                    f"r{int(r)}: {float(contrib[int(r)]):+.3f} | " +
                    " ; ".join(selector_feature_summary(
                        self.model, P_local[int(r)], self.feature_names, scaler=self.x_scaler, top_features=4
                    ))
                )

            Pbar = self.cluster_Pbar[int(g)]
            rows.append({
                "cluster": int(g),
                "cluster_contrib": cluster_contrib,
                "cluster_abs_contrib": cluster_abs_contrib,
                "n_rules": int(idx.size),
                "top_rule": top_rule,
                "top_rule_contrib": float(contrib[top_rule]),
                "cluster_summary": " || ".join(selector_feature_summary(
                    self.model, Pbar, self.feature_names, scaler=self.x_scaler, top_features=6
                )),
                "representatives": " || ".join(rep_lines),
            })

        df = pd.DataFrame(rows)
        if len(df) == 0:
            return df
        df = df.sort_values("cluster_abs_contrib", ascending=False).reset_index(drop=True)
        return df.head(int(top_clusters))

    def report(
        self,
        x_raw: np.ndarray,
        y_true01: Optional[float] = None,
        x_is_scaled: bool = False,
        top_clusters: int = 8,
    ) -> SoftLogitANDPosthocReport:
        """
        Generate a local report including:
          - multiple rule tables (one per selection strategy)
          - cluster themes table
          - optional surrogate top rules table
          - markdown summary
        """
        x_s = self._to_scaled_1d(x_raw, x_is_scaled=x_is_scaled)
        parts = forward_parts_softlogitand(self.model, x_s)

        pred_prob = float(parts["pred_prob"])
        pred_logit = float(parts["pred_logit"])

        rule_sets = self.select_rules_for_x(x_raw, x_is_scaled=x_is_scaled)

        tables: Dict[str, pd.DataFrame] = {}
        # Always include all strategies if available
        if "mmr_plus_hardcap" in rule_sets:
            tables["mmr_plus_hardcap_rules"] = self._rule_table(x_s, rule_sets["mmr_plus_hardcap"], title="MMR + Hardcap (recommended)")
        if "vanilla_top_abs_wz" in rule_sets:
            tables["vanilla_rules"] = self._rule_table(x_s, rule_sets["vanilla_top_abs_wz"], title="Vanilla top-|w*z|")
        if "mmr_topL" in rule_sets:
            tables["mmr_topL_rules"] = self._rule_table(x_s, rule_sets["mmr_topL"], title="MMR diversified (Top-L selector)")
        if "mmr_feature_mass" in rule_sets:
            tables["mmr_feature_mass_rules"] = self._rule_table(x_s, rule_sets["mmr_feature_mass"], title="MMR diversified (feature-mass)")
        if "hardcap_only" in rule_sets:
            tables["hardcap_rules"] = self._rule_table(x_s, rule_sets["hardcap_only"], title=f"Hardcap (<= {self.cap_per_feature} per dominant feature)")
        if "surrogate_top_abs_beta_z" in rule_sets and self.surrogate_coefs is not None:
            tables["surrogate_top_rules"] = self._surrogate_table(x_s, rule_sets["surrogate_top_abs_beta_z"], title="Surrogate top-|beta*z|")

        if self.cluster_labels is not None:
            tables["cluster_themes"] = self._cluster_table(x_s, top_clusters=int(top_clusters))

        meta: Dict[str, Any] = {
            "pred_prob": pred_prob,
            "pred_logit": pred_logit,
            "y_true01": None if y_true01 is None else float(y_true01),
            "k_rules": self.k_rules,
            "k_literals": self.k_literals,
            "summary_top_features": self.summary_top_features,
            "z_threshold": self.z_threshold,
            "lambda_div": self.lambda_div,
            "topL": self.topL,
            "cap_per_feature": self.cap_per_feature,
            "n_clusters": self.n_clusters,
            "use_surrogate": self.use_surrogate,
            "surrogate_alpha": self.surrogate_alpha,
        }

        md = self.render_markdown(meta, tables)
        return SoftLogitANDPosthocReport(meta=meta, tables=tables, markdown=md)

    def _surrogate_table(self, x_scaled_1d: np.ndarray, selected_rules: List[int], title: str = "") -> pd.DataFrame:
        """
        Surrogate table: contributions based on surrogate coef * z.
        """
        if self.surrogate_coefs is None:
            return pd.DataFrame()

        parts = forward_parts_softlogitand(self.model, x_scaled_1d)
        z = np.asarray(parts["z"], dtype=np.float64)
        P_local = parts["P"]
        sc = np.asarray(self.surrogate_coefs, dtype=np.float64).ravel()
        sur_contrib = sc * z

        rows: List[Dict[str, Any]] = []
        for r in selected_rules:
            r = int(r)
            if r < 0 or r >= sc.size:
                continue
            rows.append({
                "rule": r,
                "coef": float(sc[r]),
                "z": float(z[r]),
                "surrogate_contrib": float(sur_contrib[r]),
                "rule_summary": " ; ".join(selector_feature_summary(
                    self.model, P_local[r], self.feature_names, scaler=self.x_scaler, top_features=self.summary_top_features
                )),
            })

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df["abs_tmp"] = df["surrogate_contrib"].abs()
            df = df.sort_values("abs_tmp", ascending=False).drop(columns=["abs_tmp"]).reset_index(drop=True)
        if title:
            df.attrs["title"] = str(title)
        return df

    def render_markdown(self, meta: Dict[str, Any], tables: Dict[str, pd.DataFrame]) -> str:
        """
        Create a markdown report string (no external dependencies).
        """
        lines: List[str] = []
        lines.append("# SoftLogitAND post-hoc report (binary classification)")
        lines.append("")
        lines.append("## Prediction")
        lines.append(f"- pred_prob: **{meta['pred_prob']:.4f}**")
        lines.append(f"- pred_logit: **{meta['pred_logit']:+.4f}**")
        if meta.get("y_true01", None) is not None:
            y = float(meta["y_true01"])
            lines.append(f"- y_true: **{int(y)}**")
            lines.append(f"- error_prob (pred - y): **{(meta['pred_prob'] - y):+.4f}**")
        lines.append("")
        lines.append("## Settings")
        lines.append(f"- k_rules={meta['k_rules']}, k_literals={meta['k_literals']}")
        lines.append(f"- summary_top_features={meta['summary_top_features']}")
        lines.append(f"- z_threshold={meta['z_threshold']:.2f}")
        lines.append(f"- lambda_div={meta['lambda_div']}, topL={meta['topL']}, cap_per_feature={meta['cap_per_feature']}")
        lines.append(f"- n_clusters={meta['n_clusters']}")
        lines.append(f"- use_surrogate={meta['use_surrogate']} (alpha={meta['surrogate_alpha']})")
        lines.append("")

        order = [
            ("mmr_plus_hardcap_rules", "### A) Recommended: MMR + Hardcap"),
            ("vanilla_rules", "### B) Vanilla top-|w*z|"),
            ("mmr_topL_rules", "### C) MMR diversified (Top-L selector)"),
            ("mmr_feature_mass_rules", "### D) MMR diversified (feature-mass)"),
            ("hardcap_rules", "### E) Hardcap only"),
            ("cluster_themes", "### F) Cluster themes (exact contributions)"),
            ("surrogate_top_rules", "### G) Surrogate top rules (compact governance view)"),
        ]

        for key, title in order:
            if key not in tables:
                continue
            lines.append(sanitize_markdown_ascii(title))
            lines.append(df_to_markdown_ascii(tables[key], max_rows=12))
            lines.append("")

        return sanitize_markdown_ascii("\n".join(lines))

    def save_markdown(self, report: SoftLogitANDPosthocReport, path: str) -> None:
        """
        Save markdown report to file.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(sanitize_markdown_ascii(report.markdown))

    # ----------------------------
    # Fidelity table for strategies
    # ----------------------------

    def fidelity_table_for_x(self, x_raw: np.ndarray, x_is_scaled: bool = False) -> pd.DataFrame:
        """
        Compute simple fidelity metrics for each strategy:
          - coverage_abs_contrib: sum(|w*z| over selected rules) / sum(|w*z| over all rules)
          - avg_pairwise_similarity: average cosine similarity among selected in Rep_topL space
        """
        if self.Rep_topL is None:
            raise RuntimeError("Call fit_posthoc(...) first.")
        x_s = self._to_scaled_1d(x_raw, x_is_scaled=x_is_scaled)
        parts = forward_parts_softlogitand(self.model, x_s)
        contrib_abs = np.abs(np.asarray(parts["z"], dtype=np.float64) * np.asarray(parts["w"], dtype=np.float64))

        rule_sets = self.select_rules_for_x(x_raw, x_is_scaled=x_is_scaled)

        rows = []
        for name, rs in rule_sets.items():
            cov = _coverage_abs_contrib(contrib_abs, rs)
            sim = _avg_pairwise_cosine_similarity(self.Rep_topL[np.array(rs, dtype=int)]) if len(rs) > 1 else 0.0
            rows.append({
                "strategy": name,
                "k_selected": int(len(rs)),
                "coverage_abs_contrib": float(cov),
                "avg_pairwise_similarity": float(sim),
            })

        return pd.DataFrame(rows).sort_values("coverage_abs_contrib", ascending=False).reset_index(drop=True)

    # ----------------------------
    # Global interpretations
    # ----------------------------

    def global_rule_importance(self, split: str = "train", topn: int = 25, sort_by: str = "mean_abs_contrib") -> pd.DataFrame:
        """
        Global rule importance table based on rule_stats.

        sort_by:
          - "mean_abs_contrib" (requires add_mean_abs_contrib_to_rule_stats)
          - "mean_contrib"
          - "coverage"
          - "delta_y"
        """
        if self.rule_stats is None or self.P_global is None or self.w_global is None:
            raise RuntimeError("Need fit_posthoc(...) and fit_rule_stats(...) before global_rule_importance(...).")
        if self.cluster_labels is None or self.top_feature_ids is None:
            raise RuntimeError("Clustering/top_feature_ids missing. Call fit_posthoc(...).")

        rs = self.rule_stats
        thr = self.z_threshold

        col_mean = f"{split}_mean_contrib_wz"
        col_cov = f"{split}_frac_z_gt_{thr:.2f}"
        col_dy = f"{split}_delta_y_in_out"
        col_abs = f"{split}_mean_abs_contrib_abs_wz"

        if sort_by == "mean_contrib":
            key = col_mean
        elif sort_by == "coverage":
            key = col_cov
        elif sort_by == "delta_y":
            key = col_dy
        else:
            # Default: mean_abs_contrib
            if col_abs not in rs.columns:
                raise RuntimeError(f"Column '{col_abs}' not found. Call add_mean_abs_contrib_to_rule_stats(...) first.")
            key = col_abs

        order = rs.sort_values(key, ascending=False).head(int(topn)).index.astype(int).tolist()

        rows = []
        for r in order:
            P_r = self.P_global[int(r)]
            rows.append({
                "rule": int(r),
                "cluster": int(self.cluster_labels[int(r)]),
                "top_feature": self.feature_names[int(self.top_feature_ids[int(r)])],
                "w": float(self.w_global[int(r)]),
                col_cov: float(rs.loc[r, col_cov]),
                col_dy: float(rs.loc[r, col_dy]) if np.isfinite(rs.loc[r, col_dy]) else np.nan,
                col_mean: float(rs.loc[r, col_mean]),
                col_abs: float(rs.loc[r, col_abs]) if col_abs in rs.columns else np.nan,
                "rule_summary": " ; ".join(selector_feature_summary(
                    self.model, P_r, self.feature_names, scaler=self.x_scaler, top_features=8
                )),
            })

        return pd.DataFrame(rows)

    def global_cluster_importance(self, split: str = "train", use_abs: bool = True) -> pd.DataFrame:
        """
        Global cluster importance by summing per-rule metrics inside each cluster.
        Requires rule_stats and clustering.
        """
        if self.cluster_labels is None or self.rule_stats is None or self.cluster_summary is None:
            raise RuntimeError("Need fit_posthoc(...) and fit_rule_stats(...), clustering available.")
        rs = self.rule_stats

        col_mean = f"{split}_mean_contrib_wz"
        col_abs = f"{split}_mean_abs_contrib_abs_wz"
        if col_abs not in rs.columns:
            raise RuntimeError(f"Column '{col_abs}' not found. Call add_mean_abs_contrib_to_rule_stats(...) first.")

        v_mean = rs[col_mean].values.astype(np.float64)
        v_abs = rs[col_abs].values.astype(np.float64)

        rows = []
        for g in sorted(np.unique(self.cluster_labels).tolist()):
            idx = np.where(self.cluster_labels == g)[0]
            rows.append({
                "cluster": int(g),
                "n_rules": int(len(idx)),
                f"{split}_sum_mean_contrib": float(v_mean[idx].sum()),
                f"{split}_sum_mean_abs_contrib": float(v_abs[idx].sum()),
            })

        df = pd.DataFrame(rows)
        key = f"{split}_sum_mean_abs_contrib" if use_abs else f"{split}_sum_mean_contrib"
        df = df.sort_values(key, ascending=False).reset_index(drop=True)

        df = df.merge(
            self.cluster_summary[["cluster", "sum_abs_w", "top_feature_ranges"]],
            on="cluster",
            how="left",
        )
        return df

    def global_feature_importance_mass_weighted(self, split: str = "train", topn: int = 20) -> pd.DataFrame:
        """
        Global per-feature importance computed as:
          signed = sum_r mean(w*z)_r * mass_r(feature)
          abs    = sum_r mean(|w*z|)_r * mass_r(feature)
        """
        if self.rule_stats is None or self.Rep_featmass is None:
            raise RuntimeError("Need fit_posthoc(...) and fit_rule_stats(...).")
        rs = self.rule_stats

        mass = np.asarray(self.Rep_featmass, dtype=np.float64)  # [R, D]
        col_mean = f"{split}_mean_contrib_wz"
        col_abs = f"{split}_mean_abs_contrib_abs_wz"
        if col_abs not in rs.columns:
            raise RuntimeError(f"Column '{col_abs}' not found. Call add_mean_abs_contrib_to_rule_stats(...) first.")

        mean_c = rs[col_mean].values.astype(np.float64)[:, None]
        mean_a = rs[col_abs].values.astype(np.float64)[:, None]

        signed = (mean_c * mass).sum(axis=0)
        absv = (mean_a * mass).sum(axis=0)

        df = pd.DataFrame({
            "feature": self.feature_names,
            f"{split}_mass_weighted_sum": signed,
            f"{split}_mass_weighted_abs_sum": absv,
        }).sort_values(f"{split}_mass_weighted_abs_sum", ascending=False).reset_index(drop=True)

        return df.head(int(topn))

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _to_scaled_1d(self, x_raw: np.ndarray, x_is_scaled: bool) -> np.ndarray:
        x_raw = np.asarray(x_raw, dtype=np.float32).ravel()
        if x_is_scaled:
            return x_raw
        if self.x_scaler is None:
            raise ValueError("x_scaler must be provided if x_is_scaled=False.")
        return self.x_scaler.transform(x_raw.reshape(1, -1)).astype(np.float32).ravel()


# ----------------------------
# Standalone helper functions
# ----------------------------

@torch.no_grad()
def predict_logits_softlogitand(model: SoftLogitAND, X_scaled: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """
    Predict logits for SoftLogitAND given scaled inputs X_scaled.
    """
    model.eval()
    dev = next(model.parameters()).device
    X_scaled = np.asarray(X_scaled, dtype=np.float32)
    out = []
    for i in range(0, len(X_scaled), int(batch_size)):
        xb = torch.tensor(X_scaled[i:i + int(batch_size)], device=dev)
        out.append(model(xb).detach().cpu().numpy().ravel())
    return np.concatenate(out, axis=0).astype(np.float64)


def _avg_pairwise_cosine_similarity(mat: np.ndarray) -> float:
    """
    Average pairwise cosine similarity among rows of mat.
    """
    mat = np.asarray(mat, dtype=np.float64)
    k = int(mat.shape[0])
    if k <= 1:
        return 0.0
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    X = mat / norms
    C = X @ X.T
    iu = np.triu_indices(k, k=1)
    return float(C[iu].mean())


def _coverage_abs_contrib(contrib_abs: np.ndarray, idx: Sequence[int]) -> float:
    """
    Coverage of absolute contributions: sum(abs contrib on selected rules) / sum(abs contrib on all rules).
    """
    contrib_abs = np.asarray(contrib_abs, dtype=np.float64).ravel()
    denom = float(contrib_abs.sum() + 1e-12)
    idx = np.asarray(list(idx), dtype=int)
    if idx.size == 0:
        return 0.0
    return float(contrib_abs[idx].sum() / denom)