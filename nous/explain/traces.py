from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Sequence, List, Dict, Any
from ..model import NousNet
from ..prototypes import ScaledPrototypeLayer
from ..explain.aggregator import AGG_NAMES
from ..explain.facts_desc import render_fact_descriptions

@torch.no_grad()
def get_last_block_static_metadata(model: NousNet, top_k_facts_override: Optional[int] = None):
    """
    Returns:
      - agg_weights: torch.Tensor [R, A] or None
      - facts_used: np.ndarray [R, K] (indices of inputs to the last block) or None
    """
    blk = model.blocks[-1]
    agg = None
    facts_used = None

    if hasattr(blk, "aggregator_logits"):
        agg = F.softmax(blk.aggregator_logits, dim=1).detach().cpu()

    # Fact indices used per rule depend on block type
    from ..rules.blocks import SimpleNousBlock
    from ..rules.softmax import SoftmaxRuleLayer
    from ..rules.sparse import SparseRuleLayer
    from ..rules.soft_fact import SoftFactRuleLayer

    if isinstance(blk, SimpleNousBlock):
        facts_used = blk.rule.idx.detach().cpu().numpy()  # [R, 2]
    elif isinstance(blk, SoftmaxRuleLayer):
        k = top_k_facts_override or getattr(blk, "top_k_facts", 2)
        fl = F.softmax(blk.fact_logits, dim=1)
        _, topk = torch.topk(fl, k=min(k, blk.input_dim), dim=1)
        facts_used = topk.detach().cpu().numpy()  # [R, k]
    elif isinstance(blk, SoftFactRuleLayer):
        k = top_k_facts_override or getattr(blk, "top_k_facts", 2)
        mask = blk._soft_k_mask().detach()
        _, topk = torch.topk(mask, k=min(k, blk.input_dim), dim=1)
        facts_used = topk.cpu().numpy()
    elif isinstance(blk, SparseRuleLayer):
        k = top_k_facts_override or getattr(blk, "top_k_facts", 2)
        prob = blk.hard_concrete.get_proba().detach()
        _, topk = torch.topk(prob, k=min(k, blk.input_dim), dim=1)
        facts_used = topk.cpu().numpy()
    return agg, facts_used

@torch.no_grad()
def _block_fact_mapping(block, top_k_per_rule: int = 2):
    """
    Returns np.ndarray [R, k] — input indices of the given block for each rule.
    """
    from ..rules.blocks import SimpleNousBlock
    from ..rules.softmax import SoftmaxRuleLayer
    from ..rules.sparse import SparseRuleLayer
    from ..rules.soft_fact import SoftFactRuleLayer

    if isinstance(block, SimpleNousBlock):
        return block.rule.idx.detach().cpu().numpy()
    elif isinstance(block, SoftmaxRuleLayer):
        fl = F.softmax(block.fact_logits, dim=1)
        _, topk = torch.topk(fl, k=min(top_k_per_rule, block.input_dim), dim=1)
        return topk.detach().cpu().numpy()
    elif isinstance(block, SoftFactRuleLayer):
        mask = block._soft_k_mask().detach()
        _, topk = torch.topk(mask, k=min(top_k_per_rule, block.input_dim), dim=1)
        return topk.cpu().numpy()
    elif isinstance(block, SparseRuleLayer):
        prob = block.hard_concrete.get_proba().detach()
        _, topk = torch.topk(prob, k=min(top_k_per_rule, block.input_dim), dim=1)
        return topk.cpu().numpy()
    else:
        return None

@torch.no_grad()
def trace_rule_to_base_facts(model: NousNet, rule_idx_last: int, top_k_per_step: int = 2) -> List[int]:
    """
    Trace a rule from the last block down to base β-facts.
    """
    selected = {int(rule_idx_last)}
    for b in reversed(range(len(model.blocks))):
        blk = model.blocks[b]
        mapping = _block_fact_mapping(blk, top_k_per_rule=top_k_per_step)
        if mapping is None:
            return []
        prev_sel = set()
        for r in selected:
            if r < 0 or r >= mapping.shape[0]:
                continue
            prev_sel.update([int(i) for i in mapping[r].tolist()])
        selected = prev_sel
    return sorted(selected)

@torch.no_grad()
def prototype_top_rules(model: NousNet, proto_id: int, top_k_rules: int = 10):
    """
    Returns list of (rule_idx, weight) from the prototype vector sorted by |weight|.
    """
    assert isinstance(model.head, ScaledPrototypeLayer), "Prototypes head is not enabled."
    P = model.head.prototypes.detach().cpu().numpy()
    v = P[proto_id]
    idx = np.argsort(-np.abs(v))[:top_k_rules]
    return [(int(i), float(v[i])) for i in idx]

@torch.no_grad()
def prototype_contribution_df(model: NousNet, x_sample, class_names: Optional[Sequence[str]] = None, top_k: int = 5, device=None):
    """
    For a single example — contribution of prototypes to the predicted class.
    """
    import pandas as pd

    assert isinstance(model.head, ScaledPrototypeLayer), "Prototypes head is not enabled."
    device = device or next(model.parameters()).device

    h = model.encode(np.array([x_sample]), device=device)
    d, act = model.head.compute_dist_act(h.to(device))
    act = act.squeeze(0).cpu().numpy()
    d = d.squeeze(0).cpu().numpy()

    W = model.head.prototype_class.detach().cpu().numpy()  # [M, C]
    probs, logits, _ = model.forward_explain(x_sample, device=device)
    c_hat = int(np.argmax(probs))
    w_c = W[:, c_hat]
    contrib = act * w_c

    Wsm = F.softmax(model.head.prototype_class, dim=1).detach().cpu().numpy()
    ent = (-Wsm * np.clip(np.log(Wsm + 1e-12), -50, 50)).sum(axis=1)
    primary = np.argmax(Wsm, axis=1)

    order = np.argsort(-np.abs(contrib))[:top_k]
    rows = []
    for j in order:
        rows.append({
            "proto": int(j),
            "distance": float(d[j]),
            "activation": float(act[j]),
            "w_c": float(w_c[j]),
            "contribution": float(contrib[j]),
            "primary_class": (class_names[primary[j]] if class_names is not None else int(primary[j])),
            "entropy": float(ent[j])
        })
    import pandas as pd
    return pd.DataFrame(rows)

@torch.no_grad()
def prototype_report_global(
    model: NousNet, X, y: Optional[Sequence[int]] = None, class_names: Optional[Sequence[str]] = None,
    top_k_rules: int = 8, top_k_facts_per_rule: int = 2, trace_to_base: bool = True,
    k_neighbors: int = 10, chunk_size: int = 2048, device=None
):
    """
    Global prototype report:
      - primary_class probabilities and entropy,
      - average activation, top1 frequency,
      - top rules and (optionally) tracing to base β-facts,
      - nearest training examples and labels (if y provided).
    """
    import pandas as pd
    assert isinstance(model.head, ScaledPrototypeLayer), "Prototypes head is not enabled."
    device = device or next(model.parameters()).device
    model.eval()

    H = model.encode(X, device=device)
    Hn = F.normalize(H, p=2, dim=1)
    P = model.head.get_params()
    Pn = P["prototypes_norm"]
    Wsm = P["class_probs"].cpu().numpy()
    Wraw = P["class_weights"].cpu().numpy()

    M = Pn.shape[0]
    N = Hn.shape[0]
    top1_count = np.zeros(M, dtype=int)
    mean_act = np.zeros(M, dtype=float)
    neigh_idx: list[list[int]] = [[] for _ in range(M)]

    for i in range(0, N, chunk_size):
        h_chunk = Hn[i:i+chunk_size].to(device)
        d = torch.cdist(h_chunk, Pn.to(device))
        act = torch.exp(-F.softplus(model.head.temperature) * d).cpu().numpy()
        top_idx = np.argmax(act, axis=1)
        for t in top_idx:
            top1_count[int(t)] += 1
        mean_act += act.sum(axis=0)
        k = min(k_neighbors, act.shape[0])
        for m in range(M):
            order = np.argsort(d[:, m].cpu().numpy())[:k]
            neigh_idx[m].extend((i + order).tolist())

    mean_act /= max(1, N)

    agg_last, facts_last = get_last_block_static_metadata(model, top_k_facts_override=top_k_facts_per_rule)
    fd = render_fact_descriptions(model, model.config['feature_names'])

    def agg_row_to_str(aw_row):
        if aw_row is None:
            return "AND (fixed)"
        A = int(aw_row.shape[0]) if hasattr(aw_row, "shape") else len(AGG_NAMES)
        names = AGG_NAMES[:A]
        parts = []
        for i, a in enumerate(aw_row.tolist()):
            if i < len(names) and a > 1e-6:
                parts.append(f"{float(a):.2f} {names[i]}")
        return " + ".join(parts) if parts else "∅"

    rule_meta = []
    for m in range(M):
        top_rules = prototype_top_rules(model, m, top_k_rules=top_k_rules)
        rules_desc = []
        for (r, w) in top_rules:
            r = int(r)
            agg_str = "AND (fixed)"
            if agg_last is not None and 0 <= r < agg_last.shape[0]:
                aw = agg_last[r]
                agg_str = agg_row_to_str(aw)
            facts_str = ""
            if trace_to_base:
                base_facts = trace_rule_to_base_facts(model, r, top_k_per_step=top_k_facts_per_rule)
                if base_facts:
                    facts_str = " | ".join([f"[F{int(fid)+1}] {fd[int(fid)]}" for fid in base_facts[:top_k_facts_per_rule]])
            elif facts_last is not None and r < facts_last.shape[0]:
                prev_units = facts_last[r]
                facts_str = ", ".join([f"Unit{int(u)}" for u in prev_units])
            rules_desc.append(dict(rule=int(r), weight=float(w), aggregators=agg_str, facts=facts_str))
        rule_meta.append(rules_desc)

    rows = []
    for m in range(M):
        primary = int(np.argmax(Wsm[m]))
        rows.append({
            "proto": m,
            "primary_class": (class_names[primary] if class_names is not None else primary),
            "class_probs": Wsm[m].tolist(),
            "class_weights": Wraw[m].tolist(),
            "class_entropy": float((-Wsm[m] * np.log(Wsm[m] + 1e-12)).sum()),
            "mean_activation": float(mean_act[m]),
            "top1_count": int(top1_count[m]),
            "neighbors": neigh_idx[m][:k_neighbors],
            "neighbor_labels": [int(y[idx]) for idx in neigh_idx[m][:k_neighbors]] if y is not None else None,
            "top_rules_meta": rule_meta[m]
        })
    return pd.DataFrame(rows)

@torch.no_grad()
def describe_prototype(model: NousNet, proto_id: int, feature_names: Sequence[str], class_names: Optional[Sequence[str]] = None,
                       top_k_rules: int = 8, top_k_facts_per_rule: int = 2, trace_to_base: bool = True) -> str:
    """
    Human-readable description for a single prototype:
      - class affinity (weights),
      - top rules and aggregator mixtures,
      - base β-facts (if traced).
    """
    assert isinstance(model.head, ScaledPrototypeLayer), "Prototypes head is not enabled."

    P = model.head.get_params()
    Wsm = P["class_probs"].cpu().numpy()
    m = int(proto_id)
    primary = int(np.argmax(Wsm[m]))
    primary_name = class_names[primary] if class_names is not None else f"Class {primary}"

    lines = []
    lines.append(f"Prototype #{m}")
    lines.append(f"  Primary class: {primary_name} | probs={np.round(Wsm[m], 3).tolist()} | entropy={float((-Wsm[m]*np.log(Wsm[m]+1e-12)).sum()):.3f}")
    lines.append("  Top rules:")
    agg_last, _ = get_last_block_static_metadata(model, top_k_facts_override=top_k_facts_per_rule)
    fd = render_fact_descriptions(model, feature_names)

    def agg_row_to_str(aw_row):
        if aw_row is None:
            return "AND (fixed)"
        A = int(aw_row.shape[0]) if hasattr(aw_row, "shape") else len(AGG_NAMES)
        names = AGG_NAMES[:A]
        parts = []
        for i, a in enumerate(aw_row.tolist()):
            if i < len(names) and a > 1e-6:
                parts.append(f"{float(a):.2f} {names[i]}")
        return " + ".join(parts) if parts else "∅"

    for (r, w) in prototype_top_rules(model, m, top_k_rules=top_k_rules):
        r = int(r)
        agg_str = "AND (fixed)"
        if agg_last is not None and 0 <= r < agg_last.shape[0]:
            aw = agg_last[r]
            agg_str = agg_row_to_str(aw)

        facts_str = ""
        if trace_to_base:
            base_facts = trace_rule_to_base_facts(model, r, top_k_per_step=top_k_facts_per_rule)
            if base_facts:
                facts_str = " | ".join([f"[F{int(fid)+1}] {fd[int(fid)]}" for fid in base_facts[:top_k_facts_per_rule]])

        lines.append(f"    • R{r+1}: weight={w:+.3f} | {agg_str}")
        if facts_str:
            lines.append(f"        {facts_str}")
    return "\n".join(lines)