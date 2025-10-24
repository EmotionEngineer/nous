from __future__ import annotations
import os
import re
import textwrap
import importlib.util
from types import ModuleType
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

from ..model import NousNet

def export_numpy_inference(model: NousNet, file_path: str = "nous_numpy_infer.py") -> str:
    """
    Export a NousNet instance to a self-contained NumPy inference Python module.

    The generated module exposes:
      - a dict P with parameters and metadata,
      - predict(X, return_logits=False) using only numpy.

    Returns the code as a string and writes it to file_path if provided.
    """
    model.eval()
    cfg = model.config

    def npy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    def tolist(x, dtype=np.float32):
        return npy(x).astype(dtype).tolist()

    use_cals = model.calibrators is not None
    calibrators = []
    if use_cals:
        for c in model.calibrators:
            with torch.no_grad():
                deltas_sp = F.softplus(c.deltas).cpu().numpy()
                cum = np.cumsum(deltas_sp)
                cum = np.concatenate([np.array([0.0], dtype=np.float32), cum.astype(np.float32)], axis=0)
                calibrators.append(dict(
                    input_min=float(c.input_min),
                    input_max=float(c.input_max),
                    num_bins=int(c.num_bins),
                    bias=float(c.bias.item()),
                    cum=cum.tolist()
                ))

    Lw, Rw, th, k, nu = model.fact.get_rule_parameters()
    fact_dict = dict(
        L=Lw.astype(np.float32).tolist(),
        R=Rw.astype(np.float32).tolist(),
        th=th.astype(np.float32).tolist(),
        k=k.astype(np.float32).tolist(),
        nu=nu.astype(np.float32).tolist()
    )

    blocks = []
    for blk in model.blocks:
        ln_w = tolist(blk.norm.weight)
        ln_b = tolist(blk.norm.bias)
        proj_W = None
        if not isinstance(blk.proj, torch.nn.Identity):
            proj_W = tolist(blk.proj.weight)

        from ..rules.blocks import SimpleNousBlock
        from ..rules.softmax import SoftmaxRuleLayer
        from ..rules.sparse import SparseRuleLayer

        if isinstance(blk, SimpleNousBlock):
            idx_pairs = blk.rule.idx.detach().cpu().numpy().astype(np.int64).tolist()
            rule_w = tolist(blk.rule.weight)
            blocks.append(dict(
                kind="simple",
                proj_W=proj_W,
                ln_w=ln_w, ln_b=ln_b,
                idx_pairs=idx_pairs,
                rule_w=rule_w
            ))

        elif isinstance(blk, SoftmaxRuleLayer):
            with torch.no_grad():
                fl = F.softmax(blk.fact_logits, dim=1).cpu().numpy()
                kf = int(min(blk.top_k_facts, blk.input_dim))
                topk_idx = np.argpartition(fl, -kf, axis=1)[:, -kf:]
                mask = np.zeros_like(fl, dtype=np.float32)
                rows = np.arange(fl.shape[0])[:, None]
                mask[rows, topk_idx] = 1.0

                agg_w = F.softmax(blk.aggregator_logits, dim=1).cpu().numpy().astype(np.float32)  # [R, 3]
                rule_strength = torch.sigmoid(blk.rule_strength_raw).cpu().numpy().astype(np.float32)
                blocks.append(dict(
                    kind="softmax",
                    proj_W=proj_W,
                    ln_w=ln_w, ln_b=ln_b,
                    mask=mask.tolist(),
                    agg_w=agg_w.tolist(),
                    rule_strength=rule_strength.tolist(),
                    top_k_rules=int(blk.top_k_rules)
                ))
        elif isinstance(blk, SoftFactRuleLayer):
            with torch.no_grad():
                mask = blk._soft_k_mask().cpu().numpy().astype(np.float32)
                agg_w = F.softmax(blk.aggregator_logits, dim=1).cpu().numpy().astype(np.float32)
                rule_strength = torch.sigmoid(blk.rule_strength_raw).cpu().numpy().astype(np.float32)
                blocks.append(dict(
                    kind="soft_fact",
                    proj_W=proj_W,
                    ln_w=ln_w, ln_b=ln_b,
                    mask=mask.tolist(),
                    agg_w=agg_w.tolist(),
                    rule_strength=rule_strength.tolist(),
                    top_k_rules=int(blk.top_k_rules)
                ))
        elif isinstance(blk, SparseRuleLayer):
            with torch.no_grad():
                beta = blk.hard_concrete.beta.detach().cpu().numpy()
                mask = (1.0 / (1.0 + np.exp(-beta)) > 0.5).astype(np.float32)  # eval behavior
                agg_w = F.softmax(blk.aggregator_logits, dim=1).cpu().numpy().astype(np.float32)  # [R, 4]
                rule_strength = torch.sigmoid(blk.rule_strength_raw).cpu().numpy().astype(np.float32)
                blocks.append(dict(
                    kind="sparse",
                    proj_W=proj_W,
                    ln_w=ln_w, ln_b=ln_b,
                    mask=mask.tolist(),
                    agg_w=agg_w.tolist(),
                    rule_strength=rule_strength.tolist(),
                    top_k_rules=int(blk.top_k_rules)
                ))
        else:
            raise ValueError(f"Unknown block type: {type(blk)}")

    if isinstance(model.head, torch.nn.Linear):
        with torch.no_grad():
            W = model.head.weight.detach().cpu().numpy().astype(np.float32)
            b = model.head.bias.detach().cpu().numpy().astype(np.float32)
        head = dict(kind="linear", W=W.tolist(), b=b.tolist())
    else:
        from ..prototypes import ScaledPrototypeLayer
        assert isinstance(model.head, ScaledPrototypeLayer)
        with torch.no_grad():
            Pm = model.head.prototypes.detach()
            Pn = F.normalize(Pm, p=2, dim=1).cpu().numpy().astype(np.float32)
            W = model.head.prototype_class.detach().cpu().numpy().astype(np.float32)
            tau = float(F.softplus(model.head.temperature).item())
        head = dict(kind="prototypes", P_norm=Pn.tolist(), W=W.tolist(), tau=tau)

    P_dict = dict(
        task=cfg['task_type'],
        use_calibrators=bool(use_cals),
        calibrators=calibrators,
        fact=fact_dict,
        blocks=blocks,
        head=head
    )

    code = f"""# Auto-generated NumPy inference for NousNet
# This file is self-contained and requires only numpy.
import numpy as np

P = {repr(P_dict)}

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def logsigmoid(z):
    # stable: -log(1+exp(-z))
    return -np.log1p(np.exp(-z))

def layernorm(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + eps)
    return xhat * gamma + beta

def calibrate(X):
    if not P['use_calibrators']:
        return X.astype(np.float32)
    X = X.astype(np.float32)
    Xc = np.empty_like(X, dtype=np.float32)
    for j, c in enumerate(P['calibrators']):
        xmin, xmax = float(c['input_min']), float(c['input_max'])
        nb = int(c['num_bins'])
        cum = np.array(c['cum'], dtype=np.float32)
        bias = float(c['bias'])
        xj = X[:, j]
        xn = (xj - xmin) / (xmax - xmin + 1e-8)
        xn = np.clip(xn, 0.0, 1.0)
        t = xn * nb
        bin_idx = np.floor(t).astype(np.int32)
        bin_idx = np.clip(bin_idx, 0, nb-1)
        left = bias + cum[bin_idx]
        right = bias + cum[bin_idx + 1]
        frac = t - bin_idx.astype(np.float32)
        Xc[:, j] = left + frac * (right - left)
    return Xc

def beta_facts(Xc):
    L = np.array(P['fact']['L'], dtype=np.float32)
    R = np.array(P['fact']['R'], dtype=np.float32)
    th = np.array(P['fact']['th'], dtype=np.float32)
    k  = np.array(P['fact']['k'], dtype=np.float32)
    nu = np.array(P['fact']['nu'], dtype=np.float32)
    diff = Xc @ L.T - Xc @ R.T - th
    z = k * diff
    log_sig = logsigmoid(z)
    log_beta = nu * log_sig
    log_beta = np.maximum(log_beta, -80.0)
    return np.exp(log_beta).astype(np.float32)

def run_block(block, H):
    kind = block['kind']
    if block['proj_W'] is None:
        proj = H
    else:
        Wp = np.array(block['proj_W'], dtype=np.float32)
        proj = H @ Wp.T

    if kind == 'simple':
        idx = np.array(block['idx_pairs'], dtype=np.int64)
        w  = np.array(block['rule_w'], dtype=np.float32)
        rs = sigmoid(w)
        f1 = H[:, idx[:,0]]
        f2 = H[:, idx[:,1]]
        rule_act = (f1 * f2) * rs
        pre = proj + rule_act

    elif kind == 'softmax':
        mask = np.array(block['mask'], dtype=np.float32)  # [R,F]
        sel = H[:, None, :] * mask[None, :, :]
        and_agg = np.prod(sel + (1.0 - mask)[None, :, :], axis=2)
        or_agg  = 1.0 - np.prod((1.0 - sel) * mask[None, :, :] + (1.0 - mask)[None, :, :], axis=2)
        denom = np.maximum(mask.sum(axis=1), 1e-8)  # [R]
        kofn  = (sel.sum(axis=2)) / denom[None, :]
        agg_w = np.array(block['agg_w'], dtype=np.float32)  # [R,3]
        aggs = np.stack([and_agg, or_agg, kofn], axis=2)
        mixed = (aggs * agg_w[None, :, :]).sum(axis=2)
        rs = np.array(block['rule_strength'], dtype=np.float32)
        rule_act = mixed * rs[None, :]
        R = mask.shape[0]
        k_rules = int(block['top_k_rules'])
        if k_rules < R:
            gate = np.zeros_like(rule_act, dtype=np.float32)
            idx_top = np.argpartition(rule_act, -k_rules, axis=1)[:, -k_rules:]
            for i in range(rule_act.shape[0]):
                gate[i, idx_top[i]] = 1.0
            rule_act = rule_act * gate
        pre = proj + rule_act

    elif kind == 'soft_fact':
        mask = np.array(block['mask'], dtype=np.float32)  # [R,F] — soft!
        sel = H[:, None, :] * mask[None, :, :]
        and_agg = np.prod(sel + (1.0 - mask)[None, :, :], axis=2)
        or_agg  = 1.0 - np.prod((1.0 - sel) + 1e-8, axis=2)
        denom = np.maximum(mask.sum(axis=1), 1e-8)  # [R]
        kofn  = (sel.sum(axis=2)) / denom[None, :]
        agg_w = np.array(block['agg_w'], dtype=np.float32)  # [R,3]
        aggs = np.stack([and_agg, or_agg, kofn], axis=2)
        mixed = (aggs * agg_w[None, :, :]).sum(axis=2)
        rs = np.array(block['rule_strength'], dtype=np.float32)
        rule_act = mixed * rs[None, :]
        R = mask.shape[0]
        k_rules = int(block['top_k_rules'])
        if k_rules < R:
            gate = np.zeros_like(rule_act, dtype=np.float32)
            idx_top = np.argpartition(rule_act, -k_rules, axis=1)[:, -k_rules:]
            for i in range(rule_act.shape[0]):
                gate[i, idx_top[i]] = 1.0
            rule_act = rule_act * gate
        pre = proj + rule_act

    elif kind == 'sparse':
        mask = np.array(block['mask'], dtype=np.float32)  # [R,F]
        sel = H[:, None, :] * mask[None, :, :]
        and_agg = np.prod(sel + (1.0 - mask)[None, :, :], axis=2)
        or_agg  = 1.0 - np.prod((1.0 - sel) * mask[None, :, :] + (1.0 - mask)[None, :, :], axis=2)
        denom = np.maximum(mask.sum(axis=1), 1e-8)  # [R]
        kofn  = (sel.sum(axis=2)) / denom[None, :]
        not_agg = 1.0 - kofn
        agg_w = np.array(block['agg_w'], dtype=np.float32)  # [R,4]
        aggs = np.stack([and_agg, or_agg, kofn, not_agg], axis=2)
        mixed = (aggs * agg_w[None, :, :]).sum(axis=2)
        rs = np.array(block['rule_strength'], dtype=np.float32)
        rule_act = mixed * rs[None, :]
        R = mask.shape[0]
        k_rules = int(block['top_k_rules'])
        if k_rules < R:
            gate = np.zeros_like(rule_act, dtype=np.float32)
            idx_top = np.argpartition(rule_act, -k_rules, axis=1)[:, -k_rules:]
            for i in range(rule_act.shape[0]):
                gate[i, idx_top[i]] = 1.0
            rule_act = rule_act * gate
        pre = proj + rule_act

    else:
        raise ValueError("Unknown block kind: " + str(kind))

    gamma = np.array(block['ln_w'], dtype=np.float32)
    beta  = np.array(block['ln_b'], dtype=np.float32)
    return layernorm(pre, gamma, beta)

def head_forward(H):
    head = P['head']
    if head['kind'] == 'linear':
        W = np.array(head['W'], dtype=np.float32)
        b = np.array(head['b'], dtype=np.float32)
        return H @ W.T + b
    elif head['kind'] == 'prototypes':
        Pn = np.array(head['P_norm'], dtype=np.float32)
        W = np.array(head['W'], dtype=np.float32)
        Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
        dot = Hn @ Pn.T
        d = np.sqrt(np.clip(2.0 - 2.0 * dot, 1e-12, None))
        tau = float(head['tau'])
        act = np.exp(-tau * d)
        return act @ W
    else:
        raise ValueError("Unknown head kind: " + str(head['kind']))

def predict(X, return_logits=False):
    X = np.array(X, dtype=np.float32)
    Xc = calibrate(X) if P['use_calibrators'] else X
    H = beta_facts(Xc)
    for blk in P['blocks']:
        H = run_block(blk, H)
    logits = head_forward(H)
    if P['task'] == 'classification':
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        return (probs, logits) if return_logits else probs
    else:
        return logits.reshape(-1)
"""
    code = textwrap.dedent(code)
    if file_path is not None:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
    return code

def slugify(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]+', '_', s).strip('_').lower()

def load_numpy_module(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("nous_numpy_infer_mod", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def _softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def _kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=1)

def _js_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    m = 0.5*(p+q)
    return 0.5*_kl_div(p, m, eps) + 0.5*_kl_div(q, m, eps)

def validate_numpy_vs_torch(
    model: NousNet, npmod: ModuleType, X, task: str, n: int = 512,
    tol_prob_max: float = 1e-3, tol_prob_mean: float = 2e-4, tol_l1_mean: float = 3e-4, tol_js_mean: float = 5e-6,
    tol_logit_centered: float = 1e-3,
    tol_reg_max: float = 1e-4, tol_reg_mean: float = 2e-5
) -> dict:
    """
    Probability-first validation:
      - classification: PASS if prob metrics and centered-logit diff are within tolerances.
      - regression: PASS if absolute prediction diffs within tolerances.
    """
    model.eval()
    device = next(model.parameters()).device
    Xs = X[:min(len(X), n)].astype(np.float32)

    if task == "classification":
        with torch.no_grad():
            torch_logits = model(torch.tensor(Xs, device=device)).cpu().numpy()
            torch_probs = _softmax_np(torch_logits)

        np_probs, np_logits = npmod.predict(Xs, return_logits=True)
        np_probs = np.asarray(np_probs)
        np_logits = np.asarray(np_logits)

        torch_pred = np.argmax(torch_probs, axis=1)
        numpy_pred = np.argmax(np_probs, axis=1)
        fidelity = float((torch_pred == numpy_pred).mean())

        dprob = np.abs(torch_probs - np_probs)
        max_dprob  = float(dprob.max())
        mean_dprob = float(dprob.mean())
        l1_per_sample = np.sum(dprob, axis=1)
        l1_mean = float(l1_per_sample.mean())

        js = _js_div(torch_probs, np_probs)
        js_mean = float(js.mean())

        tl = torch_logits - torch_logits.mean(axis=1, keepdims=True)
        nl = np_logits    - np_logits.mean(axis=1, keepdims=True)
        dlog = np.abs(tl - nl)
        max_dlog_centered = float(dlog.max())

        passed = (
            max_dprob <= tol_prob_max and
            mean_dprob <= tol_prob_mean and
            l1_mean <= tol_l1_mean and
            js_mean <= tol_js_mean and
            max_dlog_centered <= tol_logit_centered
        )
        return {
            "fidelity_info": fidelity,
            "max_abs_prob_diff": max_dprob,
            "mean_abs_prob_diff": mean_dprob,
            "mean_L1_prob": l1_mean,
            "mean_JS": js_mean,
            "max_abs_centered_logit_diff": max_dlog_centered,
            "pass": passed
        }

    else:
        with torch.no_grad():
            torch_pred = model(torch.tensor(Xs, device=device)).cpu().numpy().ravel()
        np_pred = np.asarray(npmod.predict(Xs)).ravel()

        dp = np.abs(torch_pred - np_pred)
        max_dp = float(dp.max()) if dp.size else 0.0
        mean_dp = float(dp.mean()) if dp.size else 0.0

        passed = (max_dp <= tol_reg_max and mean_dp <= tol_reg_mean)
        return {
            "max_abs_pred_diff": max_dp,
            "mean_abs_pred_diff": mean_dp,
            "pass": passed
        }

def export_and_validate(model: NousNet, name: str, X, base_path: str = "./exports") -> dict:
    os.makedirs(base_path, exist_ok=True)
    file_path = os.path.join(base_path, f"nous_numpy_infer_{slugify(name)}.py")
    export_numpy_inference(model, file_path=file_path)
    npmod = load_numpy_module(file_path)
    return validate_numpy_vs_torch(model, npmod, X, model.config['task_type'])