"""End-to-end pipeline: blocking -> stage-2 candidate pruner -> stage-3 matcher.

Stage 1 (blocking.py)  key-based inverted index, top ``PRE_K`` per S1 record.
Stage 2 (pruner)       LightGBM on cheap similarity features; keeps pairs with
                       p2 >= tau2 (at most ``K2`` per S1). Its output is the
                       candidate set written to candidate_pairs.tsv and the
                       only pairs the matcher ever scores.
Stage 3 (matcher)      LightGBM on rich pair features + competition context
                       (how the pair ranks among the S1 record's candidates
                       and among the S2/S3 record's competing S1 records).
Decision               each S2/S3 record goes to at most one S1 record (its
                       argmax); per S1, the subset maximising expected F0.5.
"""
import gc
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import blocking as B
from . import features as F

PRE_K = 40
MAX_DF = 500
K2 = 12

LGB_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, num_threads=os.cpu_count())

NAME_COLS = ["entity_id", "name_vars", "name_full", "addr_n", "comps", "ctry"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_tables(cache_dir, split):
    rd = lambda k: pd.read_parquet(os.path.join(cache_dir, f"{split}_source{k}.parquet"),
                                   columns=NAME_COLS, dtype_backend="pyarrow")
    s1 = rd(1)
    right = pd.concat([rd(2), rd(3)], ignore_index=True)
    return s1, right


# ---------------------------------------------------------------- stage 1
def stage1(s1, right, pool):
    """-> DataFrame(i, j, kscore) sorted by i then descending kscore."""
    I, J, S = B.block(s1, right, pool, max_df=MAX_DF, pre_k=PRE_K, log=log)
    # each shard returns rows grouped by i with descending score; a stable
    # sort on i keeps that order globally
    o = np.argsort(I, kind="stable")
    c = pd.DataFrame({"i": I[o].astype(np.int32), "j": J[o].astype(np.int32),
                      "kscore": S[o].astype(np.float32)})
    return c


STAGE1_COLS = ["kscore", "k_rank", "k_gap", "k_rel", "k_n"]


def add_k_ctx(c):
    """rank/gap of the blocking score within each S1 record (c sorted by i, score desc)."""
    I, S = c["i"].values, c["kscore"].values
    new = np.empty(len(I), bool)
    new[0] = True
    np.not_equal(I[1:], I[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    cnt = np.diff(np.append(starts, len(I)))
    gid = np.cumsum(new) - 1
    top = S[starts][gid]
    c["k_rank"] = (np.arange(len(I)) - starts[gid]).astype(np.float32)
    c["k_gap"] = (top - S).astype(np.float32)
    c["k_rel"] = (S / np.maximum(top, 1e-6)).astype(np.float32)
    c["k_n"] = cnt[gid].astype(np.float32)
    return c


def stage2_matrix(s1, right, c):
    X = F.cheap_features(s1, right, c["i"].values, c["j"].values)
    for k in STAGE1_COLS:
        X[k] = c[k].values
    # name-score context among this S1 record's blocked candidates
    tmp = pd.DataFrame({"i": c["i"].values, "v": X["c_nm_tset"].values + X["c_ad_tset"].values})
    g = tmp.groupby("i")["v"]
    X["c_ctx_gap"] = (g.transform("max") - tmp["v"]).values.astype(np.float32)
    X["c_ctx_rank"] = g.rank(ascending=False, method="first").values.astype(np.float32)
    return X


def fit_lgb(X, y, rounds, valid=None):
    ds = lgb.Dataset(X, y, free_raw_data=True)
    return lgb.train(LGB_PARAMS, ds, num_boost_round=rounds)


def _chunks(c, chunk_rows):
    """row ranges cut on S1 boundaries (c sorted by i)"""
    iv = c["i"].values
    s = 0
    while s < len(c):
        e = min(len(c), s + chunk_rows)
        if e < len(c):
            e = int(np.searchsorted(iv, iv[e], side="left"))
            if e <= s:
                e = int(np.searchsorted(iv, iv[s], side="right"))
        yield s, e
        s = e


def stage2_prune(model, s1, right, c, tau2, override=None, chunk_rows=3_000_000):
    """score every blocked pair with the stage-2 model and keep p2 >= tau2
    (at most K2 per S1). ``override``: optional float array aligned with c whose
    non-NaN entries replace the model score (out-of-fold predictions in training)."""
    out = []
    for s, e in _chunks(c, chunk_rows):
        cc = add_k_ctx(c.iloc[s:e].reset_index(drop=True))
        p2 = model.predict(stage2_matrix(s1, right, cc), num_threads=os.cpu_count()).astype(np.float32)
        if override is not None:
            ov = override[s:e]
            p2 = np.where(np.isnan(ov), p2, ov)
        keep = p2 >= tau2
        cc = cc[keep].copy()
        cc["p2"] = p2[keep]
        cc.sort_values(["i", "p2"], ascending=[True, False], inplace=True, ignore_index=True)
        out.append(cc[cc.groupby("i").cumcount() < K2])
        gc.collect()
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- stage 3
def context(c):
    """competition features based on stage-2 probabilities (all S1 present)."""
    X = pd.DataFrame(index=c.index)
    gi = c.groupby("i")["p2"]
    gj = c.groupby("j")["p2"]
    X["p2"] = c["p2"].values
    X["i_rank"] = gi.rank(ascending=False, method="first").astype(np.float32)
    X["i_gap"] = (gi.transform("max") - c["p2"]).astype(np.float32)
    X["i_n"] = gi.transform("size").astype(np.float32)
    X["i_sum"] = gi.transform("sum").astype(np.float32)
    X["j_rank"] = gj.rank(ascending=False, method="first").astype(np.float32)
    X["j_gap"] = (gj.transform("max") - c["p2"]).astype(np.float32)
    X["j_n"] = gj.transform("size").astype(np.float32)
    # best competing score from another S1 record for the same S2/S3 record
    d = pd.DataFrame({"j": c["j"].values, "p": c["p2"].values})
    d["r"] = X["j_rank"].values
    second = d[d["r"] == 2].set_index("j")["p"]
    sec = second.reindex(d["j"].values).fillna(0.0).values
    top = gj.transform("max").values
    X["j_margin"] = np.where(d["r"].values == 1, d["p"].values - sec, d["p"].values - top).astype(np.float32)
    return X


def stage3_matrix(s1, right, c, ctx):
    X = F.pair_features(s1, right, c["i"].values, c["j"].values)
    for k in STAGE1_COLS:
        X[k] = c[k].values
    for k in ctx.columns:
        X[k] = ctx[k].values
    return X


def stage3_context2(c, p3):
    """second-order context on stage-3 probabilities (stacked stage 4)."""
    d = pd.DataFrame({"i": c["i"].values, "j": c["j"].values, "p": p3})
    gi, gj = d.groupby("i")["p"], d.groupby("j")["p"]
    X = pd.DataFrame({
        "p3": p3,
        "p3_i_rank": gi.rank(ascending=False, method="first").values,
        "p3_i_gap": (gi.transform("max") - d["p"]).values,
        "p3_i_sum": gi.transform("sum").values,
        "p3_j_rank": gj.rank(ascending=False, method="first").values,
        "p3_j_gap": (gj.transform("max") - d["p"]).values,
        "p3_j_sum": gj.transform("sum").values,
    }).astype(np.float32)
    return X


# ---------------------------------------------------------------- decision
def exclusive(c, p):
    """each S2/S3 record keeps only its best S1 record"""
    d = pd.DataFrame({"i": c["i"].values, "j": c["j"].values, "p": p})
    best = d.groupby("j")["p"].transform("max")
    return np.where(d["p"].values >= best.values, p, 0.0)


def select(c, p, tau, mode="exp"):
    """Return dict i -> list of j. mode 'thr': p>=tau. mode 'exp': per S1 choose
    the top-m set maximising the plug-in expected F0.5 (only among p>=tau);
    the empty set scores P(no match) = prod(1-p)."""
    I, J, p = np.asarray(c["i"].values), np.asarray(c["j"].values), np.asarray(p, np.float64)
    o = np.lexsort((-p, I))
    I, J, p = I[o], J[o], p[o]
    keep = p > 0
    I, J, p = I[keep], J[keep], p[keep]
    if len(I) == 0:
        return {}
    new = np.empty(len(I), bool)
    new[0] = True
    np.not_equal(I[1:], I[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    gid = np.cumsum(new) - 1
    m = np.arange(len(I)) - starts[gid] + 1                      # set size if cut here
    if mode == "thr":
        sel = p >= tau
    else:
        cum = np.cumsum(p)
        cum = cum - np.concatenate([[0.0], cum[starts[1:] - 1]])[gid]
        tot = np.add.reduceat(p, starts)[gid]
        val = 1.25 * cum / (0.25 * tot + m)
        val[p < tau] = -1.0
        # a prefix is only valid if every earlier item also passed tau (sorted desc -> automatic)
        empty = np.exp(np.add.reduceat(np.log1p(-np.minimum(p, 1 - 1e-9)), starts))
        best_val = np.maximum.reduceat(val, starts)
        # first position reaching the group max
        is_best = val == best_val[gid]
        first_best = np.full(len(starts), -1)
        idx = np.flatnonzero(is_best)
        g_of = gid[idx]
        _, first = np.unique(g_of, return_index=True)
        first_best[g_of[first]] = idx[first]
        best_m = np.where((best_val > empty) & (first_best >= 0), m[np.maximum(first_best, 0)], 0)
        sel = m <= best_m[gid]
    out = {}
    Is, Js = I[sel], J[sel]
    if len(Is):
        br = np.flatnonzero(np.r_[True, Is[1:] != Is[:-1]])
        for k, st in enumerate(br):
            en = br[k + 1] if k + 1 < len(br) else len(Is)
            out[int(Is[st])] = Js[st:en].tolist()
    return out


def macro_f05(pred, truth, all_i):
    tot = 0.0
    for i in all_i:
        t = truth.get(i, set())
        p = set(pred.get(i, ()))
        if not t and not p:
            tot += 1
            continue
        tp = len(t & p)
        if tp == 0:
            continue
        pr, rc = tp / len(p), tp / len(t)
        tot += 1.25 * pr * rc / (0.25 * pr + rc)
    return tot / len(all_i)
