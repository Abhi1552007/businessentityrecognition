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
    I, J, S = B.block(s1, right, pool, max_df=MAX_DF, pre_k=PRE_K, log=log)
    c = pd.DataFrame({"i": I.astype(np.int32), "j": J.astype(np.int32), "kscore": S})
    c.sort_values(["i", "kscore"], ascending=[True, False], inplace=True, ignore_index=True)
    g = c.groupby("i")["kscore"]
    c["k_rank"] = g.cumcount().astype(np.float32)
    c["k_gap"] = (g.transform("max") - c["kscore"]).astype(np.float32)
    c["k_rel"] = (c["kscore"] / g.transform("max")).astype(np.float32)
    c["k_n"] = g.transform("size").astype(np.float32)
    return c


STAGE1_COLS = ["kscore", "k_rank", "k_gap", "k_rel", "k_n"]


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


def stage2_predict(model, s1, right, c, chunk_rows=6_000_000):
    out = np.empty(len(c), np.float32)
    for s in range(0, len(c), chunk_rows):
        # chunks cut on S1 boundaries so the per-S1 context stays exact
        e = min(len(c), s + chunk_rows)
        if e < len(c):
            e = int(np.searchsorted(c["i"].values, c["i"].values[e], side="left"))
        X = stage2_matrix(s1, right, c.iloc[s:e])
        out[s:e] = model.predict(X, num_threads=os.cpu_count())
        del X
        gc.collect()
    return out


def prune(c, p2, tau2):
    c = c.assign(p2=p2)
    c = c[c["p2"] >= tau2]
    c = c.sort_values(["i", "p2"], ascending=[True, False], ignore_index=True)
    c = c[c.groupby("i").cumcount() < K2].reset_index(drop=True)
    return c


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
    second = gj.transform(lambda v: np.sort(v.values)[-2] if len(v) > 1 else 0.0)
    X["j_margin"] = np.where(X["j_rank"] == 1, c["p2"] - second, c["p2"] - gj.transform("max")).astype(np.float32)
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
    top-m maximising the plug-in expected F0.5 (only among p>=tau)."""
    d = pd.DataFrame({"i": c["i"].values, "j": c["j"].values, "p": p})
    d = d[d["p"] > 0].sort_values(["i", "p"], ascending=[True, False])
    out = {}
    if mode == "thr":
        d = d[d["p"] >= tau]
        for i, g in d.groupby("i", sort=False):
            out[i] = g["j"].tolist()
        return out
    for i, g in d.groupby("i", sort=False):
        ps = g["p"].values
        js = g["j"].values
        exp_t = ps.sum()
        best_m, best_v = 0, np.prod(1 - ps)          # F=1 iff truly no match
        cum = 0.0
        for m in range(1, len(ps) + 1):
            if ps[m - 1] < tau:
                break
            cum += ps[m - 1]
            v = 1.25 * cum / (0.25 * exp_t + m)
            if v > best_v:
                best_m, best_v = m, v
        if best_m:
            out[i] = js[:best_m].tolist()
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
