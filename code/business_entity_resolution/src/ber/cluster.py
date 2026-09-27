"""Cluster-first resolution over S2/S3 records.

1. Pair model on S2/S3 <-> S2/S3 neighbour edges ("same real entity?").
   Labels come from train ground truth: both records owned by the same S1
   entity -> positive; owned by different S1 entities, or one owned and one
   unowned (a decoy of a hidden entity) -> negative; both unowned -> unknown.
2. Connected components over confident edges -> entity clusters.
3. Cluster-level features for each (S1, record) candidate, plus new candidates
   pulled in from clusters that confidently belong to the S1 record.
"""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import features as F

RR_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
                 feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
                 verbose=-1, num_threads=os.cpu_count())


def rr_features(right, a, b):
    t = lambda col, idx: right[col].take(idx).tolist()
    na, nb = t("name_vars", a), t("name_vars", b)
    fa, fb = t("name_full", a), t("name_full", b)
    aa, ab = t("addr_n", a), t("addr_n", b)
    f = {
        "nm_tset": F._cp(na, nb, fuzz.token_set_ratio),
        "nm_ratio": F._cp(na, nb, fuzz.ratio),
        "fn_ratio": F._cp(fa, fb, fuzz.ratio),
        "nm_jw": F._cp(na, nb, JaroWinkler.normalized_similarity),
        "ad_tset": F._cp(aa, ab, fuzz.token_set_ratio),
        "ad_tsort": F._cp(aa, ab, fuzz.token_sort_ratio),
        "ad_len_a": np.fromiter((len(x) for x in aa), np.float32, len(aa)),
        "ad_len_b": np.fromiter((len(x) for x in ab), np.float32, len(ab)),
        "nm_len_a": np.fromiter((len(x) for x in na), np.float32, len(na)),
        "nm_len_b": np.fromiter((len(x) for x in nb), np.float32, len(nb)),
    }
    da = [{w.lstrip("0") for w in x.split() if w.isdigit()} for x in aa]
    db = [{w.lstrip("0") for w in x.split() if w.isdigit()} for x in ab]
    f["num_jac"] = F._jac(da, db)
    f["num_conflict"] = np.fromiter((bool(x) and bool(y) and not (x & y) for x, y in zip(da, db)), np.float32, len(da))
    ea = right["entity_id"].take(a).str.startswith("S3").to_numpy()
    eb = right["entity_id"].take(b).str.startswith("S3").to_numpy()
    f["same_src"] = (ea == eb).astype(np.float32)
    return pd.DataFrame(f)


def dedup_edges(a, n):
    lo, hi = np.minimum(a, n), np.maximum(a, n)
    k = np.unique(lo.astype(np.int64) << 32 | hi.astype(np.int64))
    return (k >> 32).astype(np.int32), (k & 0xFFFFFFFF).astype(np.int32)


def train_rr(right, a, b, owner, n_train=4_000_000, seed=0):
    oa, ob = owner[a], owner[b]
    known = (oa >= 0) | (ob >= 0)
    idx = np.flatnonzero(known)
    rng = np.random.RandomState(seed)
    idx = rng.choice(idx, min(n_train, len(idx)), replace=False)
    X = pd.concat([rr_features(right, a[idx[s:s + 1_000_000]], b[idx[s:s + 1_000_000]])
                   for s in range(0, len(idx), 1_000_000)], ignore_index=True)
    y = (oa[idx] == ob[idx]) & (oa[idx] >= 0)
    ntr = int(len(y) * 0.9)
    m = lgb.train(RR_PARAMS, lgb.Dataset(X[:ntr], y[:ntr]), 400,
                  valid_sets=[lgb.Dataset(X[ntr:], y[ntr:])],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    pv = m.predict(X[ntr:])
    return m, pv, y[ntr:]


def predict_rr(m, right, a, b, chunk=1_000_000):
    p = np.zeros(len(a), np.float32)
    for s in range(0, len(a), chunk):
        p[s:s + chunk] = m.predict(rr_features(right, a[s:s + chunk], b[s:s + chunk]),
                                   num_threads=os.cpu_count())
    return p


def components(n, a, b, p, tau):
    k = p >= tau
    g = coo_matrix((np.ones(k.sum(), np.int8), (a[k], b[k])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return lab.astype(np.int32)


def cluster_features(c, lab):
    """c: DataFrame(i, j, p3). Adds cluster-level aggregates per row."""
    d = pd.DataFrame({"i": c["i"].values, "j": c["j"].values, "p": c["p3"].values})
    d["cl"] = lab[d["j"].values]
    size = np.bincount(lab)
    X = pd.DataFrame(index=d.index)
    X["cl_size"] = size[d["cl"].values].astype(np.float32)
    g = d.groupby(["i", "cl"])["p"]
    X["cl_n_in"] = g.transform("size").astype(np.float32)
    X["cl_sum_i"] = g.transform("sum").astype(np.float32)
    X["cl_max_i"] = g.transform("max").astype(np.float32)
    X["cl_other_mean"] = ((X["cl_sum_i"] - d["p"]) / np.maximum(X["cl_n_in"] - 1, 1)).astype(np.float32)
    X["cl_frac_in"] = (X["cl_n_in"] / X["cl_size"]).astype(np.float32)
    # competition at cluster level: best total evidence another S1 has for this cluster
    tot = d.groupby(["cl", "i"])["p"].sum().rename("t").reset_index()
    tot = tot.sort_values(["cl", "t"], ascending=[True, False])
    first = tot.drop_duplicates("cl")
    second = tot[tot.duplicated("cl")].drop_duplicates("cl")
    best = pd.Series(first["t"].values, index=first["cl"].values)
    best_i = pd.Series(first["i"].values, index=first["cl"].values)
    sec = pd.Series(second["t"].values, index=second["cl"].values)
    bi = best_i.reindex(d["cl"].values).values
    bt = best.reindex(d["cl"].values).fillna(0).values
    st = sec.reindex(d["cl"].values).fillna(0).values
    X["cl_is_best"] = (bi == d["i"].values).astype(np.float32)
    X["cl_margin"] = np.where(bi == d["i"].values, X["cl_sum_i"] - st, X["cl_sum_i"] - bt).astype(np.float32)
    return X


def pull_members(c, lab, min_cl=1.5):
    """cluster members not yet candidates of i, for clusters where i has strong evidence"""
    d = pd.DataFrame({"i": c["i"].values, "j": c["j"].values, "p": c["p3"].values})
    d["cl"] = lab[d["j"].values]
    s = d.groupby(["i", "cl"])["p"].sum().reset_index()
    s = s[s["p"] >= min_cl]
    order = np.argsort(lab, kind="stable")
    sl = lab[order]
    lo = np.searchsorted(sl, s["cl"].values, "left")
    hi = np.searchsorted(sl, s["cl"].values, "right")
    cnt = hi - lo
    ok = cnt <= 50
    s, lo, cnt = s[ok], lo[ok], cnt[ok]
    rep = np.repeat(np.arange(len(s)), cnt)
    pos = np.concatenate([np.arange(l, l + k) for l, k in zip(lo, cnt)]) if len(rep) else np.array([], int)
    new = pd.DataFrame({"i": s["i"].values[rep], "j": order[pos].astype(np.int32)})
    old = c["i"].values.astype(np.int64) * 100_000_000 + c["j"].values
    k = new["i"].values.astype(np.int64) * 100_000_000 + new["j"].values
    return new[~np.isin(k, old)].reset_index(drop=True)
