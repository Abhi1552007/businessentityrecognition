"""Cluster-first stage (stage 4).

train: python cluster_run.py train <work> <gt>
test : python cluster_run.py test  <work> <out_dir>
"""
import gc
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

import run
from ber import cluster as C
from ber import features as F
from ber import pipeline as P

mode, W = sys.argv[1], sys.argv[2]
split = "train" if mode == "train" else "test"
s1, right = P.load_tables(W + "/cache", split)
z = np.load(f"{W}/neighbours_{split}.npz")
a, b = C.dedup_edges(z["a"], z["n"])
del z
P.log("edges", len(a))

if mode == "train":
    gt = sys.argv[3]
    p3 = pd.read_parquet(f"{W}/p3_train.parquet")
    _, (ti, tj) = run.label_pairs(p3.head(1), s1, right, gt)
    owner = np.full(len(right), -1, np.int64)
    owner[tj] = ti
    m, pv, yv = C.train_rr(right, a, b, owner)
    m.save_model(f"{W}/rr.txt")
    from sklearn.metrics import roc_auc_score
    P.log("rr AUC", roc_auc_score(yv, pv))
else:
    p3 = pd.read_parquet(f"{W}/p3_test.parquet")
    m = lgb.Booster(model_file=f"{W}/rr.txt")

prr = C.predict_rr(m, right, a, b)
cfg = json.load(open(f"{W}/stage4.json")) if mode != "train" else {}
taus = [cfg["tau_rr"]] if mode != "train" else [0.8, 0.9, 0.97]

if mode == "train":
    oa, ob = owner[a], owner[b]
    for t in taus:
        lab = C.components(len(right), a, b, prr, t)
        # purity: among owned records, share whose cluster mixes owners / includes decoys
        d = pd.DataFrame({"cl": lab, "o": owner})
        g = d.groupby("cl")["o"]
        mixed = (g.transform("nunique") > 1).to_numpy()
        P.log(f"tau_rr={t}: clusters={lab.max()+1}, records in mixed clusters={mixed.mean():.4f}, "
              f"max size={np.bincount(lab).max()}")


def build(c, lab):
    new = C.pull_members(c, lab)
    new["p2"] = np.float32(0)
    new["p3"] = np.float32(0)
    new["expanded"] = np.float32(2)
    cc = pd.concat([c[["i", "j", "p2", "p3", "expanded"] + (["y"] if "y" in c else [])], new], ignore_index=True)
    cc = cc.sort_values(["i", "p3"], ascending=[True, False], ignore_index=True)
    X = C.cluster_features(cc, lab)
    Xc = F.cheap_features(s1, right, cc["i"].values, cc["j"].values)
    for k in Xc.columns:
        X[k] = Xc[k].values
    gi = cc.groupby("i")["p3"]
    X["p3"] = cc["p3"].values
    X["p2"] = cc["p2"].values
    X["expanded"] = cc["expanded"].values
    X["i_rank"] = gi.rank(ascending=False, method="first").values.astype(np.float32)
    X["i_gap"] = (gi.transform("max") - cc["p3"]).values.astype(np.float32)
    gj = cc.groupby("j")["p3"]
    X["j_gap"] = (gj.transform("max") - cc["p3"]).values.astype(np.float32)
    X["j_n"] = gj.transform("size").values.astype(np.float32)
    return cc, X


if mode == "train":
    perm = np.random.RandomState(42).permutation(len(s1))
    V = perm[1250000:1450000]
    TR = np.concatenate([perm[1450000:]])
    best = None
    for t in taus:
        lab = C.components(len(right), a, b, prr, t)
        cc, X = build(p3, lab)
        cc["y"], _ = run.label_pairs(cc, s1, right, gt)
        tr = np.isin(cc["i"].values, TR)
        va = np.isin(cc["i"].values, V)
        m4 = lgb.train(P.LGB_PARAMS, lgb.Dataset(X[tr], cc["y"].values[tr]), 1500,
                       valid_sets=[lgb.Dataset(X[va], cc["y"].values[va])],
                       callbacks=[lgb.early_stopping(50, verbose=False)])
        p4 = m4.predict(X[va])
        cv = cc[va].reset_index(drop=True)
        truth = {}
        vs = set(V.tolist())
        for i, j in zip(ti.tolist(), tj.tolist()):
            if i in vs:
                truth.setdefault(i, set()).add(j)
        rec = cv["y"].sum() / sum(len(v) for v in truth.values())
        pe = P.exclusive(cv, p4)
        for tau in (0.4, 0.5, 0.6, 0.7):
            f = P.macro_f05(P.select(cv, pe, tau, "exp"), truth, V.tolist())
            P.log(f"tau_rr={t} tau={tau}: val F0.5={f:.5f} (cand recall {rec:.4f}, {len(cc)/len(s1):.2f}/S1)")
            if best is None or f > best[0]:
                best = (f, t, tau, m4)
        del cc, X
        gc.collect()
    f, t, tau, m4 = best
    m4.save_model(f"{W}/stage4.txt")
    json.dump({"tau_rr": t, "tau": tau, "val_f05": f}, open(f"{W}/stage4.json", "w"))
    P.log("BEST", best[:3])
else:
    out = sys.argv[3]
    lab = C.components(len(right), a, b, prr, cfg["tau_rr"])
    cc, X = build(p3, lab)
    m4 = lgb.Booster(model_file=f"{W}/stage4.txt")
    p4 = m4.predict(X, num_threads=os.cpu_count())
    rid = right["entity_id"].to_numpy(dtype=object)
    run.write_ids(os.path.join(out, "candidate_pairs.tsv"), "candidate_entity_ids", s1,
                  cc["i"].values, rid[cc["j"].values])
    pred = P.select(cc, P.exclusive(cc, p4), cfg["tau"], "exp")
    mi = np.array([i for i, js in pred.items() for _ in js], dtype=np.int64)
    mj = np.array([j for js in pred.values() for j in js], dtype=np.int64)
    run.write_ids(os.path.join(out, "matching_results.tsv"), "matched_entity_ids", s1, mi, rid[mj])
    P.log("wrote", len(mi), "matches;", len(cc) / len(s1), "cands/S1")
