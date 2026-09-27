"""Command-line entry point.

    python run.py train   --data ../../../dataset --work ../work
    python run.py predict --data ../../../dataset --work ../work --out ../../../output
"""
import argparse
import gc
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd


def id_num(ids):
    """'S2-123' -> 2 * 10**10 + 123 (entity ids are 'S<k>-<int>')"""
    ids = pd.Series(ids, dtype=str)
    return ids.str[1].astype(np.int64) * 10**10 + ids.str[3:].astype(np.int64)


def label_pairs(c, s1, right, gt_path):
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    ex = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")
    ex = ex[ex["m"] != ""]
    s1_idx = pd.Index(id_num(s1["entity_id"]).values)
    r_idx = pd.Index(id_num(right["entity_id"]).values)
    ti = s1_idx.get_indexer(id_num(ex["source1_entity_id"]).values)
    tj = r_idx.get_indexer(id_num(ex["m"]).values)
    del gt, ex, s1_idx, r_idx
    ok = (ti >= 0) & (tj >= 0)
    ti, tj = ti[ok].astype(np.int64), tj[ok].astype(np.int64)
    tkey = np.sort(ti * 100_000_000 + tj)
    y = np.zeros(len(c), bool)
    ci, cj = c["i"].values, c["j"].values
    for s in range(0, len(c), 10_000_000):
        ck = ci[s:s + 10_000_000].astype(np.int64) * 100_000_000 + cj[s:s + 10_000_000]
        pos = np.minimum(np.searchsorted(tkey, ck), len(tkey) - 1)
        y[s:s + 10_000_000] = tkey[pos] == ck
    return y, (ti, tj)


def run_stage12(args, P, pool, s1, right, tr_dir, grp):
    s1c = os.path.join(args.work, f"stage1_train_{args.limit}.parquet")
    if os.path.exists(s1c):
        c = pd.read_parquet(s1c, columns=["i", "j", "kscore", "s_rare", "s_addr"])
        c = pd.DataFrame({k: c[k].to_numpy().copy() for k in c.columns})
        gc.collect()
    else:
        c = P.stage1(s1, right, pool)
        c.to_parquet(s1c)
    y, truth = label_pairs(c, s1, right, os.path.join(tr_dir, "train_ground_truth.tsv"))
    n_true = len(truth[0])
    P.log(f"stage1: {len(c)} pairs, {len(c)/len(s1):.1f}/S1, recall {y.sum()/n_true:.4f}")

    cg = grp[c["i"].values]

    # ---- stage 2: OOF on A, full model for the rest
    ia = np.flatnonzero(cg == 0)
    ca = c.iloc[ia].reset_index(drop=True)
    P.add_k_ctx(ca)
    Xa = P.stage2_matrix(s1, right, ca)
    ya = y[ia]
    fold = (ca["i"].values.astype(np.int64) * 7919 % 3).astype(int)
    oof = np.zeros(len(ca), np.float32)
    for f in range(3):
        m = P.fit_lgb(Xa[fold != f], ya[fold != f], args.rounds2)
        oof[fold == f] = m.predict(Xa[fold == f])
    m2 = P.fit_lgb(Xa, ya, args.rounds2)
    m2.save_model(os.path.join(args.work, "stage2.txt"))
    del Xa, ca, cg
    gc.collect()
    # tau2: smallest threshold that loses <= loss2 of the positives blocking found
    pos_p = np.sort(oof[ya])
    tau2 = float(pos_p[int(args.loss2 * len(pos_p))])
    override = np.full(len(c), np.nan, np.float32)
    override[ia] = oof
    del oof, ia
    c["y"] = y
    del y
    gc.collect()
    c2 = P.stage2_prune(m2, s1, right, c, tau2, override)
    del c, override
    gc.collect()
    P.log(f"stage2: tau2={tau2:.4f} -> {len(c2)} pairs, {len(c2)/len(s1):.2f}/S1, "
          f"recall {c2['y'].sum()/n_true:.4f}")
    c2.to_parquet(os.path.join(args.work, "stage2_train.parquet"))

    return c2, tau2, truth, n_true


def add_siblings(args, P, pool, s1, right, c2, split):
    """second hop: near-duplicate S2/S3 neighbours of confident candidates"""
    import lightgbm as lgb
    nbp = os.path.join(args.work, f"neighbours_{split}_k{args.nb_k}.npz")
    if os.path.exists(nbp):
        z = np.load(nbp)
        nb = (z["a"], z["n"])
    else:
        a, n, _ = P.B.right_neighbors(right, pool, k=args.nb_k, log=P.log)
        np.savez(nbp, a=a, n=n)
        nb = (a, n)
    m2 = lgb.Booster(model_file=os.path.join(args.work, "stage2.txt"))
    before = len(c2)
    c2 = P.expand(c2.drop(columns=[x for x in ("expanded",) if x in c2.columns]), nb, s1, right, m2)
    P.log(f"sibling expansion: {before} -> {len(c2)} pairs ({len(c2)/len(s1):.2f}/S1)")
    return c2


def full_context(P, s1, right, c2):
    ctx = P.context(c2)
    sib = P.sibling_features(c2, s1, right)
    for k in sib.columns:
        ctx[k] = sib[k].values
    return ctx


def train(args, pool):
    from ber import cache, pipeline as P, translit as T, prep
    import lightgbm as lgb
    os.makedirs(args.work, exist_ok=True)
    tl = os.path.join(args.work, "translit.json")
    tr_dir = os.path.join(args.data, "train")
    if not os.path.exists(tl):
        P.log("learning transliteration / component maps from train ground truth")
        s1r = prep.read(os.path.join(tr_dir, "train_source1.tsv"))
        rr = pd.concat([prep.read(os.path.join(tr_dir, f"train_source{k}.tsv")) for k in (2, 3)])
        gt = prep.read(os.path.join(tr_dir, "train_ground_truth.tsv"))
        T.save(tl, *T.learn(s1r, rr, gt))
        del s1r, rr, gt
    cache.load_split(args.data, "train", os.path.join(args.work, "cache"), tl, pool=pool, keep=False)
    s1, right = P.load_tables(os.path.join(args.work, "cache"), "train")
    if args.limit:
        s1 = s1.sample(args.limit, random_state=0).reset_index(drop=True)
    P.log("train tables", len(s1), len(right))

    s2c = os.path.join(args.work, "stage2_train.parquet")
    s2j = os.path.join(args.work, "stage2.json")
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(s1))
    A = perm[: args.n_stage2]                                   # stage-2 training
    T3 = perm[args.n_stage2: args.n_stage2 + args.n_stage3]     # stage-3 training
    V = perm[args.n_stage2 + args.n_stage3: args.n_stage2 + args.n_stage3 + args.n_val]
    grp = np.full(len(s1), -1, np.int8)
    grp[A], grp[T3], grp[V] = 0, 1, 2
    if os.path.exists(s2c) and os.path.exists(s2j):
        c2 = pd.read_parquet(s2c)
        tau2 = json.load(open(s2j))["tau2"]
        _, truth = label_pairs(c2, s1, right, os.path.join(tr_dir, "train_ground_truth.tsv"))
        n_true = len(truth[0])
        P.log(f"resumed stage2 output: {len(c2)} pairs")
    else:
        c2, tau2, truth, n_true = run_stage12(args, P, pool, s1, right, tr_dir, grp)
        json.dump({"tau2": tau2}, open(s2j, "w"))

    # ---- sibling expansion + stage 3 on all surviving pairs (context needs every S1 present)
    c2 = add_siblings(args, P, pool, s1, right, c2, "train")
    c2["y"], _ = label_pairs(c2, s1, right, os.path.join(tr_dir, "train_ground_truth.tsv"))
    P.log(f"after expansion recall {c2['y'].sum()/n_true:.4f}")
    ctx = full_context(P, s1, right, c2)
    p3 = np.zeros(len(c2), np.float32)
    cg2 = grp[c2["i"].values]
    it = np.flatnonzero(cg2 == 1)
    X3 = P.stage3_matrix(s1, right, c2.iloc[it].reset_index(drop=True), ctx.iloc[it])
    y3 = c2["y"].values[it]
    ntr = int(len(it) * 0.9)
    split_i = c2["i"].values[it][ntr]
    tr = c2["i"].values[it] < split_i if False else np.arange(len(it)) < ntr
    dtr = lgb.Dataset(X3[tr], y3[tr])
    dva = lgb.Dataset(X3[~tr], y3[~tr])
    m3 = lgb.train(P.LGB_PARAMS, dtr, num_boost_round=args.rounds3, valid_sets=[dva],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    P.log("stage3 best iteration", m3.best_iteration)
    m3.save_model(os.path.join(args.work, "stage3.txt"), num_iteration=m3.best_iteration)
    imp = pd.Series(m3.feature_importance("gain"), index=X3.columns).sort_values(ascending=False)
    P.log("top features:\n" + imp.head(25).to_string())
    del X3, dtr, dva
    gc.collect()
    p3 = predict3(P, m3, s1, right, c2, ctx)

    # ---- decision tuning on V (never used for fitting)
    all_v = V.tolist()
    truth_v = {}
    vset = set(all_v)
    for i, j in zip(truth[0].tolist(), truth[1].tolist()):
        if i in vset:
            truth_v.setdefault(i, set()).add(j)
    pe = P.exclusive(c2, p3)
    vm = np.isin(c2["i"].values, V)
    cv = c2[vm].reset_index(drop=True)
    res = {}
    for mode in ("thr", "exp"):
        for tau in np.arange(0.3, 0.71, 0.05):
            for exc in (1,):
                pp = (pe if exc else p3)[vm]
                pred = P.select(cv, pp, tau, mode)
                res[(mode, round(float(tau), 3), exc)] = P.macro_f05(pred, truth_v, all_v)
    best = max(res, key=res.get)
    P.log("best decision", best, f"val macro F0.5 = {res[best]:.5f}")
    for k in sorted(res, key=res.get, reverse=True)[:8]:
        P.log("  ", k, round(res[k], 5))
    cand_ceiling = P.macro_f05({i: g["j"].tolist() for i, g in cv[cv["y"]].groupby("i")}, truth_v, all_v)
    P.log(f"val oracle-on-candidates F0.5 = {cand_ceiling:.5f}")
    cfg = {"tau2": tau2, "mode": best[0], "tau": best[1], "exclusive": best[2],
           "val_f05": res[best], "k2": P.K2, "pre_k": P.PRE_K, "max_df": P.MAX_DF}
    json.dump(cfg, open(os.path.join(args.work, "config.json"), "w"), indent=1)


def predict3(P, m3, s1, right, c2, ctx, chunk=3_000_000):
    p3 = np.zeros(len(c2), np.float32)
    for s in range(0, len(c2), chunk):
        e = min(len(c2), s + chunk)
        X = P.stage3_matrix(s1, right, c2.iloc[s:e].reset_index(drop=True), ctx.iloc[s:e])
        p3[s:e] = m3.predict(X, num_threads=os.cpu_count())
        del X
        gc.collect()
    return p3


def write_ids(path, col, s1, I, ids):
    """one row per S1 record; I (S1 positions) and ids (S2/S3 ids) aligned, pre-ordered"""
    lists = pd.Series(ids, dtype=object).groupby(np.asarray(I)).agg(",".join)
    out = pd.Series("", index=np.arange(len(s1)), dtype=object)
    out.loc[lists.index] = lists.values
    df = pd.DataFrame({"source1_entity_id": s1["entity_id"].to_numpy(dtype=object), col: out.values})
    df.to_csv(path, sep="\t", index=False, quoting=3)


def predict(args, pool):
    from ber import cache, pipeline as P
    import lightgbm as lgb
    cfg = json.load(open(os.path.join(args.work, "config.json")))
    tl = os.path.join(args.work, "translit.json")
    cache.load_split(args.data, "test", os.path.join(args.work, "cache"), tl, pool=pool, keep=False)
    s1, right = P.load_tables(os.path.join(args.work, "cache"), "test")
    P.log("test tables", len(s1), len(right))
    s2c = os.path.join(args.work, "stage2_test.parquet")
    if os.path.exists(s2c):
        c2 = pd.read_parquet(s2c)
    else:
        c = P.stage1(s1, right, pool)
        P.log(f"stage1: {len(c)} pairs")
        m2 = lgb.Booster(model_file=os.path.join(args.work, "stage2.txt"))
        c2 = P.stage2_prune(m2, s1, right, c, cfg["tau2"])
        del c
        gc.collect()
        c2.to_parquet(s2c)
    c2 = add_siblings(args, P, pool, s1, right, c2, "test")
    P.log(f"stage2+expansion: {len(c2)} candidate pairs, {len(c2)/len(s1):.2f}/S1")
    rid = right["entity_id"].to_numpy(dtype=object)
    os.makedirs(args.out, exist_ok=True)
    write_ids(os.path.join(args.out, "candidate_pairs.tsv"), "candidate_entity_ids", s1,
              c2["i"].values, rid[c2["j"].values])
    ctx = full_context(P, s1, right, c2)
    m3 = lgb.Booster(model_file=os.path.join(args.work, "stage3.txt"))
    p3 = predict3(P, m3, s1, right, c2, ctx)
    pp = P.exclusive(c2, p3) if cfg["exclusive"] else p3
    pred = P.select(c2, pp, cfg["tau"], cfg["mode"])
    mi = np.array([i for i, js in pred.items() for _ in js], dtype=np.int64)
    mj = np.array([j for js in pred.values() for j in js], dtype=np.int64)
    write_ids(os.path.join(args.out, "matching_results.tsv"), "matched_entity_ids", s1, mi, rid[mj])
    P.log(f"wrote outputs: {len(mi)} matches, {len(pred)} / {len(s1)} S1 with >=1 match")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict"])
    ap.add_argument("--data", default="../../../dataset")
    ap.add_argument("--work", default="../work")
    ap.add_argument("--out", default="../../../output")
    ap.add_argument("--n_stage2", type=int, default=250_000)
    ap.add_argument("--n_stage3", type=int, default=500_000)
    ap.add_argument("--n_val", type=int, default=150_000)
    ap.add_argument("--rounds2", type=int, default=300)
    ap.add_argument("--rounds3", type=int, default=2000)
    ap.add_argument("--nb_k", type=int, default=25)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--loss2", type=float, default=0.003)
    args = ap.parse_args()
    pool = Pool(os.cpu_count())          # fork workers before any big table is loaded
    (train if args.cmd == "train" else predict)(args, pool)
