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


def label_pairs(c, s1, right, gt_path):
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    s1_pos = pd.Series(np.arange(len(s1)), index=s1["entity_id"].astype(str).values)
    r_pos = pd.Series(np.arange(len(right)), index=right["entity_id"].astype(str).values)
    ex = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")
    ex = ex[ex["m"] != ""]
    ti = s1_pos.reindex(ex["source1_entity_id"].values).values
    tj = r_pos.reindex(ex["m"].values).values
    ok = ~(np.isnan(ti) | np.isnan(tj))
    tkey = ti[ok].astype(np.int64) * 100_000_000 + tj[ok].astype(np.int64)
    ckey = c["i"].values.astype(np.int64) * 100_000_000 + c["j"].values.astype(np.int64)
    y = np.isin(ckey, tkey)
    truth = {}
    for i, j in zip(ti[ok].astype(np.int64), tj[ok].astype(np.int64)):
        truth.setdefault(int(i), set()).add(int(j))
    return y, truth


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

    c = P.stage1(s1, right, pool)
    y, truth = label_pairs(c, s1, right, os.path.join(tr_dir, "train_ground_truth.tsv"))
    n_true = sum(len(v) for v in truth.values())
    P.log(f"stage1: {len(c)} pairs, {len(c)/len(s1):.1f}/S1, recall {y.sum()/n_true:.4f}")

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(s1))
    A = perm[: args.n_stage2]                                   # stage-2 training
    T3 = perm[args.n_stage2: args.n_stage2 + args.n_stage3]     # stage-3 training
    V = perm[args.n_stage2 + args.n_stage3: args.n_stage2 + args.n_stage3 + args.n_val]
    grp = np.full(len(s1), -1, np.int8)
    grp[A], grp[T3], grp[V] = 0, 1, 2
    cg = grp[c["i"].values]

    # ---- stage 2: OOF on A, full model for the rest
    ia = np.flatnonzero(cg == 0)
    ca = c.iloc[ia].reset_index(drop=True)
    Xa = P.stage2_matrix(s1, right, ca)
    ya = y[ia]
    fold = (ca["i"].values * 2654435761 % 3).astype(int)
    p2 = np.zeros(len(c), np.float32)
    oof = np.zeros(len(ca), np.float32)
    for f in range(3):
        m = P.fit_lgb(Xa[fold != f], ya[fold != f], args.rounds2)
        oof[fold == f] = m.predict(Xa[fold == f])
    m2 = P.fit_lgb(Xa, ya, args.rounds2)
    m2.save_model(os.path.join(args.work, "stage2.txt"))
    del Xa
    gc.collect()
    rest = np.flatnonzero(cg != 0)
    p2[rest] = P.stage2_predict(m2, s1, right, c.iloc[rest].reset_index(drop=True))
    p2[ia] = oof

    # tau2: smallest threshold that loses <= loss2 of the positives blocking found
    pos_p = np.sort(oof[ya])
    tau2 = float(pos_p[int(args.loss2 * len(pos_p))])
    c2 = P.prune(c.assign(y=y), p2, tau2)
    P.log(f"stage2: tau2={tau2:.4f} -> {len(c2)} pairs, {len(c2)/len(s1):.2f}/S1, "
          f"recall {c2['y'].sum()/n_true:.4f}")
    del c, y, p2
    gc.collect()

    # ---- stage 3 on all surviving pairs (context needs every S1 present)
    ctx = P.context(c2)
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
    truth_v = {i: truth[i] for i in all_v if i in truth}
    pe = P.exclusive(c2, p3)
    vm = np.isin(c2["i"].values, V)
    cv = c2[vm].reset_index(drop=True)
    res = {}
    for mode in ("thr", "exp"):
        for tau in np.arange(0.2, 0.8, 0.025):
            for exc in (0, 1):
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


def write_ids(path, col, s1, groups):
    ids = s1["entity_id"].astype(str).values
    with open(path, "w") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for i in range(len(s1)):
            f.write(ids[i] + "\t" + ",".join(groups.get(i, ())) + "\n")


def predict(args, pool):
    from ber import cache, pipeline as P
    import lightgbm as lgb
    cfg = json.load(open(os.path.join(args.work, "config.json")))
    tl = os.path.join(args.work, "translit.json")
    cache.load_split(args.data, "test", os.path.join(args.work, "cache"), tl, pool=pool, keep=False)
    s1, right = P.load_tables(os.path.join(args.work, "cache"), "test")
    P.log("test tables", len(s1), len(right))
    c = P.stage1(s1, right, pool)
    P.log(f"stage1: {len(c)} pairs")
    m2 = lgb.Booster(model_file=os.path.join(args.work, "stage2.txt"))
    p2 = P.stage2_predict(m2, s1, right, c)
    c2 = P.prune(c, p2, cfg["tau2"])
    del c, p2
    gc.collect()
    P.log(f"stage2: {len(c2)} candidate pairs, {len(c2)/len(s1):.2f}/S1")
    rid = right["entity_id"].astype(str).values
    os.makedirs(args.out, exist_ok=True)
    cand = {i: rid[g].tolist() for i, g in c2.groupby("i")["j"]}
    write_ids(os.path.join(args.out, "candidate_pairs.tsv"), "candidate_entity_ids", s1, cand)
    ctx = P.context(c2)
    m3 = lgb.Booster(model_file=os.path.join(args.work, "stage3.txt"))
    p3 = predict3(P, m3, s1, right, c2, ctx)
    pp = P.exclusive(c2, p3) if cfg["exclusive"] else p3
    pred = P.select(c2, pp, cfg["tau"], cfg["mode"])
    match = {i: rid[js].tolist() for i, js in pred.items()}
    write_ids(os.path.join(args.out, "matching_results.tsv"), "matched_entity_ids", s1, match)
    P.log(f"wrote outputs: {sum(len(v) for v in match.values())} matches, "
          f"{sum(1 for v in match.values() if v)} / {len(s1)} S1 with >=1 match")


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
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--loss2", type=float, default=0.003)
    args = ap.parse_args()
    pool = Pool(os.cpu_count())          # fork workers before any big table is loaded
    (train if args.cmd == "train" else predict)(args, pool)
