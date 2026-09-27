"""Score every (expanded) candidate with the stage-3 model and save i, j, p2, p3 (+y for train).
python dump_p3.py <split> <work> <gt_or_none> <nb_npz>"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import run
from ber import pipeline as P

split, W, gt, nbp = sys.argv[1:5]
s1, right = P.load_tables(W + "/cache", split)
c2 = pd.read_parquet(f"{W}/stage2_{split}.parquet")
if "y" in c2.columns:
    c2 = c2.drop(columns=["y"])
z = np.load(nbp)
m2 = lgb.Booster(model_file=W + "/stage2.txt")
c2 = P.expand(c2, (z["a"], z["n"]), s1, right, m2)
del z
ctx = run.full_context(P, s1, right, c2)
m3 = lgb.Booster(model_file=W + "/stage3.txt")
c2["p3"] = run.predict3(P, m3, s1, right, c2, ctx)
out = c2[["i", "j", "p2", "p3", "expanded"]].copy()
if gt != "none":
    out["y"], _ = run.label_pairs(out, s1, right, gt)
out.to_parquet(f"{W}/p3_{split}.parquet")
print("done", len(out), flush=True)
