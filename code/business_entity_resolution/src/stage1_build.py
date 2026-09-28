"""Multi-channel blocking in a lean standalone process (only the columns the
keys need are loaded). python stage1_build.py <cache_dir> <split> <out.parquet>"""
import sys
from multiprocessing import Pool

import pandas as pd

if __name__ == "__main__":
    cache, split, out = sys.argv[1:4]
    pool = Pool(4)
    from ber import pipeline as P
    cols = ["name_vars", "addr_n", "ctry"]
    rd = lambda k: pd.read_parquet(f"{cache}/{split}_source{k}.parquet", columns=cols, dtype_backend="pyarrow")
    s1 = rd(1)
    right = pd.concat([rd(2), rd(3)], ignore_index=True)
    c = P.stage1(s1, right, pool)
    c.to_parquet(out)
    P.log("stage1 saved", len(c), f"{len(c)/len(s1):.1f}/S1")
