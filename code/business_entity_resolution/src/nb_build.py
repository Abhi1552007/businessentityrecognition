"""Build S2/S3 near-duplicate neighbour links in a lean standalone process.
python nb_build.py <cache_dir> <split> <k> <out.npz>"""
import gc
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

if __name__ == "__main__":
    cache, split, k, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    pool = Pool(4)
    from ber import blocking as B
    cols = ["name_vars", "addr_n", "ctry"]
    right = pd.concat([pd.read_parquet(f"{cache}/{split}_source{s}.parquet", columns=cols,
                                       dtype_backend="pyarrow") for s in (2, 3)], ignore_index=True)
    rc = right["ctry"].astype(str).values
    A, N = [], []
    for c in sorted(set(rc)):
        ri = np.flatnonzero(rc == c).astype(np.int32)
        sub = right.iloc[ri].reset_index(drop=True)
        a, n, _ = B.right_neighbors(sub, pool, k=k)
        A.append(ri[a]); N.append(ri[n])
        del sub, a, n
        gc.collect()
    np.savez(out, a=np.concatenate(A), n=np.concatenate(N))
    print("done", sum(len(x) for x in A))
