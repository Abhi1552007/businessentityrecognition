"""Build (or load) normalised record tables for a split, cached as parquet."""
import gc
import os

import pandas as pd

from . import prep


def load_split(data_dir, split, cache_dir, translit_path, pool=None, keep=True):
    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    for src in ("source1", "source2", "source3"):
        cp = os.path.join(cache_dir, f"{split}_{src}.parquet")
        if not os.path.exists(cp):
            raw = prep.read(os.path.join(data_dir, split, f"{split}_{src}.tsv"))
            df = prep.prepare(raw, translit_path, pool=pool)
            del raw
            df["ctry"] = df["ctry"].astype(str)
            df.to_parquet(cp, index=False)
            del df
            gc.collect()
        if keep:
            out[src] = pd.read_parquet(cp)
    if not keep:
        return None
    right = pd.concat([out["source2"], out["source3"]], ignore_index=True)
    return out["source1"], right
