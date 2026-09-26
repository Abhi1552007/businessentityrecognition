"""Parallel normalisation of raw source frames into a compact record table."""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import normalize as N
from . import translit as T

_TRANSLIT_PATH = None


def _process(args):
    path, chunk = args
    if path and (not N.TRANSLIT or getattr(N, "_translit_path", None) != path):
        T.load(path)
        N._translit_path = path
    out = {k: [] for k in ("name_full", "name_core", "name_vars", "addr_n", "comps", "ctry")}
    for name, addr, ctry in chunk:
        c = N.fold(ctry).strip() if isinstance(ctry, str) else ""
        c = c or "_unk"
        full, core, cores = N.norm_name(name)
        comps = N.addr_components(addr, c)
        toks = N.addr_tokens(comps)
        out["name_full"].append(full)
        out["name_core"].append(core)
        out["name_vars"].append("|".join(dict.fromkeys(cores)))
        out["addr_n"].append(" ".join(toks))
        out["comps"].append("|".join(comps))
        out["ctry"].append(c)
    return out


def prepare(df, translit_path=None, pool=None, chunk=20000):
    nm, ad, ct = df["business_name"].values, df["business_address"].values, df["country"].values
    jobs = ((translit_path, list(zip(nm[i:i + chunk], ad[i:i + chunk], ct[i:i + chunk])))
            for i in range(0, len(df), chunk))
    if pool is None:
        with Pool(os.cpu_count()) as p:
            parts = list(p.imap(_process, jobs))
    else:
        parts = list(pool.imap(_process, jobs))
    out = pd.DataFrame({"entity_id": df["entity_id"].values})
    for k in parts[0]:
        out[k] = [x for part in parts for x in part[k]]
    out["ctry"] = out["ctry"].astype("category")
    return out


def read(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
