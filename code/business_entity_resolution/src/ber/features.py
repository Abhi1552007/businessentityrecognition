"""Vectorised pair features (rapidfuzz cpdist is multi-threaded C++)."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from . import normalize as N


def _cp(a, b, scorer, **kw):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def _skel(s):
    return " ".join(N.skeleton(t) for t in s.split())


def _sets(strings, fn=str.split):
    return [set(fn(s)) for s in strings]


def _jac(A, B):
    out = np.empty(len(A), np.float32)
    for k, (a, b) in enumerate(zip(A, B)):
        u = len(a | b)
        out[k] = len(a & b) / u if u else 0.0
    return out


def _best_variant(vl, vr):
    """pick the (left, right) name variant pair with the best token-set score"""
    out_l, out_r = [], []
    for a, b in zip(vl, vr):
        if "|" not in a and "|" not in b:
            out_l.append(a)
            out_r.append(b)
            continue
        best, pa, pb = -1, a, b
        for x in a.split("|"):
            for y in b.split("|"):
                s = fuzz.token_set_ratio(x, y)
                if s > best:
                    best, pa, pb = s, x, y
        out_l.append(pa)
        out_r.append(pb)
    return out_l, out_r


def _take(df, col, idx):
    return df[col].take(idx).tolist()


def cheap_features(L, R, I, J):
    """Small, fast feature set used by the stage-2 candidate pruner."""
    f = {}
    vl, vr = _take(L, "name_vars", I), _take(R, "name_vars", J)
    nl, nr = _best_variant(vl, vr)
    al, ar = _take(L, "addr_n", I), _take(R, "addr_n", J)
    f["c_nm_ratio"] = _cp(nl, nr, fuzz.ratio)
    f["c_nm_tset"] = _cp(nl, nr, fuzz.token_set_ratio)
    f["c_nm_partial"] = _cp(nl, nr, fuzz.partial_ratio)
    gl = [s.replace(" ", "") for s in nl]
    gr = [s.replace(" ", "") for s in nr]
    f["c_nm_glued"] = _cp(gl, gr, fuzz.ratio)
    f["c_nv_tset"] = _cp(vl, vr, fuzz.token_set_ratio)
    f["c_ad_tset"] = _cp(al, ar, fuzz.token_set_ratio)
    f["c_ad_len_r"] = np.fromiter((len(s) for s in ar), np.float32, len(ar))
    f["c_nm_len_l"] = np.fromiter((len(s) for s in nl), np.float32, len(nl))
    f["c_nm_len_r"] = np.fromiter((len(s) for s in nr), np.float32, len(nr))
    dl = [{t.lstrip("0") for t in s.split() if t.isdigit()} for s in al]
    dr = [{t.lstrip("0") for t in s.split() if t.isdigit()} for s in ar]
    f["c_num_jac"] = _jac(dl, dr)
    f["c_num_r"] = np.fromiter((len(b) for b in dr), np.float32, len(dr))
    return pd.DataFrame(f)


def pair_features(L, R, I, J):
    """L, R: prepared record tables; I, J: aligned index arrays of pairs."""
    f = {}
    vl, vr = _take(L, "name_vars", I), _take(R, "name_vars", J)
    nl, nr = _best_variant(vl, vr)
    fl, fr = _take(L, "name_full", I), _take(R, "name_full", J)
    al, ar = _take(L, "addr_n", I), _take(R, "addr_n", J)
    f["nm_ratio"] = _cp(nl, nr, fuzz.ratio)
    f["nm_tsort"] = _cp(nl, nr, fuzz.token_sort_ratio)
    f["nm_tset"] = _cp(nl, nr, fuzz.token_set_ratio)
    f["nm_partial"] = _cp(nl, nr, fuzz.partial_ratio)
    f["nm_jw"] = _cp(nl, nr, JaroWinkler.normalized_similarity)
    f["nm_lev"] = _cp(nl, nr, Levenshtein.distance)
    gl = [s.replace(" ", "") for s in nl]
    gr = [s.replace(" ", "") for s in nr]
    f["nm_glued"] = _cp(gl, gr, fuzz.ratio)
    f["nm_glued_partial"] = _cp(gl, gr, fuzz.partial_ratio)
    f["nm_skel"] = _cp([_skel(s) for s in nl], [_skel(s) for s in nr], fuzz.token_sort_ratio)
    f["fn_ratio"] = _cp(fl, fr, fuzz.ratio)
    f["fn_tset"] = _cp(fl, fr, fuzz.token_set_ratio)
    f["nv_tset"] = _cp(vl, vr, fuzz.token_set_ratio)
    tl, tr = _sets(nl), _sets(nr)
    f["nm_jac"] = _jac(tl, tr)
    f["nm_inter"] = np.fromiter((len(a & b) for a, b in zip(tl, tr)), np.float32, len(tl))
    f["nm_ntok_l"] = np.fromiter((len(a) for a in tl), np.float32, len(tl))
    f["nm_ntok_r"] = np.fromiter((len(b) for b in tr), np.float32, len(tr))
    f["nm_len_l"] = np.fromiter((len(s) for s in nl), np.float32, len(nl))
    f["nm_len_r"] = np.fromiter((len(s) for s in nr), np.float32, len(nr))
    f["nm_first_eq"] = np.fromiter((a.split()[:1] == b.split()[:1] for a, b in zip(nl, nr)), np.float32, len(nl))
    f["nm_eq"] = np.fromiter((a == b for a, b in zip(gl, gr)), np.float32, len(gl))
    f["nv_multi"] = np.fromiter(("|" in s for s in vr), np.float32, len(J))

    f["ad_tset"] = _cp(al, ar, fuzz.token_set_ratio)
    f["ad_tsort"] = _cp(al, ar, fuzz.token_sort_ratio)
    f["ad_partial"] = _cp(al, ar, fuzz.partial_token_set_ratio)
    f["ad_len_l"] = np.fromiter((len(s) for s in al), np.float32, len(al))
    f["ad_len_r"] = np.fromiter((len(s) for s in ar), np.float32, len(ar))
    sl, sr = _sets(al), _sets(ar)
    wl = [{t for t in s if not t.isdigit() and t not in N.ADDR_STOP} for s in sl]
    wr = [{t for t in s if not t.isdigit() and t not in N.ADDR_STOP} for s in sr]
    f["ad_wjac"] = _jac(wl, wr)
    f["ad_wcov"] = np.fromiter((len(a & b) / len(b) if b else -1 for a, b in zip(wl, wr)), np.float32, len(wl))
    dl = [{t.lstrip("0") for t in s if t.isdigit()} for s in sl]
    dr = [{t.lstrip("0") for t in s if t.isdigit()} for s in sr]
    f["num_jac"] = _jac(dl, dr)
    f["num_inter"] = np.fromiter((len(a & b) for a, b in zip(dl, dr)), np.float32, len(dl))
    f["num_r"] = np.fromiter((len(b) for b in dr), np.float32, len(dr))
    f["num_conflict"] = np.fromiter((bool(a) and bool(b) and not (a & b) for a, b in zip(dl, dr)), np.float32, len(dl))
    fnl = [s.split()[0] if s and s.split()[0].isdigit() else "" for s in al]
    fnr = [s.split()[0] if s and s.split()[0].isdigit() else "" for s in ar]
    f["num_first_eq"] = np.fromiter(((a == b) if a and b else -1 for a, b in zip(fnl, fnr)), np.float32, len(fnl))
    cl = [set(s.split("|")) for s in _take(L, "comps", I)]
    cr = [set(s.split("|")) for s in _take(R, "comps", J)]
    f["comp_jac"] = _jac(cl, cr)
    f["comp_inter"] = np.fromiter((len(a & b) for a, b in zip(cl, cr)), np.float32, len(cl))
    f["comp_n_r"] = np.fromiter((len(b) for b in cr), np.float32, len(cr))
    # cross field: name of one side inside the other's address (landmark-y noise)
    f["src3"] = np.fromiter((e.startswith("S3") for e in _take(R, "entity_id", J)), np.float32, len(J))
    return pd.DataFrame(f)


def context_features(df, score_col, prefix):
    """How a pair compares with the competing candidates of the same S1 record
    and of the same S2/S3 record."""
    g = df.groupby("i")[score_col]
    df[prefix + "rank"] = g.rank(ascending=False, method="first").astype(np.float32)
    df[prefix + "gap"] = (g.transform("max") - df[score_col]).astype(np.float32)
    df[prefix + "n"] = g.transform("size").astype(np.float32)
    h = df.groupby("j")[score_col]
    df[prefix + "rrank"] = h.rank(ascending=False, method="first").astype(np.float32)
    df[prefix + "rgap"] = (h.transform("max") - df[score_col]).astype(np.float32)
    df[prefix + "rn"] = h.transform("size").astype(np.float32)
    return df
