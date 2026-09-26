"""Scalable candidate generation (blocking).

No all-pairs comparison is ever made.

1. Every record emits a small set of blocking keys (name tokens, phonetic
   skeletons, prefixes, glued name, token pairs, acronym, house-number+street,
   postal code, street bigrams), each prefixed by the country label. Keys are
   hashed to int64.
2. S2/S3 records form an inverted index key -> records. Keys whose posting
   list is longer than ``max_df`` are dropped (stop-key pruning), so the work
   per Source 1 record is bounded by ``n_keys * max_df`` independent of corpus
   size; the index shards trivially by key hash (the same design as
   key/LSH blocking at billion-record scale).
3. The candidate score is the IDF-weighted count of shared keys, computed as a
   sparse product L @ R^T in chunks; the best ``pre_k`` per S1 record are kept.

The learned pruning step that shrinks these ``pre_k`` to the final small
candidate list lives in pipeline.py (stage-2 ranker).
"""
import hashlib
import os
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp

from . import normalize as N

_ALPHA_STOP = N.ADDR_STOP | {"street", "road", "avenue", "lane", "drive", "rue", "court", "place",
                             "boulevard", "nagar", "colony", "sector", "block", "plot", "flat",
                             "shop", "building", "phase", "main", "cross", "layout", "n", "s",
                             "e", "w", "ne", "nw", "se", "sw", "allee", "chemin", "route"}


_HOUSE_SKIP = {"unit", "number", "house", "floor", "suite", "apartment", "flat", "plot", "po",
               "box", "pmb", "n", "s", "e", "w", "ne", "nw", "se", "sw", "a", "b", "c", "d"}


def record_keys(name_vars, addr_n, ctry):
    keys = set()
    at = addr_n.split()
    nums = [t.lstrip("0") for t in at if t.isdigit()][:3]
    places = [N.skeleton(t) for t in dict.fromkeys(
        t for t in at if not t.isdigit() and t not in _ALPHA_STOP and len(t) >= 3)][-4:]
    for v in name_vars.split("|"):
        toks = v.split()
        if not toks:
            continue
        for t in toks:
            if len(t) >= 2 and not t.isdigit():
                keys.add("t" + t)
            if len(t) >= 4:
                keys.add("k" + N.skeleton(t))
        g = "".join(toks)
        keys.add("g" + g)
        # glued names ("askykonnect", "dentautobody"): index every 5-gram of a
        # long single-token name; the other side queries with its tokens' heads
        if len(toks) == 1 and len(g) >= 10:
            for k in range(len(g) - 4):
                keys.add("u" + g[k:k + 5])
        for t in toks:
            if len(t) >= 5:
                keys.add("u" + t[:5])
        if len(g) >= 8:
            keys.add("h" + g[:6])
        for a, b in zip(toks, toks[1:]):
            keys.add("b" + "_".join(sorted((a, b))))
        if len(toks) >= 3:
            keys.add("b" + "_".join(sorted((toks[0], toks[2]))))
        # name x address combinations: stay selective even when the name
        # token alone is far too common to be a usable block
        for t in [x for x in toks if len(x) >= 2][:1]:
            for n in nums:
                keys.add("x" + t + "_" + n)
            for pl in places:
                keys.add("y" + t + "_" + pl)
        if len(g) >= 6:
            keys.add("q" + "".join(sorted(g)))
        if len(toks) >= 2:
            keys.add("a" + "".join(t[0] for t in toks))
        elif 2 <= len(toks[0]) <= 5:
            keys.add("a" + toks[0])
    nh = 0
    for i, t in enumerate(at):
        if t.isdigit():
            if len(t) in (5, 6):
                keys.add("z" + t)
            if nh < 3:
                for u in at[i + 1:i + 3]:
                    if not u.isdigit() and u not in _HOUSE_SKIP and len(u) >= 2:
                        keys.add("n" + t.lstrip("0") + "_" + N.skeleton(u))
                        nh += 1
                        break
    alpha = [t for t in at if not t.isdigit() and t not in _ALPHA_STOP and len(t) >= 3]
    for a, b in zip(alpha, alpha[1:]):
        keys.add("s" + N.skeleton(a) + "_" + N.skeleton(b))
    return [_h(ctry + ":" + k) for k in keys]


def _h(s):
    # deterministic 64-bit key hash (python's hash() is salted per process)
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little", signed=True)


def _keys_chunk(args):
    base, rows = args
    rec, ks = [], []
    for i, (nv, ad, c) in enumerate(rows):
        k = record_keys(nv, ad, c)
        ks.extend(k)
        rec.extend([base + i] * len(k))
    return np.asarray(rec, np.int32), np.asarray(ks, np.int64)


def all_keys(df, pool, chunk=50000):
    nv, ad, ct = df["name_vars"].values, df["addr_n"].values, df["ctry"].astype(str).values
    jobs = ((i, list(zip(nv[i:i + chunk], ad[i:i + chunk], ct[i:i + chunk])))
            for i in range(0, len(df), chunk))
    parts = list(pool.imap(_keys_chunk, jobs))
    return np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts])


def block(s1, right, pool, max_df=500, pre_k=40, log=print):
    """Country-sharded blocking. Returns global (i, j, key_score) arrays."""
    I, J, S = [], [], []
    lc, rc = s1["ctry"].astype(str).values, right["ctry"].astype(str).values
    for c in sorted(set(lc)):
        li = np.flatnonzero(lc == c)
        ri = np.flatnonzero(rc == c)
        if len(li) == 0 or len(ri) == 0:
            continue
        rr, rk = all_keys(right.iloc[ri], pool)
        idx = KeyIndex(max_df).fit(rr, rk, len(ri))
        del rr, rk
        lr, lk = all_keys(s1.iloc[li], pool)
        i, j, sc = idx.query(lr, lk, len(li), pre_k=pre_k)
        del idx
        I.append(li[i]); J.append(ri[j]); S.append(sc)
        log(f"  blocking shard {c}: {len(li)} x {len(ri)} -> {len(i)} pairs")
    return np.concatenate(I), np.concatenate(J), np.concatenate(S)


class KeyIndex:
    def __init__(self, max_df=400):
        self.max_df = max_df

    def fit(self, right_rec, right_key, n_right):
        order = np.argsort(right_key, kind="stable")
        right_key, right_rec = right_key[order], right_rec[order]
        del order
        new = np.empty(len(right_key), bool)
        new[0] = True
        np.not_equal(right_key[1:], right_key[:-1], out=new[1:])
        starts = np.flatnonzero(new)
        df = np.diff(np.append(starts, len(right_key)))
        keep = df <= self.max_df
        self.keys = right_key[starts[keep]]
        col = np.cumsum(new) - 1                      # unique-key id per entry
        m = keep[col]
        remap = np.cumsum(keep) - 1
        self.RT = sp.csr_matrix((np.ones(int(m.sum()), np.float32), (remap[col[m]], right_rec[m])),
                                shape=(len(self.keys), n_right))
        self.idf = np.log1p(n_right / df[keep]).astype(np.float32)
        self.n_right = n_right
        return self

    def query(self, left_rec, left_key, n_left, pre_k=40, chunk=20000):
        pos = np.searchsorted(self.keys, left_key)
        pos = np.minimum(pos, len(self.keys) - 1)
        m = self.keys[pos] == left_key
        L = sp.csr_matrix((self.idf[pos[m]], (left_rec[m], pos[m])),
                          shape=(n_left, len(self.keys)))
        L.sum_duplicates()
        I, J, S = [], [], []
        for s in range(0, n_left, chunk):
            P = (L[s:s + chunk] @ self.RT).tocsr()
            P.sort_indices()
            rows = np.repeat(np.arange(P.shape[0]), np.diff(P.indptr))
            order = np.lexsort((-P.data, rows))
            r, c, d = rows[order], P.indices[order], P.data[order]
            start = np.searchsorted(r, r, side="left")
            rank = np.arange(len(r)) - start
            k = rank < pre_k
            I.append(r[k] + s)
            J.append(c[k])
            S.append(d[k])
        return np.concatenate(I), np.concatenate(J), np.concatenate(S)
