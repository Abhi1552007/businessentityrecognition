"""Learn native-script -> Latin token mappings and address-component
canonicalisations from the TRAINING ground truth only.

For every labelled pair (S1 record, S2/S3 record) whose S2/S3 name is written
in a non-Latin script (Devanagari, Tamil, ...) and has the same token count as
the Latin S1 name, tokens are aligned positionally and co-occurrence counted.
The most frequent Latin rendering becomes the dictionary entry.
"""
import json
from collections import Counter, defaultdict

from . import normalize as N


def learn(s1, right, gt, min_count=3, min_share=0.5):
    s1n = dict(zip(s1["entity_id"], s1["business_name"]))
    s1a = dict(zip(s1["entity_id"], s1["business_address"]))
    rn = dict(zip(right["entity_id"], right["business_name"]))
    ra = dict(zip(right["entity_id"], right["business_address"]))
    tok = defaultdict(Counter)
    comp = defaultdict(Counter)
    comp_n = Counter()
    s1_comp_n = Counter()
    for sid, mids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if not mids:
            continue
        a_comps = {c.strip() for c in N.fold(N.clean(s1a[sid])).split(",") if c.strip()}
        for c in a_comps:
            s1_comp_n[c] += 1
        for m in mids.split(","):
            nm = rn.get(m, "")
            if isinstance(nm, str) and not N.is_latin(nm):
                a = N.fold(s1n[sid]).split()
                b = N.fold(nm).split()
                if len(a) == len(b):
                    for x, y in zip(b, a):
                        if not N.is_latin(x):
                            tok[x][y.strip(".,()")] += 1
            ad = ra.get(m, "")
            if isinstance(ad, str):
                for c in {c.strip() for c in N.fold(N.clean(ad)).split(",") if c.strip()}:
                    if c in a_comps:
                        continue
                    if N.is_latin(c) and len(c) > 3:
                        continue
                    comp_n[c] += 1
                    for t in a_comps:
                        comp[c][t] += 1
    translit = {}
    for x, cnt in tok.items():
        y, n = cnt.most_common(1)[0]
        if n >= min_count and n / sum(cnt.values()) >= min_share:
            translit[x] = y
    component = {}
    for c, cnt in comp.items():
        if comp_n[c] < 20 or s1_comp_n.get(c, 0) > 0.05 * comp_n[c]:
            continue
        t, n = cnt.most_common(1)[0]
        if n / comp_n[c] >= 0.8:
            component[c] = t
    return translit, component


def save(path, translit, component):
    with open(path, "w") as f:
        json.dump({"translit": translit, "component": component}, f, ensure_ascii=False)


def load(path):
    with open(path) as f:
        d = json.load(f)
    N.TRANSLIT.clear()
    N.TRANSLIT.update(d["translit"])
    # component keys are learned on folded raw text; re-key them as
    # addr_components() will see them (latinised, punctuation stripped)
    N.COMPONENT.clear()
    import re
    for k, v in d["component"].items():
        kk = " ".join(re.sub(r"[^a-z0-9/\- ]", " ", N.to_latin(k)).split())
        vv = " ".join(re.sub(r"[^a-z0-9/\- ]", " ", N.to_latin(v)).split())
        if kk and vv and kk != vv and not re.search(r"\d", vv + kk):
            N.COMPONENT[kk] = vv
