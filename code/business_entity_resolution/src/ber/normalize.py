"""Country-agnostic text normalisation for business names and addresses.

Everything here is rule based or learned from the *training* files only
(see translit.py). Nothing looks anything up externally.
"""
import re
import unicodedata

from unidecode import unidecode

NAME_ABBR = {
    "corp": "corporation", "co": "company", "inc": "incorporated", "incorp": "incorporated",
    "ltd": "limited", "pvt": "private", "pte": "private", "intl": "international",
    "natl": "national", "mfg": "manufacturing", "svcs": "services", "svc": "services",
    "tech": "technologies", "technology": "technologies", "ent": "enterprises",
    "bros": "brothers", "assoc": "associates", "assocs": "associates", "mgmt": "management",
    "grp": "group", "hldgs": "holdings", "inds": "industries", "sol": "solutions",
    "solns": "solutions", "sys": "systems", "cie": "compagnie", "ste": "societe",
    "n": "and", "et": "and", "llc": "llc", "l": "l", "service": "services",
    "intl": "international", "st": "saint", "mt": "mount", "ft": "fort",
}
# Words carrying (almost) no identity: legal forms, fillers, generic suffix noise.
LEGAL = {
    "corporation", "company", "incorporated", "limited", "private", "llc", "llp", "lp", "plc",
    "public", "the", "and", "of", "pllc", "pc", "sarl", "sa", "sas", "sasu", "eurl", "sci",
    "snc", "societe", "compagnie", "gmbh", "ag", "opc", "de", "du", "des", "la", "le", "les",
    "nee", "esq", "france", "groupe", "sasu", "www", "com", "net", "org", "in", "co", "fr", "us", "india", "center", "centre",
}
ADDR_ABBR = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "ln": "lane", "dr": "drive", "ct": "court",
    "pl": "place", "sq": "square", "hwy": "highway", "pkwy": "parkway", "ste": "suite",
    "apt": "apartment", "fl": "floor", "flr": "floor", "bldg": "building", "blk": "block",
    "sec": "sector", "nr": "near", "opp": "opposite", "mkt": "market", "ngr": "nagar",
    "extn": "extension", "ext": "extension", "cir": "circle", "ter": "terrace", "terr": "terrace",
    "trl": "trail", "rte": "route", "hno": "house", "h": "house", "no": "number",
    "stn": "station", "twp": "township", "mtn": "mountain", "pt": "point", "cv": "cove",
    "saint": "street", "sainte": "sainte", "crk": "creek", "hts": "heights", "xing": "crossing", "pk": "park",
    "fwy": "freeway", "expy": "expressway", "tpke": "turnpike", "cres": "crescent",
    "plz": "plaza", "sqr": "square", "lk": "lake", "vlg": "village", "jn": "junction",
    "chem": "chemin", "ch": "chemin", "imp": "impasse", "fbg": "faubourg", "fg": "faubourg",
    "r": "rue", "all": "allee", "qu": "quai", "crs": "cours", "rte": "route", "sent": "sentier",
    "st": "street",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne",
    "northwest": "nw", "southeast": "se", "southwest": "sw",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
}
ADDR_STOP = {"near", "opposite", "behind", "beside", "next", "to", "number", "the", "of",
             "and", "de", "du", "des", "la", "le", "les", "at", "in", "off", "null", "n/a",
             "na", "none", "c/o", "co", "unit", "suite", "apartment", "floor", "house",
             "city", "po", "box", "pmb", "street", "road", "avenue", "lane", "drive"}
NULLS = re.compile(r"<null>|\bnull\b|\bn/a\b|\bnone\b|\bnan\b", re.I)

LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g",
                      "7": "t", "8": "b", "9": "g", "@": "a", "$": "s", "!": "i"})
_tok = re.compile(r"[a-z0-9]+")
_has_alpha = re.compile(r"[a-z]")
_has_digit = re.compile(r"\d")
_ordinal = re.compile(r"^(\d+)(st|nd|rd|th)$")

# learned native-script -> latin token dictionary (filled by translit.load)
TRANSLIT = {}
# address-component canonicalisation: US state names -> USPS codes (static),
# plus mappings learned from training data (e.g. "tn" -> "tamil nadu")
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc", "puerto rico": "pr",
}
COMPONENT = {}


def is_latin(s):
    for c in s:
        if ord(c) > 0x24F and c.isalpha():
            return False
    return True


def fold(s):
    """accent fold + lowercase, keeps non-latin letters."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def to_latin(s):
    if is_latin(s):
        return fold(s)
    out = []
    for t in fold(s).split():
        if is_latin(t):
            out.append(t)
        else:
            out.append(TRANSLIT.get(t) or unidecode(t).lower())
    return " ".join(out)


def _deleet(t):
    # "pub1ic" -> "public", "6urgaon" -> "gurgaon"; pure numbers untouched
    if _has_alpha.search(t) and _has_digit.search(t) and not _ordinal.match(t):
        return t.translate(LEET)
    return t


def clean(s):
    if not isinstance(s, str):
        return ""
    return NULLS.sub(" ", s)


def name_variants(raw):
    """Split 'X dba Y', 'X aka Y', 'X | www.y.com' into alternative names."""
    s = to_latin(clean(raw))
    parts = re.split(r"\s+(?:dba|d/b/a|aka|t/a|trading as)\s+|\|", s)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if re.match(r"^(www\.)?[a-z0-9\-]+\.(com|net|org|in|co\.in|fr|us|biz|info)$", p):
            p = re.sub(r"^www\.", "", p).rsplit(".", 1)[0]
            p = p.replace(".co", "")
        out.append(p)
    return out or [s]


def name_tokens(s):
    s = s.replace("&", " and ").replace("+", " ").replace("'s ", "s ").replace("'", "")
    toks = [_deleet(t) for t in _tok.findall(s.translate(str.maketrans({"@": "a", "$": "s", "!": "i"})))]
    # glue runs of single letters: "l l c" -> "llc", "i b m" -> "ibm"
    out, run = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run))
            run = []
        out.append(t)
    if run:
        out.append("".join(run))
    return [NAME_ABBR.get(t, t) for t in out]


def core(toks):
    c = [t for t in toks if t not in LEGAL and len(t) > 0]
    return c or toks


def norm_name(raw):
    """-> (full normalised name, core name, list of core variants)"""
    vars_ = name_variants(raw)
    full = [name_tokens(v) for v in vars_]
    cores = [" ".join(core(t)) for t in full]
    return " ".join(full[-1] if full else []), cores[-1] if cores else "", cores


def addr_components(raw, country=""):
    s = to_latin(clean(raw))
    states = US_STATES if country == "us" else {}
    comps = []
    for c in s.split(","):
        c = re.sub(r"[^a-z0-9/\- ]", " ", c)
        c = " ".join(c.split())
        if c:
            c = COMPONENT.get(c, c)
            comps.append(states.get(c, c))
    return comps




def addr_tokens(comps):
    toks = []
    for c in comps:
        for t in re.findall(r"[a-z0-9]+", c):
            m = _ordinal.match(t)
            if m:
                toks.append(m.group(1))
                continue
            m = re.fullmatch(r"([a-z]+)(\d+)", t)
            if m:
                toks.extend([ADDR_ABBR.get(m.group(1), m.group(1)), m.group(2)])
                continue
            m = re.fullmatch(r"(\d+)([a-z])", t)   # "1056c" -> 1056
            if m:
                toks.append(m.group(1))
                continue
            toks.append(ADDR_ABBR.get(t, t))
    return toks


def numbers(toks):
    return {t.lstrip("0") or "0" for t in toks if t.isdigit()}


def skeleton(tok):
    """Cheap phonetic key: first letter + consonant classes, runs collapsed."""
    if not tok:
        return tok
    if tok.isdigit():
        return tok
    t = tok.translate(str.maketrans("ckqgjzsxvfpbdtnmlr", "kkkkjssksfpptdnmlr"))
    rest = re.sub(r"[aeiouyhw]", "", t[1:])
    rest = re.sub(r"(.)\1+", r"\1", rest)
    return t[0] + rest
