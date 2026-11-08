"""Entity-level grounding, provision coverage, ROUGE and per-sentence features for generated summaries."""
import re, json, math
from rouge_score import rouge_scorer
from redline_core import legal_entities, act_key, WS, find_usc

SC = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

ALIAS = {
    "department of justice": ["attorney general", "department of justice"],
    "doj": ["attorney general", "department of justice"],
    "department of health and human services": ["secretary of health and human services", "health and human services", "secretary"],
    "hhs": ["health and human services"],
    "department of defense": ["secretary of defense", "department of defense"],
    "dod": ["secretary of defense", "department of defense"],
    "department of veterans affairs": ["secretary of veterans affairs", "veterans affairs"],
    "va": ["veterans affairs"],
    "department of agriculture": ["secretary of agriculture"],
    "usda": ["secretary of agriculture", "agriculture"],
    "department of the interior": ["secretary of the interior"],
    "department of education": ["secretary of education"],
    "department of energy": ["secretary of energy"],
    "department of labor": ["secretary of labor"],
    "department of transportation": ["secretary of transportation"],
    "department of homeland security": ["secretary of homeland security"],
    "department of state": ["secretary of state"],
    "department of the treasury": ["secretary of the treasury"],
    "department of commerce": ["secretary of commerce"],
    "department of housing and urban development": ["secretary of housing and urban development"],
    "internal revenue code": ["internal revenue code"],
    "government accountability office": ["comptroller general"],
    "gao": ["comptroller general"],
    "environmental protection agency": ["administrator of the environmental protection agency", "environmental protection agency"],
    "epa": ["environmental protection agency"],
}
WORDNUM = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
           "ten": "10", "twelve": "12", "thirty": "30", "sixty": "60", "ninety": "90"}


def low(s):
    return WS.sub(" ", s.lower().replace("–", "-"))


def money_key(s):
    s = s.lower().replace(",", "").replace("$", "").strip()
    m = re.match(r"([0-9\.]+)\s*(million|billion|thousand)?", s)
    if not m:
        return None
    v = float(m.group(1)) if m.group(1).replace(".", "", 1).isdigit() else None
    if v is None:
        return None
    mul = {"million": 1e6, "billion": 1e9, "thousand": 1e3}.get(m.group(2), 1)
    return round(v * mul)


class Evidence:
    """Normalised evidence index for one bill: bill text + resolved statute headings/text."""

    def __init__(self, bill, extra=""):
        self.txt = low(bill + " " + extra)
        self.money = set()
        for m in re.finditer(r"\$\s?[0-9][0-9,\.]*(?:\s+(?:million|billion|thousand))?", self.txt):
            k = money_key(m.group(0))
            if k is not None:
                self.money.add(k)
        self.years = set(re.findall(r"\b(?:19|20)[0-9]{2}\b", self.txt))
        self.nums = set(re.findall(r"\b[0-9]+(?:\.[0-9]+)?\b", self.txt))
        for w, d in WORDNUM.items():
            if re.search(r"\b" + w + r"\b", self.txt):
                self.nums.add(d)
        self.acts = set()
        for t, s in legal_entities(bill + " " + extra):
            if t == "ACT":
                self.acts.add(act_key(s))
        self.cites = set(re.findall(r"\b([0-9]{1,2}[a-z]?) u\.s\.c\. (?:app\. )?([0-9a-z\-]+)", self.txt))
        for _, disp, _, _, _ in find_usc(bill + " " + extra):
            m = re.match(r"([0-9]{1,2}[A-Za-z]?) U\.S\.C\. ([0-9A-Za-z\-\u2013]+)", disp)
            if m:
                self.cites.add((m.group(1).lower(), m.group(2).lower().replace("\u2013", "-")))

    def supported(self, typ, s):
        ls = low(s)
        if typ == "CITE":
            m = re.match(r"([0-9]{1,2}[a-z]?) u\.s\.c\. ([0-9a-z\-]+)", ls)
            return bool(m) and (m.group(1), m.group(2)) in self.cites
        if typ == "ACT":
            k = act_key(s)
            if k in self.acts or k in self.txt:
                return True
            k2 = re.sub(r" of [0-9]{4}$", "", k)
            return len(k2.split()) >= 2 and k2 in self.txt
        if typ == "AGENCY":
            if ls in self.txt:
                return True
            m = re.match(r"(?:the )?(department|secretary) of (?:the )?(.+)", ls)
            if m:
                tail = m.group(2)
                if ("secretary of " + tail) in self.txt or ("secretary of the " + tail) in self.txt or ("department of " + tail) in self.txt or ("department of the " + tail) in self.txt:
                    return True
            for a, alts in ALIAS.items():
                if a in ls and any(x in self.txt for x in alts):
                    return True
            core = re.sub(r"^(?:the )", "", ls)
            return core in self.txt
        if typ == "MONEY":
            k = money_key(s)
            return k is not None and k in self.money
        if typ == "DATE":
            ys = re.findall(r"(?:19|20)[0-9]{2}", s)
            return all(y in self.years for y in ys)
        if typ in ("DURATION", "PCT"):
            n = re.findall(r"[0-9]+(?:\.[0-9]+)?", ls)
            if not n:
                w = re.split(r"[- ]", ls)[0]
                n = [WORDNUM.get(w, w)]
            return all(x in self.nums or x in self.txt for x in n)
        return True


SENT = re.compile(r"(?<=[\.;])\s+(?=[A-Z(])")


def split_sents(s):
    s = s.strip()
    parts = [p.strip() for p in SENT.split(s) if p.strip()]
    return parts or [s]


def sentence_logprobs(pred, toks, lp, sents):
    """Map generator token log-probabilities to sentences by character alignment."""
    out = []
    pos = 0
    pieces = []
    acc = ""
    for t, l in zip(toks, lp):
        piece = t.replace("▁", " ")
        pieces.append((len(acc), len(acc) + len(piece), l))
        acc += piece
    acc_l = acc.strip()
    off = len(acc) - len(acc.lstrip())
    cur = 0
    for s in sents:
        i = acc_l.find(s[:20], cur) if s else -1
        if i < 0:
            i = cur
        j = i + len(s)
        cur = j
        ls = [l for a, b, l in pieces if a - off >= i - 1 and b - off <= j + 1]
        out.append(ls if ls else [0.0])
    return out


def provision_terms(p):
    terms = []
    if p.get("act"):
        terms.append(act_key(p["act"]))
    h = p.get("header") or ""
    if "Internal Revenue Code" in h:
        terms.append("internal revenue code")
    m = re.search(r"title ([0-9]{1,2}), United States Code", h)
    if m:
        terms.append("title %s" % m.group(1))
    if p.get("disp"):
        terms.append(low(p["disp"].split("(")[0]))
    if p.get("heading"):
        hw = [w for w in re.findall(r"[a-z]{4,}", low(p["heading"])) if w not in ("with", "from", "under", "certain", "other", "general", "relating")]
        if hw:
            terms.append(("HEAD", hw))
    return terms


def covers(text, p):
    lt = low(text)
    lt_k = act_key(text)
    for t in provision_terms(p):
        if isinstance(t, tuple):
            hw = t[1]
            if sum(w in lt for w in hw) >= max(1, math.ceil(0.6 * len(hw))):
                return True
        elif t and (t in lt or t in lt_k):
            return True
    return False


def score_summary(pred, ref, ev):
    r = SC.score(ref, pred)
    ents = legal_entities(pred)
    sup = [ev.supported(t, s) for t, s in ents]
    return {"r1": r["rouge1"].fmeasure, "r2": r["rouge2"].fmeasure, "rl": r["rougeL"].fmeasure,
            "n_ent": len(ents), "n_unsup": len(ents) - sum(sup),
            "n_cite": sum(t == "CITE" for t, _ in ents), "n_cite_unsup": sum((t == "CITE") and not s for (t, _), s in zip(ents, sup))}
