"""REDLINE core: legal entity extraction, US Code resolution, amendment compilation and execution."""
import re
from collections import defaultdict

WS = re.compile(r"\s+")


def norm(t):
    return WS.sub(" ", t.replace("’", "'").replace("“", "``").replace("”", "''")).strip()


# ---------------------------------------------------------------- citations
SEC_NUM = r"[0-9]+[A-Za-z]*(?:[-–][0-9]+[A-Za-z]*)*"
SUBDIV = r"(?:\([0-9A-Za-z]+\))*"
USC_PAREN = re.compile(r"\b([0-9]{1,2}[A-Za-z]?)\s+U\.\s?S\.\s?C\.\s+(?:App\.\s+)?(" + SEC_NUM + r")(" + SUBDIV + r")")
USC_TITLE = re.compile(r"[Ss]ection\s+(" + SEC_NUM + r")(" + SUBDIV + r")\s+of\s+title\s+([0-9]{1,2}),\s+United\s+States\s+Code")


def usc_id(title, sec):
    return "/us/usc/t%s/s%s" % (title.lower(), sec.replace("-", "–"))


def find_usc(text):
    """Return list of (identifier, display, subdiv, start, end)."""
    out = []
    for m in USC_PAREN.finditer(text):
        out.append((usc_id(m.group(1), m.group(2)), "%s U.S.C. %s%s" % (m.group(1), m.group(2), m.group(3)), m.group(3), m.start(), m.end()))
    for m in USC_TITLE.finditer(text):
        out.append((usc_id(m.group(3), m.group(1)), "%s U.S.C. %s%s" % (m.group(3), m.group(1), m.group(2)), m.group(2), m.start(), m.end()))
    out.sort(key=lambda x: x[3])
    return out


ACT = re.compile(r"\b((?:[A-Z][A-Za-z'\-\.,&]*\s+(?:of\s+|and\s+|for\s+|the\s+|on\s+|to\s+|in\s+)?){1,12}Act(?:\s+of\s+[0-9]{4})?)")
AGENCY = re.compile(r"\b((?:Department|Secretary|Office|Bureau|Administration|Agency|Commission|Service|Board|Corps|Administrator|Comptroller General|Attorney General)(?:\s+(?:of|for|on|the|and|[A-Z][a-z]+)){0,8})")
MONEY = re.compile(r"\$\s?[0-9][0-9,\.]*(?:\s+(?:million|billion|thousand))?")
YEARS = re.compile(r"\b(?:fiscal\s+years?\s+)?(?:19|20)[0-9]{2}(?:\s+(?:through|and)\s+(?:19|20)[0-9]{2})?\b")
DURATION = re.compile(r"\b([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|thirty|sixty|ninety)\s*[- ]\s*(days?|months?|years?)\b", re.I)
PCT = re.compile(r"\b[0-9]+(?:\.[0-9]+)?\s?(?:percent|%)")

STOP_ACT = {"This Act", "The Act", "Act", "Such Act", "That Act", "Act of"}


def clean_act(a):
    a = a.strip(" ,.")
    a = a.split("\n")[-1].strip()
    prev = None
    while prev != a:
        prev = a
        a = re.sub(r"^(?:The|the|This|this|Such|such|Amends|amends|Requires|Under|under|Section|In|To|A|An|And|Of|Fund\.?)\s+", "", a)
    return a


def legal_entities(text):
    """Typed legal entities: list of (type, surface)."""
    ents = []
    for i, d, s, a, b in find_usc(text):
        ents.append(("CITE", d.split("(")[0]))
    for m in ACT.finditer(text):
        a = clean_act(m.group(1))
        if a not in STOP_ACT and len(a.split()) >= 2:
            ents.append(("ACT", a))
    for m in AGENCY.finditer(text):
        a = m.group(1).strip(" ,.")
        a = re.sub(r"\s+(?:and|the|of|for|on)$", "", a)
        if len(a.split()) >= 2:
            ents.append(("AGENCY", a))
    for m in MONEY.finditer(text):
        ents.append(("MONEY", m.group(0).rstrip(".,")))
    for m in YEARS.finditer(text):
        ents.append(("DATE", m.group(0)))
    for m in DURATION.finditer(text):
        ents.append(("DURATION", m.group(0)))
    for m in PCT.finditer(text):
        ents.append(("PCT", m.group(0)))
    return ents


# ---------------------------------------------------------------- amendments
Q = r"``(.*?)''"
HEADER = re.compile(
    r"((?:[Ss]ection|[Ss]ections|[Pp]aragraph|[Ss]ubsection|[Tt]itle|[Cc]hapter|[Pp]art|[Ss]ubtitle|[Ss]ubchapter)\s+[0-9A-Za-z\-–]+[^.;]{0,260}?)\s+is\s+(?:further\s+)?(amended|repealed|hereby repealed)")
OPS = [
    ("STRIKE_THROUGH_INSERT", re.compile(r"by striking " + Q + r" and all that follows through " + Q + r"(?: and inserting " + Q + r")?")),
    ("STRIKE_INSERT", re.compile(r"by striking " + Q + r"(?: each place (?:it|such term) appears)?(?:,)? and inserting " + Q)),
    ("INSERT_AFTER", re.compile(r"by inserting " + Q + r" (after|before) " + Q)),
    ("STRIKE", re.compile(r"by striking " + Q + r"(?! and)")),
    ("ADD_END", re.compile(r"by adding at the end the following")),
    ("INSERT_STRUCT", re.compile(r"by inserting (?:after|before) ((?:section|subsection|paragraph|subparagraph|clause|part|chapter)\s+[^ ]+)")),
    ("STRIKE_STRUCT", re.compile(r"by striking ((?:section|subsection|paragraph|subparagraph|clause|part)s?\s+\([0-9A-Za-z]+\)(?:\s+(?:and|through)\s+\([0-9A-Za-z]+\))?)")),
    ("REDESIGNATE", re.compile(r"by redesignating ([^;]{1,120}?) as ([^;,]{1,80})")),
]
SCOPE = re.compile(r"\bin (subsection|paragraph|subparagraph|clause)\s+(\([0-9A-Za-z]+\)(?:\([0-9A-Za-z]+\))*)")
READ_AS = re.compile(r"is amended to read as follows")


def split_designators(s):
    return re.findall(r"\(([0-9A-Za-z]+)\)", s or "")


def act_key(a):
    return re.sub(r"[^a-z0-9 ]", "", a.lower()).replace("the ", "").strip()


def parse_amendments(text, concord=None):
    """Compile amendatory instructions into an edit program.

    Returns list of dicts {target_id, target_disp, subdiv, kind, ops:[...], header}.
    """
    t = norm(text)
    ctx = {}
    progs = []
    heads = list(HEADER.finditer(t))
    for k, h in enumerate(heads):
        end = heads[k + 1].start() if k + 1 < len(heads) else len(t)
        nxt = re.search(r"\bSEC\. [0-9]+\.", t[h.end():end])
        if nxt:
            end = h.end() + nxt.start()
        body = t[h.end():end]
        head = h.group(1)
        cites = find_usc(head)
        target = None
        how = None
        act = None
        am = re.search(r"of (?:the )?((?:[A-Z][^,()]*?)Act(?: of [0-9]{4})?)", head)
        if am:
            act = am.group(1).strip()
        sm = re.match(r"[Ss]ections?\s+(" + SEC_NUM + r")(" + SUBDIV + r")", head)
        if cites:
            target = cites[0]; how = "explicit"
        else:
            m = USC_TITLE.search(head + " " + t[h.end():h.end() + 5])
            tm = re.search(r"\btitle\s+([0-9]{1,2}),\s+United\s+States\s+Code", head)
            if m:
                target = (usc_id(m.group(3), m.group(1)), "%s U.S.C. %s%s" % (m.group(3), m.group(1), m.group(2)), m.group(2), 0, 0); how = "explicit"
            elif sm:
                ttl = None
                if "Internal Revenue Code" in head:
                    ttl = "26"; how = "code_alias"
                elif re.search(r"\b(?:such|the|that|this) Code\b", head):
                    ttl = ctx.get("code"); how = "anaphora"
                elif tm:
                    ttl = tm.group(1); how = "explicit"
                elif re.search(r"\b(?:such|that|this) title\b", head):
                    ttl = ctx.get("title"); how = "anaphora"
                elif act and concord is not None:
                    key = (act_key(act), sm.group(1))
                    if key in concord:
                        tt, ss = concord[key]
                        target = (usc_id(tt, ss), "%s U.S.C. %s%s" % (tt, ss, sm.group(2)), sm.group(2), 0, 0); how = "concordance"
                elif " of " not in head and ctx.get("last"):
                    ttl = ctx["last"]; how = "anaphora"
                if ttl and target is None:
                    target = (usc_id(ttl, sm.group(1)), "%s U.S.C. %s%s" % (ttl, sm.group(1), sm.group(2)), sm.group(2), 0, 0)
        if "Internal Revenue Code" in head:
            ctx["code"] = "26"; ctx["last"] = "26"
        tm2 = re.search(r"\btitle\s+([0-9]{1,2}),\s+United\s+States\s+Code", head)
        if tm2:
            ctx["title"] = tm2.group(1); ctx["last"] = tm2.group(1)
        if target is not None and how in ("code_alias", "anaphora", "explicit"):
            ctx["last"] = target[0].split("/t")[1].split("/")[0]
        if "et seq" in head and target is not None:
            target = (target[0], target[1], target[2], target[3], target[4])
        kind = h.group(2)
        ops = []
        if "repealed" in kind:
            ops.append({"op": "REPEAL", "scope": []})
        elif READ_AS.match(body.strip()) or body.strip().startswith("to read as follows"):
            m = re.search(r"``(.{0,600})", body)
            ops.append({"op": "READ_AS", "scope": [], "new": (m.group(1) if m else "")[:600]})
        else:
            # walk the body, tracking scope phrases
            pos = 0
            scope = []
            events = []
            for m in SCOPE.finditer(body):
                events.append((m.start(), "scope", m))
            for name, rx in OPS:
                for m in rx.finditer(body):
                    events.append((m.start(), name, m))
            events.sort(key=lambda e: e[0])
            taken = []
            for st, name, m in events:
                if any(a <= st < b for a, b in taken):
                    continue
                if name == "scope":
                    scope = split_designators(m.group(2))
                    continue
                taken.append((m.start(), m.end()))
                o = {"op": name, "scope": list(scope)}
                g = m.groups()
                if name == "STRIKE_INSERT":
                    o.update(old=g[0], new=g[1])
                elif name == "STRIKE":
                    o.update(old=g[0])
                elif name == "STRIKE_THROUGH_INSERT":
                    o.update(old=g[0], through=g[1], new=g[2] or "")
                elif name == "INSERT_AFTER":
                    o.update(new=g[0], where=g[1], anchor=g[2])
                elif name == "ADD_END":
                    q = re.search(r"``(.{0,400})", body[m.end():m.end() + 500])
                    o.update(new=(q.group(1) if q else ""))
                elif name == "INSERT_STRUCT":
                    o.update(anchor_struct=g[0])
                    q = re.search(r"``(.{0,400})", body[m.end():m.end() + 500])
                    o.update(new=(q.group(1) if q else ""))
                elif name == "STRIKE_STRUCT":
                    o.update(struct=g[0])
                elif name == "REDESIGNATE":
                    o.update(src=g[0], dst=g[1])
                ops.append(o)
        if not ops:
            continue
        progs.append({"target_id": target[0] if target else None,
                      "target_disp": target[1] if target else None,
                      "subdiv": split_designators(target[2]) if target else [],
                      "act": act, "how": how, "header": head[:300], "ops": ops})
    return progs


# ---------------------------------------------------------------- execution
def locate(stext, path):
    """Find region of statute text addressed by designator path; returns (a,b) or None."""
    a, b = 0, len(stext)
    for d in path:
        rx = re.compile(r"(?:^|\n)\s*\(" + re.escape(d) + r"\)")
        m = rx.search(stext, a, b)
        if not m:
            return None
        start = m.start()
        lvl = level(d)
        nxt = None
        for m2 in re.finditer(r"\n\s*\(([0-9A-Za-z]+)\)", stext[m.end():b]):
            if level(m2.group(1)) == lvl or levelrank(m2.group(1)) < levelrank(d):
                nxt = m.end() + m2.start()
                break
        a, b = start, (nxt if nxt else b)
    return (a, b)


ROMAN = re.compile(r"^(?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv)$")


def level(d):
    if d.isdigit():
        return "para"
    if ROMAN.match(d):
        return "clause"
    if d.islower():
        return "sub"
    if d.isupper() and ROMAN.match(d.lower()) and len(d) > 1:
        return "subclause"
    return "subpara"


def levelrank(d):
    return {"sub": 0, "para": 1, "subpara": 2, "clause": 3, "subclause": 4}[level(d)]


def wnorm(s):
    return WS.sub(" ", s).strip()


def tolerant_find(hay, needle):
    """Case-insensitive, whitespace-flexible match with optional trailing punctuation."""
    core = needle.strip().rstrip(".,;")
    if len(core) < 2:
        return -1
    rx = re.compile(r"\s+".join(re.escape(w) for w in core.split()) + r"[.,;]?", re.I)
    m = rx.search(hay)
    return m.start() if m else -1


TOLERANT = False


def execute(ops, stext, subdiv):
    """Execute an edit program on one statute text. Returns (n_exec, n_total, hunks, applied_already)."""
    hunks = []
    n_exec = 0
    already = 0
    flat = stext
    for o in ops:
        path = list(subdiv) + [d for d in o.get("scope", []) if d not in subdiv]
        reg = locate(flat, path) if path else (0, len(flat))
        if reg is None:
            reg = locate(flat, subdiv) if subdiv else None
        if reg is None:
            reg = (0, len(flat))
        region = wnorm(flat[reg[0]:reg[1]])
        whole = wnorm(flat)
        ok = False
        ctx = ""
        op = o["op"]
        if op in ("STRIKE_INSERT", "STRIKE", "STRIKE_THROUGH_INSERT"):
            old = wnorm(o["old"])
            i = region.find(old)
            if i < 0:
                i2 = whole.find(old)
                if i2 >= 0:
                    i, region = i2, whole
            if i < 0 and TOLERANT and old:
                i = tolerant_find(whole, old)
                if i >= 0:
                    region = whole
            if i >= 0 and old:
                ok = True
                ctx = region[max(0, i - 120):i + len(old) + 120]
            elif o.get("new") and wnorm(o["new"]) and whole.find(wnorm(o["new"])) >= 0:
                already += 1
        elif op == "INSERT_AFTER":
            anc = wnorm(o["anchor"])
            i = region.find(anc)
            if i < 0:
                i = whole.find(anc)
                region = whole
            if i >= 0 and anc:
                ok = True
                ctx = region[max(0, i - 120):i + len(anc) + 120]
                if whole.find(wnorm(o["new"])) >= 0 and len(o["new"]) > 12:
                    already += 1
        elif op in ("STRIKE_STRUCT", "INSERT_STRUCT"):
            ds = split_designators(o.get("struct") or o.get("anchor_struct") or "")
            if not ds:
                m = re.search(r"\s([0-9]+[a-z]*)$", o.get("anchor_struct") or "")
                ok = bool(m)
            else:
                r2 = locate(flat, list(subdiv) + ds[:1])
                ok = r2 is not None
                if ok:
                    ctx = wnorm(flat[r2[0]:r2[1]])[:240]
        elif op == "REDESIGNATE":
            ds = split_designators(o["src"])
            ok = bool(ds) and locate(flat, list(subdiv) + ds[:1]) is not None
        elif op in ("ADD_END", "READ_AS", "REPEAL"):
            ok = len(region) > 0
            ctx = region[:240]
        if ok:
            n_exec += 1
        hunks.append({"op": op, "ok": ok, "ctx": ctx, "old": o.get("old", ""), "new": o.get("new", ""),
                      "path": path})
    return n_exec, len(ops), hunks, already


def select_version(ops, subdiv, versions):
    """Satisfiability-based temporal anchoring: choose the provision version under which the
    edit program is most executable; ties broken toward the earliest release."""
    best = None
    for v in sorted(versions, key=lambda r: r["first_release_seq"] or 0):
        ne, nt, h, al = execute(ops, v["text"] or "", subdiv)
        score = ne / max(nt, 1)
        if best is None or score > best[0] + 1e-9:
            best = (score, v, h, al, ne, nt)
    return best


def render_redline(prog, best, heading, max_chars=420):
    """Serialise one executed edit program as a compact redline string for the generator."""
    disp = prog["target_disp"] or prog["header"][:60]
    parts = ["[%s | %s]" % (disp, (heading or "").strip()[:90])]
    for hk in best[2][:4]:
        op = hk["op"]
        if op == "STRIKE_INSERT":
            parts.append("replace \"%s\" with \"%s\"" % (hk["old"][:80], hk["new"][:80]))
        elif op == "STRIKE":
            parts.append("strike \"%s\"" % hk["old"][:80])
        elif op == "STRIKE_THROUGH_INSERT":
            parts.append("strike from \"%s\" and insert \"%s\"" % (hk["old"][:60], hk["new"][:60]))
        elif op == "INSERT_AFTER":
            parts.append("insert \"%s\"" % hk["new"][:80])
        elif op == "READ_AS":
            parts.append("rewrite provision")
        elif op == "REPEAL":
            parts.append("repeal")
        else:
            parts.append(op.lower().replace("_", " "))
        if hk["ok"] and hk["ctx"]:
            parts.append("current law: \"%s\"" % hk["ctx"][:160])
    s = " ; ".join(parts)
    return s[:max_chars]


CONC = re.compile(r"[Ss]ection\s+(" + SEC_NUM + r")" + SUBDIV + r"\s+of\s+(?:the\s+)?((?:[A-Z][^,()]*?)Act(?:\s+of\s+[0-9]{4})?)\s+\(([0-9]{1,2})\s+U\.S\.C\.\s+(" + SEC_NUM + r")")


def build_concordance(texts):
    """Corpus-induced act-to-Code concordance: (act, act section) -> (title, code section)."""
    votes = defaultdict(lambda: defaultdict(int))
    for tx in texts:
        for m in CONC.finditer(norm(tx)):
            votes[(act_key(m.group(2)), m.group(1))][(m.group(3), m.group(4))] += 1
    return {k: max(v.items(), key=lambda kv: kv[1])[0] for k, v in votes.items()}
