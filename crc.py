"""Two-sided conformal risk control over legal-entity hallucination and amended-provision omission."""
import json, math, re
import numpy as np
from sklearn.linear_model import LogisticRegression
from evaluate import Evidence, split_sents, sentence_logprobs, legal_entities, covers, low

ETYPES = ["CITE", "ACT", "AGENCY", "MONEY", "DATE", "DURATION", "PCT"]


def lexical_support(sent, evtxt):
    w = re.findall(r"[a-z0-9]{3,}", low(sent))
    if not w:
        return 1.0
    return sum(x in evtxt for x in w) / len(w)


def doc_units(row, gen, use_statute=True):
    """Sentence units with features/labels, and provision units with features/labels."""
    ev = Evidence(row["text"], row["evidence"] if use_statute else "")
    sents = split_sents(gen["pred"])
    lps = sentence_logprobs(gen["pred"], gen.get("toks", []), gen.get("lp", []), sents) if gen.get("lp") else [[0.0]] * len(sents)
    S = []
    for j, (s, lp) in enumerate(zip(sents, lps)):
        ents = legal_entities(s)
        sup = [ev.supported(t, x) for t, x in ents]
        lp = np.array(lp, dtype=float)
        f_conf = [float(lp.mean()), float(lp.min()), float(np.percentile(lp, 10))]
        # grounding features computable at inference: lexical overlap with the evidence and entity-type counts
        f_ground = [lexical_support(s, ev.txt)] + [sum(1 for t, _ in ents if t == et) for et in ETYPES]
        f_misc = [j / max(len(sents) - 1, 1), len(s.split()) / 40.0, len(ents)]
        S.append({"conf": f_conf, "ground": f_ground, "misc": f_misc, "n_ent": len(ents), "n_bad": len(ents) - sum(sup)})
    P = []
    for p in row["provs"]:
        if not (p.get("resolved") or p.get("disp")):
            continue
        if not p.get("resolved") and row.get("n_resolved", 0) > 0:
            continue
        ops = p.get("ops", [])
        feat = [p.get("pos", 0.5), p.get("rank", 0) / max(p.get("n_prog", 1), 1), math.log1p(p.get("edit_chars", 0)), p.get("n_ops", 1),
                float(any(o in ("ADD_END", "READ_AS", "INSERT_STRUCT") for o in ops)), float(any(o in ("STRIKE_INSERT", "STRIKE") for o in ops)),
                float(bool(p.get("act")) and low(p["act"]) in low(row["title"])), 1.0 / max(p.get("n_prog", 1), 1)]
        P.append({"p": p, "feat": feat, "y": int(covers(row["summary"], p))})
    return S, P


def sent_X(S, mode):
    if mode == "conf":
        return [u["conf"] + u["misc"] for u in S]
    return [u["conf"] + u["ground"] + u["misc"] for u in S]


class Controller:
    def __init__(self, mode="full"):
        self.mode = mode

    def fit(self, docs):
        Xs, ys, Xp, yp = [], [], [], []
        for S, P, _ in docs:
            Xs += sent_X(S, self.mode); ys += [int(u["n_bad"] > 0) for u in S]
            Xp += [q["feat"] for q in P]; yp += [q["y"] for q in P]
        self.hs = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(np.array(Xs), np.array(ys)) if len(set(ys)) > 1 else None
        self.hp = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(np.array(Xp), np.array(yp))
        return self

    def score(self, S, P):
        rs = self.hs.predict_proba(np.array(sent_X(S, self.mode)))[:, 1] if (S and self.hs is not None) else np.zeros(len(S))
        qs = self.hp.predict_proba(np.array([q["feat"] for q in P]))[:, 1] if P else np.zeros(0)
        return rs, qs


GRID = np.linspace(0, 1, 401)


def hall_loss(S, rs, lam):
    n = sum(u["n_ent"] for u in S)
    if n == 0:
        return 0.0
    return sum(u["n_bad"] for u, r in zip(S, rs) if r <= lam) / n


def omit_loss(P, qs, covered, mu):
    if not P:
        return 0.0
    return sum(1 for q, s, c in zip(P, qs, covered) if q["y"] == 1 and not c and s < mu) / len(P)


def published_text(sents, rs, lam):
    return " ".join(s for s, r in zip(sents, rs) if r <= lam)


def crc_threshold(losses_by_grid, alpha, B=1.0):
    """losses_by_grid: array [n_docs, len(GRID)] nondecreasing along grid. Largest admissible threshold."""
    n = losses_by_grid.shape[0]
    R = losses_by_grid.mean(0)
    ok = (n / (n + 1)) * R + B / (n + 1) <= alpha
    idx = np.where(ok)[0]
    return GRID[idx.max()] if len(idx) else -1.0


def prepare(rows, gens, ctrl, use_statute=True):
    """Precompute per-document loss curves over the grid for both controllers."""
    out = []
    for r in rows:
        g = gens[r["i"]]
        S, P = doc_units(r, g, use_statute)
        rs, qs = ctrl.score(S, P)
        sents = split_sents(g["pred"])
        u = np.array([x["n_bad"] for x in S], float); ntot = max(sum(x["n_ent"] for x in S), 1)
        Lh = ((rs[None, :] <= GRID[:, None]) * u[None, :]).sum(1) / ntot if len(S) else np.zeros(len(GRID))
        out.append({"S": S, "P": P, "rs": rs, "qs": qs, "sents": sents, "Lh": Lh, "row": r})
    return out


def coverage(d, lam):
    c = d.setdefault("_cov", {})
    if lam not in c:
        pub = published_text(d["sents"], d["rs"], lam)
        c[lam] = [covers(pub, q["p"]) for q in d["P"]]
    return c[lam]


def omission_curves(D, lam):
    for d in D:
        cov = coverage(d, lam)
        d["cov"] = cov
        if d["P"]:
            rel = np.array([s for q, s, c in zip(d["P"], d["qs"], cov) if q["y"] == 1 and not c], float)
            d["Lo"] = (rel[None, :] < GRID[:, None]).sum(1) / len(d["P"]) if len(rel) else np.zeros(len(GRID))
        else:
            d["Lo"] = np.zeros(len(GRID))
    return D


def evaluate_policy(D, lam, mu):
    Rh, Ro, load_s, load_p, kept = [], [], [], [], []
    for d in D:
        Rh.append(hall_loss(d["S"], d["rs"], lam))
        cov = coverage(d, lam)
        Ro.append(omit_loss(d["P"], d["qs"], cov, mu))
        load_s.append(sum(1 for r in d["rs"] if r > lam))
        load_p.append(sum(1 for q, s, c in zip(d["P"], d["qs"], cov) if (not c) and s >= mu))
        kept.append(sum(1 for r in d["rs"] if r <= lam) / max(len(d["rs"]), 1))
    return {"risk_h": float(np.mean(Rh)), "risk_o": float(np.mean(Ro)), "flag_sent": float(np.mean(load_s)),
            "flag_prov": float(np.mean(load_p)), "load": float(np.mean(np.array(load_s) + np.array(load_p))), "kept": float(np.mean(kept))}
