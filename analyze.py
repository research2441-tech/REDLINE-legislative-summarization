"""All quantitative results: generator comparison, paired statistics, two-sided CRC, transfer, sensitivity."""
import json, os, sys, random, collections
import numpy as np
from scipy.stats import wilcoxon
from evaluate import Evidence, legal_entities, covers, SC, split_sents
import crc as C

RNG = np.random.default_rng(2026)


def load_rows(tag):
    return [json.loads(l) for l in open("data/eval_%s.jsonl" % tag)]


def load_gen(arm, tag):
    p = "results/gen_%s_%s.jsonl" % (arm, tag)
    if not os.path.exists(p):
        return None
    return {g["i"]: g for g in (json.loads(l) for l in open(p))}


def per_bill(rows, gens, use_statute=True):
    out = {}
    for r in rows:
        g = gens.get(r["i"])
        if g is None:
            continue
        ev = Evidence(r["text"], r["evidence"] if use_statute else "")
        pred = g["pred"]
        sc = SC.score(r["summary"], pred)
        ents = legal_entities(pred)
        sup = [ev.supported(t, s) for t, s in ents]
        cites = [(t, ok) for (t, s), ok in zip(ents, sup) if t == "CITE"]
        provs = [p for p in r["provs"] if p.get("resolved") or (not r.get("n_resolved") and p.get("disp"))]
        sal = [p for p in provs if covers(r["summary"], p)]
        out[r["i"]] = {"r1": sc["rouge1"].fmeasure, "r2": sc["rouge2"].fmeasure, "rl": sc["rougeL"].fmeasure,
                       "n_ent": len(ents), "n_sup": sum(sup), "n_unsup": len(ents) - sum(sup),
                       "n_cite": len(cites), "n_cite_bad": sum(1 for _, ok in cites if not ok),
                       "n_sal": len(sal), "n_sal_cov": sum(covers(pred, p) for p in sal),
                       "words": len(pred.split()), "sec": g.get("sec", 0.0), "in_tokens": g.get("in_tokens", 0)}
    return out


def boot_ci(x, n=10000):
    x = np.asarray(x, float)
    idx = RNG.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def ratio_ci(num, den, n=10000):
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = RNG.integers(0, len(num), (n, len(num)))
    v = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def summarize(pb):
    v = list(pb.values())
    a = lambda k: [x[k] for x in v]
    efp = sum(a("n_sup")) / max(sum(a("n_ent")), 1)
    chr_ = sum(a("n_cite_bad")) / max(sum(a("n_cite")), 1)
    spr = sum(a("n_sal_cov")) / max(sum(a("n_sal")), 1)
    return {"n": len(v), "r1": float(np.mean(a("r1"))), "r1_ci": boot_ci(a("r1")), "r2": float(np.mean(a("r2"))), "r2_ci": boot_ci(a("r2")),
            "rl": float(np.mean(a("rl"))), "rl_ci": boot_ci(a("rl")), "efp": efp, "efp_ci": ratio_ci(a("n_sup"), a("n_ent")),
            "unsup": float(np.mean(a("n_unsup"))), "ents": float(np.mean(a("n_ent"))), "cites": float(np.mean(a("n_cite"))),
            "chr": chr_, "spr": spr, "spr_ci": ratio_ci(a("n_sal_cov"), a("n_sal")), "words": float(np.mean(a("words"))),
            "sec": float(np.mean(a("sec")))}


def paired(pa, pb_, keys=("r1", "r2", "rl", "efp_b", "unsup", "spr_b")):
    ids = sorted(set(pa) & set(pb_))
    res = {}
    for k in keys:
        if k == "efp_b":
            ii = [i for i in ids if pa[i]["n_ent"] > 0 and pb_[i]["n_ent"] > 0]
            x = np.array([pa[i]["n_sup"] / pa[i]["n_ent"] for i in ii]); y = np.array([pb_[i]["n_sup"] / pb_[i]["n_ent"] for i in ii])
        elif k == "spr_b":
            ii = [i for i in ids if pa[i]["n_sal"] > 0]
            x = np.array([pa[i]["n_sal_cov"] / pa[i]["n_sal"] for i in ii]); y = np.array([pb_[i]["n_sal_cov"] / pb_[i]["n_sal"] for i in ii])
        elif k == "unsup":
            x = np.array([pa[i]["n_unsup"] for i in ids], float); y = np.array([pb_[i]["n_unsup"] for i in ids], float)
        else:
            x = np.array([pa[i][k] for i in ids]); y = np.array([pb_[i][k] for i in ids])
        dlt = x - y
        nz = dlt[dlt != 0]
        if len(nz) == 0:
            p, rb = 1.0, 0.0
        else:
            p = float(wilcoxon(x, y, zero_method="wilcox").pvalue)
            rk = np.argsort(np.argsort(np.abs(nz))) + 1.0
            # average ranks for ties
            from scipy.stats import rankdata
            rk = rankdata(np.abs(nz))
            wp, wm = rk[nz > 0].sum(), rk[nz < 0].sum()
            rb = float((wp - wm) / (wp + wm))
        res[k] = {"n": len(dlt), "mean_a": float(x.mean()), "mean_b": float(y.mean()), "diff": float(dlt.mean()), "ci": boot_ci(dlt), "p": p, "rb": rb}
    # Holm
    ps = sorted(res.items(), key=lambda kv: kv[1]["p"])
    m = len(ps); run = 0
    for j, (k, v) in enumerate(ps):
        adj = min(1.0, (m - j) * v["p"])
        run = max(run, adj)
        v["p_holm"] = run
    return res


def main():
    R = {}
    rows = load_rows("usamd")
    test = [r for r in rows if r["role"] == "test"]
    ca = load_rows("ca")
    arms = ["LEAD", "TEXTRANK", "B0", "BILL", "ENT", "OPS", "REDLINE", "REDLINE_ADD"]
    PB = {}
    for a in arms:
        g = load_gen(a, "usamd")
        if g:
            PB[a] = per_bill(test, g)
            R.setdefault("main", {})[a] = summarize(PB[a])
    # reference floor of the checker
    ref_pb = per_bill(test, {r["i"]: {"pred": r["summary"]} for r in test})
    R["reference"] = summarize(ref_pb)
    tot = sum(len([p for p in r["provs"] if p.get("resolved")]) for r in test)
    sal = sum(sum(covers(r["summary"], p) for p in r["provs"] if p.get("resolved")) for r in test)
    R["salient_share"] = sal / max(tot, 1)
    R["n_resolved_provs_test"] = tot
    # paired tests
    R["paired"] = {}
    for a, b in [("REDLINE", "BILL"), ("REDLINE", "B0"), ("REDLINE", "ENT"), ("REDLINE", "OPS"), ("OPS", "BILL"), ("ENT", "BILL"), ("BILL", "B0"), ("REDLINE_ADD", "BILL"), ("REDLINE_ADD", "REDLINE")]:
        if a in PB and b in PB:
            R["paired"]["%s_vs_%s" % (a, b)] = paired(PB[a], PB[b])
    # external subset (first 100 test bills)
    sub_ids = None
    for a in ["PEGASUS", "QWEN", "QWEN_RL"]:
        g = load_gen(a, "usamd")
        if g:
            sub_ids = set(g) if sub_ids is None else sub_ids & set(g)
    if sub_ids:
        sub = [r for r in test if r["i"] in sub_ids]
        R["external"] = {}
        XPB = {}
        for a in ["PEGASUS", "QWEN", "QWEN_RL", "B0", "BILL", "REDLINE", "REDLINE_ADD", "LEAD"]:
            g = load_gen(a, "usamd")
            if g:
                XPB[a] = per_bill(sub, g)
                R["external"][a] = summarize(XPB[a])
        if "QWEN_RL" in XPB and "QWEN" in XPB:
            R["paired"]["QWEN_RL_vs_QWEN"] = paired(XPB["QWEN_RL"], XPB["QWEN"])
    # California transfer (evidence = bill only for every arm)
    R["ca"] = {}
    PBC = {}
    for a in ["LEAD", "TEXTRANK", "BILL", "REDLINE"]:
        g = load_gen(a, "ca")
        if g:
            PBC[a] = per_bill(ca, g, use_statute=False)
            R["ca"][a] = summarize(PBC[a])
    if "REDLINE" in PBC and "BILL" in PBC:
        R["ca_paired"] = paired(PBC["REDLINE"], PBC["BILL"])
    # by number of resolved provisions (failure / heterogeneity)
    if "REDLINE" in PB and "BILL" in PB:
        het = {}
        for lo, hi, name in [(1, 1, "1"), (2, 3, "2-3"), (4, 999, ">=4")]:
            ids = [r["i"] for r in test if lo <= r["n_resolved"] <= hi]
            sa = {i: PB["REDLINE"][i] for i in ids}; sb = {i: PB["BILL"][i] for i in ids}
            het[name] = {"n": len(ids), "REDLINE": summarize(sa), "BILL": summarize(sb)}
        R["by_nres"] = het
    json.dump(R, open("results/summary_gen.json", "w"), indent=1)
    print(json.dumps({k: {a: {m: round(v[m], 4) for m in ["r1", "r2", "rl", "efp", "unsup", "chr", "spr", "words"]} for a, v in R[k].items()} for k in ["main", "ca"] if k in R}, indent=1))
    print("ref", {m: round(R["reference"][m], 4) for m in ["efp", "unsup", "spr", "chr"]}, "salient", R["salient_share"])


if __name__ == "__main__":
    main()
