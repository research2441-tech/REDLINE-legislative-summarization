"""E1 detailed: per-operation executability under three version policies, and executor robustness to
instruction-text noise (whitespace, quote style, case, trailing punctuation)."""
import pickle, json, collections, random, re
from redline_core import execute, select_version
D = pickle.load(open("data/store_test.pkl", "rb"))
store, parsed = D["store"], D["parsed"]["test"]
pol = {k: collections.Counter() for k in ["anchor", "current", "earliest"]}
tot = collections.Counter()
progs = []
for b in parsed:
    for p in b["progs"]:
        vs = store.get(p["target_id"]) if p["target_id"] else None
        if not vs:
            continue
        best = select_version(p["ops"], p["subdiv"], vs)
        cur = [v for v in vs if v["is_current"]] or [max(vs, key=lambda v: v["last_release_seq"] or 0)]
        ear = min(vs, key=lambda v: v["first_release_seq"] or 0)
        hc = execute(p["ops"], cur[0]["text"] or "", p["subdiv"])[2]
        he = execute(p["ops"], ear["text"] or "", p["subdiv"])[2]
        for ha, hcc, hee in zip(best[2], hc, he):
            tot[ha["op"]] += 1
            pol["anchor"][ha["op"]] += ha["ok"]; pol["current"][ha["op"]] += hcc["ok"]; pol["earliest"][ha["op"]] += hee["ok"]
        progs.append((p, best[1]))
out = {"total": dict(tot), **{k: dict(v) for k, v in pol.items()}, "n_versions_mean": sum(len(store[p["target_id"]]) for p, _ in progs) / len(progs)}
# robustness to instruction noise on text-anchored ops
R = random.Random(3)
def noisy(s, kind):
    if kind == "ws":
        return re.sub(" ", lambda m: "  " if R.random() < .3 else " ", s)
    if kind == "case":
        return s[:1].swapcase() + s[1:]
    if kind == "punct":
        return s.rstrip(".,;") if s[-1:] in ".,;" else s + ","
    if kind == "quote":
        return s.replace("'", "’")
    return s
import redline_core
rob = {}
for tol in [False, True]:
    redline_core.TOLERANT = tol
    for kind in ["none", "ws", "case", "punct", "quote"]:
        ok = n = 0
        for p, v in progs:
            ops = []
            for o in p["ops"]:
                o2 = dict(o)
                if o["op"] in ("STRIKE", "STRIKE_INSERT") and o.get("old"):
                    o2["old"] = noisy(o["old"], kind)
                    ops.append(o2)
            if not ops:
                continue
            ne, nt, _, _ = redline_core.execute(ops, v["text"] or "", p["subdiv"])
            ok += ne; n += nt
        rob[("tol_" if tol else "strict_") + kind] = [ok, n]
out["robust"] = rob
json.dump(out, open("results/exec_stats2.json", "w"), indent=1)
print(json.dumps(out["robust"]))
