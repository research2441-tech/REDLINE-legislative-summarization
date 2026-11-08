"""E1: resolution and execution of amendment programs; temporal anchoring vs fixed-version baselines."""
import pickle, json, collections, time
from redline_core import execute, select_version

D = pickle.load(open("data/store_test.pkl", "rb"))
store, parsed = D["store"], D["parsed"]


def run(split):
    res = collections.Counter()
    opc = collections.Counter(); opok = collections.Counter()
    per_bill = []
    for b in parsed[split]:
        bill = {"n_prog": 0, "n_res": 0, "ops": 0, "ex_anchor": 0, "ex_current": 0, "ex_earliest": 0, "already": 0}
        for p in b["progs"]:
            bill["n_prog"] += 1
            res["prog"] += 1
            if p["target_id"] is None:
                res["no_cite"] += 1
                continue
            vs = store.get(p["target_id"])
            if not vs:
                res["unresolved"] += 1
                continue
            res["resolved"] += 1
            res["how_" + str(p.get("how"))] += 1
            bill["n_res"] += 1
            best = select_version(p["ops"], p["subdiv"], vs)
            cur = [v for v in vs if v["is_current"]] or [max(vs, key=lambda v: v["last_release_seq"] or 0)]
            ne_c, nt, hc, _ = execute(p["ops"], cur[0]["text"] or "", p["subdiv"])
            ear = min(vs, key=lambda v: v["first_release_seq"] or 0)
            ne_e, _, _, _ = execute(p["ops"], ear["text"] or "", p["subdiv"])
            bill["ops"] += nt
            bill["ex_anchor"] += best[4]; bill["ex_current"] += ne_c; bill["ex_earliest"] += ne_e
            bill["already"] += best[3]
            if (cur[0]["status"] or "") in ("transferred", "repealed", "omitted", "renumbered"):
                res["current_status_" + cur[0]["status"]] += 1
            if best[1]["identifier"] and not best[1]["is_current"]:
                res["anchored_noncurrent"] += 1
            for hk in best[2]:
                opc[hk["op"]] += 1; opok[hk["op"]] += hk["ok"]
        per_bill.append(bill)
    return res, opc, opok, per_bill


out = {}
for split in ["test"]:
    t = time.time()
    res, opc, opok, pb = run(split)
    ops = sum(b["ops"] for b in pb)
    s = {"programs": res["prog"], "with_citation": res["prog"] - res["no_cite"], "resolved": res["resolved"],
         "unresolved": res["unresolved"], "anchored_noncurrent": res["anchored_noncurrent"],
         "transferred_current": res["current_status_transferred"], "repealed_current": res["current_status_repealed"],
         "ops": ops, "exec_anchor": sum(b["ex_anchor"] for b in pb), "exec_current": sum(b["ex_current"] for b in pb),
         "exec_earliest": sum(b["ex_earliest"] for b in pb), "already_applied": sum(b["already"] for b in pb),
         "bills": len(pb), "how": {k: v for k, v in res.items() if k.startswith("how_")}, "bills_with_prog": sum(b["n_prog"] > 0 for b in pb), "bills_with_resolved": sum(b["n_res"] > 0 for b in pb),
         "op_counts": dict(opc), "op_exec": dict(opok), "sec": time.time() - t}
    out[split] = s
    print(split, json.dumps(s, indent=1))
    json.dump(pb, open("results/exec_perbill_%s.json" % split, "w"))
json.dump(out, open("results/exec_stats.json", "w"), indent=1)
