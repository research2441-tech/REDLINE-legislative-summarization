"""Per-bill latency of resolution, anchoring/execution, serialization and grounding on amendatory test bills."""
import pickle, json, time, statistics
from redline_core import parse_amendments, select_version, render_redline, norm
from evaluate import Evidence, legal_entities
concord = pickle.load(open("data/concord.pkl", "rb"))
D = pickle.load(open("data/store_test.pkl", "rb")); store = D["store"]
rows = [json.loads(l) for l in open("data/eval_usamd.jsonl")][:150]
raw = {json.loads(l)["i"]: None for l in []}
bills = [json.loads(l) for l in open("data/billsum_test.jsonl")]
tr, ta, tg = [], [], []
for r in rows:
    txt = bills[r["i"]]["text"]
    t0 = time.perf_counter(); progs = parse_amendments(txt, concord); t1 = time.perf_counter()
    for p in progs:
        vs = store.get(p["target_id"]) if p["target_id"] else None
        if vs:
            b = select_version(p["ops"], p["subdiv"], vs); render_redline(p, b, b[1]["heading"])
    t2 = time.perf_counter()
    ev = Evidence(r["text"], r["evidence"]); legal_entities(r["summary"]); t3 = time.perf_counter()
    tr.append(t1 - t0); ta.append(t2 - t1); tg.append(t3 - t2)
q = lambda v: {"mean_ms": 1000 * statistics.mean(v), "median_ms": 1000 * statistics.median(v), "p95_ms": 1000 * sorted(v)[int(0.95 * len(v))]}
out = {"resolve": q(tr), "anchor_execute": q(ta), "grounding": q(tg), "n": len(rows)}
json.dump(out, open("results/timing.json", "w"), indent=1); print(out)
