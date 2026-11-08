"""Slim statute store restricted to identifiers needed by one split (memory-light rebuild)."""
import json, pickle, glob, sys
import pyarrow.parquet as pq, pyarrow as pa, pyarrow.compute as pc
from redline_core import parse_amendments, find_usc, norm
concord = pickle.load(open("data/concord.pkl", "rb"))
split = sys.argv[1]
rows = [json.loads(l) for l in open("data/billsum_%s.jsonl" % split)]
parsed, need = [], set()
for r in rows:
    progs = parse_amendments(r["text"], concord)
    cites = [c[0] for c in find_usc(norm(r["text"]))]
    need.update(cites); need.update(p["target_id"] for p in progs if p["target_id"])
    parsed.append({"progs": progs, "cites": cites})
store = {}
cols = ["identifier", "text", "heading", "status", "first_release_seq", "last_release_seq", "first_currency_date", "is_current"]
na = pa.array(sorted(need))
for f in sorted(glob.glob("data/uscode/versions/*.parquet")):
    t = pq.read_table(f, columns=cols)
    t = t.filter(pc.is_in(t.column("identifier"), value_set=na))
    for r in t.to_pylist():
        r["first_currency_date"] = str(r["first_currency_date"])
        store.setdefault(r["identifier"], []).append(r)
    del t
pickle.dump({"parsed": {split: parsed}, "store": store}, open("data/store_%s.pkl" % split, "wb"))
print(split, len(need), len(store))
