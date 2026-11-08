"""Parse all BillSum bills, collect cited US Code identifiers and load every stored version of them."""
import json, pickle, glob, sys, time
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
from redline_core import parse_amendments, find_usc, norm
concord = pickle.load(open("data/concord.pkl", "rb"))

t0 = time.time()
parsed = {}
need = set()
for split in ["train", "test", "ca_test"]:
    rows = [json.loads(l) for l in open("data/billsum_%s.jsonl" % split)]
    out = []
    for r in rows:
        txt = norm(r["text"])
        progs = parse_amendments(r["text"], concord) if split != "ca_test" else []
        cites = [c[0] for c in find_usc(txt)]
        need.update(cites)
        need.update(p["target_id"] for p in progs if p["target_id"])
        out.append({"progs": progs, "cites": cites})
    parsed[split] = out
    print(split, len(rows), "parsed", time.time() - t0, flush=True)
print("needed ids", len(need))
store = {}
cols = ["identifier", "text", "heading", "status", "first_release_seq", "last_release_seq", "first_currency_date", "is_current"]
needarr = pa.array(sorted(need))
for f in sorted(glob.glob("data/uscode/versions/*.parquet")):
    t = pq.read_table(f, columns=cols)
    t = t.filter(pc.is_in(t.column("identifier"), value_set=needarr))
    for r in t.to_pylist():
        r["first_currency_date"] = str(r["first_currency_date"])
        store.setdefault(r["identifier"], []).append(r)
    print(f, len(store), time.time() - t0, flush=True)
pickle.dump({"parsed": parsed, "store": store}, open("data/store.pkl", "wb"))
print("found", len(store), "of", len(need))
