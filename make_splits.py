"""Evaluation subsets and the generator training pool (fixed seeds)."""
import json, random
R = random.Random(7)
te = [json.loads(l) for l in open("data/inputs_test.jsonl")]
amd = [r for r in te if r["n_resolved"] > 0]; R.shuffle(amd)
for k, r in enumerate(amd):
    r["role"] = "fit" if k < 135 else ("cal" if k < 335 else "test")
with open("data/eval_usamd.jsonl", "w") as f:
    for r in amd:
        f.write(json.dumps(r) + "\n")
ca = [json.loads(l) for l in open("data/inputs_ca_test.jsonl")]; R.shuffle(ca)
with open("data/eval_ca.jsonl", "w") as f:
    for r in ca[:250]:
        f.write(json.dumps(r) + "\n")
tr = [json.loads(l) for l in open("data/inputs_train.jsonl")]
res = [r for r in tr if r["n_resolved"] > 0]; oth = [r for r in tr if r["n_resolved"] == 0]
R2 = random.Random(13); R2.shuffle(res); R2.shuffle(oth)
with open("data/train_pool.jsonl", "w") as f:
    for r in res[:1200] + oth[:400]:
        f.write(json.dumps({k: r[k] for k in ["i", "text", "summary", "ent_table", "redline_full", "redline_ops"]}) + "\n")
