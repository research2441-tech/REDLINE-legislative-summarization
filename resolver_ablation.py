"""Cumulative resolver ablation on the test split: programs and bills reachable per resolution route."""
import pickle, json
D = pickle.load(open("data/store_test.pkl", "rb"))
store, parsed = D["store"], D["parsed"]["test"]
order = ["explicit", "code_alias", "anaphora", "concordance"]
out = []
for k in range(1, 5):
    allowed = set(order[:k])
    np_, nb, nops = 0, 0, 0
    for b in parsed:
        hit = False
        for p in b["progs"]:
            if p["target_id"] and p.get("how") in allowed and p["target_id"] in store:
                np_ += 1; nops += len(p["ops"]); hit = True
        nb += hit
    out.append({"routes": "+".join(order[:k]), "programs": np_, "ops": nops, "bills": nb})
tot = sum(len(b["progs"]) for b in parsed)
json.dump({"rows": out, "total_programs": tot}, open("results/resolver_ablation.json", "w"), indent=1)
print(out, tot)
