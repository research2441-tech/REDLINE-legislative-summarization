"""Build per-bill REDLINE artefacts: entity table, executed redline, provision metadata, evidence."""
import json, pickle, re, sys
from redline_core import (norm, find_usc, select_version, render_redline, execute, legal_entities, act_key,
                          parse_amendments)

D = pickle.load(open("data/store.pkl", "rb"))
store, parsed = D["store"], D["parsed"]


def pick(vs, prefer=None):
    if prefer is not None:
        return prefer
    cur = [v for v in vs if v["is_current"]]
    return cur[0] if cur else max(vs, key=lambda v: v["last_release_seq"] or 0)


CA_HEAD = re.compile(r"(Section|Sections)\s+([0-9\.]+(?:,\s*[0-9\.]+)*)\s+of\s+(?:the\s+)?([A-Z][A-Za-z ]+? Code)\s+(?:is|are)\s+(amended|added|repealed)")


def ca_programs(text):
    t = norm(text)
    out = []
    for m in CA_HEAD.finditer(t):
        out.append({"target_id": None, "target_disp": "%s %s" % (m.group(3), m.group(2)), "act": m.group(3), "subdiv": [],
                    "header": m.group(0), "ops": [{"op": {"amended": "READ_AS", "added": "ADD_END", "repealed": "REPEAL"}[m.group(4)], "scope": []}],
                    "pos": m.start() / max(len(t), 1), "how": "ca"})
    return out


def build(split, idx_list=None):
    rows = [json.loads(l) for l in open("data/billsum_%s.jsonl" % split)]
    out = []
    for i, r in enumerate(rows):
        if idx_list is not None and i not in idx_list:
            continue
        txt = norm(r["text"])
        progs = parsed[split][i]["progs"] if split != "ca_test" else ca_programs(r["text"])
        ev = []
        provs = []
        red_full, red_ops = [], []
        chosen = {}
        for k, p in enumerate(progs):
            meta = {"act": p.get("act"), "disp": p.get("target_disp"), "n_ops": len(p["ops"]),
                    "ops": [o["op"] for o in p["ops"]], "header": p["header"][:200], "resolved": False,
                    "heading": None, "edit_chars": sum(len(o.get("new", "") or "") + len(o.get("old", "") or "") for o in p["ops"]),
                    "rank": k, "n_prog": len(progs)}
            pos = txt.find(p["header"][:60])
            meta["pos"] = pos / max(len(txt), 1) if pos >= 0 else p.get("pos", 0.5)
            vs = store.get(p["target_id"]) if p.get("target_id") else None
            if vs:
                best = select_version(p["ops"], p["subdiv"], vs)
                v = best[1]
                chosen[p["target_id"]] = v
                meta.update(resolved=True, heading=v["heading"], exec=best[4], already=best[3])
                red_full.append(render_redline(p, best, v["heading"]))
                # ops-only rendering (no statute heading, no current-law context)
                b2 = (best[0], best[1], [dict(h, ctx="") for h in best[2]], best[3], best[4], best[5])
                red_ops.append(render_redline(p, b2, "").replace(" | ]", "]"))
                ev.append(v["heading"] or "")
                ev.append((v["text"] or "")[:3000])
            else:
                if split == "ca_test":
                    red_ops.append("[%s] %s" % (p["target_disp"], p["ops"][0]["op"].lower().replace("_", " ")))
                    red_full.append(red_ops[-1])
            provs.append(meta)
        ents = []
        seen = set()
        targets = [p["target_id"] for p in progs if p.get("target_id")]
        cites = [c[0] for c in find_usc(txt)]
        for cid in targets + cites:
            if cid in seen or cid not in store:
                continue
            seen.add(cid)
            v = chosen.get(cid) or pick(store[cid])
            disp = (v["identifier"].replace("/us/usc/t", "").replace("/s", " U.S.C. ").replace("–", "-"))
            ents.append("%s: %s" % (disp, (v["heading"] or "").strip()[:80]))
            ev.append(v["heading"] or "")
            if len(ents) >= 8:
                break
        out.append({"i": i, "title": r["title"], "text": txt, "summary": r["summary"],
                    "ent_table": " ; ".join(ents), "redline_full": " || ".join(red_full),
                    "redline_ops": " || ".join(red_ops), "provs": provs, "evidence": " ".join(ev)[:20000],
                    "n_resolved": sum(p["resolved"] for p in provs)})
    return out


if __name__ == "__main__":
    split = sys.argv[1]
    o = build(split)
    with open("data/inputs_%s.jsonl" % split, "w") as f:
        for r in o:
            f.write(json.dumps(r) + "\n")
    print(split, len(o), sum(r["n_resolved"] > 0 for r in o), sum(bool(r["ent_table"]) for r in o))
