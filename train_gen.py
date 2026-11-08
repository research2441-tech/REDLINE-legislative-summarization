"""Equal-budget continued fine-tuning and generation for the generator arms.

Arms (same initial checkpoint, same training bills and order, same 512-token input budget):
  B0      initial BillSum checkpoint, bill text only, no further tuning
  BILL    bill text only, tuned
  ENT     resolved entity table + bill
  OPS     compiled edit operations without statute context + bill
  REDLINE entity table + executed redline with current-law context + bill
"""
import json, sys, time, random, math, os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

INIT = "antonkurylo/centralized-t5-small-billsum"
MAXIN, MAXOUT = 512, 192
MAXIN_ADD = 800


_TOK = None


def bill_prefix(bill, n=506):
    """Bill text truncated to its first n tokens (used by the additive arm so that bill content equals BILL)."""
    global _TOK
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained(INIT)
    ids = _TOK(bill, add_special_tokens=False, truncation=True, max_length=n).input_ids
    return _TOK.decode(ids, skip_special_tokens=True)


def serialise(r, arm):
    bill = r["text"]
    if arm == "REDLINE_ADD":
        parts = ["summarize:"]
        if r.get("ent_table"):
            parts.append("ENTITIES: " + r["ent_table"][:420])
        if r.get("redline_full"):
            parts.append("CHANGES: " + r["redline_full"][:900])
        parts.append("BILL: " + bill_prefix(bill))
        return " ".join(parts)
    if arm in ("B0", "BILL"):
        return "summarize: " + bill
    parts = ["summarize:"]
    if arm in ("ENT", "REDLINE") and r.get("ent_table"):
        parts.append("ENTITIES: " + r["ent_table"][:420])
    if arm == "OPS" and r.get("redline_ops"):
        parts.append("CHANGES: " + r["redline_ops"][:700])
    if arm == "REDLINE" and r.get("redline_full"):
        parts.append("CHANGES: " + r["redline_full"][:900])
    parts.append("BILL: " + bill)
    return " ".join(parts)


def load(path):
    return [json.loads(l) for l in open(path)]


def train(arm, steps, seed=13, bs=4, lr=2e-4):
    torch.manual_seed(seed); random.seed(seed)
    tok = AutoTokenizer.from_pretrained(INIT)
    model = AutoModelForSeq2SeqLM.from_pretrained(INIT)
    if arm == "B0":
        return tok, model
    tr = load("data/train_pool.jsonl")
    order = list(range(len(tr)))
    random.Random(seed).shuffle(order)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * max(0.1, 1 - s / steps))
    model.train()
    t0 = time.time()
    log = []
    for s in range(steps):
        batch = [tr[order[(s * bs + j) % len(tr)]] for j in range(bs)]
        # the additive arm uses two micro-batches of two bills (same effective batch of four) to fit in memory
        micro = [batch[:2], batch[2:]] if arm == "REDLINE_ADD" else [batch]
        tot = 0.0
        for mb in micro:
            x = tok([serialise(b, arm) for b in mb], max_length=(MAXIN_ADD if arm == "REDLINE_ADD" else MAXIN), truncation=True, padding=True, return_tensors="pt")
            y = tok([b["summary"] for b in mb], max_length=MAXOUT, truncation=True, padding=True, return_tensors="pt").input_ids
            y[y == tok.pad_token_id] = -100
            l_ = model(**x, labels=y).loss / len(micro)
            l_.backward()
            tot += float(l_.detach())
        loss = torch.tensor(tot)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        log.append(float(loss.detach()))
        if s % 25 == 0:
            print(arm, s, round(sum(log[-25:]) / len(log[-25:]), 4), round(time.time() - t0, 1), flush=True)
    json.dump({"loss": log, "sec": time.time() - t0}, open("results/trainlog_%s.json" % arm, "w"))
    model.save_pretrained("ckpt/%s" % arm); tok.save_pretrained("ckpt/%s" % arm)
    return tok, model


def generate(tok, model, arm, rows, tag):
    model.eval()
    out = open("results/gen_%s_%s.jsonl" % (arm, tag), "w")
    t0 = time.time()
    for k, r in enumerate(rows):
        x = tok(serialise(r, arm), max_length=(MAXIN_ADD if arm == "REDLINE_ADD" else MAXIN), truncation=True, return_tensors="pt")
        t1 = time.time()
        with torch.no_grad():
            g = model.generate(**x, max_new_tokens=160, num_beams=1, no_repeat_ngram_size=3,
                               output_scores=True, return_dict_in_generate=True)
        seq = g.sequences[0][1:]
        lp = []
        for j, sc in enumerate(g.scores):
            if j >= len(seq):
                break
            lp.append(float(torch.log_softmax(sc[0], -1)[seq[j]]))
        toks = tok.convert_ids_to_tokens(seq)
        out.write(json.dumps({"i": r["i"], "pred": tok.decode(seq, skip_special_tokens=True), "toks": toks,
                              "lp": lp, "sec": time.time() - t1, "in_tokens": int(x.input_ids.shape[1])}) + "\n")
        out.flush()
        if k % 50 == 0:
            print(arm, tag, k, round(time.time() - t0, 1), flush=True)
    out.close()


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("NT", "2")))
    arm, steps = sys.argv[1], int(sys.argv[2])
    tags = sys.argv[3].split(",")
    tok, model = train(arm, steps)
    for tag in tags:
        role = None
        if ":" in tag:
            tag, role = tag.split(":")
        rows = load("data/eval_%s.jsonl" % tag)
        if role:
            rows = [r for r in rows if r.get("role") == role]
        generate(tok, model, arm, rows, tag)
