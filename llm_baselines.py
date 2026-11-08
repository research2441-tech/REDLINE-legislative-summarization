"""Strong external baselines on a fixed subset: PEGASUS-BillSum and a zero-shot instruction-tuned small LLM
(bill only, and bill plus the REDLINE evidence block in the prompt)."""
import json, sys, time, os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

torch.set_num_threads(int(os.environ.get("NT", "2")))
tag, which, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [r for r in (json.loads(l) for l in open("data/eval_%s.jsonl" % tag)) if r.get("role", "test") == "test"][:n]

if which == "PEGASUS":
    name = "google/pegasus-billsum"
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForSeq2SeqLM.from_pretrained(name).eval()
    out = open("results/gen_PEGASUS_%s.jsonl" % tag, "w")
    for k, r in enumerate(rows):
        t = time.time()
        x = tok(r["text"], truncation=True, max_length=1024, return_tensors="pt")
        with torch.no_grad():
            g = m.generate(**x, max_new_tokens=160, num_beams=1, no_repeat_ngram_size=3)
        out.write(json.dumps({"i": r["i"], "pred": tok.decode(g[0], skip_special_tokens=True), "toks": [], "lp": [], "sec": time.time() - t}) + "\n")
        out.flush()
        if k % 20 == 0:
            print(which, k, flush=True)
else:
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
    for arm in ["QWEN", "QWEN_RL"]:
        out = open("results/gen_%s_%s.jsonl" % (arm, tag), "w")
        for k, r in enumerate(rows):
            t = time.time()
            bill = tok.decode(tok(r["text"]).input_ids[:1100])
            ev = ""
            if arm == "QWEN_RL":
                ev = "Resolved provisions of existing law: %s\nChanges made by the bill to existing law: %s\n\n" % (r["ent_table"][:500], r["redline_full"][:900])
            msg = [{"role": "system", "content": "You write Congressional Research Service style bill summaries."},
                   {"role": "user", "content": ev + "Bill text:\n" + bill + "\n\nWrite a concise summary (3 to 5 sentences) of what this bill does and how it changes existing law. Mention the statutes it amends."}]
            p = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            x = tok(p, return_tensors="pt")
            with torch.no_grad():
                g = m.generate(**x, max_new_tokens=160, do_sample=False, repetition_penalty=1.1)
            pred = tok.decode(g[0][x.input_ids.shape[1]:], skip_special_tokens=True)
            out.write(json.dumps({"i": r["i"], "pred": pred, "toks": [], "lp": [], "sec": time.time() - t}) + "\n")
            out.flush()
            if k % 20 == 0:
                print(arm, k, round(time.time() - t, 1), flush=True)
