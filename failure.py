"""Failure analysis: bill tokens displaced by the evidence block vs. per-bill change in ROUGE-L and entity precision."""
import json
import numpy as np
from scipy.stats import spearmanr
from transformers import AutoTokenizer
from analyze import load_rows, load_gen, per_bill
import train_gen as T
tok = AutoTokenizer.from_pretrained(T.INIT)
rows = [r for r in load_rows("usamd") if r["role"] == "test"]
pr = per_bill(rows, load_gen("REDLINE", "usamd")); pb = per_bill(rows, load_gen("BILL", "usamd"))
disp, drl, dw, bill_len = [], [], [], []
for r in rows:
    pre = T.serialise(dict(r, text=""), "REDLINE")
    n_ev = len(tok(pre, add_special_tokens=False).input_ids)
    n_bill = len(tok(r["text"], add_special_tokens=False).input_ids)
    kept_bill_red = max(0, min(n_bill, 511 - n_ev)); kept_bill = min(n_bill, 511 - 4)
    disp.append(kept_bill - kept_bill_red); bill_len.append(n_bill)
    drl.append(pr[r["i"]]["rl"] - pb[r["i"]]["rl"]); dw.append(pr[r["i"]]["words"] - pb[r["i"]]["words"])
disp = np.array(disp); drl = np.array(drl)
rho, p = spearmanr(disp, drl)
q = np.quantile(disp, [0, 1 / 3, 2 / 3, 1])
bins = []
for lo, hi in zip(q[:-1], q[1:]):
    m = (disp >= lo) & (disp <= hi)
    bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "d_rl": float(drl[m].mean())})
out = {"spearman_disp_drl": float(rho), "p": float(p), "disp_mean": float(disp.mean()), "disp_median": float(np.median(disp)),
       "share_bills_longer_than_budget": float(np.mean(np.array(bill_len) > 507)), "bins": bins, "d_words_mean": float(np.mean(dw))}
json.dump(out, open("results/failure.json", "w"), indent=1); print(out)
