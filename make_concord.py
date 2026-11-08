"""Induce the act-to-Code concordance from BillSum training bills (Eq. 1)."""
import json, pickle
from redline_core import build_concordance
texts = [json.loads(l)["text"] for l in open("data/billsum_train.jsonl")]
c = build_concordance(texts)
pickle.dump(c, open("data/concord.pkl", "wb"))
print("concordance entries", len(c))
