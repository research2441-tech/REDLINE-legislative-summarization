"""Extractive baselines: LEAD-k (after the short-title clause) and TextRank over TF-IDF sentence graphs."""
import json, re, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

SPL = re.compile(r"(?<=[\.;:])\s+(?=[A-Z(`])")


def sents(text):
    s = [x.strip() for x in SPL.split(text) if len(x.split()) >= 6]
    return [x for x in s if "may be cited as" not in x and not x.startswith("SECTION 1.")]


def lead(text, k=4):
    return " ".join(sents(text)[:k])


def textrank(text, k=4, d=0.85, it=50):
    ss = sents(text)[:120]
    if len(ss) <= k:
        return " ".join(ss)
    X = TfidfVectorizer(stop_words="english").fit_transform(ss)
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 0)
    S = S / np.maximum(S.sum(1, keepdims=True), 1e-9)
    r = np.ones(len(ss)) / len(ss)
    for _ in range(it):
        r = (1 - d) / len(ss) + d * S.T @ r
    top = sorted(np.argsort(-r)[:k])
    return " ".join(ss[i] for i in top)


if __name__ == "__main__":
    tag = sys.argv[1]
    rows = [json.loads(l) for l in open("data/eval_%s.jsonl" % tag)]
    for name, f in [("LEAD", lead), ("TEXTRANK", textrank)]:
        with open("results/gen_%s_%s.jsonl" % (name, tag), "w") as o:
            for r in rows:
                o.write(json.dumps({"i": r["i"], "pred": f(r["text"]), "toks": [], "lp": [], "sec": 0.0}) + "\n")
    print("done", tag, len(rows))
