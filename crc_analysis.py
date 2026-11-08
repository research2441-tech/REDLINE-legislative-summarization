"""Two-sided conformal risk control experiments: main test, 500 re-splits, risk levels, calibration size,
scorer ablation, arm comparison (REDLINE vs BILL), US-to-California transfer."""
import json, os, sys
import numpy as np
import crc as C
from analyze import load_rows, load_gen

RNG = np.random.default_rng(7)
AH = [0.02, 0.05, 0.10]
AO = [0.05, 0.10, 0.20]


def prep(arm, mode, rows, tag="usamd", use_statute=True):
    gens = load_gen(arm, tag)
    fit = [r for r in rows if r["role"] == "fit"]
    docs_fit = []
    for r in fit:
        S, P = C.doc_units(r, gens[r["i"]], use_statute)
        docs_fit.append((S, P, r))
    ctrl = C.Controller(mode).fit(docs_fit)
    rest = [r for r in rows if r["role"] in ("cal", "test")]
    D = C.prepare(rest, gens, ctrl, use_statute)
    return ctrl, D


def calib(DA, DB, ah, ao):
    lam = C.crc_threshold(np.stack([d["Lh"] for d in DA]), ah)
    DB = C.omission_curves(DB, lam)
    mu = C.crc_threshold(np.stack([d["Lo"] for d in DB]), ao)
    return lam, mu


def split_eval(D, idx_cal, idx_te, ah, ao):
    cal = [D[i] for i in idx_cal]
    half = len(cal) // 2
    lam, mu = calib(cal[:half], cal[half:], ah, ao)
    ev = C.evaluate_policy([D[i] for i in idx_te], lam, mu)
    ev.update(lam=float(lam), mu=float(mu))
    return ev


def resplits(D, ah, ao, n_rep=500, n_cal=200):
    out = []
    N = len(D)
    for _ in range(n_rep):
        perm = RNG.permutation(N)
        out.append(split_eval(D, perm[:n_cal], perm[n_cal:], ah, ao))
    k = ["risk_h", "risk_o", "load", "kept", "flag_sent", "flag_prov"]
    agg = {m: float(np.mean([o[m] for o in out])) for m in k}
    agg.update({m + "_sd": float(np.std([o[m] for o in out])) for m in k})
    agg["viol_h"] = float(np.mean([o["risk_h"] > ah for o in out]))
    agg["viol_o"] = float(np.mean([o["risk_o"] > ao for o in out]))
    agg["dist_h"] = [o["risk_h"] for o in out]
    agg["dist_o"] = [o["risk_o"] for o in out]
    agg["dist_load"] = [o["load"] for o in out]
    return agg


def main():
    rows = load_rows("usamd")
    R = {}
    arms = sys.argv[1].split(",") if len(sys.argv) > 1 else ["REDLINE", "BILL"]
    modes = sys.argv[2].split(",") if len(sys.argv) > 2 else ["full", "conf"]
    outp = sys.argv[3] if len(sys.argv) > 3 else "results/summary_crc.json"
    for arm in arms:
        for mode in modes:
            if load_gen(arm, "usamd") is None:
                continue
            ctrl, D = prep(arm, mode, rows)
            key = "%s_%s" % (arm, mode)
            cal_idx = [i for i, d in enumerate(D) if d["row"]["role"] == "cal"]
            te_idx = [i for i, d in enumerate(D) if d["row"]["role"] == "test"]
            res = {"fixed": {}, "resplit": {}}
            # uncontrolled and heuristic baselines on the fixed test split
            te = [D[i] for i in te_idx]
            res["none"] = C.evaluate_policy(te, 1.0, 2.0)
            res["heur"] = C.evaluate_policy(te, 0.5, 0.5)
            for ah in AH:
                for ao in AO:
                    res["fixed"]["%g_%g" % (ah, ao)] = split_eval(D, cal_idx, te_idx, ah, ao)
                    if (ah, ao) in [(0.05, 0.10), (0.02, 0.05), (0.10, 0.20)] or mode == "full" and arm == "REDLINE":
                        res["resplit"]["%g_%g" % (ah, ao)] = resplits(D, ah, ao, n_rep=500 if (ah, ao) == (0.05, 0.10) else 200)
            # calibration size sensitivity at (0.05, 0.10)
            if mode == "full":
                res["ncal"] = {}
                for n_cal in [20, 50, 100, 200, 300]:
                    res["ncal"][str(n_cal)] = resplits(D, 0.05, 0.10, n_rep=200, n_cal=n_cal)
                    for k in ["dist_h", "dist_o", "dist_load"]:
                        res["ncal"][str(n_cal)].pop(k)
            R[key] = res
            print(key, "none", res["none"], "\nheur", res["heur"], "\nfixed .05/.10", res["fixed"]["0.05_0.1"],
                  "\nresplit", {k: v for k, v in res["resplit"]["0.05_0.1"].items() if not k.startswith("dist")}, flush=True)
            if arm == "REDLINE" and mode == "full":
                # transfer to California: US-calibrated thresholds vs CA recalibration (evidence = bill only)
                ca = load_rows("ca")
                gca = load_gen("REDLINE", "ca")
                if gca:
                    for r in ca:
                        r["role"] = "ca"
                    Dca = C.prepare(ca, gca, ctrl, use_statute=False)
                    tr = {}
                    lam, mu = calib([D[i] for i in cal_idx[:100]], [D[i] for i in cal_idx[100:]], 0.05, 0.10)
                    tr["us_cal"] = C.evaluate_policy(Dca, lam, mu); tr["us_cal"].update(lam=float(lam), mu=float(mu))
                    reps = []
                    for _ in range(200):
                        perm = RNG.permutation(len(Dca))
                        cal = [Dca[i] for i in perm[:100]]
                        l2, m2 = calib(cal[:50], cal[50:], 0.05, 0.10)
                        e = C.evaluate_policy([Dca[i] for i in perm[100:]], l2, m2); e.update(lam=float(l2), mu=float(m2))
                        reps.append(e)
                    tr["ca_recal"] = {m: float(np.mean([e[m] for e in reps])) for m in reps[0]}
                    tr["ca_recal"]["viol_h"] = float(np.mean([e["risk_h"] > 0.05 for e in reps]))
                    tr["ca_recal"]["viol_o"] = float(np.mean([e["risk_o"] > 0.10 for e in reps]))
                    tr["none"] = C.evaluate_policy(Dca, 1.0, 2.0)
                    R["transfer_ca"] = tr
                    print("transfer", tr, flush=True)
    json.dump(R, open(outp, "w"))


if __name__ == "__main__":
    main()
