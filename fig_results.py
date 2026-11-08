"""Result figures (grayscale, no overlapping legends)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 7.5, "axes.linewidth": 0.7, "mathtext.fontset": "dejavuserif",
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False})
GR = ["#111111", "#666666", "#aaaaaa", "#dddddd"]
HAT = ["", "///", "...", "xxx"]


def fig_exec(path):
    E = json.load(open("results/exec_stats2.json"))
    A = json.load(open("results/resolver_ablation.json"))
    ops = ["STRIKE_INSERT", "STRIKE", "INSERT_AFTER", "STRIKE_STRUCT", "INSERT_STRUCT", "REDESIGNATE", "STRIKE_THROUGH_INSERT"]
    lab = ["strike +\ninsert", "strike", "insert\nafter", "strike\nunit", "insert\nunit", "redesig-\nnate", "strike\nthrough"]
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.55), dpi=300, gridspec_kw={"width_ratios": [2.5, 1.1, 1.2], "wspace": 0.38})
    ax = axs[0]
    x = np.arange(len(ops)); w = 0.26
    for k, (pol, name) in enumerate([("current", "current text"), ("earliest", "earliest stored"), ("anchor", "anchored (4)")]):
        v = [E[pol].get(o, 0) / E["total"][o] for o in ops]
        ax.bar(x + (k - 1) * w, v, w, color=GR[k + 0] if k < 2 else GR[0], edgecolor="black", lw=0.5, hatch=["///", "...", ""][k],
               label=name, facecolor=["#ffffff", "#bbbbbb", "#333333"][k])
    ax.set_xticks(x); ax.set_xticklabels([l + "\n(%d)" % E["total"][o] for l, o in zip(lab, ops)], fontsize=5.6)
    ax.set_ylabel("executable share of operations"); ax.set_ylim(0, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=6.4, handlelength=1.4)
    ax.set_title("(a) Executability by version policy", fontsize=7.5)
    ax = axs[1]
    rows = A["rows"]; names = ["explicit", "+alias", "+anaphora", "+concord."]
    pv = [r["programs"] for r in rows]; bv = [r["bills"] for r in rows]
    xx = np.arange(4)
    ax.bar(xx, pv, 0.6, facecolor="#555555", edgecolor="black", lw=0.5)
    for i, (p_, b_) in enumerate(zip(pv, bv)):
        ax.text(i, p_ + 60, "%d\nbills" % b_, ha="center", va="bottom", fontsize=5.6, linespacing=1.0)
    ax.set_xticks(xx); ax.set_xticklabels(names, rotation=30, fontsize=6.3, ha="right")
    ax.set_ylabel("resolved clauses (test split)"); ax.set_ylim(0, 3600)
    ax.set_title("(b) Resolver routes", fontsize=7.5)
    ax = axs[2]
    kinds = ["none", "ws", "case", "punct", "quote"]; kl = ["clean", "space", "case", "punct.", "quote"]
    rb = E["robust"]
    s = [rb["strict_" + k][0] / rb["strict_" + k][1] for k in kinds]
    t = [rb["tol_" + k][0] / rb["tol_" + k][1] for k in kinds]
    xx = np.arange(len(kinds))
    ax.bar(xx - 0.19, s, 0.36, facecolor="#ffffff", edgecolor="black", hatch="...", lw=0.5, label="strict")
    ax.bar(xx + 0.19, t, 0.36, facecolor="#555555", edgecolor="black", lw=0.5, label="tolerant")
    ax.set_xticks(xx); ax.set_xticklabels(kl, rotation=30, fontsize=6.3, ha="right")
    ax.set_ylim(0, 1.0); ax.set_ylabel("executable share (strike ops)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=6.4)
    ax.set_title("(c) Instruction noise", fontsize=7.5)
    for a in axs:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    fig_exec("figs/fig3_execution.png")


def fig_gen(path):
    S = json.load(open("results/summary_gen.json"))
    M = S["main"]
    arms = [a for a in ["LEAD", "TEXTRANK", "B0", "BILL", "ENT", "OPS", "REDLINE", "REDLINE_ADD"] if a in M]
    names = {"LEAD": "LEAD-4", "TEXTRANK": "TextRank", "B0": "B0", "BILL": "BILL", "ENT": "ENT", "OPS": "OPS", "REDLINE": "REDLINE", "REDLINE_ADD": "RL-ADD"}
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.35), dpi=300, gridspec_kw={"wspace": 0.45})
    x = np.arange(len(arms))
    fc = {"LEAD": "#ffffff", "TEXTRANK": "#ffffff", "B0": "#dddddd", "BILL": "#aaaaaa", "ENT": "#888888", "OPS": "#666666", "REDLINE": "#222222", "REDLINE_ADD": "#000000"}
    hat = {"LEAD": "///", "TEXTRANK": "...", "B0": "", "BILL": "", "ENT": "", "OPS": "", "REDLINE": "", "REDLINE_ADD": "xxx"}
    for ax, (k, kci, lab, lo) in zip(axs, [("rl", "rl_ci", "ROUGE-L F1", 0.15), ("efp", "efp_ci", "entity factual precision", None), ("spr", "spr_ci", "salient provision recall", None)]):
        v = [M[a][k] for a in arms]
        lo_ = [M[a][k] - M[a][kci][0] for a in arms]; hi_ = [M[a][kci][1] - M[a][k] for a in arms]
        for i, a in enumerate(arms):
            ax.bar(i, v[i], 0.68, facecolor=fc[a], edgecolor="black", lw=0.5, hatch=hat[a])
        ax.errorbar(x, v, yerr=[lo_, hi_], fmt="none", ecolor="black", elinewidth=0.6, capsize=1.6)
        ax.set_xticks(x); ax.set_xticklabels([names[a] for a in arms], rotation=40, ha="right", fontsize=6.2)
        ax.set_ylabel(lab)
        mn = min(M[a][kci][0] for a in arms); mx = max(M[a][kci][1] for a in arms)
        pad = (mx - mn) * 0.25 + 1e-3
        ax.set_ylim(max(0, mn - pad), min(1.0, mx + pad))
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.axhline(S["reference"][k], color="black", lw=0.6, ls=(0, (3, 2))) if k == "efp" else None
    axs[0].set_title("(a) Overlap with CRS summary", fontsize=7.5)
    axs[1].set_title("(b) Entity faithfulness", fontsize=7.5)
    axs[2].set_title("(c) Coverage of amended law", fontsize=7.5)
    axs[1].text(len(arms) - 0.5, S["reference"]["efp"], "reference floor", fontsize=5.6, ha="right", va="bottom")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_crc_dist(path):
    C = json.load(open("results/summary_crc.json"))
    K = C["REDLINE_full"]["resplit"]
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.3), dpi=300, gridspec_kw={"wspace": 0.55})
    keys_h = [("0.02_0.05", 0.02), ("0.05_0.1", 0.05), ("0.1_0.2", 0.10)]
    for ax, side, title in [(axs[0], "h", "(a) Hallucination"), (axs[1], "o", "(b) Omission")]:
        data, lev = [], []
        for key, ah in keys_h:
            ao = float(key.split("_")[1])
            data.append(K[key]["dist_" + side]); lev.append(ah if side == "h" else ao)
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=0.8),
                        boxprops=dict(facecolor="#cccccc", lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        for i, a in enumerate(lev):
            ax.plot([i + 0.6, i + 1.4], [a, a], color="black", lw=0.9, ls=(0, (3, 2)))
        ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["%.2f" % a for a in lev], fontsize=6.4)
        ax.set_xlabel("nominal level $\\alpha_{%s}$" % side)
        ax.set_ylabel("realized test risk")
        ax.set_title(title, fontsize=7.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_load(path):
    C = json.load(open("results/summary_crc.json"))
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.3), dpi=300, gridspec_kw={"wspace": 0.55})
    styles = {"REDLINE_full": ("-", "o", "#000000"), "REDLINE_conf": ("--", "s", "#555555"), "BILL_full": (":", "^", "#000000"), "BILL_conf": ("-.", "v", "#777777")}
    lab = {"REDLINE_full": "REDLINE, grounded", "REDLINE_conf": "REDLINE, confidence", "BILL_full": "BILL, grounded", "BILL_conf": "BILL, confidence"}
    ah = [0.02, 0.05, 0.10]
    for key, (ls, mk, col) in styles.items():
        if key not in C:
            continue
        F = C[key]["fixed"]
        axs[0].plot(ah, [F["%g_0.1" % a]["load"] for a in ah], ls=ls, marker=mk, ms=3, color=col, lw=0.9, label=lab[key])
        axs[1].plot(ah, [F["%g_0.1" % a]["kept"] for a in ah], ls=ls, marker=mk, ms=3, color=col, lw=0.9)
    for ax, yl, t in [(axs[0], "flags per bill", "(a) Review load"), (axs[1], "share of sentences published", "(b) Retained text")]:
        ax.set_xlabel("$\\alpha_{h}$ (with $\\alpha_{o}=0.10$)"); ax.set_ylabel(yl); ax.set_title(t, fontsize=7.5)
        ax.set_xticks(ah)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=6.2, frameon=False)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_ncal(path):
    C = json.load(open("results/summary_crc.json"))
    N = C["REDLINE_full"]["ncal"]
    ns = sorted(int(k) for k in N)
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.2), dpi=300, gridspec_kw={"wspace": 0.55})
    for side, a, ls, mk in [("h", 0.05, "-", "o"), ("o", 0.10, "--", "s")]:
        m = [N[str(n)]["risk_" + side] for n in ns]; s = [N[str(n)]["risk_" + side + "_sd"] for n in ns]
        off = 0.94 if side == "h" else 1.06
        axs[0].errorbar([n * off for n in ns], m, yerr=s, ls=ls, marker=mk, ms=3, color="black", lw=0.8, capsize=1.5, elinewidth=0.6, label="$L_{%s}$ (level %.2f)" % (side, a))
        axs[0].axhline(a, color="#888888", lw=0.6, ls=ls)
        axs[1].plot(ns, [N[str(n)]["viol_" + side] for n in ns], ls=ls, marker=mk, ms=3, color="black", lw=0.8)
    axs[0].set_xscale("log"); axs[1].set_xscale("log")
    for ax, yl, t in [(axs[0], "realized risk (mean $\\pm$ sd)", "(a) Risk"), (axs[1], "share of re-splits above level", "(b) Violations")]:
        ax.set_xlabel("calibration bills $n$"); ax.set_ylabel(yl); ax.set_title(t, fontsize=7.5)
        ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns], fontsize=5.8, rotation=45)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=6.2, frameon=False)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_het(path):
    S = json.load(open("results/summary_gen.json"))
    H = S["by_nres"]; groups = list(H.keys())
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.2), dpi=300, gridspec_kw={"wspace": 0.42})
    x = np.arange(len(groups))
    for ax, k, yl in [(axs[0], "efp", "entity factual precision"), (axs[1], "spr", "salient provision recall")]:
        for j, (arm, fc) in enumerate([("BILL", "#bbbbbb"), ("REDLINE", "#333333")]):
            v = [H[g][arm][k] for g in groups]
            ax.bar(x + (j - 0.5) * 0.36, v, 0.36, facecolor=fc, edgecolor="black", lw=0.5, label=arm)
        ax.set_xticks(x); ax.set_xticklabels(["%s\n(n=%d)" % (g, H[g]["n"]) for g in groups], fontsize=6.2)
        ax.set_xlabel("resolved clauses per bill"); ax.set_ylabel(yl)
        vals = [H[g][a][k] for g in groups for a in ("BILL", "REDLINE")]
        ax.set_ylim(max(0, min(vals) - 0.1), min(1, max(vals) + 0.08))
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=6.3)
    axs[0].set_title("(a) Faithfulness by bill type", fontsize=7.5); axs[1].set_title("(b) Coverage by bill type", fontsize=7.5)
    ax = axs[2]
    for arm, ls in [("BILL", ":"), ("ENT", "-."), ("OPS", "--"), ("REDLINE", "-"), ("REDLINE_ADD", (0, (1, 1)))]:
        p = "results/trainlog_%s.json" % arm
        if os.path.exists(p):
            l = np.array(json.load(open(p))["loss"]); w = 25
            sm = np.convolve(l, np.ones(w) / w, mode="valid")
            ax.plot(np.arange(len(sm)) + w, sm, ls=ls, color="black", lw=0.8, label=arm)
    ax.set_xlabel("training step"); ax.set_ylabel("$\\mathcal{L}_{\\mathrm{gen}}$ (moving mean)")
    ax.set_title("(c) Adaptation loss", fontsize=7.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=6.0, handlelength=1.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
