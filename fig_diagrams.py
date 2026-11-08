"""Architecture (Fig. 1) and training/calibration workflow (Fig. 2) diagrams, grayscale."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 7.2, "mathtext.fontset": "dejavuserif"})


def box(ax, x, y, w, h, title, body=None, fill="#ffffff", lw=0.9, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=fill, ec="black", lw=lw, ls=ls)
    ax.add_patch(p)
    if body:
        ax.text(x + w / 2, y + h - 0.022, title, ha="center", va="top", fontsize=7.4, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.022, body, ha="center", va="center", fontsize=6.4, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=7.4, fontweight="bold", linespacing=1.2)
    return (x, y, w, h)


def arrow(ax, p, q, dashed=False, lw=0.9, rad=0.0, label=None, lpos=0.5, loff=(0, 0.018), style="-|>"):
    a = FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, lw=lw, color="black",
                        linestyle=(0, (3.5, 2.2)) if dashed else "-", connectionstyle="arc3,rad=%s" % rad, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    if label:
        x = p[0] + (q[0] - p[0]) * lpos + loff[0]
        y = p[1] + (q[1] - p[1]) * lpos + loff[1]
        ax.text(x, y, label, ha="center", va="center", fontsize=6.2, style="italic",
                bbox=dict(fc="white", ec="none", pad=0.6))


def poly(ax, pts, dashed=False, lw=0.9):
    """Orthogonal connector: line through pts, arrowhead on the last segment."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=lw, ls=(0, (3.5, 2.2)) if dashed else "-", solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], dashed=dashed, lw=lw)


def R(b):  # right-middle
    return (b[0] + b[2], b[1] + b[3] / 2)


def L(b):
    return (b[0], b[1] + b[3] / 2)


def T(b, f=0.5):
    return (b[0] + b[2] * f, b[1] + b[3])


def B(b, f=0.5):
    return (b[0] + b[2] * f, b[1])


def fig1(path):
    fig = plt.figure(figsize=(7.16, 3.75), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    g1, g2, g3 = "#f2f2f2", "#e6e6e6", "#d9d9d9"
    # lane labels
    ax.text(0.012, 0.955, "(a) Statute-resolved evidence construction", fontsize=7.4, fontweight="bold", ha="left")
    ax.text(0.012, 0.600, "(b) Redline-conditioned generation", fontsize=7.4, fontweight="bold", ha="left")
    ax.text(0.012, 0.302, "(c) Two-sided conformal risk control", fontsize=7.4, fontweight="bold", ha="left")
    h1, y1 = 0.215, 0.705
    b_bill = box(ax, 0.012, y1, 0.150, h1, "Bill text $x$", "amendatory clauses\nheaders $h$, operations", g1)
    b_res = box(ax, 0.205, y1, 0.170, h1, "Citation resolver", "explicit U.S.C. cite\ncode alias, anaphora\nact-to-Code concordance", g1)
    b_anc = box(ax, 0.418, y1, 0.170, h1, "Temporal anchoring", "versions $\\mathcal{V}(\\tau)$ of each\ntarget provision\n$v^{*}=\\arg\\max$ executability", g1)
    b_exe = box(ax, 0.631, y1, 0.165, h1, "Edit executor", "strike / insert / add\nredesignate / read-as\nscoped by designators", g1)
    b_red = box(ax, 0.838, y1, 0.150, h1, "Redline $\\mathcal{R}$ and\nentity table $\\mathcal{E}$", None, g2)
    for a, b in [(b_bill, b_res), (b_res, b_anc), (b_anc, b_exe), (b_exe, b_red)]:
        arrow(ax, R(a), L(b))
    arrow(ax, (0.503, 0.668), B(b_anc))
    ax.text(0.512, 0.676, "versioned U.S. Code store (381 release points)", ha="left", va="center", fontsize=6.0, style="italic")
    # lane b (right to left)
    h2, y2 = 0.165, 0.405
    b_ser = box(ax, 0.808, y2, 0.180, h2, "Budgeted serializer", "$z=[\\mathcal{E}\\,|\\,\\mathcal{R}\\,|\\,x]$, 512 tokens", g1)
    b_gen = box(ax, 0.520, y2, 0.200, h2, "Seq2seq generator $p_{\\theta}$", "token log-probabilities\nretained for scoring", g2)
    b_dr = box(ax, 0.245, y2, 0.190, h2, "Draft summary $\\hat{y}$", "sentences $s_{1},\\ldots,s_{J}$", g1)
    arrow(ax, B(b_red), T(b_ser, 0.583))
    poly(ax, [B(b_bill, 0.5), (0.087, 0.648), (0.853, 0.648), T(b_ser, 0.25)])
    ax.text(0.25, 0.662, "bill text $x$", fontsize=6.0, style="italic", ha="center")
    arrow(ax, L(b_ser), R(b_gen))
    arrow(ax, L(b_gen), R(b_dr))
    # lane c (left to right)
    h3, y3 = 0.180, 0.050
    b_gr = box(ax, 0.012, y3, 0.200, h3, "Claim grounding", "legal entities per sentence\nevidence: bill + statute\nprovision coverage $\\kappa$", g1)
    b_sc = box(ax, 0.255, y3, 0.180, h3, "Risk and salience", "sentence risk $r_{j}$\nprovision salience $q_{p}$", g1)
    b_crc = box(ax, 0.478, y3, 0.200, h3, "Two-sided CRC", "$\\hat{\\lambda}$ on split A ($\\alpha_{h}$)\n$\\hat{\\mu}$ on split B ($\\alpha_{o}$)", g3)
    b_pub = box(ax, 0.738, 0.160, 0.250, 0.085, "Published summary ($r_{j}\\leq\\hat{\\lambda}$)", None, g2)
    b_q = box(ax, 0.738, 0.045, 0.250, 0.085, "Analyst queue (flags)", None, "#ffffff")
    poly(ax, [B(b_dr, 0.8), (0.397, 0.262), (0.112, 0.262), T(b_gr)])
    arrow(ax, R(b_gr), L(b_sc))
    arrow(ax, R(b_sc), L(b_crc))
    arrow(ax, (0.678, 0.2025), L(b_pub))
    arrow(ax, (0.678, 0.0875), L(b_q))
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def fig2(path):
    fig = plt.figure(figsize=(3.45, 3.35), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.02, 0.965, "(a) Equal-budget generator adaptation", fontsize=7.2, fontweight="bold")
    b_pool = box(ax, 0.03, 0.745, 0.27, 0.165, "Training pool", "1,600 bills\n$(z,y)$ pairs", "#f2f2f2")
    b_gen = box(ax, 0.37, 0.745, 0.27, 0.165, "Generator $p_{\\theta}$", "shared start\ncheckpoint", "#e6e6e6")
    b_loss = box(ax, 0.71, 0.745, 0.26, 0.165, "Loss", "$\\mathcal{L}_{\\mathrm{gen}}(\\theta)$\ncross-entropy", "#f2f2f2")
    arrow(ax, R(b_pool), L(b_gen))
    arrow(ax, R(b_gen), L(b_loss))
    # dashed reverse-mode gradient arrow: from loss back to generator, routed below the boxes
    poly(ax, [B(b_loss, 0.5), (0.84, 0.675), (0.505, 0.675), B(b_gen, 0.5)], dashed=True)
    ax.text(0.672, 0.640, "reverse-mode gradient $\\nabla_{\\theta}\\mathcal{L}_{\\mathrm{gen}}$, AdamW", fontsize=6.0, style="italic", ha="center")
    ax.text(0.02, 0.575, "(b) Calibration and evaluation protocol", fontsize=7.2, fontweight="bold")
    b_fit = box(ax, 0.03, 0.335, 0.27, 0.185, "Fit split", "135 bills\nscorers $r$, $q$", "#f2f2f2")
    b_ca = box(ax, 0.37, 0.335, 0.27, 0.185, "Calibration A", "100 bills\n$\\hat{\\lambda}$ at $\\alpha_{h}$", "#e6e6e6")
    b_cb = box(ax, 0.71, 0.335, 0.26, 0.185, "Calibration B", "100 bills\n$\\hat{\\mu}\\mid\\hat{\\lambda}$ at $\\alpha_{o}$", "#e6e6e6")
    b_te = box(ax, 0.03, 0.050, 0.94, 0.170, "Held-out test split (400 bills)", "realized risks and review load; 500 random re-splits", "#d9d9d9")
    arrow(ax, R(b_fit), L(b_ca))
    arrow(ax, R(b_ca), L(b_cb))
    arrow(ax, B(b_fit, 0.5), (0.165, 0.220))
    arrow(ax, B(b_ca, 0.5), (0.505, 0.220))
    arrow(ax, B(b_cb, 0.5), (0.840, 0.220))
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs("figs", exist_ok=True)
    fig1("figs/fig1_architecture.png")
    fig2("figs/fig2_workflow.png")
