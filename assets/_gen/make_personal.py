import os

import matplotlib.pyplot as plt
import numpy as np
from common import (
    AMBER,
    BLUE,
    GRID,
    INK,
    OUT_PERSONAL,
    PURPLE,
    RED,
    SLATE,
    SUBINK,
    TEAL,
    WHITE,
    load,
    pu_trajectory,
    savefig,
    takeaway,
    titleblock,
)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PANEL = WHITE
PAPER = WHITE

O = OUT_PERSONAL


def p01_pipeline():
    fig, ax = plt.subplots(figsize=(13, 6.0))
    ax.set_xlim(0, 100); ax.set_ylim(5, 79); ax.axis("off"); ax.margins(0)

    def box(x, y, w, h, num, title, module, lines, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.4",
                                    fc=fc, ec=ec, lw=1.8))
        ax.text(x + 2.0, y + h - 3.2, num, ha="left", va="top", fontsize=15,
                fontweight="bold", color=ec)
        ax.text(x + w / 2, y + h - 3.4, title, ha="center", va="top", fontsize=11,
                fontweight="bold")
        ax.text(x + w / 2, y + h - 7.6, module, ha="center", va="top", fontsize=8.2,
                color=ec, family="monospace")
        ax.text(x + w / 2, y + h - 11.4, "\n".join(lines), ha="center", va="top",
                fontsize=8.3, color=SUBINK)

    def arrow(x1, y1, x2, y2, label=""):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                                     lw=1.6, color=SUBINK))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 2.4, label, ha="center", fontsize=8,
                    color=INK, style="italic")

    top, bot, w, h = 46, 8, 29.5, 31
    box(1.5, top, w, h, "1", "FORWARD", "generation/",
        ["known symbolic L  ->  EOM",
         "RK4 over 150 random ICs",
         "+ Gaussian noise, noise% x std",
         "-> noisy trajectory CSV"], "#eef3fd", BLUE)
    box(35.25, top, w, h, "2", "LIBRARY  +  GRAM", "finding_L/",
        ["monomials in (q_i, q_i'), deg 1..4",
         "one EL(theta) column each",
         "stream G = Theta^T Theta",
         "Theta ~23 GB, never built"], "#eef3fd", BLUE)
    box(69, top, w, h, "3", "MODEL SELECTION", "main_streaming / regularized_select",
        ["debiased LASSO path (default)",
         "or greedy OMP",
         "frozen tolerances, all systems",
         "-> sparse coefficient set"], "#eef3fd", BLUE)
    box(69, bot, w, h, "4", "READABLE REPORT", "report.py",
        ["snap coeffs to small rationals",
         "-0.24999 -> -1/4",
         "group terms by degree",
         "-> DiscoveredLagrangian"], "#e9f7f3", TEAL)
    box(35.25, bot, w, h, "5", "EQUIVALENCE CLASS", "equivalence_class.py",
        ["dL = L_recovered - L_true",
         "is EL(dL) identically 0 ?",
         "same theory  vs  look-alike",
         "coeff proximity answers neither"], "#e9f7f3", TEAL)
    box(1.5, bot, w, h, "6", "ORDER  +  GHOST", "higher_order_* / ghost_detection",
        ["infer Lagrangian order",
         "Ostrogradski H bounded below?",
         "dynamics oscillatory?",
         "-> ghost verdict + 3 confidences"], "#fdeef0", RED)

    arrow(31, top + h / 2, 35.25, top + h / 2)
    arrow(64.75, top + h / 2, 69, top + h / 2)
    arrow(83.75, top, 83.75, bot + h)
    arrow(69, bot + h / 2, 64.75, bot + h / 2, "compare")
    arrow(35.25, bot + h / 2, 31, bot + h / 2)

    titleblock(fig, "The pipeline, end to end",
               "A known Lagrangian makes noisy data (1); stages 2-6 recover a Lagrangian from that data alone")
    takeaway(fig, "One CSV in  ->  a Lagrangian, a same-theory verdict and a ghost flag out. "
                  "No ground truth is used after step 1.", color=SLATE)
    savefig(fig, os.path.join(O, "P01_pipeline_overview.png"))


def p02_forward():
    t = np.linspace(0, 12, 1600)
    q, _ = pu_trajectory(t, q0=1.0, v0=0.2, a0=-0.5, j0=0.1)
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(t, q, color=BLUE, lw=2.0, label="true trajectory  q(t)  (from the known L)")
    noisy = q + rng.normal(0, 0.05 * q.std(), q.shape)
    ax.plot(t, noisy, color=AMBER, lw=0.9, alpha=0.85, label="measured  =  true + 5% x std noise")
    ax.set_xlabel("time  t"); ax.set_ylabel("position  q")
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_title("Pais-Uhlenbeck benchmark  ·  one of 150 integrated trajectories", fontsize=11)

    ax = axes[1]
    levels = np.array([0, 1, 2, 5, 10, 25])
    ax.bar(range(len(levels)), levels, color=[BLUE if l == 0 else AMBER for l in levels],
           width=0.62)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"{l}%" for l in levels])
    ax.set_ylabel("noise  (% of per-column signal std)")
    ax.set_title("noise is an explicit, controlled quantity", fontsize=11)
    ax.grid(True, axis="y", color=GRID, lw=0.6)

    titleblock(fig, "Step 1 - forward: a known L becomes noisy data",
               "Integrate the true Euler-Lagrange equations, then add measurement noise scaled to each signal")
    takeaway(fig, "Because the true L is known here, every later recovery can be scored exactly and the "
                  "noise level is a dial we set - not an unknown we estimate.", color=BLUE)
    savefig(fig, os.path.join(O, "P02_forward_generation.png"))


def p03_library():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    deg = np.array([1, 2, 3, 4])
    # de-duplicated monomials in (q_i,v_i) for 6 DOF, pure-velocity dropped - representative counts
    ncand = np.array([6, 63, 210, 714])
    ax.plot(deg, ncand, "o-", color=BLUE, lw=2, ms=8)
    for d, n in zip(deg, ncand):
        ax.annotate(f"{n}", (d, n), textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=10, fontweight="bold")
    ax.set_xlabel("library max degree"); ax.set_ylabel("candidate monomials  (6 DOF)")
    ax.set_xticks(deg)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_title("'find a function'  ->  'find sparse coefficients over a fixed basis'", fontsize=10.5)

    ax = axes[1]
    rng = np.random.default_rng(1)
    n = 26
    G = rng.normal(0, 0.12, (n, n)); G = G @ G.T
    d = np.sqrt(np.diag(G)); G = G / np.outer(d, d)
    im = ax.imshow(G, cmap="BuPu", vmin=-0.2, vmax=1)
    ax.set_title("Gram matrix  G = Theta^T Theta   (schematic)", fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="normalised inner product")
    ax.text(0.5, -0.13, "Theta (design matrix) for deg-4 / 6 DOF  ~  23 GB and never materialised.\n"
            "Everything downstream needs only G  (n_cand x n_cand)  and  b = -G[:, kinetic].",
            transform=ax.transAxes, ha="center", fontsize=8.6, color=SUBINK)

    titleblock(fig, "Steps 2-3 - the linear problem behind recovery",
               "Applying the Euler-Lagrange operator to each candidate turns discovery into sparse regression on a Gram matrix")
    savefig(fig, os.path.join(O, "P03_library_and_gram.png"))


def p04_snapping():
    row = load("noise_curve_isotropic_quartic_calibration.json")["rows"][0]
    text = row["discoveredText"]
    pairs = []
    for line in text.splitlines():
        line = line.strip()
        if "->" in line and ":" in line and "q" in line:
            key, rest = line.split(":", 1)
            raw, clean = rest.split("->")
            try:
                pairs.append((key.strip(), float(raw), float(clean)))
            except ValueError:
                pass
    raw = np.array([p[1] for p in pairs])
    clean = np.array([p[2] for p in pairs])
    err = raw - clean

    families = [(-1.0, "sum q_i^2", "-1", 12), (-0.3, "sum q_i^2 q_j^2", "-3/10", 4),
                (-0.15, "sum q_i^4", "-3/20", 2)]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6), gridspec_kw={"width_ratios": [1.4, 1]})

    ax = axes[0]
    for i, (target, name, rat, snappct) in enumerate(families):
        sel = np.isclose(clean, target)
        dev = (raw[sel] - target) * 1e3
        jit = rng.uniform(-0.16, 0.16, len(dev))
        win = abs(target) * snappct  # 1% relative window, in units of 1e-3
        ax.axvspan(-win, win, ymin=(i + 0.05) / 3, ymax=(i + 0.95) / 3, color=TEAL, alpha=0.10)
        ax.axvline(0, color=SLATE, lw=0.7, ls=":")
        ax.scatter(dev, i + jit, color=AMBER, s=48, alpha=0.9, ec="white", lw=0.6, zorder=3)
        ax.text(0.015, i + 0.34, f"{name}   ->   snap to {rat}", transform=ax.get_yaxis_transform(),
                fontsize=9.5, color=INK, family="monospace")
    ax.set_yticks([]); ax.set_ylim(-0.6, len(families) - 0.1)
    ax.set_xlim(-13, 13)
    ax.set_xlabel("raw fitted coefficient  -  exact rational      ( x 10^-3 )")
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.text(0.98, 0.03, "shaded band = 1% relative snap window", transform=ax.transAxes,
            ha="right", fontsize=8.5, color=TEAL)
    ax.set_title("isotropic quartic calibration  ·  1% position noise  ·  27 coefficients", fontsize=10)

    ax = axes[1]; ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.02, 0.93, "recovered Lagrangian", fontsize=11, fontweight="bold", va="top")
    L = ("L  =  sum v_i^2\n"
         "      -  1 * sum q_i^2\n"
         "      -  3/10 * sum_{i<j} q_i^2 q_j^2\n"
         "      -  3/20 * sum q_i^4")
    ax.text(0.02, 0.82, L, fontsize=10.5, va="top", family="monospace", color=INK,
            bbox={"boxstyle": "round,pad=0.6", "fc": "#f4faf8", "ec": TEAL, "lw": 1.6})
    ax.text(0.02, 0.50, f"max coefficient error   {np.abs(err).max():.4f}   "
            f"( {np.abs(err).max()*100:.2f}% )", fontsize=10, va="top", color=SUBINK)
    ax.text(0.02, 0.40, "equivalence class:  dL = 0\nstructurally identical to the true L",
            fontsize=10.5, va="top", color=TEAL, fontweight="bold")

    titleblock(fig, "Step 4 - snapping the coefficients to rationals",
               "Physical Lagrangians have simple rational coefficients; snapping makes the same-theory check exact, not approximate")
    takeaway(fig, "At 1% noise every one of the 27 fitted coefficients lands within 0.1% of its true "
                  "rational value  ->  the recovered term set is exact.", color=TEAL)
    savefig(fig, os.path.join(O, "P04_coefficient_snapping.png"))


def p05_selectors():
    data = load("model_selection_comparison.json")
    systems = list(data)
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.9), sharey=True)
    for ax, sysname in zip(axes, systems):
        rows = data[sysname]
        noise = np.array([r["noiseLevel"] * 100 for r in rows])
        for sel, col in [("greedy", RED), ("stlsq", AMBER), ("lasso", TEAL)]:
            rec = [r[sel]["recoveredFraction"] for r in rows]
            ax.plot(noise, rec, "o-", color=col, lw=2, ms=6, label=sel)
        ax.set_title(sysname.replace("_", " "), fontsize=9.5)
        ax.set_xlabel("position noise  (%)")
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_ylim(-0.05, 1.08)
    axes[0].set_ylabel("fraction of seeds\nrecovered exactly")
    axes[0].legend(fontsize=9, loc="lower left")
    titleblock(fig, "Step 3 - which selector, and how far into noise",
               "Greedy vs sequential-thresholded least squares vs debiased LASSO  ·  3 seeds per point, degree-4 library")
    takeaway(fig, "Greedy breaks from ~2% noise (spurious velocity-quartics out-correlate the true cubics). "
                  "Debiased LASSO holds the exact sparse term set to 5% on all three systems - it is the production default.",
             color=TEAL)
    savefig(fig, os.path.join(O, "P05_selector_comparison.png"))


def p06_noise_robustness():
    # README table, isotropic quartic calibration, single seed
    noise = [0, 1, 2, 5, 10]
    greedy_missing = [0, 0, 7, 20, 21]
    greedy_spurious = [0, 0, 19, 20, 11]
    lasso_missing = [0, 0, 0, 0, 4]
    lasso_spurious = [0, 0, 0, 0, 33]
    lasso_dcoef = [0.0, 0.0006, 0.0028, 0.0175, 0.199]
    verdict = ["same theory", "same theory", "coeffs close,\nnot same theory",
               "coeffs close,\nnot same theory", "failed"]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))
    x = np.arange(len(noise)); w = 0.2
    ax = axes[0]
    ax.bar(x - 1.5 * w, greedy_missing, w, color=RED, label="greedy missing")
    ax.bar(x - 0.5 * w, greedy_spurious, w, color="#f0a6b0", label="greedy spurious")
    ax.bar(x + 0.5 * w, lasso_missing, w, color=TEAL, label="LASSO missing")
    ax.bar(x + 1.5 * w, lasso_spurious, w, color="#9bd9cd", label="LASSO spurious")
    ax.set_xticks(x); ax.set_xticklabels([f"{n}%" for n in noise])
    ax.set_ylabel("wrong terms  (count)"); ax.set_xlabel("position noise")
    ax.legend(fontsize=8.5); ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_title("term-set errors  ·  greedy vs LASSO", fontsize=10.5)

    ax = axes[1]
    ax.plot(x, lasso_dcoef, "o-", color=BLUE, lw=2, ms=7)
    ax.set_ylim(-0.02, 0.26)
    for xi, dd, v in zip(x, lasso_dcoef, verdict):
        off = -34 if xi == x[-1] else 12
        ax.annotate(v, (xi, dd), textcoords="offset points", xytext=(0, off), ha="center",
                    fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([f"{n}%" for n in noise])
    ax.set_xlabel("position noise"); ax.set_ylabel("LASSO  max |coef error|")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_title("coefficient error and the same-theory verdict", fontsize=10.5)

    titleblock(fig, "Steps 3-5 - noise robustness of 2nd-order recovery",
               "Isotropic quartic calibration, 6 DOF, single seed  (README results table)")
    takeaway(fig, "LASSO keeps the exact sparse structure to ~5% noise. The strict same-theory verdict "
                  "still turns over near 1-2% - every EL column is built from noisy derivatives (errors-in-variables).",
             color=AMBER)
    savefig(fig, os.path.join(O, "P06_noise_robustness.png"))


def p07_differentiation():
    d = load("differentiation_study.json")
    noise = np.array(d["noiseLevels"]) * 100
    rec = d["records"]
    methods = ["finite_difference", "savitzky_golay", "smoothing_spline"]
    mcol = {"finite_difference": RED, "savitzky_golay": AMBER, "smoothing_spline": TEAL}
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 4.6), sharex=True, sharey=True)
    for order, ax in zip([1, 2, 3, 4], axes):
        for m in methods:
            ys = [r["relativeL2Error"] for r in rec if r["method"] == m and r["order"] == order]
            ax.plot(noise[:len(ys)], ys, "o-", color=mcol[m], lw=1.8, ms=5,
                    label=m.replace("_", " "))
        ax.set_yscale("log")
        ax.axhline(0.5, color=SLATE, ls="--", lw=1)
        ax.set_title(f"derivative order {order}", fontsize=10)
        ax.set_xlabel("noise (%)")
        ax.grid(True, which="both", color=GRID, lw=0.5)
    axes[0].set_ylabel("relative L2 error  (log)")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[3].text(0.95, 0.06, "dashed = breakdown\n(error > 0.5)", transform=axes[3].transAxes,
                 ha="right", fontsize=8, color=SLATE)
    titleblock(fig, "The bottleneck - estimating derivatives from noisy positions",
               "Relative L2 error of estimated q', q'', q''', q'''' for a Pais-Uhlenbeck trajectory")
    takeaway(fig, "Finite differences explode past 0.1% noise by 2nd order. The quintic smoothing spline "
                  "is the only method that survives to 3rd / 4th order - which is what higher-derivative recovery needs.",
             color=TEAL)
    savefig(fig, os.path.join(O, "P07_differentiation_breakdown.png"))


def p08_order_inference():
    d = load("order_inference.json")
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.0), sharey=True)
    fig.subplots_adjust(top=0.74, wspace=0.12)
    for ax, entry in zip(axes, d):
        orders = [p["order"] for p in entry["perOrder"]]
        res = [max(p["scaledResidual"], 1e-17) for p in entry["perOrder"]]
        degen = [p["degenerate"] for p in entry["perOrder"]]
        cols = [PURPLE if g else BLUE for g in degen]
        ax.bar(orders, res, color=cols, width=0.5)
        ax.set_yscale("log")
        ax.set_xlim(0.4, max(2.6, max(orders) + 0.6))
        ax.set_ylim(1e-17, 30)
        for o, r in zip(orders, res):
            tag = "EL eq\nsatisfied" if r < 0.01 else "no order-%d\nEL eq" % o
            yy = max(r * 4, 3e-16) if r < 0.01 else 2.2
            ax.text(o, yy, tag, ha="center", va="bottom", fontsize=7.5, color=SUBINK)
        ax.axhline(0.01, color=RED, ls="--", lw=1.2, label="Condition A tolerance")
        ax.set_title(f"{entry['system'].replace('_', ' ')}\ntrue {entry['trueOrder']}  ->  inferred {entry['inferredOrder']}",
                     fontsize=9.5, pad=8)
        ax.set_xlabel("assumed Lagrangian order")
        ax.set_xticks(orders)
        ax.grid(True, axis="y", which="major", color=GRID, lw=0.5)
    axes[0].set_ylabel("feasibility residual")
    axes[0].legend(fontsize=7.5, loc="lower left")
    axes[2].text(0.96, 0.5, "purple = degenerate\n(rank-deficient library:\nzero residual carries\nno evidence)",
                 transform=axes[2].transAxes, ha="right", va="center", fontsize=7.5, color=PURPLE)
    titleblock(fig, "Step 6a - inferring the Lagrangian order from data alone",
               "For each candidate order: does the data satisfy an order-n Euler-Lagrange equation?")
    takeaway(fig, "Pais-Uhlenbeck needs 2 derivatives; the anharmonic oscillator needs 1. A purely linear "
                  "order-1 system (harmonic) tests feasible at every order - correctly inferred, but flagged as a degenerate route.",
             color=BLUE)
    savefig(fig, os.path.join(O, "P08_order_inference.png"))


def p09_ghost_anatomy():
    ref = load("ghost_detection.json")["reference"]
    keys = ["healthy_second_order_oscillator", "pais_uhlenbeck_ghost",
            "pais_uhlenbeck_plus_total_derivative"]
    fig = plt.figure(figsize=(13.6, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.5], wspace=0.32)

    ax = fig.add_subplot(gs[0]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 11)
    steps = [
        ("build Ostrogradski H", "Legendre transform of L(q, q', q'', ...)", "#eef3fd", BLUE, 8.7),
        ("H bounded below?", "Hessian eigenvalues (or polynomial test)", "#fdf3e6", AMBER, 6.1),
        ("dynamics oscillatory?", "characteristic roots of the EOM", "#fdf3e6", AMBER, 3.5),
        ("GHOST  iff  unbounded below\nAND oscillatory", "bounded-motion negative-energy mode", "#fdeef0", RED, 0.7),
    ]
    for title, sub, fc, ec, y in steps:
        ax.add_patch(FancyBboxPatch((0.3, y), 9.4, 1.9, boxstyle="round,pad=0.12,rounding_size=0.3",
                                    fc=fc, ec=ec, lw=1.6))
        ax.text(5, y + 1.28, title, ha="center", va="center", fontsize=9.5, fontweight="bold")
        ax.text(5, y + 0.5, sub, ha="center", va="center", fontsize=7.6, color=SUBINK)
    for y0, y1 in [(8.7, 8.0), (6.1, 5.4), (3.5, 2.6)]:
        ax.add_patch(FancyArrowPatch((5, y0), (5, y1), arrowstyle="-|>", mutation_scale=13,
                                     color=SUBINK, lw=1.4))
    ax.set_title("the verdict logic", fontsize=11)

    ax = fig.add_subplot(gs[1])
    nice = {"healthy_second_order_oscillator": "healthy 2nd-order\noscillator",
            "pais_uhlenbeck_ghost": "Pais-Uhlenbeck",
            "pais_uhlenbeck_plus_total_derivative": "PU + total\nderivative"}
    for i, k in enumerate(keys):
        eig = ref[k]["hamiltonianEigenvalues"]
        gh = ref[k]["ghost"]
        col = RED if gh else TEAL
        ax.axhline(i, color=GRID, lw=0.8, zorder=0)
        ax.scatter(eig, [i] * len(eig), color=col, s=80, zorder=3, ec="white", lw=0.8)
        ax.text(11.2, i, "GHOST" if gh else "healthy", ha="left", va="center",
                fontsize=9.5, color=col, fontweight="bold")
    ax.axvline(0, color=INK, lw=1.3)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([nice[k] for k in keys], fontsize=9)
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.set_xlim(-11, 16)
    ax.set_xlabel("Hessian(H) eigenvalues     negative  =  energy unbounded below")
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_title("H spectra  ·  negative eigenvalues are the Ostrogradski signature", fontsize=10)

    titleblock(fig, "Step 6b - the Ostrogradski ghost verdict",
               "A non-degenerate higher-derivative L has a Hamiltonian unbounded below - a negative-energy 'ghost' mode")
    takeaway(fig, "PU has 2 negative Hessian directions and oscillatory dynamics  ->  ghost = True. "
                  "Adding a total time derivative changes H but not the verdict - it is the same physical theory.",
             color=RED)
    savefig(fig, os.path.join(O, "P09_ghost_verdict_anatomy.png"))


def p10_ghost_robustness():
    g = load("ghost_detection.json")
    nb = g["noiseBoundary"]
    roc = g["roc"]["perNoise"]

    def pct(s):
        return 0.0 if s == "ground_truth" else float(s.replace("pct", ""))
    noise = [pct(r["scenario"]) for r in nb]
    mineig = [min(r["hamiltonianEigenvalues"]) for r in nb]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    ax = axes[0]
    ax.plot(noise, mineig, "o-", color=RED, lw=2, ms=6)
    ax.axhline(0, color=INK, lw=1)
    ax.fill_between(noise, mineig, 0, color=RED, alpha=0.12)
    for n, e, r in zip(noise, mineig, nb):
        pass
    ax.set_xlabel("measurement noise (%)")
    ax.set_ylabel("most negative eigenvalue of H(L_recovered)")
    ax.set_title("verdict = ghost at every level tested, 0 -> 35%", fontsize=10.5)
    ax.grid(True, color=GRID, lw=0.6)
    ax.text(0.5, 0.1, "recovered coefficients drift with noise;\nthe sign structure of H does not",
            transform=ax.transAxes, ha="center", fontsize=9, color=SUBINK)

    ax = axes[1]
    ns = sorted(roc, key=float)
    xn = [float(k) * 100 for k in ns]
    fp = [roc[k]["falsePositiveRate"] for k in ns]
    fn = [roc[k]["falseNegativeRate"] for k in ns]
    un = [roc[k]["undeterminedRate"] for k in ns]
    ax.plot(xn, fp, "o-", color=RED, lw=2, label="false positive")
    ax.plot(xn, fn, "s-", color=AMBER, lw=2, label="false negative")
    ax.plot(xn, un, "^-", color=SLATE, lw=2, label="undetermined")
    ax.set_ylim(-0.03, 0.7)
    ax.set_xlabel("measurement noise (%)")
    ax.set_ylabel("rate")
    ax.legend(fontsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_title("labelled battery  ·  3 healthy + 4 ghost systems x 3 seeds", fontsize=10.5)

    titleblock(fig, "Step 6c - how robust is the ghost flag",
               "Noise boundary on a data-recovered PU Lagrangian, and error rates on a labelled battery")
    takeaway(fig, "Zero false positives and zero false negatives across the battery. The 'undetermined' rate "
                  "rises with noise (rational H, recovery-quality gaps) - the method abstains rather than guesses.",
             color=RED)
    savefig(fig, os.path.join(O, "P10_ghost_robustness_roc.png"))


if __name__ == "__main__":
    p01_pipeline()
    p02_forward()
    p03_library()
    p04_snapping()
    p05_selectors()
    p06_noise_robustness()
    p07_differentiation()
    p08_order_inference()
    p09_ghost_anatomy()
    p10_ghost_robustness()
