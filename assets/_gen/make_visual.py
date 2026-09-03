import os

import matplotlib.pyplot as plt
import numpy as np
from common import (
    AMBER,
    BLUE,
    GREEN,
    GRID,
    INK,
    OUT_VISUAL,
    PURPLE,
    RED,
    SLATE,
    SPINE,
    SUBINK,
    TEAL,
    WHITE,
    load,
    pu_modes,
    pu_trajectory,
    savefig,
)
from matplotlib.patches import FancyArrowPatch, Rectangle

O = OUT_VISUAL


def head(fig, title, sub=None):
    h = fig.get_size_inches()[1]
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.0 + (0.52 if sub else 0.34) / h)
    if sub:
        fig.text(0.5, 1.0 + 0.12 / h, sub, fontsize=11, ha="center", va="bottom",
                 color=SUBINK)


def stage(fig, rect=(0.03, 0.05, 0.94, 0.86)):
    ax = fig.add_axes(rect); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return ax


def chip(ax, x, y, w, h, lines, ec=SPINE, fc=WHITE, fs=12, mono=True, label=None,
         label_color=None):
    ax.add_patch(Rectangle((x, y), w, h, transform=ax.transAxes, clip_on=False,
                           fc=fc, ec=ec, lw=1.6))
    if label:
        ax.text(x + 0.02, y + h + 0.03, label, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=fs - 1, fontweight="bold", color=label_color or ec)
    ax.text(x + w / 2, y + h / 2, "\n".join(lines), transform=ax.transAxes, ha="center",
            va="center", fontsize=fs, color=INK,
            family="monospace" if mono else "DejaVu Sans")


def bare(ax, title=None):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(SPINE)
    ax.grid(False)
    ax.set_facecolor(WHITE)
    if title:
        ax.set_title(title, fontsize=11, color=SUBINK, pad=6)


# ---------------------------------------------------------------------------

def v01():
    fig = plt.figure(figsize=(9.6, 4.6))
    head(fig, "Recovering the law behind the motion")
    axp = fig.add_axes([0.05, 0.16, 0.44, 0.70])
    t = np.linspace(0, 10, 800)
    q, _ = pu_trajectory(t, q0=1.0, v0=0.2, a0=-0.5, j0=0.1)
    rng = np.random.default_rng(1)
    axp.scatter(t[::16], q[::16] + rng.normal(0, 0.05, len(t[::16])), s=16, color=BLUE)
    bare(axp, "measured positions   q(t)")

    ax = stage(fig, (0.0, 0.0, 1.0, 1.0))
    ax.add_patch(FancyArrowPatch((0.52, 0.5), (0.60, 0.5), arrowstyle="-|>",
                                 mutation_scale=26, lw=2.4, color=INK))
    chip(ax, 0.62, 0.30, 0.34, 0.40, ["L  =  T( q' )", "        -  V( q )"],
         ec=INK, fs=14, label="the Lagrangian", label_color=INK)
    ax.text(0.79, 0.24, "symmetries · conserved quantities · the action",
            transform=ax.transAxes, ha="center", fontsize=9, color=SUBINK, style="italic")
    savefig(fig, os.path.join(O, "V01_the_inverse_problem.png"))


def v02():
    fig = plt.figure(figsize=(8.8, 4.3))
    head(fig, "How the method is tested")
    ax = stage(fig, (0.03, 0.06, 0.94, 0.82))
    nodes = [(0.06, 0.55, "known law", BLUE), (0.55, 0.55, "simulate\n+ noise", AMBER),
             (0.55, 0.08, "recover\nfrom data", TEAL), (0.06, 0.08, "same theory?", PURPLE)]
    for x, y, txt, c in nodes:
        ax.add_patch(Rectangle((x, y), 0.39, 0.30, transform=ax.transAxes, clip_on=False,
                               fc=WHITE, ec=c, lw=2))
        ax.text(x + 0.195, y + 0.15, txt, transform=ax.transAxes, ha="center", va="center",
                fontsize=13, color=INK)
    for a, b in [((0.45, 0.70), (0.55, 0.70)), ((0.745, 0.55), (0.745, 0.38)),
                 ((0.55, 0.23), (0.45, 0.23))]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=20, lw=2,
                                     color=SLATE))
    ax.add_patch(FancyArrowPatch((0.045, 0.23), (0.045, 0.55), arrowstyle="-|>",
                                 mutation_scale=18, lw=1.6, color=SLATE,
                                 connectionstyle="arc3,rad=-0.45"))
    ax.text(0.5, -0.02, "one fixed set of settings for every system   ·   blind hold-out   ·   no peeking",
            transform=ax.transAxes, ha="center", fontsize=9.5, color=SUBINK, style="italic")
    savefig(fig, os.path.join(O, "V02_hide_and_grade.png"))


def v03():
    fig = plt.figure(figsize=(9.4, 4.2))
    head(fig, "Same theory  ≠  same-looking formula")
    ax = stage(fig, (0.04, 0.04, 0.92, 0.84))
    chip(ax, 0.04, 0.60, 0.38, 0.34, ["L  =  q q''"], ec=SPINE, fs=16)
    chip(ax, 0.58, 0.60, 0.38, 0.34, ["L  =  - (q')^2"], ec=SPINE, fs=16)
    ax.text(0.5, 0.77, "=", transform=ax.transAxes, ha="center", va="center", fontsize=32,
            color=GREEN, fontweight="bold")
    ax.text(0.5, 0.52, "identical equations of motion", transform=ax.transAxes, ha="center",
            fontsize=12, color=GREEN)

    chip(ax, 0.04, 0.08, 0.38, 0.32, ["fitted:  0.500"], ec=SPINE, fs=15)
    chip(ax, 0.58, 0.08, 0.38, 0.32, ["true:    0.520"], ec=SPINE, fs=15)
    ax.text(0.5, 0.25, "≠", transform=ax.transAxes, ha="center", va="center", fontsize=32,
            color=RED, fontweight="bold")
    ax.text(0.5, 0.00, "different equations of motion", transform=ax.transAxes, ha="center",
            fontsize=12, color=RED)
    savefig(fig, os.path.join(O, "V03_same_physics.png"))


def v04():
    fig = plt.figure(figsize=(9.4, 2.7))
    head(fig, "Measurement noise the recovery survives")
    ax = fig.add_axes([0.06, 0.42, 0.88, 0.28])
    ax.set_xlim(0, 12); ax.set_ylim(0, 1)
    zones = [(0, 1, GREEN, "exact"), (1, 5, "#8fd19e", "structure holds"),
             (5, 10, AMBER, "bending"), (10, 12, RED, "breaks")]
    for a, b, c, lab in zones:
        ax.axvspan(a, b, color=c, alpha=0.9)
        if b - a < 1.5:
            ax.text((a + b) / 2, 1.35, lab, ha="center", va="bottom", fontsize=10, color=INK)
        else:
            ax.text((a + b) / 2, 0.5, lab, ha="center", va="center", fontsize=11,
                    color="white", fontweight="bold")
    ax.set_yticks([])
    ax.set_xticks([0, 1, 5, 10])
    ax.set_xticklabels(["0", "1%", "5%", "10%"])
    ax.set_xlabel("position noise  (% of signal)")
    for s in ax.spines.values():
        s.set_edgecolor(SPINE)
    ax.grid(False)
    savefig(fig, os.path.join(O, "V04_noise_budget.png"))


def v05():
    fig = plt.figure(figsize=(9.0, 2.8))
    head(fig, "Naive vs debiased term selection", "recovered the exact law?")
    ax = stage(fig, (0.04, 0.04, 0.92, 0.84))
    noise = ["clean", "1%", "2%", "5%"]
    rows = [("naive", [1, 1, 0, 0], 0.60), ("debiased", [1, 1, 1, 1], 0.16)]
    xs = np.linspace(0.34, 0.92, 4)
    for lbl, row, yc in rows:
        ax.text(0.04, yc, lbl, transform=ax.transAxes, fontsize=13, fontweight="bold",
                va="center", color=INK)
        for x, ok in zip(xs, row):
            ax.text(x, yc, "✓" if ok else "✗", transform=ax.transAxes, ha="center",
                    va="center", fontsize=23, fontweight="bold",
                    color=(GREEN if ok else RED))
    for x, n in zip(xs, noise):
        ax.text(x, 0.98, n, transform=ax.transAxes, ha="center", fontsize=11, color=SUBINK)
    savefig(fig, os.path.join(O, "V05_old_vs_new.png"))


def v06():
    fig = plt.figure(figsize=(10.4, 3.5))
    head(fig, "Why higher derivatives are the hard part")
    t = np.linspace(0, 6, 500)
    clean = np.sin(1.3 * t)
    rng = np.random.default_rng(4)
    noisy = clean + rng.normal(0, 0.05, t.shape)
    from scipy.interpolate import make_smoothing_spline
    sp = make_smoothing_spline(t, noisy, lam=1e-3)

    a1 = fig.add_axes([0.04, 0.14, 0.28, 0.66])
    a1.plot(t, noisy, color=SLATE, lw=0.7); a1.plot(t, clean, color=BLUE, lw=2)
    bare(a1, "noisy signal")
    a2 = fig.add_axes([0.37, 0.14, 0.28, 0.66])
    a2.plot(t, np.gradient(noisy, t), color=RED, lw=0.7)
    bare(a2, "slope, taken directly")
    a3 = fig.add_axes([0.70, 0.14, 0.28, 0.66])
    a3.plot(t, 1.3 * np.cos(1.3 * t), color=BLUE, lw=2)
    a3.plot(t, sp.derivative()(t), color=TEAL, lw=1.6, ls="--")
    bare(a3, "slope, after smoothing")
    savefig(fig, os.path.join(O, "V06_noisy_slopes.png"))


def v07():
    fig = plt.figure(figsize=(8.2, 3.9))
    head(fig, "Derivative levels the law uses")
    ax = fig.add_axes([0.16, 0.15, 0.74, 0.76])
    ax.set_axisbelow(True)
    ax.grid(False); ax.grid(True, axis="y", color=GRID)
    ax.bar(["pendulum-type", "Pais-Uhlenbeck"], [1, 2], color=[BLUE, RED], width=0.5)
    ax.set_ylim(0, 2.4); ax.set_yticks([1, 2])
    ax.text(0, 1.08, "1", ha="center", fontsize=14, fontweight="bold")
    ax.text(1, 2.08, "2", ha="center", fontsize=14, fontweight="bold")
    for s in ax.spines.values():
        s.set_edgecolor(SPINE)
    savefig(fig, os.path.join(O, "V07_how_deep.png"))


def v08():
    fig = plt.figure(figsize=(9.6, 3.7))
    head(fig, "The Ostrogradski ghost")
    t = np.linspace(0, 12, 800)
    _, _, tot = pu_modes(t)
    a1 = fig.add_axes([0.05, 0.16, 0.42, 0.64])
    a1.plot(t, tot, color=BLUE, lw=2)
    bare(a1, "observed motion  -  bounded")
    a2 = fig.add_axes([0.55, 0.16, 0.42, 0.64])
    tt = np.linspace(0, 10, 200)
    a2.plot(tt, -0.35 * tt ** 2, color=RED, lw=2.4)
    a2.annotate("", xy=(9.3, -0.35 * 9.3 ** 2), xytext=(7, -0.35 * 7 ** 2),
                arrowprops={"arrowstyle": "-|>", "lw": 2, "color": RED})
    bare(a2, "hidden mode's energy  -  no floor")
    savefig(fig, os.path.join(O, "V08_the_ghost.png"))


def v09():
    fig = plt.figure(figsize=(9.4, 3.2))
    head(fig, "Stability verdict vs measurement noise")
    nb = load("ghost_detection.json")["noiseBoundary"]

    def pct(s):
        return 0.0 if s == "ground_truth" else float(s.replace("pct", ""))
    noise = [pct(r["scenario"]) for r in nb]
    ax = fig.add_axes([0.06, 0.26, 0.9, 0.5])
    ax.plot(noise, [1] * len(noise), color=RED, lw=3, zorder=1)
    ax.scatter(noise, [1] * len(noise), s=70, color=RED, zorder=3)
    ax.set_ylim(0.6, 1.4); ax.set_yticks([1]); ax.set_yticklabels(["GHOST"], fontsize=12,
                                                                  fontweight="bold")
    ax.set_xlim(-1.5, 37); ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35])
    ax.set_xlabel("measurement noise  (%)")
    for s in ax.spines.values():
        s.set_edgecolor(SPINE)
    ax.grid(True, axis="x", color=GRID)
    savefig(fig, os.path.join(O, "V09_warning_survives.png"))


def v10():
    fig = plt.figure(figsize=(10.2, 4.8))
    head(fig, "Noisy positions in  —  law, proof and warning out")
    axp = fig.add_axes([0.04, 0.16, 0.30, 0.66])
    t = np.linspace(0, 8, 400)
    q, _ = pu_trajectory(t, q0=1, v0=0.3, a0=-0.4, j0=0.1)
    rng = np.random.default_rng(2)
    axp.scatter(t[::5], q[::5] + rng.normal(0, 0.06, len(t[::5])), s=11, color=BLUE)
    bare(axp, "measured positions")

    ax = stage(fig, (0.0, 0.0, 1.0, 1.0))
    ax.add_patch(FancyArrowPatch((0.36, 0.5), (0.42, 0.5), arrowstyle="-|>",
                                 mutation_scale=22, lw=2.2, color=INK))
    chip(ax, 0.45, 0.63, 0.52, 0.19, ["L = 4 q^2 + 5 q q'' + (q'')^2"], ec=TEAL, fs=12,
         label="recovered law", label_color=TEAL)
    chip(ax, 0.45, 0.36, 0.52, 0.17, ["same theory as the true one"], ec=BLUE, fs=11.5,
         mono=False, label="checked, not guessed", label_color=BLUE)
    chip(ax, 0.45, 0.09, 0.52, 0.17, ["contains an Ostrogradski ghost"], ec=RED, fs=11.5,
         mono=False, label="stability", label_color=RED)
    savefig(fig, os.path.join(O, "V10_full_story.png"))


if __name__ == "__main__":
    for f in (v01, v02, v03, v04, v05, v06, v07, v08, v09, v10):
        f()
