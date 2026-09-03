import os

import matplotlib.pyplot as plt
import numpy as np
from common import (
    AMBER,
    BLUE,
    INK,
    OUT_GIFS,
    RED,
    SLATE,
    SPINE,
    SUBINK,
    TEAL,
    WHITE,
    pu_modes,
    pu_trajectory,
)
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

O = OUT_GIFS
plt.rcParams.update({"figure.facecolor": WHITE, "savefig.facecolor": WHITE,
                     "axes.facecolor": WHITE, "font.family": "DejaVu Sans"})


def save(anim, name, fps=14, dpi=100):
    path = os.path.join(O, name)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(anim._fig)
    print("wrote", os.path.relpath(path, "/home/kaitro/Projects/work/Something"), "  ",
          f"{os.path.getsize(path)/1e6:.1f} MB")


def title(fig, big, small=None):
    fig.text(0.5, 0.955, big, fontsize=16, fontweight="bold", ha="center", va="top", color=INK)
    if small:
        fig.text(0.5, 0.895, small, fontsize=10.5, ha="center", va="top", color=SUBINK)


def bare(ax, t=None):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(SPINE)
    ax.grid(False)
    if t:
        ax.set_title(t, fontsize=10.5, color=SUBINK, pad=6)


# ---------------------------------------------------------------------------
# G1 - recovering the law, term by term
# ---------------------------------------------------------------------------

def g1_recovery():
    fig = plt.figure(figsize=(8.4, 5.2))
    fig._fig = fig
    title(fig, "Recovering the law, one term at a time",
          "the sparse fit adds only the terms the data demands - the rest stay at zero")
    axL = fig.add_axes([0.16, 0.16, 0.46, 0.60])
    axR = fig.add_axes([0.72, 0.16, 0.24, 0.60])

    terms = ["q'^2 (motion)", "q^2", "q^4", "q^2 q_j^2", "q'^4", "q q'"]
    true_vals = np.array([1.0, -1.0, -0.15, -0.30, 0.0, 0.0])
    order = [0, 1, 2, 3]
    resid = [1.0, 0.55, 0.22, 0.07, 0.015]

    hold = 9
    steps = len(order)
    frames = steps * hold + 12
    y = np.arange(len(terms))

    def ease(x):
        return x * x * (3 - 2 * x)

    def draw(f):
        axL.clear(); axR.clear()
        k = min(f // hold, steps)          # fully-in terms
        frac = ease(min((f % hold) / hold, 1.0)) if k < steps else 1.0
        vals = np.zeros(len(terms))
        for j, idx in enumerate(order):
            if j < k:
                vals[idx] = true_vals[idx]
            elif j == k:
                vals[idx] = true_vals[idx] * frac
        colors = [TEAL if (vals[i] != 0) else "#dddddd" for i in range(len(terms))]
        axL.barh(y, vals, color=colors, height=0.6)
        axL.axvline(0, color=INK, lw=1)
        axL.set_yticks(y); axL.set_yticklabels(terms, fontsize=9)
        axL.set_xlim(-1.25, 1.35)
        axL.set_axisbelow(True)
        axL.invert_yaxis()
        axL.set_xlabel("coefficient in the recovered L", fontsize=10)
        for s in axL.spines.values():
            s.set_edgecolor(SPINE)
        axL.set_title(f"terms selected: {min(k + (frac > 0.2), steps)}", fontsize=10, color=SUBINK)

        r0 = resid[min(k, len(resid) - 1)]
        r1 = resid[min(k + 1, len(resid) - 1)]
        r = r0 + (r1 - r0) * frac
        axR.add_patch(Rectangle((0.1, 0.02), 0.8, 0.96, fc=WHITE, ec=SPINE, lw=1.2))
        axR.add_patch(Rectangle((0.1, 0.02), 0.8, 0.96 * r, ec="none",
                                fc=(TEAL if r < 0.02 else AMBER if r < 0.3 else RED)))
        axR.set_xlim(0, 1); axR.set_ylim(0, 1); axR.axis("off")
        axR.set_title("mismatch\nwith the data", fontsize=10, color=SUBINK)
        axR.text(0.5, -0.08, f"{r*100:4.1f}%", ha="center", fontsize=12, fontweight="bold",
                 color=INK, transform=axR.transAxes)
        if k >= steps:
            axL.text(0.5, 1.12, "structural match to the true law   ✓", transform=axL.transAxes,
                     ha="center", fontsize=11, color=TEAL, fontweight="bold")

    anim = FuncAnimation(fig, draw, frames=frames, interval=70)
    save(anim, "G1_recovery_term_by_term.gif")


# ---------------------------------------------------------------------------
# G2 - turn up the noise
# ---------------------------------------------------------------------------

def g2_noise_dial():
    fig = plt.figure(figsize=(8.6, 5.0))
    fig._fig = fig
    title(fig, "Turn up the noise",
          "recover the law from measurements of steadily worse quality")
    axT = fig.add_axes([0.07, 0.16, 0.55, 0.62])
    axP = fig.add_axes([0.66, 0.16, 0.30, 0.62]); axP.axis("off")

    t = np.linspace(0, 10, 800)
    q, _ = pu_trajectory(t, q0=1.0, v0=0.2, a0=-0.5, j0=0.1)
    base_std = q.std()
    rng = np.random.default_rng(7)
    noise_seq = np.concatenate([np.linspace(0, 12, 44), np.full(8, 12)])

    def verdict(n):
        if n <= 1.2:
            return "exact law recovered", TEAL, "L = v^2 - q^2 - 0.15 q^4 - 0.3 q^2 q_j^2"
        if n <= 5.2:
            return "same terms, numbers drifting", TEAL, "L ~ v^2 - q^2 - 0.15 q^4 - ..."
        if n <= 9.5:
            return "structure bending", AMBER, "spurious q^2 v^2 terms creeping in"
        return "recovery broken", RED, "wrong term set"

    def draw(f):
        n = noise_seq[f]
        axT.clear()
        noisy = q + rng.normal(0, (n / 100) * base_std, q.shape)
        axT.plot(t, q, color=BLUE, lw=1.0, alpha=0.35)
        axT.plot(t, noisy, color=AMBER, lw=0.8)
        axT.set_ylim(-1.8, 1.8)
        axT.set_xticks([]); axT.set_yticks([])
        for s in axT.spines.values():
            s.set_edgecolor(SPINE)
        axT.set_title(f"measurement noise  =  {n:4.1f}%", fontsize=11, color=INK)

        lab, col, formula = verdict(n)
        axP.clear(); axP.axis("off")
        axP.add_patch(Rectangle((0.0, 0.42), 1.0, 0.46, fc=WHITE, ec=SPINE, lw=1.2,
                                transform=axP.transAxes))
        axP.text(0.5, 0.78, lab, transform=axP.transAxes, ha="center", va="center",
                 fontsize=11, color=col, fontweight="bold", wrap=True)
        axP.text(0.5, 0.56, formula, transform=axP.transAxes, ha="center", va="center",
                 fontsize=8.2, family="monospace", color=INK, wrap=True)
        axP.text(0.5, 0.3, "same theory?  " + ("yes" if n <= 1.6 else "no"),
                 transform=axP.transAxes, ha="center", fontsize=10,
                 color=(TEAL if n <= 1.6 else RED), fontweight="bold")

    anim = FuncAnimation(fig, draw, frames=len(noise_seq), interval=90)
    save(anim, "G2_turn_up_the_noise.gif")


# ---------------------------------------------------------------------------
# G3 - the Ostrogradski ghost
# ---------------------------------------------------------------------------

def g3_ghost():
    fig = plt.figure(figsize=(8.4, 5.4))
    fig._fig = fig
    title(fig, "The Ostrogradski ghost",
          "the visible motion stays bounded while a hidden mode's energy runs to minus infinity")
    axO = fig.add_axes([0.09, 0.44, 0.84, 0.34])
    axE = fig.add_axes([0.09, 0.12, 0.84, 0.26])

    t = np.linspace(0, 14, 900)
    _, _, tot = pu_modes(t)
    xs = np.linspace(-1, 1, 200)
    well = -3.2 * xs ** 2

    frames = 70

    def draw(f):
        prog = f / (frames - 1)
        axO.clear(); axE.clear()
        cut = int(60 + prog * (len(t) - 60))
        axO.plot(t[:cut], tot[:cut], color=BLUE, lw=2)
        axO.scatter([t[cut - 1]], [tot[cut - 1]], color=BLUE, s=40, zorder=3)
        axO.set_xlim(0, 14); axO.set_ylim(-1.9, 1.9)
        axO.set_xticks([]); axO.set_yticks([])
        for s in axO.spines.values():
            s.set_edgecolor(SPINE)
        axO.set_title("what you observe: a bounded wiggle", fontsize=10, color=SUBINK)

        axE.plot(xs, well, color=SLATE, lw=2)
        px = -0.15 - 0.8 * prog
        axE.scatter([px], [-3.2 * px ** 2], s=180, color=RED, zorder=3, ec="white", lw=1.5)
        axE.annotate("", xy=(px - 0.12, -3.2 * px ** 2 - 0.9),
                     xytext=(px, -3.2 * px ** 2),
                     arrowprops={"arrowstyle": "-|>", "lw": 2, "color": RED})
        axE.set_xlim(-1.05, 1.05); axE.set_ylim(-3.6, 0.4)
        axE.set_xticks([]); axE.set_yticks([])
        for s in axE.spines.values():
            s.set_edgecolor(SPINE)
        axE.set_title("the ghost mode's energy: no floor to stop it", fontsize=10, color=RED)
        if prog > 0.6:
            axE.text(0.5, 0.12, "verdict:  GHOST  -  theory is unstable", transform=axE.transAxes,
                     ha="center", fontsize=11, color=RED, fontweight="bold")

    anim = FuncAnimation(fig, draw, frames=frames, interval=75)
    save(anim, "G3_ostrogradski_ghost.gif")


# ---------------------------------------------------------------------------
# G4 - noisy dots -> law + warning
# ---------------------------------------------------------------------------

def g4_pipeline():
    fig = plt.figure(figsize=(8.6, 5.2))
    fig._fig = fig
    title(fig, "From noisy dots to a law and a warning",
          "position measurements in  ->  a Lagrangian, a same-theory proof, and a stability flag out")
    axP = fig.add_axes([0.06, 0.14, 0.42, 0.62])
    axC = fig.add_axes([0.53, 0.12, 0.44, 0.66]); axC.axis("off")

    t = np.linspace(0, 8, 260)
    q, _ = pu_trajectory(t, q0=1, v0=0.3, a0=-0.4, j0=0.1)
    rng = np.random.default_rng(2)
    noisy = q + rng.normal(0, 0.06, q.shape)

    stages = [
        (0.0, None),
        (0.30, ("1  recovered law", "L = 4 q^2 + 5 q q'' + (q'')^2", TEAL)),
        (0.58, ("2  same theory as the true one", "checked with the equations-of-motion test", BLUE)),
        (0.82, ("3  stability", "contains an Ostrogradski ghost  ->  unstable", RED)),
    ]
    frames = 78

    def draw(f):
        prog = f / (frames - 1)
        axP.clear()
        npts = int(6 + prog * (len(t) - 6))
        axP.scatter(t[:npts], noisy[:npts], s=10, color=AMBER)
        if prog > 0.32:
            k = int((prog - 0.32) / 0.68 * len(t))
            axP.plot(t[:k], q[:k], color=BLUE, lw=2)
        axP.set_xlim(0, 8); axP.set_ylim(noisy.min() - 0.2, noisy.max() + 0.2)
        axP.set_xticks([]); axP.set_yticks([])
        for s in axP.spines.values():
            s.set_edgecolor(SPINE)
        axP.set_title("noisy positions" + ("  ->  fitted curve" if prog > 0.32 else ""),
                      fontsize=10, color=SUBINK)

        axC.clear(); axC.axis("off")
        yv = 0.82
        for thr, card in stages[1:]:
            if prog >= thr and card:
                ti, body, ec = card
                axC.add_patch(Rectangle((0.0, yv - 0.19), 1.0, 0.20, fc=WHITE, ec=SPINE,
                                        lw=1.2, transform=axC.transAxes))
                axC.add_patch(Rectangle((0.0, yv - 0.19), 0.015, 0.20, fc=ec, ec="none",
                                        transform=axC.transAxes))
                axC.text(0.5, yv - 0.03, ti, transform=axC.transAxes, ha="center", va="top",
                         fontsize=10.5, fontweight="bold", color=ec)
                axC.text(0.5, yv - 0.10, body, transform=axC.transAxes, ha="center", va="top",
                         fontsize=8.4, color=INK)
            yv -= 0.29

    anim = FuncAnimation(fig, draw, frames=frames, interval=80)
    save(anim, "G4_dots_to_law_and_warning.gif")


if __name__ == "__main__":
    g1_recovery()
    g2_noise_dial()
    g3_ghost()
    g4_pipeline()
