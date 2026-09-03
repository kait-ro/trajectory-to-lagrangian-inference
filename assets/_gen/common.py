import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kaitro/Projects/work/Something"
RESULTS = os.path.join(ROOT, "src/experiments/results")
OUT_PERSONAL = os.path.join(ROOT, "assets/reports/personal")
OUT_VISUAL = os.path.join(ROOT, "assets/reports/visual")
OUT_GIFS = os.path.join(ROOT, "assets/reports/gifs")

INK = "#1a1a1a"
SUBINK = "#555555"
WHITE = "#ffffff"
AXFACE = "#fbfbfb"
GRID = "#e4e4e4"
SPINE = "#c9c9c9"

BLUE = "#2077d6"
TEAL = "#1a9e8f"
GREEN = "#2ca02c"
PURPLE = "#8e2f9e"
RED = "#d62728"
AMBER = "#e08214"
SLATE = "#7f7f7f"

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "text.color": INK,
    "axes.edgecolor": SPINE,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": SUBINK,
    "ytick.color": SUBINK,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "figure.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "axes.facecolor": AXFACE,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": SPINE,
    "legend.fontsize": 9,
})


def load(name):
    with open(os.path.join(RESULTS, name)) as fh:
        return json.load(fh)


def savefig(fig, path, dpi=170):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def titleblock(fig, title, subtitle=None, tag=None):
    h = fig.get_size_inches()[1]
    fig.suptitle(title, fontsize=19, fontweight="bold", y=1.0 + 0.78 / h)
    if subtitle:
        fig.text(0.5, 1.0 + 0.24 / h, subtitle, fontsize=12, ha="center", va="bottom",
                 color=SUBINK, style="italic")


def takeaway(fig, text, y=None, color=None):
    import textwrap
    h = fig.get_size_inches()[1]
    yy = -0.34 / h if y is None else y
    fig.text(0.5, yy, "\n".join(textwrap.wrap(text, 122)), fontsize=11, ha="center",
             va="top", color=SUBINK, style="italic")


def panel(ax):
    for s in ax.spines.values():
        s.set_edgecolor(SPINE)
    ax.grid(True, color=GRID, lw=0.8)


# ----- physics -----------------------------------------------------------------

def pu_trajectory(t, q0=1.0, v0=0.0, a0=0.0, j0=0.0, w1=1.0, w2=2.0):
    from numpy.linalg import solve
    W = np.array([
        [1, 0, 1, 0],
        [0, w1, 0, w2],
        [-w1**2, 0, -w2**2, 0],
        [0, -w1**3, 0, -w2**3],
    ], float)
    c = solve(W, np.array([q0, v0, a0, j0], float))
    q = c[0]*np.cos(w1*t) + c[1]*np.sin(w1*t) + c[2]*np.cos(w2*t) + c[3]*np.sin(w2*t)
    return q, c


def pu_modes(t, amp1=1.0, amp2=0.6, w1=1.0, w2=2.0):
    m1 = amp1*np.cos(w1*t)
    m2 = amp2*np.cos(w2*t)
    return m1, m2, m1 + m2


def pu_ostrogradski_energy(t, amp1=1.0, amp2=0.6, w1=1.0, w2=2.0):
    pref = (w2**2 - w1**2)
    e1 = 0.5 * pref * w1**2 * amp1**2 * np.ones_like(t)
    e2 = 0.5 * pref * w2**2 * amp2**2 * np.ones_like(t)
    return e1, e2, e1 - e2


def load_traj_csv(name, tid=0, cols=("q0", "q1", "q2")):
    import pandas as pd
    path = os.path.join(ROOT, "assets", name)
    df = pd.read_csv(path, nrows=4000)
    df = df[df.trajectory_id == tid]
    return df["t"].to_numpy(), {c: df[c].to_numpy() for c in cols}
