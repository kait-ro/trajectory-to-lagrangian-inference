import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "assets"
RESULTS = REPO / "src" / "experiments" / "results"

OK_GREEN = "#2e7d32"
BAD_RED = "#c62828"
UNDETERMINED = "#8a8a8a"

GREEDY = "#c62828"
LASSO = "#1565c0"
STLSQ = "#f9a825"

HEALTHY = "#1f6feb"
GHOST = "#d1495b"
POSITIVE = "#2e8b57"
NEGATIVE = "#c62828"

MODE_LOW = "#1565c0"
MODE_HIGH = "#8e24aa"
EXPANSION = "#6a1b9a"
ACCENT = "#00695c"

SEQUENTIAL = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]

METHOD_COLOR = {
    "greedy": GREEDY,
    "stlsq": STLSQ,
    "lasso": LASSO,
    "finite_difference": "#c62828",
    "savitzky_golay": "#f9a825",
    "smoothing_spline": "#1565c0",
}

METHOD_MARKER = {
    "greedy": "o",
    "stlsq": "s",
    "lasso": "^",
    "finite_difference": "o",
    "savitzky_golay": "s",
    "smoothing_spline": "D",
}

METHOD_LABEL = {
    "greedy": "greedy forward-selection",
    "stlsq": "SINDy STLSQ",
    "lasso": "debiased LASSO",
    "finite_difference": "finite difference",
    "savitzky_golay": "Savitzky-Golay",
    "smoothing_spline": "quintic smoothing spline",
}

_LIGHT = {
    "figure.facecolor": "#ffffff",
    "figure.edgecolor": "#ffffff",
    "savefig.facecolor": "#ffffff",
    "axes.facecolor": "#fcfcfd",
    "axes.edgecolor": "#b6bcc4",
    "axes.labelcolor": "#1b1f24",
    "axes.titlecolor": "#12161b",
    "text.color": "#1b1f24",
    "xtick.color": "#3b424b",
    "ytick.color": "#3b424b",
    "grid.color": "#dfe3e8",
}

_DARK = {
    "figure.facecolor": "#12161c",
    "figure.edgecolor": "#12161c",
    "savefig.facecolor": "#12161c",
    "axes.facecolor": "#191f27",
    "axes.edgecolor": "#3a434f",
    "axes.labelcolor": "#e4e8ee",
    "axes.titlecolor": "#f2f5f9",
    "text.color": "#e4e8ee",
    "xtick.color": "#aab3bf",
    "ytick.color": "#aab3bf",
    "grid.color": "#2b333d",
}


def applyStyle(dark=False):
    palette = _DARK if dark else _LIGHT
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlepad": 10,
            "axes.labelsize": 11,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.7,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#c9ced6" if not dark else "#3a434f",
            "legend.fontsize": 9.5,
            "lines.linewidth": 2.1,
            "lines.markersize": 6.5,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )
    plt.rcParams.update(palette)


def provenance(figure, text, y=0.012):
    figure.text(
        0.5,
        y,
        text,
        ha="center",
        va="bottom",
        fontsize=7.6,
        color="#7b828c",
        style="italic",
    )


def thresholdLine(ax, value, label, axis="x", color="#12161b", side="right"):
    if axis == "x":
        ax.axvline(value, color=color, ls="--", lw=1.2, zorder=1)
        ax.annotate(
            label,
            xy=(value, 1.0),
            xytext=(4 if side == "right" else -4, -6),
            textcoords="offset points",
            xycoords=("data", "axes fraction"),
            ha="left" if side == "right" else "right",
            va="top",
            fontsize=8.2,
            color=color,
        )
    else:
        ax.axhline(value, color=color, ls="--", lw=1.2, zorder=1)
        ax.annotate(
            label,
            xy=(1.0, value),
            xytext=(-4, 4),
            textcoords="offset points",
            xycoords=("axes fraction", "data"),
            ha="right",
            va="bottom",
            fontsize=8.2,
            color=color,
        )


def verdictColor(ghost):
    if ghost is True:
        return GHOST
    if ghost is False:
        return POSITIVE
    return UNDETERMINED


def loadResult(name):
    return json.loads((RESULTS / name).read_text())


def badge(ax, text, xy=(0.5, 0.5), facecolor="#e6f4ea", edgecolor="#137333"):
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11.5,
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": facecolor,
            "edgecolor": edgecolor,
            "linewidth": 1.4,
        },
        zorder=6,
    )


def savefig(figure, path, dpi=150):
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return Path(path).stat().st_size
