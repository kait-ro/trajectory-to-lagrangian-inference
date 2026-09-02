import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments import vizstyle

ASSETS = vizstyle.ASSETS
OUTPUT = ASSETS / "pipeline_confidence.png"
TRUE_LAGRANGIAN = "4*s0**2 - 5*s1**2 + s2**2   (PU, omega = 1, 2)"


def panelConfidence(ax, rows):
    noise = [row["noiseLevel"] * 100 for row in rows]
    x = np.arange(len(rows))
    width = 0.26
    order = [row["orderConfidence"] for row in rows]
    ghost = [row["ghostConfidence"] for row in rows]
    coefficient = [row["coefficientConfidence"] for row in rows]
    overall = [min(o, g) for o, g in zip(order, ghost)]

    ax.bar(x - width, order, width, color=vizstyle.OK_GREEN, label="order confidence (cross-method agreement)")
    ax.bar(x, ghost, width, color=vizstyle.GHOST, label="ghost confidence")
    ax.bar(x + width, coefficient, width, color=vizstyle.STLSQ, label="coefficient confidence (spread + plausibility)")
    ax.plot(x, overall, "o--", color="#1b1f24", lw=1.6, label="overall = min(order, ghost)")

    for xi, value in zip(x + width, coefficient):
        ax.text(xi, value + 0.02, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:g}%" for n in noise])
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("position noise (% of signal std)")
    ax.set_ylabel("confidence")
    ax.set_title("Order and ghost verdict stay maximally confident;\ncoefficients are reported separately (differentiation-limited)")
    ax.legend(loc="lower center", fontsize=8.4)


def panelCards(ax, rows):
    ax.axis("off")
    ax.text(0.0, 1.0, "true L (hidden from the pipeline):", fontsize=10, fontweight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.95, TRUE_LAGRANGIAN, fontsize=10, family="monospace", transform=ax.transAxes, color="#3b424b")

    y = 0.86
    for row in rows:
        ax.text(0.0, y, f"{row['noiseLevel'] * 100:g}% noise", fontsize=10, fontweight="bold", transform=ax.transAxes)
        ax.text(
            0.30,
            y,
            f"order {row['order']}  ({row['orderConfidence']:.2f})     ghost {row['ghost']}  ({row['ghostConfidence']:.2f})",
            fontsize=9,
            transform=ax.transAxes,
            color=vizstyle.verdictColor(row["ghost"]),
            fontweight="bold",
        )
        y -= 0.055
        ax.text(0.02, y, f"L = {row['discoveredLagrangian']}", fontsize=9.3, family="monospace", transform=ax.transAxes)
        y -= 0.05
        ax.text(0.02, y, f"diff method selected: {row['differentiationMethod']}", fontsize=8.4, style="italic", color="#5f6368", transform=ax.transAxes)
        y -= 0.085

    vizstyle.badge(
        ax,
        "recovered from noisy positions ONLY:\norder 2 + ghost = True at every noise level, full cross-method agreement",
        xy=(0.5, 0.06),
        facecolor="#fdecec",
        edgecolor=vizstyle.BAD_RED,
    )


def main():
    vizstyle.applyStyle()
    rows = vizstyle.loadResult("end_to_end_pipeline.json")

    figure = plt.figure(figsize=(16, 8))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.16)
    panelConfidence(figure.add_subplot(grid[0]), rows)
    panelCards(figure.add_subplot(grid[1]), rows)

    figure.suptitle(
        "End-to-end pipeline: noisy positions  ->  Lagrangian  +  ghost verdict  +  calibrated confidence",
        fontsize=15,
        fontweight="bold",
    )
    vizstyle.provenance(
        figure,
        "source: src/experiments/results/end_to_end_pipeline.json  (end_to_end_pipeline_validation.py; 3 differentiation methods, majority vote)",
    )
    figure.subplots_adjust(top=0.9, bottom=0.1, left=0.06, right=0.97)
    size = vizstyle.savefig(figure, OUTPUT)
    print(f"wrote {OUTPUT}  ({size} bytes)")


if __name__ == "__main__":
    main()
