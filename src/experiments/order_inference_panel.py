import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import vizstyle

ASSETS = vizstyle.ASSETS
OUTPUT = ASSETS / "order_inference_panel.png"
CONDITION_A = 0.01
FLOOR = 1e-17

SYSTEM_LABEL = {
    "pais_uhlenbeck": "Pais-Uhlenbeck\n(true order 2, ghost)",
    "anharmonic_oscillator": "anharmonic oscillator\n(true order 1)",
    "harmonic_oscillator": "harmonic oscillator\n(true order 1, linear)",
}


def panelResiduals(ax, records):
    groupWidth = 0.8
    xTicks = []
    xLabels = []
    for groupIndex, record in enumerate(records):
        perOrder = record["perOrder"]
        barWidth = groupWidth / max(len(perOrder), 1)
        for orderIndex, entry in enumerate(perOrder):
            x = groupIndex + (orderIndex - (len(perOrder) - 1) / 2) * barWidth
            height = max(entry["scaledResidual"], FLOOR)
            feasible = entry["scaledResidual"] < CONDITION_A
            colour = vizstyle.OK_GREEN if feasible else vizstyle.UNDETERMINED
            hatch = "xxx" if entry["degenerate"] else None
            ax.bar(x, height, barWidth * 0.92, bottom=FLOOR, color=colour, hatch=hatch, edgecolor="#1b1f24", linewidth=0.6)
            tag = f"order {entry['order']}: {'feasible' if feasible else 'infeasible'}"
            ax.text(x, height * (0.35 if not feasible else 4), tag, ha="center", fontsize=8, va="top" if not feasible else "bottom", color="white" if not feasible else "#1b1f24")
            if entry["degenerate"]:
                ax.text(x, 1e-12, "degenerate\nrank-deficient", ha="center", fontsize=7.8, color=vizstyle.BAD_RED, fontweight="bold")
        xTicks.append(groupIndex)
        xLabels.append(SYSTEM_LABEL[record["system"]])
        ax.text(groupIndex, 15.0, f"inferred order {record['inferredOrder']}", ha="center", fontsize=10, fontweight="bold", color=vizstyle.ACCENT)

    vizstyle.thresholdLine(ax, CONDITION_A, "Condition A: residualRmsTolerance = 0.01", axis="y")
    ax.set_yscale("log")
    ax.set_ylim(FLOOR, 40.0)
    ax.set_xlim(-0.6, len(records) - 0.4)
    ax.set_xticks(xTicks)
    ax.set_xticklabels(xLabels)
    ax.set_ylabel("EL feasibility residual  (log)")
    ax.set_title("Smallest order whose kinetic EL column is spanned by the other EL columns")


def panelExplainer(ax, records):
    ax.axis("off")
    lines = [
        ("feasibility test", "does the data satisfy an order-n Euler-Lagrange equation?  residual -> 0 means yes."),
        ("Pais-Uhlenbeck", "order 1 residual 0.50 (infeasible) -> order 2 residual 5e-17 (feasible). Clean order-2 verdict; the higher mode is excited."),
        ("anharmonic oscillator", "order 1 residual 8e-16, and the degree-2 library spans many EL directions -> genuine feasible order-1 verdict."),
        ("harmonic oscillator", "order 1 residual 2e-16 BUT the kept EL matrix has numerical rank <= 1: every order tests as feasible, so the zero residual carries no evidence. Flagged degenerate (Open problem F); reduceOrderToPrior refuses to lower an order on a degenerate inference."),
    ]
    y = 1.0
    for head, body in lines:
        ax.text(0.0, y, head, fontsize=10.5, fontweight="bold", color=vizstyle.ACCENT, transform=ax.transAxes)
        y -= 0.075
        wrapped = _wrap(body, 118)
        for segment in wrapped:
            ax.text(0.02, y, segment, fontsize=9.2, transform=ax.transAxes)
            y -= 0.058
        y -= 0.03


def _wrap(text, width):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def main():
    vizstyle.applyStyle()
    records = vizstyle.loadResult("order_inference.json")

    figure = plt.figure(figsize=(15.5, 9.5))
    grid = figure.add_gridspec(2, 1, height_ratios=[1.5, 1.0], hspace=0.34)
    panelResiduals(figure.add_subplot(grid[0]), records)
    panelExplainer(figure.add_subplot(grid[1]), records)

    figure.suptitle(
        "Order inference: reading the model order off trajectory data",
        fontsize=16,
        fontweight="bold",
    )
    vizstyle.provenance(
        figure,
        "source: src/experiments/results/order_inference.json  (order_inference_validation.py; libraryMaxDegree = 2)",
    )
    figure.subplots_adjust(top=0.92, bottom=0.06, left=0.12, right=0.96)
    size = vizstyle.savefig(figure, OUTPUT)
    print(f"wrote {OUTPUT}  ({size} bytes)")


if __name__ == "__main__":
    main()
