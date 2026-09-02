import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments import vizstyle

ASSETS = vizstyle.ASSETS
OUTPUT = ASSETS / "noise_robustness.png"
METHODS = ["greedy", "stlsq", "lasso"]
SYSTEM_TITLE = {
    "isotropic_quartic_calibration": "isotropic quartic calibration  (reference, 6 DOF)",
    "anharmonic_chain_blind": "anharmonic chain  (blind holdout, 6 DOF)",
}


def firstFailureNoise(rows, method):
    for row in rows:
        if not row[method]["success"]:
            return row["noiseLevel"] * 100.0
    return None


def panelError(ax, system, rows):
    noise = np.array([row["noiseLevel"] * 100.0 for row in rows])
    for method in METHODS:
        errors = np.array([max(row[method]["maxAbsError"], 1e-15) for row in rows])
        success = [row[method]["success"] for row in rows]
        ax.plot(
            noise,
            errors,
            marker=vizstyle.METHOD_MARKER[method],
            color=vizstyle.METHOD_COLOR[method],
            label=vizstyle.METHOD_LABEL[method],
            zorder=3,
        )
        failed = ~np.array(success)
        if failed.any():
            ax.scatter(
                noise[failed],
                errors[failed],
                s=170,
                facecolors="none",
                edgecolors=vizstyle.METHOD_COLOR[method],
                linewidths=2.0,
                zorder=4,
            )
    ceiling = firstFailureNoise(rows, "greedy")
    if ceiling is not None:
        ax.axvspan(ceiling, noise.max() + 1, color=vizstyle.GREEDY, alpha=0.07)
        vizstyle.thresholdLine(ax, ceiling, f"greedy fails from {ceiling:g}%", color=vizstyle.GREEDY)
    ax.set_yscale("log")
    ax.set_xlim(-0.2, noise.max() + 0.4)
    ax.set_ylim(1e-15, 3.0)
    ax.set_xlabel("position noise (% of signal std)")
    ax.set_ylabel("max |coefficient error| on shared terms")
    ax.set_title(SYSTEM_TITLE[system])
    ax.legend(loc="lower right")
    ax.annotate(
        "hollow ring = wrong sparse term set",
        xy=(0.02, 0.04),
        xycoords="axes fraction",
        fontsize=8,
        color="#6a6f77",
    )


def panelCounts(ax, system, rows):
    noise = [f"{row['noiseLevel'] * 100:g}%" for row in rows]
    x = np.arange(len(noise))
    width = 0.26
    for offset, method in zip((-1, 0, 1), METHODS):
        missing = [row[method]["missing"] for row in rows]
        spurious = [row[method]["spurious"] for row in rows]
        base = x + offset * width
        ax.bar(base, missing, width, color=vizstyle.METHOD_COLOR[method], label=f"{method} missing")
        ax.bar(
            base,
            spurious,
            width,
            bottom=missing,
            color=vizstyle.METHOD_COLOR[method],
            alpha=0.45,
            hatch="////",
            label=f"{method} spurious",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(noise)
    ax.set_xlabel("position noise")
    ax.set_ylabel("wrong terms  (solid = missing, hatched = spurious)")
    ax.set_title("Term-set errors per selector")
    ax.legend(ncol=3, fontsize=7.4, loc="upper left")


def panelVerdict(ax, data):
    allNoise = sorted({row["noiseLevel"] for rows in data.values() for row in rows})
    rowLabels = []
    cellColour = []
    cellText = []
    for system in data:
        rows = {row["noiseLevel"]: row for row in data[system]}
        shortName = "isotropic quartic" if "isotropic" in system else "anharmonic chain"
        for method in METHODS:
            rowLabels.append(f"{shortName}  /  {method}")
            colourRow = []
            textRow = []
            for noiseLevel in allNoise:
                record = rows.get(noiseLevel, {}).get(method)
                if record is None:
                    colourRow.append("#e9ecef")
                    textRow.append("")
                elif record["structurallyEquivalent"]:
                    colourRow.append(vizstyle.OK_GREEN)
                    textRow.append("same theory")
                elif record["success"]:
                    colourRow.append(vizstyle.STLSQ)
                    textRow.append("coeffs close")
                else:
                    colourRow.append(vizstyle.BAD_RED)
                    textRow.append("wrong terms")
            cellColour.append(colourRow)
            cellText.append(textRow)

    ax.set_xlim(0, len(allNoise))
    ax.set_ylim(0, len(rowLabels))
    for r in range(len(rowLabels)):
        for c in range(len(allNoise)):
            ax.add_patch(
                plt.Rectangle((c + 0.03, r + 0.08), 0.94, 0.84, color=cellColour[r][c], alpha=0.85)
            )
            ax.text(
                c + 0.5,
                r + 0.5,
                cellText[r][c],
                ha="center",
                va="center",
                fontsize=8.5,
                color="white",
                fontweight="bold",
            )
    ax.set_xticks(np.arange(len(allNoise)) + 0.5)
    ax.set_xticklabels([f"{n * 100:g}%  noise" for n in allNoise], fontsize=9.5)
    ax.set_yticks(np.arange(len(rowLabels)) + 0.5)
    ax.set_yticklabels(rowLabels, fontsize=9)
    ax.invert_yaxis()
    ax.grid(False)
    ax.tick_params(length=0)
    ax.set_title("Equivalence-class verdict:  is EL(L_recovered - L_true) identically zero?")


def main():
    vizstyle.applyStyle()
    data = vizstyle.loadResult("model_selection_comparison.json")

    figure = plt.figure(figsize=(17, 12.5))
    grid = figure.add_gridspec(3, 2, height_ratios=[1.15, 0.9, 0.72], hspace=0.42, wspace=0.24)
    for column, system in enumerate(data):
        panelError(figure.add_subplot(grid[0, column]), system, data[system])
        panelCounts(figure.add_subplot(grid[1, column]), system, data[system])
    panelVerdict(figure.add_subplot(grid[2, :]), data)

    figure.suptitle(
        "The ~1-2% greedy noise ceiling is a selector artifact - debiased LASSO holds the term set to 5%",
        fontsize=16,
        fontweight="bold",
    )
    vizstyle.provenance(
        figure,
        "source: src/experiments/results/model_selection_comparison.json  "
        "(model_selection_comparison.py, degree-4 library, one seed per noise level, frozen tolerances)",
    )
    figure.subplots_adjust(top=0.93, bottom=0.06)
    size = vizstyle.savefig(figure, OUTPUT)
    print(f"wrote {OUTPUT}  ({size} bytes)")


if __name__ == "__main__":
    main()
