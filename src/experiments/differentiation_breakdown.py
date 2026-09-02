import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments import vizstyle

ASSETS = vizstyle.ASSETS
OUTPUT = ASSETS / "differentiation_breakdown.png"
BREAKDOWN = 0.5
METHODS = ["finite_difference", "savitzky_golay", "smoothing_spline"]
ORDER_NAME = {1: "velocity", 2: "acceleration", 3: "jerk", 4: "snap"}


def organise(payload):
    noiseLevels = payload["noiseLevels"]
    table = {}
    for record in payload["records"]:
        key = (record["method"], record["order"])
        table.setdefault(key, {})[record["noiseLevel"]] = record["relativeL2Error"]
    series = {}
    for (method, order), byNoise in table.items():
        series[(method, order)] = [byNoise[n] for n in noiseLevels]
    return noiseLevels, series


def panelPerMethod(ax, method, noiseLevels, series):
    noisePercent = np.array(noiseLevels) * 100.0
    cmap = plt.get_cmap("viridis")
    for order in (1, 2, 3, 4):
        values = np.array(series[(method, order)])
        colour = cmap(0.08 + 0.24 * (order - 1))
        ax.plot(
            noisePercent,
            values,
            marker="ov^s"[order - 1],
            color=colour,
            label=f"d{order}  ({ORDER_NAME[order]})",
        )
    ax.axhspan(BREAKDOWN, ax.get_ylim()[1] if False else 1e12, color=vizstyle.BAD_RED, alpha=0.05)
    vizstyle.thresholdLine(ax, BREAKDOWN, "breakdown  (rel. L2 > 0.5)", axis="y", color=vizstyle.BAD_RED)
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_yscale("log")
    ax.set_xlim(-0.01, noisePercent.max() * 1.4)
    ax.set_ylim(3e-6, 3e7)
    ax.set_xticks([0, 0.1, 0.3, 1, 3, 10])
    ax.set_xticklabels(["0", "0.1", "0.3", "1", "3", "10"])
    ax.set_xlabel("position noise (% of signal std)")
    ax.set_ylabel("relative L2 error of estimated derivative")
    ax.set_title(vizstyle.METHOD_LABEL[method])
    ax.legend(loc="lower right", ncol=2)


def panelSurvival(ax, noiseLevels, series):
    noisePercent = np.array(noiseLevels) * 100.0
    grid = np.zeros((len(METHODS) * 4, len(noiseLevels)))
    labels = []
    for methodIndex, method in enumerate(METHODS):
        for order in (1, 2, 3, 4):
            row = methodIndex * 4 + (order - 1)
            grid[row] = np.log10(np.clip(series[(method, order)], 1e-6, 1e12))
            labels.append(f"{vizstyle.METHOD_LABEL[method].split()[0]}  d{order}")
    mesh = ax.pcolormesh(
        np.arange(len(noiseLevels) + 1),
        np.arange(len(labels) + 1),
        grid,
        cmap="RdYlGn_r",
        vmin=-3,
        vmax=3,
    )
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            broken = series[_rowKey(row)][col] > BREAKDOWN
            ax.plot(col + 0.5, row + 0.5, "x" if broken else ".", color="#111417" if broken else "#33772f", ms=7 if broken else 4)
    ax.set_xticks(np.arange(len(noiseLevels)) + 0.5)
    ax.set_xticklabels([f"{p:g}%" for p in noisePercent])
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("position noise")
    ax.set_title("Survival grid  (x = broken, dot = usable)")
    ax.invert_yaxis()
    ax.grid(False)
    bar = plt.colorbar(mesh, ax=ax, pad=0.02)
    bar.set_label("log10 relative L2 error")


def _rowKey(row):
    method = METHODS[row // 4]
    order = row % 4 + 1
    return (method, order)


def main():
    vizstyle.applyStyle()
    payload = vizstyle.loadResult("differentiation_study.json")
    noiseLevels, series = organise(payload)

    figure = plt.figure(figsize=(18, 9.2))
    grid = figure.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.3)
    for index, method in enumerate(METHODS):
        panelPerMethod(figure.add_subplot(grid[0, index]), method, noiseLevels, series)
    panelSurvival(figure.add_subplot(grid[1, :]), noiseLevels, series)

    figure.suptitle(
        "Numerical differentiation is the bottleneck of higher-derivative recovery",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.925,
        "Pais-Uhlenbeck q(t): only the quintic smoothing spline keeps 3rd/4th derivatives usable under noise; "
        "finite differences explode past 0.1%",
        ha="center",
        fontsize=10.5,
        style="italic",
        color="#4a515b",
    )
    vizstyle.provenance(
        figure,
        "source: src/experiments/results/differentiation_study.json  (differentiation_method_study.py, PU omega=1,2)",
    )
    figure.subplots_adjust(top=0.88, bottom=0.08)
    size = vizstyle.savefig(figure, OUTPUT)
    print(f"wrote {OUTPUT}  ({size} bytes)")


if __name__ == "__main__":
    main()
