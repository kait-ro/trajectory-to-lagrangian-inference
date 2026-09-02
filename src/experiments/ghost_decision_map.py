import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments import vizstyle

ASSETS = vizstyle.ASSETS
OUTPUT = ASSETS / "ghost_decision_map.png"

REFERENCE_LABEL = {
    "healthy_second_order_oscillator": "healthy 2nd-order oscillator",
    "pais_uhlenbeck_ghost": "Pais-Uhlenbeck (ghost)",
    "pais_uhlenbeck_plus_total_derivative": "PU + total derivative",
}


def scenarioNoise(name):
    if name in ("ground_truth", "ground_truth_derivatives"):
        return 0.0
    return float(name.replace("pct", ""))


def panelSpectrum(ax, reference):
    names = list(reference)
    width = 0.8
    offset = 0.0
    ticks = []
    for name in names:
        eig = np.array(reference[name]["hamiltonianEigenvalues"])
        xs = offset + np.arange(len(eig))
        colours = [vizstyle.POSITIVE if v > 0 else vizstyle.NEGATIVE for v in eig]
        ax.bar(xs, eig, width=width, color=colours, edgecolor="#2b2b2b", linewidth=0.6)
        for x, v in zip(xs, eig):
            ax.text(x, v + (0.5 if v >= 0 else -0.5), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
        ticks.append((xs.mean(), REFERENCE_LABEL[name], reference[name]["ghost"]))
        offset += len(eig) + 1.2
    ax.axhline(0.0, color="#2b2b2b", lw=1.1)
    ax.set_xticks([t[0] for t in ticks])
    ax.set_xticklabels([f"{t[1]}\nghost = {t[2]}" for t in ticks], fontsize=9.5)
    ax.set_ylabel("eigenvalues of the Ostrogradski Hessian(H)")
    ax.set_title("A single negative eigenvalue of H is the ghost signature")
    ax.text(
        0.98,
        0.04,
        "negative eigenvalue  =>  H unbounded below;\nwith oscillatory dynamics  =>  Ostrogradski ghost",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.6,
        color=vizstyle.NEGATIVE,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#fdecec", "edgecolor": vizstyle.NEGATIVE, "linewidth": 0.9},
    )


def panelNoiseBoundary(ax, boundary):
    noise = np.array([scenarioNoise(row["scenario"]) for row in boundary])
    order = np.argsort(noise)
    noise = noise[order]
    boundary = [boundary[i] for i in order]
    eigStack = np.array([sorted(row["hamiltonianEigenvalues"]) for row in boundary])
    for column in range(eigStack.shape[1]):
        colour = vizstyle.NEGATIVE if eigStack[:, column].mean() < 0 else vizstyle.POSITIVE
        ax.plot(noise, eigStack[:, column], marker="o", color=colour, lw=1.8, alpha=0.85)
    vizstyle.thresholdLine(ax, 0.0, "decision boundary (eig = 0)", axis="y")
    ax.fill_between(noise, -12, 0, color=vizstyle.NEGATIVE, alpha=0.05)
    ax.set_xlabel("measurement noise (% of signal std)")
    ax.set_ylabel("Hessian(H) eigenvalues of the recovered PU Lagrangian")
    ax.set_ylim(-12, 6)
    ax.set_title("Ghost verdict survives to 35% noise")
    ax.annotate(
        "recovered L stays order-2 with oscillatory dynamics and\nan indefinite H at every noise level tested -> ghost = True throughout",
        xy=(0.03, 0.05),
        xycoords="axes fraction",
        fontsize=8.3,
        color="#4a515b",
    )


def panelRoc(ax, roc):
    noise = sorted(float(k) for k in roc["perNoise"])
    keys = [_key(roc, n) for n in noise]
    fp = [roc["perNoise"][k]["falsePositiveRate"] for k in keys]
    fn = [roc["perNoise"][k]["falseNegativeRate"] for k in keys]
    undetermined = [roc["perNoise"][k]["undeterminedRate"] for k in keys]
    noisePercent = np.array(noise) * 100.0
    x = np.arange(len(noise))
    ax.bar(x - 0.25, fp, 0.25, color=vizstyle.BAD_RED, label="false positive rate")
    ax.bar(x, fn, 0.25, color="#8e24aa", label="false negative rate")
    ax.bar(x + 0.25, undetermined, 0.25, color=vizstyle.UNDETERMINED, label="undetermined rate")
    for xi, value in zip(x + 0.25, undetermined):
        ax.text(xi, value + 0.02, f"{value:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:g}%" for p in noisePercent])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("measurement noise")
    ax.set_ylabel("rate over the labelled battery")
    ax.set_title("FP = FN = 0;  cost is abstention, not error")
    ax.legend(loc="upper left")


def _key(roc, value):
    for k in roc["perNoise"]:
        if abs(float(k) - value) < 1e-12:
            return k
    raise KeyError(value)


def panelTrials(ax, trials):
    systems = []
    for trial in trials:
        if trial["system"] not in systems:
            systems.append(trial["system"])
    noiseSeed = sorted({(t["noiseLevel"], t["seed"]) for t in trials})
    grid = np.full((len(systems), len(noiseSeed)), np.nan)
    verdictCode = {True: 2.0, False: 0.0, None: 1.0}
    lookup = {(t["system"], (t["noiseLevel"], t["seed"])): t["ghost"] for t in trials}
    for r, system in enumerate(systems):
        for c, key in enumerate(noiseSeed):
            if (system, key) in lookup:
                grid[r, c] = verdictCode[lookup[(system, key)]]
    cmap = matplotlib.colors.ListedColormap([vizstyle.POSITIVE, vizstyle.UNDETERMINED, vizstyle.GHOST])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    labelledGhost = {t["system"]: t["label"] for t in trials}
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels([f"{s.replace('_', ' ')}  [{labelledGhost[s]}]" for s in systems], fontsize=8)
    ax.set_xticks(range(len(noiseSeed)))
    ax.set_xticklabels([f"{n * 100:g}%\ns{s}" for n, s in noiseSeed], fontsize=7)
    ax.set_title("Per-trial verdict  (green healthy, grey undetermined, red ghost)")
    ax.grid(False)
    for r in range(len(systems)):
        for c in range(len(noiseSeed)):
            ax.axhline(r - 0.5, color="white", lw=0.6)
            ax.axvline(c - 0.5, color="white", lw=0.6)


def main():
    vizstyle.applyStyle()
    payload = vizstyle.loadResult("ghost_detection.json")

    figure = plt.figure(figsize=(17, 11.5))
    grid = figure.add_gridspec(2, 2, hspace=0.42, wspace=0.24, height_ratios=[1.0, 1.05])
    panelSpectrum(figure.add_subplot(grid[0, 0]), payload["reference"])
    panelNoiseBoundary(figure.add_subplot(grid[0, 1]), payload["noiseBoundary"])
    panelRoc(figure.add_subplot(grid[1, 0]), payload["roc"])
    panelTrials(figure.add_subplot(grid[1, 1]), payload["roc"]["trials"])

    figure.suptitle(
        "Ostrogradski-ghost detection: decision boundary, noise robustness, and error rates",
        fontsize=16,
        fontweight="bold",
    )
    vizstyle.provenance(
        figure,
        "source: src/experiments/results/ghost_detection.json  "
        "(ghost_detection_validation.py; battery = 3 healthy + 4 ghost systems, 3 seeds each)",
    )
    figure.subplots_adjust(top=0.92, bottom=0.07)
    size = vizstyle.savefig(figure, OUTPUT)
    print(f"wrote {OUTPUT}  ({size} bytes)")


if __name__ == "__main__":
    main()
