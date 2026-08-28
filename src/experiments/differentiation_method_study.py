import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from generation.numerical_diff import (
    finiteDifferenceDerivatives,
    relativeL2Error,
    savitzkyGolayDerivatives,
    smoothingSplineDerivatives,
)

from experiments.pu_system import groundTruthTrajectory

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MAX_ORDER = 4
NOISE_LEVELS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]
BREAKDOWN_THRESHOLD = 0.5

METHODS = {
    "finite_difference": lambda signal, dt: finiteDifferenceDerivatives(signal, dt, MAX_ORDER),
    "savitzky_golay": lambda signal, dt: savitzkyGolayDerivatives(signal, dt, MAX_ORDER),
    "smoothing_spline": lambda signal, dt: smoothingSplineDerivatives(signal, dt, MAX_ORDER),
}


def runStudy():
    dt, cleanPosition, groundTruth = groundTruthTrajectory()
    positionStd = cleanPosition.std()
    rng = np.random.default_rng(12345)

    records = []
    for noiseLevel in NOISE_LEVELS:
        noisyPosition = cleanPosition + rng.normal(0.0, noiseLevel * positionStd, cleanPosition.shape)
        for methodName, method in METHODS.items():
            estimated = method(noisyPosition, dt)
            for order in range(1, MAX_ORDER + 1):
                error = relativeL2Error(estimated[order], groundTruth[order])
                records.append(
                    {
                        "noiseLevel": noiseLevel,
                        "method": methodName,
                        "order": order,
                        "relativeL2Error": float(error),
                    }
                )
    return records


def breakdownTable(records):
    lines = []
    for methodName in METHODS:
        for order in range(1, MAX_ORDER + 1):
            series = sorted(
                (record for record in records if record["method"] == methodName and record["order"] == order),
                key=lambda record: record["noiseLevel"],
            )
            breakdown = None
            for record in series:
                if record["relativeL2Error"] > BREAKDOWN_THRESHOLD:
                    breakdown = record["noiseLevel"]
                    break
            label = "never" if breakdown is None else f"{breakdown*100:g}%"
            lines.append(f"  {methodName:>17}  d^{order}: breakdown at noise >= {label}")
    return "\n".join(lines)


def writeArtifacts(records):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "differentiation_study.json"), "w") as handle:
        json.dump({"noiseLevels": NOISE_LEVELS, "records": records}, handle, indent=2)

    header = f"{'noise':>7} " + " ".join(f"{m[:10]:>11}" for m in METHODS)
    lines = ["Relative L2 error of estimated derivatives (Pais-Uhlenbeck q(t))", ""]
    for order in range(1, MAX_ORDER + 1):
        lines.append(f"-- derivative order {order} --")
        lines.append(header)
        for noiseLevel in NOISE_LEVELS:
            cells = []
            for methodName in METHODS:
                value = next(
                    record["relativeL2Error"]
                    for record in records
                    if record["method"] == methodName and record["order"] == order and record["noiseLevel"] == noiseLevel
                )
                cells.append(f"{value:>11.3e}")
            lines.append(f"{noiseLevel*100:6.1f}% " + " ".join(cells))
        lines.append("")
    lines.append("Breakdown thresholds (relative L2 error > 0.5):")
    lines.append(breakdownTable(records))
    tableText = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "differentiation_study.txt"), "w") as handle:
        handle.write(tableText + "\n")

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for order, axis in zip(range(1, MAX_ORDER + 1), axes.flat):
        for methodName in METHODS:
            series = sorted(
                (record for record in records if record["method"] == methodName and record["order"] == order),
                key=lambda record: record["noiseLevel"],
            )
            noiseAxis = [max(record["noiseLevel"], 1e-4) * 100 for record in series]
            errorAxis = [record["relativeL2Error"] for record in series]
            axis.loglog(noiseAxis, errorAxis, "o-", label=methodName)
        axis.axhline(BREAKDOWN_THRESHOLD, color="grey", linestyle="--", linewidth=1)
        axis.set_title(f"derivative order {order}")
        axis.set_xlabel("position noise (%), 0 shown at 0.01%")
        axis.set_ylabel("relative L2 error")
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=8)
    figure.suptitle("Noisy higher-order differentiation: finite difference vs Savitzky-Golay vs smoothing spline")
    figure.tight_layout()
    figurePath = os.path.join(RESULTS_DIR, "differentiation_study.png")
    figure.savefig(figurePath, dpi=130)
    plt.close(figure)

    return tableText, figurePath


if __name__ == "__main__":
    records = runStudy()
    tableText, figurePath = writeArtifacts(records)
    print(tableText)
    print()
    print(f"figure: {figurePath}")
