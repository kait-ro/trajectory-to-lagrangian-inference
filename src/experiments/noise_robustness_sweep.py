import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.discovery import (
    CALIBRATION_SYSTEM,
    LOCKED_TOLERANCES,
    compareToExpected,
    lockedTolerancesReport,
    runSystemDiscovery,
)
from experiments.generate_dataset import datasetPath, generateSystemDatasets
from experiments.systems import SYSTEMS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _finalScaledResidual(logFrame):
    if logFrame is None or logFrame.empty or "scaledResidual" not in logFrame:
        return float("nan")
    return float(logFrame["scaledResidual"].iloc[-1])


def runNoiseCurve(systemName, noiseLevels=None, chunkRows=150_000):
    system = SYSTEMS[systemName]
    levels = list(system.noiseLevels if noiseLevels is None else noiseLevels)
    isHoldout = system.name != CALIBRATION_SYSTEM

    generateSystemDatasets(systemName, levels)
    expected = system.expectedScaledLagrangian(system.noCoords)

    rows = []
    for noiseLevel in levels:
        csvPath = datasetPath(system, noiseLevel)
        # enforceLocked=True (default): the holdout runs on the tolerances frozen
        # from CALIBRATION_SYSTEM, and raises if this system overrode any of them.
        discovered, logFrame = runSystemDiscovery(system, csvPath, chunkRows=chunkRows)
        comparison = compareToExpected(discovered, expected, noCoords=system.noCoords)
        verdict = comparison.equivalenceVerdict
        rows.append(
            {
                "noiseLevel": noiseLevel,
                "success": comparison.success,
                "structurallyEquivalent": bool(comparison.structurallyEquivalent),
                "equivalenceDetail": verdict.detail if verdict is not None else "",
                "missingCount": len(comparison.missingMonomials),
                "spuriousCount": len(comparison.spuriousMonomials),
                "maxAbsoluteCoefficientError": comparison.maxAbsoluteError,
                "finalScaledResidual": _finalScaledResidual(logFrame),
                "missing": [str(monomial) for monomial in comparison.missingMonomials],
                "spurious": [str(monomial) for monomial in comparison.spuriousMonomials],
                "discoveredText": discovered.text,
            }
        )
    return system, rows, isHoldout


def _classify(row):
    if row["success"]:
        return "recovered"
    if row["missingCount"] == 0 and row["spuriousCount"] <= 2:
        return "degraded"
    return "failed"


def writeArtifacts(system, rows, isHoldout):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stem = os.path.join(RESULTS_DIR, f"noise_curve_{system.name}")

    roleLine = (
        "ROLE: BLIND HOLDOUT -- tolerances frozen from the calibration system, no retuning"
        if isHoldout
        else "ROLE: CALIBRATION SYSTEM -- tolerances were fit here, then locked"
    )

    with open(f"{stem}.json", "w") as handle:
        json.dump(
            {
                "system": system.name,
                "role": "holdout" if isHoldout else "calibration",
                "lockedTolerances": LOCKED_TOLERANCES,
                "rows": rows,
            },
            handle,
            indent=2,
        )

    header = (
        f"{'noise':>7} {'outcome':>10} {'missing':>8} {'spurious':>9} "
        f"{'max|dcoef|':>11} {'finalResid':>11} {'equiv?':>8}"
    )
    lines = [
        system.name,
        system.description,
        "",
        roleLine,
        lockedTolerancesReport(),
        "",
        "equiv? = is (discovered - expected) a genuine null Lagrangian (EL operator identically zero)",
        "",
        header,
        "-" * len(header),
    ]
    for row in rows:
        if not row["structurallyEquivalent"]:
            equivCell = "no"
        elif "identical" in row["equivalenceDetail"]:
            equivCell = "exact"
        else:
            equivCell = "null-L"
        lines.append(
            f"{row['noiseLevel']*100:6.0f}% {_classify(row):>10} {row['missingCount']:>8} "
            f"{row['spuriousCount']:>9} {row['maxAbsoluteCoefficientError']:>11.4f} "
            f"{row['finalScaledResidual']:>11.4f} {equivCell:>8}"
        )
    tableText = "\n".join(lines)
    with open(f"{stem}.txt", "w") as handle:
        handle.write(tableText + "\n")

    noiseAxis = [row["noiseLevel"] * 100 for row in rows]
    colours = {"recovered": "#2e7d32", "degraded": "#f9a825", "failed": "#c62828"}

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(noiseAxis, [row["maxAbsoluteCoefficientError"] for row in rows], "o-", color="#1565c0")
    axes[0].set_xlabel("noise level (%)")
    axes[0].set_ylabel("max |coefficient error| on recovered terms")
    axes[0].set_title("Coefficient accuracy vs noise")
    axes[0].grid(alpha=0.3)

    axes[1].bar(
        [x - 1.2 for x in noiseAxis],
        [row["missingCount"] for row in rows],
        width=2.4,
        label="missing terms",
        color="#6a1b9a",
    )
    axes[1].bar(
        [x + 1.2 for x in noiseAxis],
        [row["spuriousCount"] for row in rows],
        width=2.4,
        label="spurious terms",
        color="#ef6c00",
    )
    for x, row in zip(noiseAxis, rows):
        axes[1].scatter([x], [-0.6], marker="s", s=90, color=colours[_classify(row)])
    axes[1].set_xlabel("noise level (%)")
    axes[1].set_ylabel("term count")
    axes[1].set_title("Structural recovery vs noise")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.suptitle(f"Noise-robustness curve: {system.name}")
    figure.tight_layout()
    figure.savefig(f"{stem}.png", dpi=130)
    plt.close(figure)

    return tableText, f"{stem}.png"


def _parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("system", nargs="?", default="isotropic_quartic_calibration", choices=sorted(SYSTEMS))
    parser.add_argument("--noise", type=float, nargs="*", default=None)
    parser.add_argument("--chunk-rows", type=int, default=150_000)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parseArgs()
    system, rows, isHoldout = runNoiseCurve(arguments.system, arguments.noise, arguments.chunk_rows)
    tableText, figurePath = writeArtifacts(system, rows, isHoldout)
    print(tableText)
    print()
    print(f"figure: {figurePath}")
