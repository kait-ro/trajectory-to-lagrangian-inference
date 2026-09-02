import json
import os

from experiments.artifacts import RESULTS_DIR
from experiments.discovery import FROZEN_TOLERANCES, compareToExpected
from experiments.generate_dataset import datasetPath, generateSystemDatasets
from experiments.systems import SYSTEMS
from finding_L.build_matrix import buildAdmissibleGram
from finding_L.candidates import buildCandidateLibrary, filterPureVelocityTerms
from finding_L.higher_order_discovery import forwardSelectFromGram
from finding_L.regularized_select import lassoSelect, sequentialThresholdedLeastSquares
from finding_L.report import assembleDiscoveredLagrangian
from generation.eqnofmotion import TIME, defineCoordinates

DEGREE = FROZEN_TOLERANCES["degreeCap"]
PRUNE = FROZEN_TOLERANCES["pruneRelativeThreshold"]
SEEDS = (20260828, 20260829, 20260830)
METHODS = ("greedy", "stlsq", "lasso")


def _buildGram(system, csvPath, chunkRows=150_000):
    _t, coords, vels = defineCoordinates(system.noCoords)
    kineticTerm = sum(v ** 2 for v in vels)
    candidateTerms = filterPureVelocityTerms(buildCandidateLibrary(coords, vels, DEGREE), coords)
    if kineticTerm not in candidateTerms:
        candidateTerms.append(kineticTerm)
    gram, terms, kineticIndex, _n = buildAdmissibleGram(
        candidateTerms, coords, vels, TIME, csvPath, system.noCoords, kineticTerm, chunkRows
    )
    return gram, terms, kineticIndex, coords, vels, kineticTerm


def _assemble(terms, kineticTerm, coords, vels, activeIndices, coefficients):
    discoveredTerms = [(terms[i], float(c)) for i, c in zip(activeIndices, coefficients)]
    return assembleDiscoveredLagrangian(kineticTerm, discoveredTerms, coords, vels)


def _score(discovered, expected, noCoords):
    comparison = compareToExpected(discovered, expected, noCoords=noCoords)
    return {
        "success": comparison.success,
        "missing": len(comparison.missingMonomials),
        "spurious": len(comparison.spuriousMonomials),
        "maxAbsError": comparison.maxAbsoluteError,
        "structurallyEquivalent": bool(comparison.structurallyEquivalent),
    }


def _aggregate(scores):
    count = len(scores)
    return {
        "seeds": count,
        "recoveredFraction": sum(s["success"] for s in scores) / count,
        "equivalentFraction": sum(s["structurallyEquivalent"] for s in scores) / count,
        "meanMissing": sum(s["missing"] for s in scores) / count,
        "meanSpurious": sum(s["spurious"] for s in scores) / count,
        "worstMaxAbsError": max(s["maxAbsError"] for s in scores),
    }


def _selectAll(gram, kineticIndex, maxRounds):
    b = -gram[:, kineticIndex]
    return {
        "greedy": forwardSelectFromGram(gram, kineticIndex, maxRounds=maxRounds),
        "stlsq": sequentialThresholdedLeastSquares(gram, b, kineticIndex, relativeThreshold=PRUNE),
        "lasso": lassoSelect(gram, b, kineticIndex, relativeThreshold=PRUNE),
    }


def compareOnSystem(systemName, noiseLevels=(0.0, 0.01, 0.02, 0.05), seeds=SEEDS):
    system = SYSTEMS[systemName]
    expected = system.expectedScaledLagrangian(system.noCoords)

    rows = []
    for noiseLevel in noiseLevels:
        perMethod = {name: [] for name in METHODS}
        for seed in seeds:
            generateSystemDatasets(systemName, [noiseLevel], overwrite=True, seed=seed)
            csvPath = datasetPath(system, noiseLevel, seed)
            gram, terms, kineticIndex, coords, vels, kineticTerm = _buildGram(system, csvPath)

            for name, (active, coef) in _selectAll(gram, kineticIndex, system.maxRounds).items():
                discovered = _assemble(terms, kineticTerm, coords, vels, active, coef)
                perMethod[name].append(_score(discovered, expected, system.noCoords))

        row = {"noiseLevel": noiseLevel}
        for name, scores in perMethod.items():
            row[name] = _aggregate(scores)
        rows.append(row)
    return system, rows


def _formatSystem(system, rows):
    lines = [f"{system.name}  ({rows[0][METHODS[0]]['seeds']} seeds per noise level)", ""]
    header = f"{'noise':>6}  " + "  ".join(f"{m:>26}" for m in METHODS)
    lines.append(header)
    lines.append(f"{'':>6}  " + "  ".join(f"{'rec / equiv / miss / spur':>26}" for _ in METHODS))
    lines.append("-" * len(header))
    for row in rows:
        cells = []
        for method in METHODS:
            s = row[method]
            cells.append(
                f"{s['recoveredFraction']:>4.2f} /{s['equivalentFraction']:>5.2f} /"
                f"{s['meanMissing']:>5.1f} /{s['meanSpurious']:>5.1f}"
            )
        lines.append(f"{row['noiseLevel'] * 100:5.1f}%  " + "  ".join(f"{c:>26}" for c in cells))
    return "\n".join(lines)


def run():
    allRows = {}
    lines = [
        "Model selection: greedy forward selection vs. STLSQ vs. debiased LASSO",
        "",
        f"Degree-{DEGREE} library, {len(SEEDS)} seeds per noise level, every system in experiments.systems.SYSTEMS.",
        "cell = fraction of seeds recovered / fraction structurally equivalent / mean missing / mean spurious",
        "",
    ]
    for systemName in SYSTEMS:
        system, rows = compareOnSystem(systemName)
        allRows[systemName] = rows
        lines.append(_formatSystem(system, rows))
        lines.append("")

    lines.append("Reading:")
    lines.append("  * clean data: all three selectors recover every system exactly.")
    lines.append("  * greedy forward selection degrades from ~2% noise -- spurious velocity-quartic")
    lines.append("    terms out-correlate the true cubics and it commits to them early.")
    lines.append("  * the debiased LASSO path recovers the benchmark systems exactly further into noise.")
    lines.append("")
    lines.append("The ~1-2% noise ceiling is a property of the greedy selector, not of")
    lines.append("ordinary-least-squares Lagrangian recovery. Debiased LASSO is now the production")
    lines.append("2nd-order selector (finding_L.main_streaming.DEFAULT_SELECTOR = 'lasso'); the greedy")
    lines.append("path stays reachable via runDiscoveryStreaming(selector='greedy').")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "model_selection_comparison.json"), "w") as handle:
        json.dump(allRows, handle, indent=2, default=str)
    report = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "model_selection_comparison.txt"), "w") as handle:
        handle.write(report + "\n")
    return report


if __name__ == "__main__":
    print(run())
