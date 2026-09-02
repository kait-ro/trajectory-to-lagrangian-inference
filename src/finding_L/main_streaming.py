from pathlib import Path

import sympy as sp
from finding_L.build_matrix import buildAdmissibleGram, buildGramMatrixChunked
from finding_L.candidates import buildCandidateLibrary, filterPureVelocityTerms
from finding_L.gram_forward_select import (
    checkResidualToleranceFromGram,
    fitActiveCoefficientsFromGram,
    pruneNearZeroCoefficients,
    residualNormSquaredFromGram,
    scoreReserveCandidatesFromGram,
)
from finding_L.regularized_select import lassoSelect
from finding_L.report import assembleDiscoveredLagrangian
from finding_L.selection_logger import toSelectionLogFrame
from finding_L.stopping_conditions import (
    checkCorrelationCutoff,
    checkDegreeExpansionNeeded,
    checkResidualStagnation,
)
from generation.eqnofmotion import defineCoordinates

DEFAULT_SELECTOR = "lasso"


def zeroVarianceMask(G, colSum, n):
    variance = G.diagonal() / n - (colSum / n) ** 2
    return variance > 1e-12


def admissibleReserve(candidateTerms, kineticTerm, G, colSum, n):
    kineticIndex = candidateTerms.index(kineticTerm)
    variance = zeroVarianceMask(G, colSum, n)
    reserveIndices = [index for index, ok in enumerate(variance) if ok and index != kineticIndex]
    return kineticIndex, reserveIndices


def runDiscoveryStreaming(
    csvPath: str,
    noCoords: int = 3,
    startingMaxDegree: int = 2,
    maxRounds: int = 20,
    chunkRows: int = 200_000,
    degreeCap: int = 6,
    residualRmsTolerance: float = 0.01,
    stagnationTolerance: float = 0.01,
    stagnationPatience: int = 3,
    correlationCutoff: float = 0.1,
    pruneRelativeThreshold: float = 1e-2,
    roundCallback=None,
    selector: str = DEFAULT_SELECTOR,
):
    t, coords, vels = defineCoordinates(noCoords)
    kineticTerm = sp.expand(sum(v ** 2 for v in vels))

    if selector == "lasso":
        return runLassoDiscoveryStreaming(
            csvPath,
            coords,
            vels,
            t,
            kineticTerm,
            noCoords,
            degreeCap,
            chunkRows,
            residualRmsTolerance,
            pruneRelativeThreshold,
            roundCallback,
        )
    if selector != "greedy":
        raise ValueError(f"unknown selector {selector!r}; expected 'greedy' or 'lasso'")

    currentMaxDegree = startingMaxDegree
    candidateTerms = filterPureVelocityTerms(buildCandidateLibrary(coords, vels, currentMaxDegree), coords)
    if kineticTerm not in candidateTerms:
        candidateTerms.append(kineticTerm)

    lambdifiedCache = {}
    print(f"Streaming Gram matrix over {len(candidateTerms)} candidate terms...")
    n, colSum, G = buildGramMatrixChunked(
        candidateTerms, coords, vels, t, csvPath, noCoords, chunkRows, lambdifiedCache=lambdifiedCache
    )

    kineticIndex, reserveIndices = admissibleReserve(candidateTerms, kineticTerm, G, colSum, n)
    targetNormSq = G[kineticIndex, kineticIndex]
    b = -G[:, kineticIndex]

    activeIndices = []
    selectionLog = []
    residualHistory = []

    for roundNumber in range(maxRounds):
        if not reserveIndices:
            print("Stopping: reserve exhausted.")
            break

        coefficients = fitActiveCoefficientsFromGram(G, b, activeIndices)
        residualNormSq = residualNormSquaredFromGram(targetNormSq, b, activeIndices, coefficients)

        converged, scaledResidual = checkResidualToleranceFromGram(residualNormSq, targetNormSq, residualRmsTolerance)

        if roundCallback is not None:
            roundCallback(
                {
                    "round": roundNumber,
                    "activeTerms": [candidateTerms[index] for index in activeIndices],
                    "coefficients": [float(value) for value in coefficients],
                    "kineticTerm": kineticTerm,
                    "scaledResidual": float(scaledResidual),
                    "converged": bool(converged),
                    "currentMaxDegree": currentMaxDegree,
                }
            )

        if converged:
            print(f"round {roundNumber}: scaledResidual={scaledResidual:.5f}")
            print("Stopping: residual converged (Condition A).")
            break

        residualHistory.append(scaledResidual)
        if checkResidualStagnation(residualHistory, stagnationTolerance, stagnationPatience):
            print(f"round {roundNumber}: scaledResidual={scaledResidual:.5f}")
            print("Stopping: residual no longer improving (Condition C).")
            break

        bestLocalIndex, bestScore, _ = scoreReserveCandidatesFromGram(
            G, b, activeIndices, reserveIndices, coefficients, residualNormSq
        )
        bestReserveIndex = reserveIndices[bestLocalIndex]
        selectionLog.append({"round": roundNumber, "bestReserveScore": bestScore, "scaledResidual": scaledResidual})
        stalled, _ = checkCorrelationCutoff(bestScore, correlationCutoff)

        print(
            f"round {roundNumber}: candidate {candidateTerms[bestReserveIndex]}, "
            f"score={bestScore:.4f}, scaledResidual={scaledResidual:.5f}"
        )

        if stalled:
            expandNeeded, newMaxDegree = checkDegreeExpansionNeeded(converged, stalled, currentMaxDegree, degreeCap)
            if not expandNeeded:
                print("Stopping: no reserve candidate meaningfully helps (Condition B).")
                break

            print(f"Expanding candidate library to max degree {newMaxDegree}. Re-streaming for new terms...")
            currentMaxDegree = newMaxDegree
            expandedTerms = filterPureVelocityTerms(buildCandidateLibrary(coords, vels, currentMaxDegree), coords)
            newTerms = [term for term in expandedTerms if term not in candidateTerms]
            if not newTerms:
                print("Stopping: degree expansion produced no new terms; cap reached or exhausted.")
                break

            candidateTerms.extend(newTerms)
            n, colSum, G = buildGramMatrixChunked(
                candidateTerms, coords, vels, t, csvPath, noCoords, chunkRows, lambdifiedCache=lambdifiedCache
            )
            kineticIndex, reserveCandidates = admissibleReserve(candidateTerms, kineticTerm, G, colSum, n)
            targetNormSq = G[kineticIndex, kineticIndex]
            b = -G[:, kineticIndex]
            reserveIndices = [index for index in reserveCandidates if index not in activeIndices]
            residualHistory = []
            continue

        activeIndices.append(bestReserveIndex)
        reserveIndices = [index for index in reserveIndices if index != bestReserveIndex]

    activeIndices, finalCoefficients = pruneNearZeroCoefficients(
        G, b, activeIndices, threshold=pruneRelativeThreshold
    )
    discoveredTerms = [(candidateTerms[index], coefficient) for index, coefficient in zip(activeIndices, finalCoefficients)]

    discovered = assembleDiscoveredLagrangian(kineticTerm, discoveredTerms, coords, vels)
    print()
    print(discovered.text)

    return discovered, toSelectionLogFrame(selectionLog)


def runLassoDiscoveryStreaming(
    csvPath,
    coords,
    vels,
    t,
    kineticTerm,
    noCoords,
    degreeCap,
    chunkRows,
    residualRmsTolerance,
    pruneRelativeThreshold,
    roundCallback=None,
):
    candidateTerms = filterPureVelocityTerms(buildCandidateLibrary(coords, vels, degreeCap), coords)
    if kineticTerm not in candidateTerms:
        candidateTerms.append(kineticTerm)

    print(
        f"Streaming Gram matrix over {len(candidateTerms)} candidate terms "
        f"(selector=lasso, degree {degreeCap})..."
    )
    G, terms, kineticIndex, _n = buildAdmissibleGram(
        candidateTerms, coords, vels, t, csvPath, noCoords, kineticTerm, chunkRows
    )
    b = -G[:, kineticIndex]
    targetNormSq = G[kineticIndex, kineticIndex]

    selectedIndices, _selectedCoefficients = lassoSelect(
        G, b, kineticIndex, relativeThreshold=pruneRelativeThreshold
    )
    activeIndices, finalCoefficients = pruneNearZeroCoefficients(
        G, b, list(selectedIndices), threshold=pruneRelativeThreshold
    )

    residualNormSq = residualNormSquaredFromGram(targetNormSq, b, activeIndices, finalCoefficients)
    converged, scaledResidual = checkResidualToleranceFromGram(
        residualNormSq, targetNormSq, residualRmsTolerance
    )

    discoveredTerms = [
        (terms[index], coefficient) for index, coefficient in zip(activeIndices, finalCoefficients)
    ]
    discovered = assembleDiscoveredLagrangian(kineticTerm, discoveredTerms, coords, vels)

    if roundCallback is not None:
        roundCallback(
            {
                "round": 0,
                "activeTerms": [terms[index] for index in activeIndices],
                "coefficients": [float(value) for value in finalCoefficients],
                "kineticTerm": kineticTerm,
                "scaledResidual": float(scaledResidual),
                "converged": bool(converged),
                "currentMaxDegree": degreeCap,
            }
        )

    print(f"selector=lasso: {len(activeIndices)} terms, scaledResidual={scaledResidual:.5f}")
    print()
    print(discovered.text)

    selectionLog = [
        {
            "round": 0,
            "selector": "lasso",
            "scaledResidual": float(scaledResidual),
            "converged": bool(converged),
        }
    ]
    return discovered, toSelectionLogFrame(selectionLog)


if __name__ == "__main__":
    repoRoot = Path(__file__).resolve().parents[2]
    csvPath = str(repoRoot / "assets/anharmonic_chain_blind_n6_noise0.csv")
    runDiscoveryStreaming(csvPath, noCoords=6, maxRounds=80, chunkRows=120_000, degreeCap=4)
