import warnings

import numpy as np


def fitActiveCoefficientsFromGram(G: np.ndarray, b: np.ndarray, activeIndices: list) -> np.ndarray:
    if not activeIndices:
        return np.array([])
    subG = G[np.ix_(activeIndices, activeIndices)]
    subB = b[activeIndices]
    try:
        coefficients = np.linalg.solve(subG, subB)
    except np.linalg.LinAlgError:
        # The active Gram block is singular: two or more selected candidate EL
        # columns are (numerically) collinear -- typically a velocity alias that
        # slipped past dropKineticAliasColumns' cosine tolerance. lstsq still
        # returns a minimum-norm fit, but the coefficient split between the
        # collinear terms is then arbitrary, so make the event visible.
        warnings.warn(
            f"fitActiveCoefficientsFromGram: singular active Gram block "
            f"(active set size {len(activeIndices)}); falling back to lstsq. "
            f"Coefficients on collinear terms are not individually identifiable.",
            RuntimeWarning,
            stacklevel=2,
        )
        coefficients, _, _, _ = np.linalg.lstsq(subG, subB, rcond=None)
    return coefficients


def residualNormSquaredFromGram(targetNormSq: float, b: np.ndarray, activeIndices: list, coefficients: np.ndarray) -> float:
    if not activeIndices:
        return targetNormSq
    return float(targetNormSq - coefficients @ b[activeIndices])


def scoreReserveCandidatesFromGram(
    G: np.ndarray,
    b: np.ndarray,
    activeIndices: list,
    reserveIndices: list,
    coefficients: np.ndarray,
    residualNormSq: float,
):
    scores = np.empty(len(reserveIndices))
    residualNorm = np.sqrt(max(residualNormSq, 1e-30))

    for localIndex, j in enumerate(reserveIndices):
        thetaJ_dot_residual = b[j]
        if activeIndices:
            thetaJ_dot_residual = thetaJ_dot_residual - coefficients @ G[j, activeIndices]

        thetaJNorm = np.sqrt(max(G[j, j], 1e-30))
        scores[localIndex] = thetaJ_dot_residual / (thetaJNorm * residualNorm)

    bestLocalIndex = int(np.argmax(np.abs(scores)))
    bestScore = scores[bestLocalIndex]
    return bestLocalIndex, bestScore, scores


def pruneNearZeroCoefficients(
    G: np.ndarray,
    b: np.ndarray,
    activeIndices: list,
    threshold: float = 1e-2,
    relative: bool = True,
):
    activeIndices = list(activeIndices)

    while activeIndices:
        coefficients = fitActiveCoefficientsFromGram(G, b, activeIndices)
        if coefficients.size == 0:
            break

        maxAbs = np.max(np.abs(coefficients))
        cutoff = threshold * maxAbs if relative else threshold

        keepMask = np.abs(coefficients) >= cutoff
        if keepMask.all():
            return activeIndices, coefficients

        activeIndices = [idx for idx, keep in zip(activeIndices, keepMask) if keep]

    coefficients = fitActiveCoefficientsFromGram(G, b, activeIndices)
    return activeIndices, coefficients


def checkResidualToleranceFromGram(residualNormSq: float, targetNormSq: float, tolerance: float = 0.01):
    scaledResidual = np.sqrt(max(residualNormSq, 0.0) / max(targetNormSq, 1e-30))
    hasConverged = scaledResidual < tolerance
    return hasConverged, scaledResidual