"""Regularisation-path model selection from a Gram matrix -- an additive
alternative to the greedy forward selection in `gram_forward_select.py`.

Both routines here solve the same problem the greedy path does: find sparse `c`
with  theta_kinetic + sum_j c_j theta_j ~= 0  , given only  G = Theta^T Theta  and
b = -G[:, kinetic]  (so `Theta` is never materialised, same as the streaming path).

Nothing in this module is imported by the existing discovery pipeline; it exists
to be compared against it (see experiments/model_selection_comparison.py).
"""

import numpy as np


def _lstsqOnActive(gram, b, active):
    if not active:
        return np.array([])
    return np.linalg.lstsq(gram[np.ix_(active, active)], b[active], rcond=None)[0]


def sequentialThresholdedLeastSquares(gram, b, kineticIndex, relativeThreshold=1e-2, maxIterations=12):
    """SINDy-style STLSQ. Start from the full ordinary least-squares fit, zero the
    coefficients below `relativeThreshold * max|c|`, refit on the survivors, repeat
    to a fixed point. Returns (activeIndices, coefficients)."""
    candidates = [index for index in range(gram.shape[0]) if index != kineticIndex]
    active = list(candidates)

    for _ in range(maxIterations):
        coefficients = _lstsqOnActive(gram, b, active)
        if coefficients.size == 0:
            break
        cutoff = relativeThreshold * np.max(np.abs(coefficients))
        survivors = [index for index, value in zip(active, coefficients) if abs(value) >= cutoff]
        if survivors == active:
            return active, coefficients
        active = survivors

    return active, _lstsqOnActive(gram, b, active)


def lassoPathFromGram(gram, b, kineticIndex, nLambdas=40, lambdaRatio=1e-3, maxIterations=400, tolerance=1e-8):
    """Coordinate-descent LASSO regularisation path from the Gram matrix.

    Minimises  1/2 c^T G c - b^T c + lambda ||c||_1  over the non-kinetic columns,
    for a geometric sequence of lambda from lambda_max (all-zero solution) down to
    lambdaRatio * lambda_max. Returns (lambdas, coefficientPaths) where
    coefficientPaths[k] is the coefficient vector (length n-1) at lambdas[k].
    """
    candidates = [index for index in range(gram.shape[0]) if index != kineticIndex]
    subGram = gram[np.ix_(candidates, candidates)]
    subB = b[candidates]
    diagonal = np.diag(subGram).copy()
    diagonal[diagonal < 1e-30] = 1e-30

    lambdaMax = np.max(np.abs(subB))
    lambdas = lambdaMax * lambdaRatio ** (np.arange(nLambdas) / max(nLambdas - 1, 1))

    coefficients = np.zeros(len(candidates))
    paths = []
    for penalty in lambdas:
        for _ in range(maxIterations):
            maxChange = 0.0
            for j in range(len(candidates)):
                residualCorrelation = subB[j] - subGram[j] @ coefficients + subGram[j, j] * coefficients[j]
                updated = np.sign(residualCorrelation) * max(abs(residualCorrelation) - penalty, 0.0) / diagonal[j]
                maxChange = max(maxChange, abs(updated - coefficients[j]))
                coefficients[j] = updated
            if maxChange < tolerance:
                break
        paths.append(coefficients.copy())
    return lambdas, np.array(paths), candidates


def lassoSelect(gram, b, kineticIndex, relativeThreshold=1e-2, **pathKwargs):
    """Pick the sparsest LASSO-path solution whose least-squares refit residual is
    within 5% of the densest solution's, then hard-threshold and refit (debiased
    LASSO). Returns (activeIndices, coefficients)."""
    lambdas, paths, candidates = lassoPathFromGram(gram, b, kineticIndex, **pathKwargs)

    targetNormSq = gram[kineticIndex, kineticIndex]

    def refitResidual(active):
        coefficients = _lstsqOnActive(gram, b, active)
        if coefficients.size == 0:
            return targetNormSq
        return float(targetNormSq - coefficients @ b[active])

    densestActive = [candidates[j] for j in np.nonzero(np.abs(paths[-1]) > 0)[0]]
    bestResidual = refitResidual(densestActive)

    chosen = densestActive
    for row in paths:
        active = [candidates[j] for j, value in enumerate(row) if value != 0.0]
        if not active:
            continue
        if refitResidual(active) <= bestResidual * 1.05 + 1e-12:
            chosen = active
            break

    coefficients = _lstsqOnActive(gram, b, chosen)
    if coefficients.size:
        cutoff = relativeThreshold * np.max(np.abs(coefficients))
        keep = [index for index, value in zip(chosen, coefficients) if abs(value) >= cutoff]
        if keep and keep != chosen:
            chosen = keep
            coefficients = _lstsqOnActive(gram, b, chosen)
    return chosen, coefficients
