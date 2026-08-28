import sympy as sp
from finding_L.gram_forward_select import (
    checkResidualToleranceFromGram,
    fitActiveCoefficientsFromGram,
    pruneNearZeroCoefficients,
    residualNormSquaredFromGram,
    scoreReserveCandidatesFromGram,
)
from finding_L.higher_order_candidates import (
    buildEulerLagrangeMatrix,
    buildHigherOrderLibrary,
    dropKineticAliasColumns,
    stateVariableSymbols,
)
from finding_L.report import snapCoefficient
from finding_L.stopping_conditions import checkCorrelationCutoff
from generation.ostrogradski import TIME


def stateToCoordinate(stateExpression, noStateVars, coordinate):
    substitution = {
        sp.Symbol(f"s{index}"): (coordinate if index == 0 else sp.diff(coordinate, TIME, index))
        for index in range(noStateVars)
    }
    return sp.expand(sp.sympify(stateExpression).subs(substitution))


def forwardSelectFromGram(gramMatrix, kineticIndex, maxRounds=25):
    b = -gramMatrix[:, kineticIndex]
    targetNormSq = gramMatrix[kineticIndex, kineticIndex]
    activeIndices = []
    reserveIndices = [index for index in range(gramMatrix.shape[0]) if index != kineticIndex]

    for _ in range(maxRounds):
        if not reserveIndices:
            break
        coefficients = fitActiveCoefficientsFromGram(gramMatrix, b, activeIndices)
        residualNormSq = residualNormSquaredFromGram(targetNormSq, b, activeIndices, coefficients)
        converged, _scaled = checkResidualToleranceFromGram(residualNormSq, targetNormSq)
        if converged:
            break
        bestLocal, bestScore, _scores = scoreReserveCandidatesFromGram(
            gramMatrix, b, activeIndices, reserveIndices, coefficients, residualNormSq
        )
        stalled, _magnitude = checkCorrelationCutoff(bestScore)
        if stalled:
            break
        chosen = reserveIndices[bestLocal]
        activeIndices.append(chosen)
        reserveIndices = [index for index in reserveIndices if index != chosen]

    return pruneNearZeroCoefficients(gramMatrix, b, activeIndices)


def recoverHigherOrderLagrangian(
    derivativeColumns,
    noStateVars,
    lagrangianOrder,
    libraryMaxDegree=2,
    snapRelativeTolerance=0.05,
):
    library = buildHigherOrderLibrary(noStateVars, libraryMaxDegree)
    coordinate = sp.Function("q0")(TIME)
    matrix, _elExpressions = buildEulerLagrangeMatrix(
        library, coordinate, lagrangianOrder, noStateVars, derivativeColumns
    )

    columnStd = matrix.std(axis=0)
    keepMask = columnStd > 1e-10
    keptLibrary = [monomial for monomial, keep in zip(library, keepMask) if keep]
    keptMatrix = matrix[:, keepMask]

    kineticMonomial = sp.expand(stateVariableSymbols(noStateVars)[2] ** 2)
    kineticIndex = keptLibrary.index(kineticMonomial)
    keptMatrix, keptLibrary, kineticIndex = dropKineticAliasColumns(keptMatrix, keptLibrary, kineticIndex)

    gramMatrix = keptMatrix.T @ keptMatrix
    activeIndices, coefficients = forwardSelectFromGram(gramMatrix, kineticIndex)

    expression = kineticMonomial
    selected = []
    for index, coefficient in zip(activeIndices, coefficients):
        selected.append((keptLibrary[index], float(coefficient)))
        expression = expression + snapCoefficient(float(coefficient), relativeTolerance=snapRelativeTolerance) * keptLibrary[index]

    return sp.expand(expression), selected
