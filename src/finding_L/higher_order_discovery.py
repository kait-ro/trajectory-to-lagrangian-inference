import numpy as np
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
    buildMultiFieldElMatrix,
    dropKineticAliasColumns,
    multiFieldLibrary,
    stateGridSymbols,
    stateVariableSymbols,
)
from finding_L.report import snapCoefficient
from finding_L.stopping_conditions import checkCorrelationCutoff
from generation.eqnofmotion import defineCoordinates
from generation.ostrogradski import TIME


def stateToCoordinate(stateExpression, noStateVars, coordinate):
    substitution = {
        sp.Symbol(f"s{index}"): (coordinate if index == 0 else sp.diff(coordinate, TIME, index))
        for index in range(noStateVars)
    }
    return sp.expand(sp.sympify(stateExpression).subs(substitution))


def multiFieldStateToCoordinates(stateExpression, noFields, lagrangianOrder, coords=None):
    """s{i}_{k} -> (k-th time derivative of coords[i])."""
    if coords is None:
        _t, coords, _v = defineCoordinates(noFields)
    grid = stateGridSymbols(noFields, lagrangianOrder)
    substitution = {}
    for field in range(noFields):
        for level in range(lagrangianOrder + 1):
            substitution[grid[field][level]] = (
                coords[field] if level == 0 else sp.diff(coords[field], TIME, level)
            )
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


def recoverMultiFieldHigherOrderLagrangian(
    derivativeData,
    noFields,
    lagrangianOrder,
    libraryMaxDegree=2,
    kineticLevel=None,
    snapRelativeTolerance=0.05,
):
    """Multi-coordinate analogue of recoverHigherOrderLagrangian.

    `derivativeData` is a list over derivative levels 0..2*lagrangianOrder, each an
    array of shape (rows, noFields). The isotropic kinetic term
    sum_i (d^k q_i / dt^k)^2 (k = kineticLevel, default lagrangianOrder) is fixed
    to coefficient 1; forward selection recovers everything else, including
    cross-field coupling monomials.
    """
    kineticLevel = lagrangianOrder if kineticLevel is None else kineticLevel
    grid = stateGridSymbols(noFields, lagrangianOrder)
    library = multiFieldLibrary(noFields, lagrangianOrder, libraryMaxDegree)
    matrix, _elExpressions = buildMultiFieldElMatrix(library, noFields, lagrangianOrder, derivativeData)

    kineticMonomials = [sp.expand(grid[i][kineticLevel] ** 2) for i in range(noFields)]
    kineticIndices = [library.index(monomial) for monomial in kineticMonomials]
    kineticColumn = matrix[:, kineticIndices].sum(axis=1)

    keep = [
        i for i in range(len(library))
        if i not in kineticIndices and matrix[:, i].std() > 1e-10
    ]
    designLibrary = [library[i] for i in keep]
    design = matrix[:, keep]

    augmented = np.column_stack([design, kineticColumn])
    augmentedLibrary = designLibrary + [sp.expand(sum(kineticMonomials))]
    augmentedKineticIndex = design.shape[1]
    augmented, augmentedLibrary, augmentedKineticIndex = dropKineticAliasColumns(
        augmented, augmentedLibrary, augmentedKineticIndex
    )

    gramMatrix = augmented.T @ augmented
    activeIndices, coefficients = forwardSelectFromGram(gramMatrix, augmentedKineticIndex)

    expression = sp.expand(sum(kineticMonomials))
    selected = []
    for index, coefficient in zip(activeIndices, coefficients):
        selected.append((augmentedLibrary[index], float(coefficient)))
        expression += snapCoefficient(float(coefficient), relativeTolerance=snapRelativeTolerance) * augmentedLibrary[index]

    return sp.expand(expression), selected
