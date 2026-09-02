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


def _orderFitResidual(derivativeColumns, lagrangianOrder, libraryMaxDegree=2):
    noStateVars = lagrangianOrder + 1
    library = buildHigherOrderLibrary(noStateVars, libraryMaxDegree)
    coordinate = sp.Function("q0")(TIME)
    matrix, _elExpressions = buildEulerLagrangeMatrix(
        library, coordinate, lagrangianOrder, noStateVars, derivativeColumns
    )
    keepMask = matrix.std(axis=0) > 1e-10
    keptLibrary = [monomial for monomial, keep in zip(library, keepMask) if keep]
    keptMatrix = matrix[:, keepMask]

    kineticMonomial = sp.expand(stateVariableSymbols(noStateVars)[lagrangianOrder] ** 2)
    if kineticMonomial not in keptLibrary:
        return 1.0, False
    kineticIndex = keptLibrary.index(kineticMonomial)

    kineticColumn = keptMatrix[:, kineticIndex]
    design = np.delete(keptMatrix, kineticIndex, axis=1)
    if design.shape[1] == 0:
        return 1.0, False
    coefficients, *_ = np.linalg.lstsq(design, -kineticColumn, rcond=None)
    residual = np.linalg.norm(design @ coefficients + kineticColumn)
    scaledResidual = float(residual / max(np.linalg.norm(kineticColumn), 1e-30))

    columnNorms = np.linalg.norm(keptMatrix, axis=0)
    normalized = keptMatrix[:, columnNorms > 0] / columnNorms[columnNorms > 0]
    degenerate = bool(np.linalg.matrix_rank(normalized) <= 1)
    return scaledResidual, degenerate


def inferLagrangianOrder(derivativeColumns, maxOrder=3, libraryMaxDegree=2, convergenceTolerance=None, stagnationTolerance=None):
    convergenceTolerance = (
        checkResidualToleranceFromGram.__defaults__[0]
        if convergenceTolerance is None
        else convergenceTolerance
    )
    stagnationTolerance = (
        checkCorrelationCutoff.__defaults__[0]
        if stagnationTolerance is None
        else stagnationTolerance
    )

    perOrder = []
    for order in range(1, maxOrder + 1):
        if len(derivativeColumns) < 2 * order + 1:
            break
        residual, degenerate = _orderFitResidual(derivativeColumns[: 2 * order + 1], order, libraryMaxDegree)
        converged = residual < convergenceTolerance
        perOrder.append({"order": order, "scaledResidual": residual, "converged": converged, "degenerate": degenerate})
        if converged:
            return order, perOrder

    if not perOrder:
        raise ValueError("not enough derivative levels to test even order 1")

    for i in range(1, len(perOrder)):
        earlier = perOrder[i - 1]["scaledResidual"]
        later = perOrder[i]["scaledResidual"]
        if earlier > 0 and (earlier - later) / earlier < stagnationTolerance:
            return perOrder[i - 1]["order"], perOrder
    return perOrder[-1]["order"], perOrder


def reduceOrderToPrior(derivativeColumns, lagrangianOrder, libraryMaxDegree=2):
    if lagrangianOrder < 2 or len(derivativeColumns) < 2 * lagrangianOrder + 1:
        return lagrangianOrder
    inferred, perOrder = inferLagrangianOrder(
        derivativeColumns, maxOrder=lagrangianOrder, libraryMaxDegree=libraryMaxDegree
    )
    inferredRecord = next((entry for entry in perOrder if entry["order"] == inferred), None)
    if inferredRecord is not None and inferredRecord["degenerate"]:
        return lagrangianOrder
    return min(inferred, lagrangianOrder)


def recoverHigherOrderLagrangian(
    derivativeColumns,
    noStateVars,
    lagrangianOrder,
    libraryMaxDegree=2,
    snapRelativeTolerance=0.05,
    kineticLevel=None,
    orderPrior=False,
):
    if orderPrior:
        inferred = reduceOrderToPrior(derivativeColumns, lagrangianOrder, libraryMaxDegree)
        if inferred < lagrangianOrder:
            noStateVars = noStateVars - (lagrangianOrder - inferred)
            derivativeColumns = derivativeColumns[: 2 * inferred + 1]
            if kineticLevel is not None:
                kineticLevel = min(kineticLevel, inferred)
            lagrangianOrder = inferred

    if kineticLevel is None:
        kineticLevel = min(2, lagrangianOrder)

    library = buildHigherOrderLibrary(noStateVars, libraryMaxDegree)
    coordinate = sp.Function("q0")(TIME)
    matrix, _elExpressions = buildEulerLagrangeMatrix(
        library, coordinate, lagrangianOrder, noStateVars, derivativeColumns
    )

    columnStd = matrix.std(axis=0)
    keepMask = columnStd > 1e-10
    keptLibrary = [monomial for monomial, keep in zip(library, keepMask) if keep]
    keptMatrix = matrix[:, keepMask]

    kineticMonomial = sp.expand(stateVariableSymbols(noStateVars)[kineticLevel] ** 2)
    if kineticMonomial not in keptLibrary:
        return kineticMonomial, []
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
