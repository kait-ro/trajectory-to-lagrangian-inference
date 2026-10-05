import numpy as np
import sympy as sp

from generation.eqnofmotion import TIME


def highestTimeDerivativeOrder(expression, coords):
    coordSet = set(coords)
    order = 1
    for derivative in sp.sympify(expression).atoms(sp.Derivative):
        if derivative.expr not in coordSet:
            continue
        for variable, count in derivative.variable_count:
            if variable == TIME:
                order = max(order, int(count))
    return order


def lagrangianOrder(lagrangian, coords):
    return highestTimeDerivativeOrder(lagrangian, coords)


def timeDerivative(coordinate, k):
    return coordinate if k == 0 else sp.diff(coordinate, TIME, k)


def isNullLagrangianLocal(deltaL, coords, order):
    deltaL = sp.expand(sp.sympify(deltaL))
    if deltaL == 0:
        return True
    for coordinate in coords:
        residual = sp.expand(eulerLagrangeExpression(deltaL, coordinate, order, pipelineSign=True))
        if residual != 0 and sp.simplify(residual) != 0:
            return False
    return True


def _reduceSameFieldBilinearTerm(term, levels):
    factors = sp.Mul.make_args(term)
    levelDegrees = {}
    otherFactors = []
    for factor in factors:
        base, exponent = factor.as_base_exp()
        matchedIndex = None
        for index, level in enumerate(levels):
            if base == level:
                matchedIndex = index
                break
        if matchedIndex is None:
            otherFactors.append(factor)
        else:
            levelDegrees[matchedIndex] = levelDegrees.get(matchedIndex, 0) + exponent

    involved = [(index, degree) for index, degree in levelDegrees.items() if degree != 0]
    if len(involved) != 2 or involved[0][1] != 1 or involved[1][1] != 1:
        return term

    indexA, indexB = involved[0][0], involved[1][0]
    rest = sp.Mul(*otherFactors) if otherFactors else sp.Integer(1)
    gap = abs(indexA - indexB)
    if gap % 2 == 1:
        return sp.Integer(0)

    steps = gap // 2
    midIndex = (indexA + indexB) // 2
    sign = sp.Integer(-1) ** steps
    return sp.expand(rest * sign * levels[midIndex] ** 2)


def _canonicalizeForCoordinate(lagrangian, coordinate, order):
    levels = [timeDerivative(coordinate, k) for k in range(order + 1)]
    terms = sp.Add.make_args(sp.expand(lagrangian))
    return sp.expand(sp.Add(*[_reduceSameFieldBilinearTerm(term, levels) for term in terms]))


def canonicalizeLagrangian(lagrangian, coords, order):
    result = sp.expand(sp.sympify(lagrangian))
    for coordinate in coords:
        result = _canonicalizeForCoordinate(result, coordinate, order)
    return sp.expand(result)


def eulerLagrangeExpression(lagrangian, coordinate, order, pipelineSign=False):
    lagrangian = sp.sympify(lagrangian)
    accumulated = sp.Integer(0)
    for k in range(order + 1):
        partial = sp.diff(lagrangian, timeDerivative(coordinate, k))
        accumulated = accumulated + sp.Integer(-1) ** k * sp.diff(partial, TIME, k)
    accumulated = sp.expand(accumulated)
    return -accumulated if pipelineSign else accumulated


def eulerLagrangeSystem(lagrangian, coords, order=None, pipelineSign=False):
    resolvedOrder = lagrangianOrder(lagrangian, coords) if order is None else order
    expressions = [
        eulerLagrangeExpression(lagrangian, coordinate, resolvedOrder, pipelineSign) for coordinate in coords
    ]
    return expressions, resolvedOrder


def _stateSymbol(coordinateIndex, derivativeOrder):
    return sp.Symbol(f"q{coordinateIndex}_d{derivativeOrder}")


def solveTopDerivatives(lagrangian, coords, order=None, constants=None):
    elSystem, resolvedOrder = eulerLagrangeSystem(lagrangian, coords, order)
    equationOrder = 2 * resolvedOrder

    topSymbols = [_stateSymbol(index, equationOrder) for index in range(len(coords))]
    topSubstitution = {sp.diff(coord, TIME, equationOrder): symbol for coord, symbol in zip(coords, topSymbols)}
    substitutedEquations = [equation.subs(topSubstitution) for equation in elSystem]

    massMatrix, forcing = sp.linear_eq_to_matrix(substitutedEquations, topSymbols)
    solution = massMatrix.inv() * forcing
    if constants:
        solution = solution.subs(constants)

    return [sp.expand(component) for component in solution], resolvedOrder, equationOrder


def buildStateDerivative(lagrangian, coords, order=None, constants=None):
    topSolution, _resolvedOrder, equationOrder = solveTopDerivatives(lagrangian, coords, order, constants)
    noCoords = len(coords)

    lowerSubstitution = {}
    flatSymbols = []
    for derivativeOrder in range(equationOrder):
        for coordinateIndex in range(noCoords):
            symbol = _stateSymbol(coordinateIndex, derivativeOrder)
            lowerSubstitution[timeDerivative(coords[coordinateIndex], derivativeOrder)] = symbol
            flatSymbols.append(symbol)

    topExpressions = [component.subs(lowerSubstitution) for component in topSolution]
    topFunctions = [sp.lambdify(flatSymbols, expression, modules="numpy") for expression in topExpressions]

    def stateDerivative(state):
        stateArray = np.asarray(state, dtype=float)
        blocks = [stateArray[level * noCoords:(level + 1) * noCoords] for level in range(equationOrder)]
        topValues = np.array([function(*stateArray) for function in topFunctions])
        return np.concatenate(blocks[1:] + [topValues])

    return stateDerivative, equationOrder, noCoords
