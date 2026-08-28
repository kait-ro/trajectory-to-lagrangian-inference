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
    topSolution, resolvedOrder, equationOrder = solveTopDerivatives(lagrangian, coords, order, constants)
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
