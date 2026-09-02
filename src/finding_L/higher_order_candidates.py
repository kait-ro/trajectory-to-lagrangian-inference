import numpy as np
import sympy as sp
from finding_L.candidates import monomialLibrary
from generation.eqnofmotion import defineCoordinates
from generation.ostrogradski import TIME, eulerLagrangeExpression


def stateVariableSymbols(noStateVars):
    return [sp.Symbol(f"s{index}") for index in range(noStateVars)]


def buildHigherOrderLibrary(noStateVars, maxDegree):
    return monomialLibrary(stateVariableSymbols(noStateVars), maxDegree)


def monomialToCoordinate(monomial, coordinate, noStateVars):
    substitution = {
        sp.Symbol(f"s{index}"): (coordinate if index == 0 else sp.diff(coordinate, TIME, index))
        for index in range(noStateVars)
    }
    return monomial.subs(substitution)


def eulerLagrangeColumnFunction(monomial, coordinate, lagrangianOrder, noStateVars, columnOrder):
    termInCoordinate = monomialToCoordinate(monomial, coordinate, noStateVars)
    elExpression = eulerLagrangeExpression(termInCoordinate, coordinate, lagrangianOrder, pipelineSign=True)

    dataSymbols = [sp.Symbol(f"d{k}") for k in range(columnOrder + 1)]
    toDataSymbols = {
        (coordinate if k == 0 else sp.diff(coordinate, TIME, k)): dataSymbols[k]
        for k in range(columnOrder + 1)
    }
    substituted = elExpression.subs(toDataSymbols)
    if substituted.atoms(sp.Derivative):
        raise ValueError(
            f"candidate {monomial} produces derivatives above column order {columnOrder}; "
            f"raise columnOrder / lagrangianOrder"
        )
    lambdified = sp.lambdify(dataSymbols, substituted, modules="numpy")
    return lambdified, elExpression


def dropKineticAliasColumns(matrix, library, kineticIndex, collinearityTolerance=1e-6):
    kineticColumn = matrix[:, kineticIndex]
    kineticNorm = np.linalg.norm(kineticColumn)

    keptIndices = []
    for columnIndex in range(matrix.shape[1]):
        column = matrix[:, columnIndex]
        norm = np.linalg.norm(column)
        if columnIndex != kineticIndex and norm < collinearityTolerance * max(kineticNorm, 1.0):
            continue
        if columnIndex != kineticIndex and kineticNorm > 0 and norm > 0:
            cosine = abs(float(column @ kineticColumn) / (norm * kineticNorm))
            if cosine > 1.0 - collinearityTolerance:
                continue
        keptIndices.append(columnIndex)

    keptLibrary = [library[index] for index in keptIndices]
    keptMatrix = matrix[:, keptIndices]
    newKineticIndex = keptIndices.index(kineticIndex)
    return keptMatrix, keptLibrary, newKineticIndex


def buildEulerLagrangeMatrix(library, coordinate, lagrangianOrder, noStateVars, derivativeColumns):
    columnOrder = len(derivativeColumns) - 1
    noRows = len(derivativeColumns[0])
    matrix = np.zeros((noRows, len(library)))
    elExpressions = []
    for columnIndex, monomial in enumerate(library):
        lambdified, elExpression = eulerLagrangeColumnFunction(
            monomial, coordinate, lagrangianOrder, noStateVars, columnOrder
        )
        elExpressions.append(elExpression)
        evaluated = lambdified(*derivativeColumns)
        matrix[:, columnIndex] = np.broadcast_to(np.asarray(evaluated, dtype=float), (noRows,))
    return matrix, elExpressions


def stateGridSymbols(noFields, lagrangianOrder):
    return [
        [sp.Symbol(f"s{field}_{level}") for level in range(lagrangianOrder + 1)]
        for field in range(noFields)
    ]


def multiFieldLibrary(noFields, lagrangianOrder, maxDegree):
    flat = [symbol for row in stateGridSymbols(noFields, lagrangianOrder) for symbol in row]
    return monomialLibrary(flat, maxDegree)


def multiFieldMonomialToCoordinates(monomial, coords, noFields, lagrangianOrder):
    grid = stateGridSymbols(noFields, lagrangianOrder)
    substitution = {}
    for field in range(noFields):
        for level in range(lagrangianOrder + 1):
            substitution[grid[field][level]] = (
                coords[field] if level == 0 else sp.diff(coords[field], TIME, level)
            )
    return sp.sympify(monomial).subs(substitution)


def buildMultiFieldElMatrix(library, noFields, lagrangianOrder, derivativeData):
    columnOrder = len(derivativeData) - 1
    noRows = derivativeData[0].shape[0]
    _t, coords, _v = defineCoordinates(noFields)

    dataSymbols = [[sp.Symbol(f"d{field}_{level}") for level in range(columnOrder + 1)] for field in range(noFields)]
    toData = {}
    for field in range(noFields):
        for level in range(columnOrder + 1):
            key = coords[field] if level == 0 else sp.diff(coords[field], TIME, level)
            toData[key] = dataSymbols[field][level]
    flatDataSymbols = [symbol for row in dataSymbols for symbol in row]
    evaluationArgs = [derivativeData[level][:, field] for field in range(noFields) for level in range(columnOrder + 1)]

    matrix = np.zeros((noRows * noFields, len(library)))
    elExpressions = []
    for columnIndex, monomial in enumerate(library):
        termInCoords = multiFieldMonomialToCoordinates(monomial, coords, noFields, lagrangianOrder)
        perField = []
        for j in range(noFields):
            elExpression = eulerLagrangeExpression(termInCoords, coords[j], lagrangianOrder, pipelineSign=True)
            substituted = elExpression.subs(toData)
            if substituted.atoms(sp.Derivative):
                raise ValueError(
                    f"candidate {monomial} produces derivatives above column order {columnOrder}; "
                    f"raise columnOrder / lagrangianOrder"
                )
            perField.append(substituted)
            function = sp.lambdify(flatDataSymbols, substituted, modules="numpy")
            evaluated = function(*evaluationArgs)
            matrix[j * noRows:(j + 1) * noRows, columnIndex] = np.broadcast_to(
                np.asarray(evaluated, dtype=float), (noRows,)
            )
        elExpressions.append(perField)
    return matrix, elExpressions
