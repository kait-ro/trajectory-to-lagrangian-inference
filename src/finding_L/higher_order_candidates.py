import itertools

import numpy as np
import sympy as sp

from generation.ostrogradski import TIME, eulerLagrangeExpression


def stateVariableSymbols(noStateVars):
    return [sp.Symbol(f"s{index}") for index in range(noStateVars)]


def buildHigherOrderLibrary(noStateVars, maxDegree):
    variables = stateVariableSymbols(noStateVars)
    library = []
    seenExponents = set()
    for degree in range(1, maxDegree + 1):
        for combo in itertools.combinations_with_replacement(range(noStateVars), degree):
            exponents = tuple(sorted(combo))
            if exponents in seenExponents:
                continue
            seenExponents.add(exponents)
            monomial = sp.Integer(1)
            for index in combo:
                monomial = monomial * variables[index]
            library.append(sp.expand(monomial))
    return library


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
