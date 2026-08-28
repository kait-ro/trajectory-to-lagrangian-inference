import numpy as np
import pandas as pd
import sympy as sp

from generation.eqnofmotion import TIME, EulerLagrangeEqn

GRAM_DENSE_CELL_BUDGET = 48_000_000


def computeCandidateElColumns(candidateTerm: sp.Expr, coords: list, vels: list, t: sp.Symbol = TIME):
    elExpressions = EulerLagrangeEqn(candidateTerm, coords, vels)

    qddotSymbols = [sp.symbols(f"q{i}ddot") for i in range(len(coords))]
    substitutedExpressions = []
    for expr in elExpressions:
        for j in range(len(coords)):
            expr = expr.subs(sp.diff(coords[j], t, 2), qddotSymbols[j])
        substitutedExpressions.append(expr)

    return [
        sp.lambdify(coords + vels + qddotSymbols, expr, modules="numpy")
        for expr in substitutedExpressions
    ]


def lambdifiedColumnsForTerms(candidateTerms, coords, vels, t=TIME, cache=None):
    if cache is None:
        return [computeCandidateElColumns(term, coords, vels, t) for term in candidateTerms]

    result = []
    for term in candidateTerms:
        key = sp.srepr(term)
        if key not in cache:
            cache[key] = computeCandidateElColumns(term, coords, vels, t)
        result.append(cache[key])
    return result


def evaluateChunkColumns(lambdifiedFuncsPerTerm: list, dataFrame, noCoords: int) -> np.ndarray:
    qColumns = [dataFrame[f"q{i}"].to_numpy() for i in range(noCoords)]
    qdotColumns = [dataFrame[f"q{i}dot"].to_numpy() for i in range(noCoords)]
    qddotColumns = [dataFrame[f"q{i}ddot"].to_numpy() for i in range(noCoords)]

    noRows = len(dataFrame)
    thetaChunk = np.empty((noRows * noCoords, len(lambdifiedFuncsPerTerm)))

    for candidateIndex, lambdifiedFuncs in enumerate(lambdifiedFuncsPerTerm):
        for coordIndex, func in enumerate(lambdifiedFuncs):
            evaluatedColumn = func(*qColumns, *qdotColumns, *qddotColumns)
            rowStart = coordIndex * noRows
            thetaChunk[rowStart:rowStart + noRows, candidateIndex] = evaluatedColumn

    return thetaChunk


def denseBlockRowLimit(noCandidates: int, noCoords: int, cellBudget: int = GRAM_DENSE_CELL_BUDGET) -> int:
    cellsPerRow = max(noCandidates * noCoords, 1)
    return max(1, min(200_000, cellBudget // cellsPerRow))


def buildGramMatrixChunked(
    candidateTerms: list,
    coords: list,
    vels: list,
    t: sp.Symbol,
    csvPath: str,
    noCoords: int,
    chunkRows: int = 200_000,
    cellBudget: int = GRAM_DENSE_CELL_BUDGET,
    lambdifiedCache: dict = None,
):
    noCandidates = len(candidateTerms)
    lambdifiedFuncsPerTerm = lambdifiedColumnsForTerms(candidateTerms, coords, vels, t, lambdifiedCache)

    n = 0
    colSum = np.zeros(noCandidates)
    G = np.zeros((noCandidates, noCandidates))

    rowLimit = denseBlockRowLimit(noCandidates, noCoords, cellBudget)
    for chunk in pd.read_csv(csvPath, chunksize=chunkRows):
        for start in range(0, len(chunk), rowLimit):
            subChunk = chunk.iloc[start:start + rowLimit]
            thetaChunk = evaluateChunkColumns(lambdifiedFuncsPerTerm, subChunk, noCoords)

            n += thetaChunk.shape[0]
            colSum += thetaChunk.sum(axis=0)
            G += thetaChunk.T @ thetaChunk
    return n, colSum, G
