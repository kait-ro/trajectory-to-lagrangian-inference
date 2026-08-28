import numpy as np
import sympy as sp

TIME = sp.Symbol("t")


def defineCoordinates(no_coords: int) -> list:
    coords = []
    vels = []
    for i in range(no_coords):
        coords.append(sp.Function(f"q{i}")(TIME))
    for j in coords:
        vels.append(sp.diff(j, TIME))
    return TIME, coords, vels


def EulerLagrangeEqn(L: sp.Expr, coords: list, vels: list) -> list[sp.Expr]:
    Term1 = [sp.diff(sp.diff(L, qdot), TIME) for qdot in vels]
    Term2 = [sp.diff(L, q) for q in coords]
    ExprList = np.array(Term1) - np.array(Term2)
    return ExprList


def solveEulerLagrangeEqn(EulerLagrangeExprList: np.ndarray, coords: list, t: sp.Symbol = TIME):
    qddot = [sp.symbols(f"q{i}ddot") for i in range(len(coords))]
    equations = []
    for i in range(len(coords)):
        eq = EulerLagrangeExprList[i]
        for j in range(len(coords)):
            eq = eq.subs(sp.diff(coords[j], t, 2), qddot[j])
        equations.append(eq)
    M, F = sp.linear_eq_to_matrix(equations, qddot)
    accel_solution = M.inv() * (F)
    return accel_solution, qddot


def ConvertSymbolsForNumpy(accelExpressions, coords, vels):
    accelExprToFunctions = []
    symbols = coords + vels
    for expression in accelExpressions:
        f = sp.lambdify(symbols, expression, modules='numpy')
        accelExprToFunctions.append(f)
    return accelExprToFunctions
