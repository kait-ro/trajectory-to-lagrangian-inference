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

