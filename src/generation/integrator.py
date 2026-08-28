import numpy as np
import sympy as sp
from generation.eqnofmotion import (
    ConvertSymbolsForNumpy,
    EulerLagrangeEqn,
    solveEulerLagrangeEqn,
)


def GetAccelFunctions(L: sp.Expr, coords: list, vels: list, t: sp.Symbol, constants: dict | None = None):
    ELterms = EulerLagrangeEqn(L, coords, vels)
    accelSolutions, _qddot = solveEulerLagrangeEqn(ELterms, coords, t)

    if constants:
        accelList = [expr.subs(constants) for expr in accelSolutions]
    else:
        accelList = list(accelSolutions)

    accelFuncs = ConvertSymbolsForNumpy(accelList, coords, vels)
    return accelFuncs


def simulateStep(state: np.ndarray, dt: float, accelFunctions: list):
    n = len(accelFunctions)

    def stateDerivative(s):
        q = s[:n]
        v = s[n:]
        a = np.array([f(*q, *v) for f in accelFunctions])
        return np.concatenate([v, a])

    k1 = stateDerivative(state)
    k2 = stateDerivative(state + 0.5 * dt * k1)
    k3 = stateDerivative(state + 0.5 * dt * k2)
    k4 = stateDerivative(state + dt * k3)

    next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return next_state


def simulateTrajectory(initial_state: np.ndarray, accelFunctions: list, dt: float, steps: int):
    n = len(accelFunctions)
    state = np.array(initial_state, dtype=float)

    t_arr = np.zeros(steps)
    q_arr = np.zeros((steps, n))
    qdot_arr = np.zeros((steps, n))
    qddot_arr = np.zeros((steps, n))

    for i in range(steps):
        q = state[:n]
        v = state[n:]
        a = np.array([f(*q, *v) for f in accelFunctions])

        t_arr[i] = i * dt
        q_arr[i] = q
        qdot_arr[i] = v
        qddot_arr[i] = a

        state = simulateStep(state, dt, accelFunctions)

    return t_arr, q_arr, qdot_arr, qddot_arr
