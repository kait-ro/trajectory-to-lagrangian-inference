import sympy as sp

from generation.eqnofmotion import TIME, defineCoordinates
from generation.higher_order_integrator import simulateHigherOrderTrajectory
from generation.ostrogradski import buildStateDerivative
from finding_L.higher_order_candidates import stateVariableSymbols

OMEGA1, OMEGA2 = sp.symbols("omega1 omega2", positive=True)


def paisUhlenbeckLagrangian(coordinate):
    velocity = sp.diff(coordinate, TIME)
    acceleration = sp.diff(coordinate, TIME, 2)
    return sp.Rational(1, 2) * (
        acceleration ** 2 - (OMEGA1 ** 2 + OMEGA2 ** 2) * velocity ** 2 + OMEGA1 ** 2 * OMEGA2 ** 2 * coordinate ** 2
    )


def paisUhlenbeckStateLagrangian(noStateVars, omega1=1.0, omega2=2.0):
    s0, s1, s2 = stateVariableSymbols(noStateVars)[:3]
    return sp.expand(s2 ** 2 - (omega1 ** 2 + omega2 ** 2) * s1 ** 2 + omega1 ** 2 * omega2 ** 2 * s0 ** 2)


def groundTruthColumns(maxOrder, omega1=1.0, omega2=2.0, dt=0.004, steps=15000, initialState=(1.0, 0.4, 0.0, 0.0)):
    _t, coords, _vels = defineCoordinates(1)
    lagrangian = paisUhlenbeckLagrangian(coords[0])
    constants = {OMEGA1: omega1, OMEGA2: omega2}

    stateDerivative, equationOrder, noCoords = buildStateDerivative(lagrangian, coords, constants=constants)
    _times, perDerivative = simulateHigherOrderTrajectory(list(initialState), stateDerivative, dt, steps, equationOrder, noCoords)

    columns = [perDerivative[level][:, 0] for level in range(equationOrder)]
    while len(columns) <= maxOrder:
        columns.append(
            -(omega1 ** 2 + omega2 ** 2) * columns[-2] - omega1 ** 2 * omega2 ** 2 * columns[-4]
        )

    return dt, columns[0], columns[: maxOrder + 1]


def groundTruthTrajectory(omega1=1.0, omega2=2.0, dt=0.005, steps=12000, initialState=(1.0, 0.4, 0.0, 0.0)):
    dtOut, position, columns = groundTruthColumns(4, omega1, omega2, dt, steps, initialState)
    return dtOut, position, columns
