import numpy as np
import sympy as sp

from generation.eqnofmotion import TIME, defineCoordinates
from generation.higher_order_integrator import simulateHigherOrderTrajectory
from generation.ostrogradski import buildStateDerivative
from finding_L.higher_order_candidates import stateGridSymbols, stateVariableSymbols

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


def multiFieldPaisUhlenbeckLagrangian(coords, omega1=1.0, omega2=2.0, coupling=0.0):
    total = sp.Integer(0)
    for coordinate in coords:
        velocity = sp.diff(coordinate, TIME)
        acceleration = sp.diff(coordinate, TIME, 2)
        total += sp.Rational(1, 2) * (
            acceleration ** 2
            - (omega1 ** 2 + omega2 ** 2) * velocity ** 2
            + omega1 ** 2 * omega2 ** 2 * coordinate ** 2
        )
    for i in range(len(coords) - 1):
        total += coupling * coords[i] * coords[i + 1]
    return sp.expand(total)


def multiFieldPaisUhlenbeckStateLagrangian(noFields, lagrangianOrder=2, omega1=1.0, omega2=2.0, coupling=0.0):
    grid = stateGridSymbols(noFields, lagrangianOrder)
    total = sp.Integer(0)
    for field in range(noFields):
        s0, s1, s2 = grid[field][0], grid[field][1], grid[field][2]
        total += s2 ** 2 - (omega1 ** 2 + omega2 ** 2) * s1 ** 2 + omega1 ** 2 * omega2 ** 2 * s0 ** 2
    for i in range(noFields - 1):
        total += 2 * coupling * grid[i][0] * grid[i + 1][0]
    return sp.expand(total)


def multiFieldGroundTruthColumns(
    noFields,
    maxLevel,
    omega1=1.0,
    omega2=2.0,
    coupling=0.0,
    dt=0.004,
    steps=15000,
    noTrajectories=6,
    seed=11,
):
    _t, coords, _v = defineCoordinates(noFields)
    lagrangian = multiFieldPaisUhlenbeckLagrangian(coords, omega1, omega2, coupling)
    stateDerivative, equationOrder, _n = buildStateDerivative(lagrangian, coords)

    rng = np.random.default_rng(seed)
    perLevel = [[] for _ in range(equationOrder)]
    topLevel = []
    for _ in range(noTrajectories):
        initialState = rng.uniform(-0.6, 0.6, size=equationOrder * noFields)
        _times, perDerivative = simulateHigherOrderTrajectory(
            list(initialState), stateDerivative, dt, steps, equationOrder, noFields
        )
        for level in range(equationOrder):
            perLevel[level].append(perDerivative[level])
        states = np.hstack(perDerivative)
        top = np.array([stateDerivative(state)[-noFields:] for state in states])
        topLevel.append(top)

    columns = [np.vstack(level) for level in perLevel] + [np.vstack(topLevel)]
    while len(columns) <= maxLevel:
        higher = -(omega1 ** 2 + omega2 ** 2) * columns[-2] - omega1 ** 2 * omega2 ** 2 * columns[-4]
        columns.append(higher)
    return dt, columns[: maxLevel + 1]
