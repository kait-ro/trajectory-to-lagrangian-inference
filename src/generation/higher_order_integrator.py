import numpy as np


def rk4Step(state, dt, stateDerivative):
    k1 = stateDerivative(state)
    k2 = stateDerivative(state + 0.5 * dt * k1)
    k3 = stateDerivative(state + 0.5 * dt * k2)
    k4 = stateDerivative(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulateHigherOrderTrajectory(initialState, stateDerivative, dt, steps, equationOrder, noCoords):
    state = np.array(initialState, dtype=float)
    recorded = np.zeros((steps, equationOrder * noCoords))
    times = np.zeros(steps)

    for stepIndex in range(steps):
        times[stepIndex] = stepIndex * dt
        recorded[stepIndex] = state
        state = rk4Step(state, dt, stateDerivative)

    perDerivative = [recorded[:, level * noCoords:(level + 1) * noCoords] for level in range(equationOrder)]
    return times, perDerivative
