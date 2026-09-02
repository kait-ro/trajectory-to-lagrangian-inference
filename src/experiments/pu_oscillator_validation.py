import numpy as np
import sympy as sp
from experiments.pu_system import OMEGA1, OMEGA2, paisUhlenbeckLagrangian
from generation.eqnofmotion import EulerLagrangeEqn, defineCoordinates
from generation.higher_order_integrator import simulateHigherOrderTrajectory
from generation.ostrogradski import (
    TIME,
    buildStateDerivative,
    eulerLagrangeExpression,
    lagrangianOrder,
)
from generation.ostrogradski_hamiltonian import (
    hamiltonianOnTrajectory,
    ostrogradskiHamiltonian,
)


def regressionAgainstSecondOrder():
    _t, coords, vels = defineCoordinates(2)
    lagrangian = sp.Rational(1, 2) * sum(v ** 2 for v in vels) - sp.Rational(1, 2) * sum(q ** 2 for q in coords) - sp.Rational(1, 10) * coords[0] ** 2 * coords[1] ** 2

    existing = list(EulerLagrangeEqn(lagrangian, coords, vels))
    generalised = [eulerLagrangeExpression(lagrangian, coordinate, 1, pipelineSign=True) for coordinate in coords]
    mismatch = [sp.simplify(a - b) for a, b in zip(existing, generalised)]
    return all(term == 0 for term in mismatch), mismatch


def paisUhlenbeckEquationOfMotion():
    _t, coords, _vels = defineCoordinates(1)
    coordinate = coords[0]
    lagrangian = paisUhlenbeckLagrangian(coordinate)

    order = lagrangianOrder(lagrangian, coords)
    elExpression = eulerLagrangeExpression(lagrangian, coordinate, order)

    expected = (
        sp.diff(coordinate, TIME, 4)
        + (OMEGA1 ** 2 + OMEGA2 ** 2) * sp.diff(coordinate, TIME, 2)
        + OMEGA1 ** 2 * OMEGA2 ** 2 * coordinate
    )
    return sp.simplify(elExpression - expected) == 0, elExpression, order


def integratePaisUhlenbeck(omega1, omega2, dt, steps, initialState):
    _t, coords, _vels = defineCoordinates(1)
    lagrangian = paisUhlenbeckLagrangian(coords[0])
    constants = {OMEGA1: omega1, OMEGA2: omega2}

    stateDerivative, equationOrder, noCoords = buildStateDerivative(lagrangian, coords, constants=constants)
    times, perDerivative = simulateHigherOrderTrajectory(initialState, stateDerivative, dt, steps, equationOrder, noCoords)

    hamiltonianData = ostrogradskiHamiltonian(lagrangian, coords)
    derivativeArrays = list(perDerivative)
    hamiltonianSeries = hamiltonianOnTrajectory(hamiltonianData, coords, constants, derivativeArrays)

    return times, perDerivative, hamiltonianData, hamiltonianSeries, lagrangian, coords, constants


def dominantFrequencies(signal, dt, count=2):
    windowed = signal - signal.mean()
    spectrum = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(len(signal), dt)
    peaks = np.argsort(spectrum)[::-1]
    selected = []
    for index in peaks:
        angular = 2 * np.pi * frequencies[index]
        if all(abs(angular - existing) > 0.05 for existing in selected):
            selected.append(angular)
        if len(selected) == count:
            break
    return sorted(selected)


def run():
    lines = []

    regressionOk, regressionMismatch = regressionAgainstSecondOrder()
    lines.append(f"[Ostrogradski EL vs legacy 2nd-order EL] agree on an order-1 Lagrangian: {regressionOk}")
    if not regressionOk:
        lines.append(f"      mismatch: {regressionMismatch}")

    eomOk, elExpression, order = paisUhlenbeckEquationOfMotion()
    lines.append(f"[PU equation of motion] detected Lagrangian order: {order}")
    lines.append(f"[PU equation of motion] Euler-Lagrange expression: {elExpression}")
    lines.append(f"[PU equation of motion] matches q'''' + (w1^2+w2^2) q'' + w1^2 w2^2 q: {eomOk}")

    omega1, omega2 = 1.0, 2.0
    dt, steps = 0.005, 24000
    initialState = [1.0, 0.0, 0.0, 0.0]
    _times, perDerivative, hamiltonianData, hamiltonianSeries, _lagrangian, _coords, _constants = integratePaisUhlenbeck(
        omega1, omega2, dt, steps, initialState
    )

    positionSignal = perDerivative[0][:, 0]
    recoveredFrequencies = dominantFrequencies(positionSignal, dt, count=2)
    lines.append(f"[higher-order RK4 integration] state = (q, q', q'', q'''), steps={steps}, dt={dt}")
    lines.append(f"[higher-order RK4 integration] dominant angular frequencies from q(t): {np.round(recoveredFrequencies, 4)}  (expected [1.0, 2.0])")
    lines.append(f"[higher-order RK4 integration] max |q(t)| over trajectory: {np.abs(positionSignal).max():.3f}")

    hamiltonianDrift = (hamiltonianSeries.max() - hamiltonianSeries.min()) / max(abs(hamiltonianSeries.mean()), 1e-12)
    lines.append(f"[Ostrogradski Hamiltonian] H = {sp.nsimplify(hamiltonianData['hamiltonian'])}")
    lines.append(f"[Ostrogradski Hamiltonian] H mean = {hamiltonianSeries.mean():.6f}, relative drift over trajectory = {hamiltonianDrift:.2e}")

    frequencyOk = (
        abs(recoveredFrequencies[0] - min(omega1, omega2)) < 0.02
        and abs(recoveredFrequencies[1] - max(omega1, omega2)) < 0.02
    )
    conservationOk = abs(hamiltonianDrift) < 1e-3

    lines.append("")
    lines.append(f"SUMMARY: regression={regressionOk}  PU-EOM={eomOk}  frequencies={frequencyOk}  H-conserved={conservationOk}")
    return "\n".join(lines), {
        "regressionOk": regressionOk,
        "eomOk": eomOk,
        "frequencyOk": frequencyOk,
        "conservationOk": conservationOk,
    }


if __name__ == "__main__":
    report, _ = run()
    print(report)
