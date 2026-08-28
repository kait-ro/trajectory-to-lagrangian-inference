import json
import os

import numpy as np
import sympy as sp

from experiments.pu_system import groundTruthColumns, paisUhlenbeckStateLagrangian
from finding_L.higher_order_discovery import recoverHigherOrderLagrangian, stateToCoordinate
from generation.eqnofmotion import defineCoordinates
from generation.ghost_detection import detectGhost
from generation.numerical_diff import smoothingSplineDerivatives
from generation.ostrogradski import TIME

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
NO_STATE_VARS = 3
LAGRANGIAN_ORDER = 2


def _stateToCoordinate(stateExpression, coordinate):
    return stateToCoordinate(stateExpression, NO_STATE_VARS, coordinate)


def referenceSystems():
    t, coords, vels = defineCoordinates(1)
    coordinate = coords[0]
    velocity = sp.diff(coordinate, TIME)
    acceleration = sp.diff(coordinate, TIME, 2)

    healthyOscillator = sp.Rational(1, 2) * velocity ** 2 - sp.Rational(1, 2) * 4 * coordinate ** 2
    paisUhlenbeck = _stateToCoordinate(paisUhlenbeckStateLagrangian(NO_STATE_VARS, 1.0, 2.0), coordinate)
    higherDerivativeTotalDerivative = sp.expand(paisUhlenbeck + 3 * sp.diff(coordinate * velocity, TIME))

    return {
        "healthy_second_order_oscillator": (healthyOscillator, coords),
        "pais_uhlenbeck_ghost": (paisUhlenbeck, coords),
        "pais_uhlenbeck_plus_total_derivative": (higherDerivativeTotalDerivative, coords),
    }


def referenceReport():
    lines = ["Ghost detection on reference systems", ""]
    records = {}
    for name, (lagrangian, coords) in referenceSystems().items():
        verdict = detectGhost(lagrangian, coords)
        records[name] = {
            "ghost": verdict["ghost"],
            "order": verdict["order"],
            "dynamicalStability": verdict["dynamicalStability"],
            "hamiltonianEigenvalues": verdict.get("hamiltonianEigenvalues"),
            "detail": verdict["detail"],
        }
        lines.append(f"[{name}]")
        lines.append(f"  Lagrangian order: {verdict['order']}   dynamics: {verdict['dynamicalStability']}")
        lines.append(f"  H = {sp.nsimplify(verdict['hamiltonian'])}")
        if verdict.get("hamiltonianEigenvalues") is not None:
            lines.append(f"  Hessian(H) eigenvalues: {np.round(verdict['hamiltonianEigenvalues'], 4).tolist()}")
        lines.append(f"  ghost: {verdict['ghost']}  --  {verdict['detail']}")
        lines.append("")
    return "\n".join(lines), records


def noiseBoundaryReport():
    columnOrder = 2 * LAGRANGIAN_ORDER
    dt, cleanPosition, groundTruth = groundTruthColumns(columnOrder, dt=0.004, steps=15000)
    cleanColumns = [np.asarray(component, dtype=float) for component in groundTruth]
    coordinate = defineCoordinates(1)[1][0]

    lines = ["Ghost verdict vs measurement noise (Pais-Uhlenbeck, omega=1,2)", ""]
    rows = []

    scenarios = [("ground_truth", cleanColumns)]
    positionStd = cleanPosition.std()
    for noiseLevel in [0.001, 0.005, 0.01, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35]:
        rng = np.random.default_rng(4242)
        noisy = cleanPosition + rng.normal(0.0, noiseLevel * positionStd, cleanPosition.shape)
        estimated = smoothingSplineDerivatives(noisy, dt, columnOrder)
        scenarios.append((f"{noiseLevel*100:g}pct", estimated))

    for name, columns in scenarios:
        recoveredState, _selected = recoverHigherOrderLagrangian(columns, NO_STATE_VARS, LAGRANGIAN_ORDER)
        recoveredLagrangian = _stateToCoordinate(recoveredState, coordinate)
        verdict = detectGhost(recoveredLagrangian, defineCoordinates(1)[1])
        rows.append(
            {
                "scenario": name,
                "recovered": str(recoveredState),
                "order": verdict["order"],
                "dynamicalStability": verdict["dynamicalStability"],
                "ghost": verdict["ghost"],
                "hamiltonianEigenvalues": verdict.get("hamiltonianEigenvalues"),
            }
        )
        eigenText = np.round(verdict.get("hamiltonianEigenvalues") or [], 3).tolist()
        lines.append(
            f"  {name:>12}: recovered L = {recoveredState}    order={verdict['order']}  "
            f"dynamics={verdict['dynamicalStability']}  ghost={verdict['ghost']}  eig(H)={eigenText}"
        )

    lines.append("")
    lines.append("Interpretation: the ghost verdict is stable as long as the recovered Lagrangian stays order-2")
    lines.append("with oscillatory dynamics; it fails when noise pushes the recovered coefficients to a")
    lines.append("configuration whose characteristic roots gain a real part (spurious runaway) or whose")
    lines.append("highest-derivative term is lost.")
    return "\n".join(lines), rows


def run():
    referenceText, referenceRecords = referenceReport()
    boundaryText, boundaryRows = noiseBoundaryReport()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "ghost_detection.json"), "w") as handle:
        json.dump({"reference": referenceRecords, "noiseBoundary": boundaryRows}, handle, indent=2, default=str)

    report = referenceText + "\n" + boundaryText + "\n"
    with open(os.path.join(RESULTS_DIR, "ghost_detection.txt"), "w") as handle:
        handle.write(report)
    return report


if __name__ == "__main__":
    print(run())
