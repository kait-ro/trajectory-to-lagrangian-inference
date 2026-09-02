import json
import os

import numpy as np
import sympy as sp
from experiments.artifacts import RESULTS_DIR
from experiments.pu_system import groundTruthColumns, paisUhlenbeckStateLagrangian
from finding_L.higher_order_discovery import (
    recoverHigherOrderLagrangian,
    stateToCoordinate,
)
from finding_L.pipeline import endToEndPipeline
from generation.eqnofmotion import TIME, defineCoordinates
from generation.ghost_detection import detectGhost
from generation.higher_order_integrator import simulateHigherOrderTrajectory
from generation.numerical_diff import smoothingSplineDerivatives
from generation.ostrogradski import buildStateDerivative

NO_STATE_VARS = 3
LAGRANGIAN_ORDER = 2


def _stateToCoordinate(stateExpression, coordinate):
    return stateToCoordinate(stateExpression, NO_STATE_VARS, coordinate)


def referenceSystems():
    _t, coords, _vels = defineCoordinates(1)
    coordinate = coords[0]
    velocity = sp.diff(coordinate, TIME)

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


def _cleanPositionSignal(lagrangian, coords, dt=0.004, steps=13000, seed=0):
    stateDerivative, equationOrder, noCoords = buildStateDerivative(lagrangian, coords)
    rng = np.random.default_rng(seed)
    initialState = rng.uniform(-0.8, 0.8, size=equationOrder * noCoords)
    _times, perDerivative = simulateHigherOrderTrajectory(
        list(initialState), stateDerivative, dt, steps, equationOrder, noCoords
    )
    return dt, perDerivative[0][:, 0]


def _ghostBattery():
    _t, coords, _v = defineCoordinates(1)
    q = coords[0]
    v = sp.diff(q, TIME)
    systems = []

    systems.append(("sho_w1", "healthy", sp.Rational(1, 2) * v ** 2 - sp.Rational(1, 2) * q ** 2))
    systems.append(("sho_w3", "healthy", sp.Rational(1, 2) * v ** 2 - sp.Rational(9, 2) * q ** 2))
    systems.append(
        ("anharmonic", "healthy", sp.Rational(1, 2) * v ** 2 - sp.Rational(1, 2) * q ** 2 - sp.Rational(1, 4) * q ** 4)
    )

    for w1, w2, tag in [(1.0, 2.0, "12"), (1.0, 3.0, "13"), (0.7, 1.6, "07_16")]:
        pu = _stateToCoordinate(paisUhlenbeckStateLagrangian(NO_STATE_VARS, w1, w2), q)
        systems.append((f"pais_uhlenbeck_{tag}", "ghost", pu))
    puBase = _stateToCoordinate(paisUhlenbeckStateLagrangian(NO_STATE_VARS, 1.0, 2.0), q)
    systems.append(("pais_uhlenbeck_plus_total_derivative", "ghost", sp.expand(puBase + 3 * sp.diff(q * v, TIME))))

    return systems


def rocReport(noiseLevels=(0.0, 0.002, 0.005, 0.01), seeds=(0, 1, 2)):
    battery = _ghostBattery()
    perNoise = {}
    trials = []

    for noiseLevel in noiseLevels:
        falsePositives = falseNegatives = healthyTrials = ghostTrials = undetermined = 0
        for name, label, lagrangian in battery:
            _t, coords, _v = defineCoordinates(1)
            for seed in seeds:
                dt, clean = _cleanPositionSignal(lagrangian, coords, seed=seed)
                rng = np.random.default_rng(1000 + seed)
                noisy = clean + rng.normal(0.0, noiseLevel * clean.std(), clean.shape)
                try:
                    result = endToEndPipeline(noisy, dt, maxOrder=2, libraryMaxDegree=4)
                    verdict = result.ghost
                except Exception:
                    verdict = None
                trials.append({"system": name, "label": label, "noiseLevel": noiseLevel, "seed": seed, "ghost": verdict})

                if label == "healthy":
                    healthyTrials += 1
                    if verdict is True:
                        falsePositives += 1
                    elif verdict is None:
                        undetermined += 1
                else:
                    ghostTrials += 1
                    if verdict is False:
                        falseNegatives += 1
                    elif verdict is None:
                        undetermined += 1

        perNoise[noiseLevel] = {
            "falsePositiveRate": falsePositives / max(healthyTrials, 1),
            "falseNegativeRate": falseNegatives / max(ghostTrials, 1),
            "undeterminedRate": undetermined / max(healthyTrials + ghostTrials, 1),
            "healthyTrials": healthyTrials,
            "ghostTrials": ghostTrials,
        }

    _t, coords2, _v = defineCoordinates(2)
    q1, q2 = coords2
    degenerate = sp.diff(q1, TIME) * q2 - sp.Rational(1, 2) * q2 ** 2 - sp.Rational(1, 2) * q1 ** 2
    degenerateVerdict = detectGhost(degenerate, coords2)
    degenerateFlagged = bool(degenerateVerdict.get("degenerate"))

    lines = ["Ghost detection -- aggregate statistics over a labelled battery", ""]
    lines.append(f"battery: {sum(1 for _n, l, _s in battery if l == 'healthy')} healthy + "
                 f"{sum(1 for _n, l, _s in battery if l == 'ghost')} ghost systems, {len(seeds)} seeds each")
    lines.append("")
    lines.append(f"{'noise':>7} {'FP rate':>9} {'FN rate':>9} {'undet.':>8}")
    lines.append("-" * 36)
    for noiseLevel, stats in perNoise.items():
        lines.append(
            f"{noiseLevel * 100:6.1f}% {stats['falsePositiveRate']:>9.2f} "
            f"{stats['falseNegativeRate']:>9.2f} {stats['undeterminedRate']:>8.2f}"
        )
    lines.append("")
    lines.append(f"borderline: degenerate Lagrangian flagged as degenerate (not FP/FN): {degenerateFlagged}")
    return "\n".join(lines), {"perNoise": {str(k): v for k, v in perNoise.items()}, "trials": trials, "degenerateFlagged": degenerateFlagged}


def run():
    referenceText, referenceRecords = referenceReport()
    boundaryText, boundaryRows = noiseBoundaryReport()
    rocText, rocRecords = rocReport()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "ghost_detection.json"), "w") as handle:
        json.dump(
            {"reference": referenceRecords, "noiseBoundary": boundaryRows, "roc": rocRecords},
            handle,
            indent=2,
            default=str,
        )

    report = referenceText + "\n" + boundaryText + "\n\n" + rocText + "\n"
    with open(os.path.join(RESULTS_DIR, "ghost_detection.txt"), "w") as handle:
        handle.write(report)
    return report


if __name__ == "__main__":
    print(run())
