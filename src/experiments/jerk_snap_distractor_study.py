import json
import os

import numpy as np
from experiments.artifacts import RESULTS_DIR
from experiments.pu_system import groundTruthColumns, paisUhlenbeckStateLagrangian
from finding_L.equivalence_class import isNullLagrangian
from finding_L.higher_order_discovery import (
    recoverHigherOrderLagrangian,
    stateToCoordinate,
)
from generation.eqnofmotion import defineCoordinates
from generation.numerical_diff import savitzkyGolayDerivatives

NO_STATE_VARS = 4
LAGRANGIAN_ORDER = 3
COLUMN_ORDER = 2 * LAGRANGIAN_ORDER


def _jerkTermsPresent(selectedTerms):
    return [str(monomial) for monomial, _ in selectedTerms if "s3" in str(monomial)]


def _scenario(name, derivativeColumns, orderPrior=False):
    recovered, selected = recoverHigherOrderLagrangian(
        derivativeColumns, NO_STATE_VARS, LAGRANGIAN_ORDER, orderPrior=orderPrior
    )
    expected = paisUhlenbeckStateLagrangian(NO_STATE_VARS)

    _t, coords, vels = defineCoordinates(1)
    difference = stateToCoordinate(recovered - expected, NO_STATE_VARS, coords[0])
    equivalent, _residual = isNullLagrangian(difference, coords, vels, order=LAGRANGIAN_ORDER)

    return {
        "scenario": name,
        "orderPrior": orderPrior,
        "recovered": str(recovered),
        "selectedTerms": [(str(monomial), coefficient) for monomial, coefficient in selected],
        "spuriousJerkTerms": _jerkTermsPresent(selected),
        "equivalentUpToNullLagrangian": bool(equivalent),
    }


def run():
    dt, cleanPosition, columns = groundTruthColumns(COLUMN_ORDER, dt=0.004, steps=15000)
    expected = paisUhlenbeckStateLagrangian(NO_STATE_VARS)

    groundTruth = [np.asarray(c, dtype=float) for c in columns]
    scenarios = [
        _scenario("ground_truth", groundTruth),
        _scenario("ground_truth+prior", groundTruth, orderPrior=True),
    ]
    positionStd = cleanPosition.std()
    for noiseLevel in [0.001, 0.005, 0.01, 0.03]:
        rng = np.random.default_rng(77)
        noisy = cleanPosition + rng.normal(0.0, noiseLevel * positionStd, cleanPosition.shape)
        estimated = savitzkyGolayDerivatives(noisy, dt, COLUMN_ORDER, polyOrder=8)
        scenarios.append(_scenario(f"savgol_{noiseLevel * 100:g}pct", estimated))
        scenarios.append(_scenario(f"savgol_{noiseLevel * 100:g}pct+prior", estimated, orderPrior=True))

    lines = [
        "Jerk/snap distractor library study (Pais-Uhlenbeck)",
        f"library: degree<=2 monomials in (q, q', q'', q'''), Lagrangian order {LAGRANGIAN_ORDER}, EL columns to order {COLUMN_ORDER}",
        f"expected order-2 Lagrangian: {expected}",
        "+prior: recoverHigherOrderLagrangian(orderPrior=True) reduces the library",
        "        order to inferLagrangianOrder's verdict before selecting",
        "",
    ]
    for record in scenarios:
        lines.append(f"[{record['scenario']}]")
        lines.append(f"  recovered: {record['recovered']}")
        lines.append(f"  jerk/snap distractor terms selected: {record['spuriousJerkTerms'] or 'none'}")
        lines.append(f"  equivalence-class match to true PU Lagrangian: {record['equivalentUpToNullLagrangian']}")
        lines.append("")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "jerk_distractor.json"), "w") as handle:
        json.dump(scenarios, handle, indent=2)
    report = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "jerk_distractor.txt"), "w") as handle:
        handle.write(report + "\n")
    return report


if __name__ == "__main__":
    print(run())
