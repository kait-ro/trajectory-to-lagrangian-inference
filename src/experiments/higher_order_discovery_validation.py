import json
import os

import numpy as np
import sympy as sp
from experiments.artifacts import RESULTS_DIR
from experiments.pu_system import groundTruthColumns, paisUhlenbeckStateLagrangian
from finding_L.equivalence_class import isNullLagrangian
from finding_L.higher_order_discovery import (
    recoverHigherOrderLagrangian,
    stateToCoordinate,
)
from generation.eqnofmotion import defineCoordinates

NO_STATE_VARS = 3
LAGRANGIAN_ORDER = 2
COLUMN_ORDER = 2 * LAGRANGIAN_ORDER


def _equivalenceToTruePu(recoveredState, expectedState):
    _t, coords, vels = defineCoordinates(1)
    difference = stateToCoordinate(recoveredState - expectedState, NO_STATE_VARS, coords[0])
    isNull, residual = isNullLagrangian(difference, coords, vels, order=LAGRANGIAN_ORDER)
    return isNull, residual[0]


def _scenario(name, derivativeColumns):
    recovered, _selected = recoverHigherOrderLagrangian(derivativeColumns, NO_STATE_VARS, LAGRANGIAN_ORDER)
    expected = paisUhlenbeckStateLagrangian(NO_STATE_VARS)
    equivalent, residual = _equivalenceToTruePu(recovered, expected)

    expectedTerms = expected.as_coefficients_dict()
    recoveredTerms = recovered.as_coefficients_dict()
    monomials = sorted(set(expectedTerms) | set(recoveredTerms), key=sp.default_sort_key)

    return {
        "scenario": name,
        "recovered": str(recovered),
        "expected": str(expected),
        "coefficientRows": [
            (str(monomial), float(expectedTerms.get(monomial, 0)), float(recoveredTerms.get(monomial, 0)))
            for monomial in monomials
        ],
        "equivalentUpToNullLagrangian": bool(equivalent),
        "eulerLagrangeResidualOfDifference": str(residual),
        "exactMatch": bool(sp.expand(recovered - expected) == 0),
    }


def run():
    dt, cleanPosition, columns = groundTruthColumns(COLUMN_ORDER, dt=0.004, steps=15000)
    scenarios = [_scenario("ground_truth_derivatives", [np.asarray(c, dtype=float) for c in columns])]

    from generation.numerical_diff import smoothingSplineDerivatives

    positionStd = cleanPosition.std()
    for noiseLevel in [0.001, 0.005, 0.01, 0.03]:
        rng = np.random.default_rng(2026)
        noisyPosition = cleanPosition + rng.normal(0.0, noiseLevel * positionStd, cleanPosition.shape)
        estimated = smoothingSplineDerivatives(noisyPosition, dt, COLUMN_ORDER)
        scenarios.append(_scenario(f"spline_from_{noiseLevel * 100:g}pct_noise", estimated))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "higher_order_discovery.json"), "w") as handle:
        json.dump(scenarios, handle, indent=2)

    lines = ["Pais-Uhlenbeck higher-derivative Lagrangian recovery", ""]
    lines.append(f"expected (kinetic q''^2 locked to 1): {scenarios[0]['expected']}")
    lines.append("")
    for record in scenarios:
        lines.append(f"[{record['scenario']}]")
        lines.append(f"  recovered: {record['recovered']}")
        lines.append(f"  exact match: {record['exactMatch']}   equivalence-class match: {record['equivalentUpToNullLagrangian']}")
        for name, expectedValue, recoveredValue in record["coefficientRows"]:
            lines.append(f"    {name:>10}: expected {expectedValue:+.4f}  recovered {recoveredValue:+.4f}")
        if not record["equivalentUpToNullLagrangian"]:
            lines.append(f"    EL residual of (recovered - expected): {record['eulerLagrangeResidualOfDifference']}")
        lines.append("")

    tableText = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "higher_order_discovery.txt"), "w") as handle:
        handle.write(tableText + "\n")
    return tableText


if __name__ == "__main__":
    print(run())
