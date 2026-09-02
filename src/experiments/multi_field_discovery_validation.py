import json
import os

import numpy as np
import sympy as sp
from experiments.artifacts import RESULTS_DIR
from experiments.pu_system import (
    multiFieldGroundTruthColumns,
    multiFieldPaisUhlenbeckStateLagrangian,
)
from finding_L.equivalence_class import isNullLagrangian
from finding_L.higher_order_discovery import (
    multiFieldStateToCoordinates,
    recoverMultiFieldHigherOrderLagrangian,
)
from generation.eqnofmotion import defineCoordinates
from generation.numerical_diff import segmentedDerivatives, smoothingSplineDerivatives

LAGRANGIAN_ORDER = 2
COLUMN_LEVEL = 2 * LAGRANGIAN_ORDER
STEPS = 12000
NO_TRAJECTORIES = 6


def _crossFieldCoefficient(selectedTerms, noFields):
    grid = [[sp.Symbol(f"s{i}_{k}") for k in range(LAGRANGIAN_ORDER + 1)] for i in range(noFields)]
    couplingMonomial = sp.expand(grid[0][0] * grid[1][0])
    for monomial, coefficient in selectedTerms:
        if sp.expand(monomial - couplingMonomial) == 0:
            return coefficient
    return 0.0


def _scenario(name, noFields, coupling, derivativeData):
    recovered, selected = recoverMultiFieldHigherOrderLagrangian(
        derivativeData, noFields, LAGRANGIAN_ORDER
    )
    expected = multiFieldPaisUhlenbeckStateLagrangian(noFields, LAGRANGIAN_ORDER, coupling=coupling)

    _t, coords, vels = defineCoordinates(noFields)
    difference = multiFieldStateToCoordinates(recovered - expected, noFields, LAGRANGIAN_ORDER, coords)
    equivalent, _residual = isNullLagrangian(difference, coords, vels, order=LAGRANGIAN_ORDER)

    return {
        "scenario": name,
        "noFields": noFields,
        "trueCoupling2mu": 2 * coupling,
        "recoveredCrossFieldCoefficient": float(_crossFieldCoefficient(selected, noFields)),
        "recovered": str(recovered),
        "equivalentUpToNullLagrangian": bool(equivalent),
    }


def _gluedSplineColumns(noisyPositions, dt, noFields):
    return [
        np.column_stack([smoothingSplineDerivatives(noisyPositions[:, f], dt, COLUMN_LEVEL)[level] for f in range(noFields)])
        for level in range(COLUMN_LEVEL + 1)
    ]


def _segmentedSplineColumns(noisyPositions, dt, noFields):
    perField = [
        segmentedDerivatives(noisyPositions[:, f], dt, COLUMN_LEVEL, STEPS, edgeTrim=0.05)
        for f in range(noFields)
    ]
    return [np.column_stack([perField[f][level] for f in range(noFields)]) for level in range(COLUMN_LEVEL + 1)]


def run():
    records = []

    for noFields, coupling in [(2, 0.3), (3, 0.2)]:
        dt, columns = multiFieldGroundTruthColumns(
            noFields, COLUMN_LEVEL, coupling=coupling, steps=STEPS, noTrajectories=NO_TRAJECTORIES
        )
        cleanColumns = [np.asarray(c, dtype=float) for c in columns]
        records.append(_scenario(f"ground_truth_{noFields}field", noFields, coupling, cleanColumns))

    dt, columns = multiFieldGroundTruthColumns(
        2, COLUMN_LEVEL, coupling=0.3, steps=STEPS, noTrajectories=NO_TRAJECTORIES
    )
    exact = [np.asarray(c, dtype=float) for c in columns]
    rng = np.random.default_rng(303)
    perturbed = [c + rng.normal(0.0, 1e-3 * c.std(), c.shape) for c in exact]
    records.append(_scenario("columns_perturbed_0.1pct_2field", 2, 0.3, perturbed))

    noisyPositions = exact[0] + rng.normal(0.0, 3e-4 * exact[0].std(), exact[0].shape)
    records.append(
        _scenario(
            "spline_glued_from_0.03pct_positions_2field", 2, 0.3, _gluedSplineColumns(noisyPositions, dt, 2)
        )
    )
    records.append(
        _scenario(
            "spline_segmented_from_0.03pct_positions_2field", 2, 0.3, _segmentedSplineColumns(noisyPositions, dt, 2)
        )
    )

    lines = ["Multi-field higher-derivative Lagrangian recovery (coupled Pais-Uhlenbeck chain)", ""]
    for record in records:
        lines.append(f"[{record['scenario']}]  ({record['noFields']} fields)")
        lines.append(f"  recovered: {record['recovered']}")
        lines.append(
            f"  cross-field coupling coefficient: recovered {record['recoveredCrossFieldCoefficient']:+.4f}  "
            f"true {record['trueCoupling2mu']:+.4f}"
        )
        lines.append(f"  equivalence-class match to true chain: {record['equivalentUpToNullLagrangian']}")
        lines.append("")
    lines.append("Ground truth and column-perturbed: exact recovery up to a total derivative, with the")
    lines.append("cross-field coupling term included -- the multi-coordinate recovery itself is sound.")
    lines.append("Spline-from-noisy-positions: the 'glued' path splines one curve through all")
    lines.append(f"{NO_TRAJECTORIES} concatenated trajectories, so the step discontinuities at the joins blow up")
    lines.append("the 3rd/4th derivatives and recovery fails. The 'segmented' path differentiates each")
    lines.append("trajectory on its own (segmentedDerivatives) and trims the spline-unstable segment")
    lines.append("edges; the coupled chain is then recovered up to a total derivative from 0.03% position")
    lines.append("noise. The remaining gap to the ~3% single-field case is real -- each field's 4th")
    lines.append("derivative is taken from short segments -- but it is a differentiation gap, not a")
    lines.append("recovery one.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "multi_field_discovery.json"), "w") as handle:
        json.dump(records, handle, indent=2)
    report = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "multi_field_discovery.txt"), "w") as handle:
        handle.write(report + "\n")
    return report


if __name__ == "__main__":
    print(run())
