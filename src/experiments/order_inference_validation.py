"""Infer the Lagrangian order straight from trajectory data.

`inferLagrangianOrder` tries orders 1..maxOrder; for each it measures how well an
order-n Euler-Lagrange equation can account for the q^(n)-squared kinetic term
(least-squares projection onto the other EL columns). The smallest order that
satisfies an EL equation to tolerance (Condition A) is the answer; failing that,
the order after which the residual stops improving (Condition C).
"""

import json
import os

import numpy as np
import sympy as sp

from experiments.pu_system import groundTruthColumns
from finding_L.higher_order_discovery import inferLagrangianOrder
from generation.eqnofmotion import TIME, defineCoordinates
from generation.higher_order_integrator import simulateHigherOrderTrajectory
from generation.ostrogradski import buildStateDerivative

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _anharmonicOscillatorColumns(steps=9000, noTrajectories=8, dt=0.003, seed=5):
    """L = 1/2 q'^2 - 1/2 q^2 - 1/4 q^4  (order 1). Derivative levels 0..4, exact
    via repeated application of the equation of motion q'' = -q - q^3."""
    _t, coords, _v = defineCoordinates(1)
    q = coords[0]
    lagrangian = sp.Rational(1, 2) * sp.diff(q, TIME) ** 2 - sp.Rational(1, 2) * q ** 2 - sp.Rational(1, 4) * q ** 4
    stateDerivative, equationOrder, _n = buildStateDerivative(lagrangian, coords)

    rng = np.random.default_rng(seed)
    q0, q1, q2, q3, q4 = [], [], [], [], []
    for _ in range(noTrajectories):
        state = rng.uniform(-1.0, 1.0, size=equationOrder)
        _times, perDerivative = simulateHigherOrderTrajectory(list(state), stateDerivative, dt, steps, equationOrder, 1)
        position = perDerivative[0][:, 0]
        velocity = perDerivative[1][:, 0]
        acceleration = -position - position ** 3
        jerk = -velocity - 3 * position ** 2 * velocity
        snap = -acceleration - 6 * position * velocity ** 2 - 3 * position ** 2 * acceleration
        q0.append(position); q1.append(velocity); q2.append(acceleration); q3.append(jerk); q4.append(snap)
    return [np.concatenate(level) for level in (q0, q1, q2, q3, q4)]


def run():
    records = []

    dt, _position, puColumns = groundTruthColumns(6, dt=0.004, steps=15000)
    puOrder, puPerOrder = inferLagrangianOrder([np.asarray(c, dtype=float) for c in puColumns], maxOrder=3)
    records.append({"system": "pais_uhlenbeck", "trueOrder": 2, "inferredOrder": puOrder, "perOrder": puPerOrder})

    anharmonic = _anharmonicOscillatorColumns()
    anhOrder, anhPerOrder = inferLagrangianOrder(anharmonic, maxOrder=2, libraryMaxDegree=4)
    records.append({"system": "anharmonic_oscillator", "trueOrder": 1, "inferredOrder": anhOrder, "perOrder": anhPerOrder})

    lines = ["Lagrangian-order inference from trajectory data", ""]
    for record in records:
        mark = "ok" if record["inferredOrder"] == record["trueOrder"] else "WRONG"
        lines.append(f"[{record['system']}]  true order {record['trueOrder']}  ->  inferred {record['inferredOrder']}  ({mark})")
        for entry in record["perOrder"]:
            flag = "  <- satisfies an EL equation" if entry["converged"] else ""
            lines.append(f"    order {entry['order']}: scaled residual {entry['scaledResidual']:.5f}{flag}")
        lines.append("")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "order_inference.json"), "w") as handle:
        json.dump(records, handle, indent=2)
    report = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "order_inference.txt"), "w") as handle:
        handle.write(report + "\n")
    return report


if __name__ == "__main__":
    print(run())
