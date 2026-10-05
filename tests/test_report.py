import numpy as np
import sympy as sp
from experiments.pu_system import groundTruthColumns, multiFieldGroundTruthColumns
from finding_L.higher_order_discovery import (
    assembleDiscoveredHigherOrderLagrangian,
    assembleDiscoveredMultiFieldLagrangian,
)
from finding_L.report import assembleDiscoveredLagrangian, assembleDiscoveredLagrangianFromState
from generation.eqnofmotion import defineCoordinates


def test_assemble_from_state_matches_functional_wrapper_for_2nd_order():
    _t, coords, vels = defineCoordinates(2)
    kineticTerm = sp.expand(sum(v ** 2 for v in vels))
    discoveredTerms = [(coords[0] ** 2, 4.0), (coords[0] * coords[1], -0.5)]

    viaFunctional = assembleDiscoveredLagrangian(kineticTerm, discoveredTerms, coords, vels)

    q0, q1, v0, v1 = sp.symbols("q0 q1 v0 v1")
    kineticState = v0 ** 2 + v1 ** 2
    discoveredStateTerms = [(q0 ** 2, 4.0), (q0 * q1, -0.5)]
    viaState = assembleDiscoveredLagrangianFromState(kineticState, discoveredStateTerms)

    assert sp.expand(viaFunctional.expression - viaState.expression) == 0
    assert viaFunctional.text == viaState.text


def test_assemble_discovered_higher_order_lagrangian_is_readable_in_native_s_notation():
    dt, _position, columns = groundTruthColumns(4, dt=0.004, steps=8000)
    result = assembleDiscoveredHigherOrderLagrangian(
        [np.asarray(c, dtype=float) for c in columns[:5]], 3, 2
    )

    s0, s2 = sp.symbols("s0 s2")
    assert "s0" in result.text
    assert "_" not in result.text
    coefficients = sp.expand(result.expression).as_coefficients_dict()
    assert abs(float(coefficients[s0 ** 2]) - 4.0) < 0.3
    assert abs(float(coefficients[s2 ** 2]) - 1.0) < 1e-6


def test_assemble_discovered_multi_field_lagrangian_is_readable_in_native_s_field_notation():
    dt, columns = multiFieldGroundTruthColumns(2, 4, coupling=0.3, dt=0.004, steps=8000)
    result = assembleDiscoveredMultiFieldLagrangian(
        [np.asarray(c, dtype=float) for c in columns[:5]], 2, 2
    )

    assert "s0_0" in result.text and "s1_0" in result.text
    s0_0, s1_0 = sp.symbols("s0_0 s1_0")
    coefficients = sp.expand(result.expression).as_coefficients_dict()
    assert abs(float(coefficients[s0_0 * s1_0]) - 0.6) < 1e-3
