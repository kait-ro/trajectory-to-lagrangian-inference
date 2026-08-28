"""Equivalence-class classifier: null-Lagrangian pairs vs physically distinct pairs."""

import sympy as sp

from finding_L.equivalence_class import classifyLagrangianPair, isNullLagrangian
from generation.eqnofmotion import TIME, defineCoordinates


def _base_lagrangian(coords, vels):
    return sp.Rational(1, 2) * sum(v ** 2 for v in vels) - sp.Rational(1, 2) * sum(q ** 2 for q in coords)


def test_total_time_derivative_difference_is_equivalent():
    _t, coords, vels = defineCoordinates(4)
    base = _base_lagrangian(coords, vels)
    # 3/10 d/dt( q0^2/2 + q1^2/2 ) written as its Lagrangian form
    nullAddition = sp.Rational(3, 10) * (coords[0] * vels[0] + coords[1] * vels[1])

    verdict = classifyLagrangianPair(base + nullAddition, base, coords, vels)

    assert verdict.equivalent
    assert all(component == 0 for component in verdict.eulerLagrangeResidual)
    assert verdict.boundaryPotential is not None
    assert sp.expand(verdict.boundaryPotential - sp.Rational(3, 20) * (coords[0] ** 2 + coords[1] ** 2)) == 0


def test_equation_of_motion_changing_difference_is_not_equivalent():
    _t, coords, vels = defineCoordinates(4)
    base = _base_lagrangian(coords, vels)
    physicalAddition = sp.Rational(1, 5) * coords[0] ** 2 + sp.Rational(1, 10) * coords[0] * coords[1]

    verdict = classifyLagrangianPair(base + physicalAddition, base, coords, vels)

    assert not verdict.equivalent
    assert any(component != 0 for component in verdict.eulerLagrangeResidual)
    assert verdict.boundaryPotential is None


def test_identical_lagrangians_are_equivalent():
    _t, coords, vels = defineCoordinates(2)
    base = _base_lagrangian(coords, vels)
    verdict = classifyLagrangianPair(base, base, coords, vels)
    assert verdict.equivalent
    assert "identical" in verdict.detail


def test_higher_order_total_derivative_is_null():
    _t, coords, vels = defineCoordinates(1)
    q = coords[0]
    # d/dt( q q' ) = q'^2 + q q''  -- a genuine null Lagrangian for an order-2 theory
    difference = sp.diff(q, TIME) ** 2 + q * sp.diff(q, TIME, 2)

    isNull, residual = isNullLagrangian(difference, coords, vels, order=2)
    assert isNull
    assert all(component == 0 for component in residual)
