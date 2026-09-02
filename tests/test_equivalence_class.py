import sympy as sp
from finding_L.equivalence_class import (
    classifyLagrangianPair,
    isNullLagrangian,
    verifyEquivalenceClass,
)
from generation.eqnofmotion import TIME, defineCoordinates


def _base_lagrangian(coords, vels):
    return sp.Rational(1, 2) * sum(v ** 2 for v in vels) - sp.Rational(1, 2) * sum(q ** 2 for q in coords)


def test_total_time_derivative_difference_is_equivalent():
    _t, coords, vels = defineCoordinates(4)
    base = _base_lagrangian(coords, vels)
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
    difference = sp.diff(q, TIME) ** 2 + q * sp.diff(q, TIME, 2)

    isNull, residual = isNullLagrangian(difference, coords, vels, order=2)
    assert isNull
    assert all(component == 0 for component in residual)


def test_verify_scale_factor_equivalence():
    _t, coords, vels = defineCoordinates(2)
    base = _base_lagrangian(coords, vels)

    result = verifyEquivalenceClass(base, 3 * base, coords, vels)

    assert result.verdict == "equivalent-by-scale"
    assert sp.simplify(result.scale - 3) == 0
    assert sp.expand(result.boundaryFunction) == 0
    assert all(component == 0 for component in result.eulerLagrangeResidual)


def test_verify_hand_constructed_total_derivative():
    _t, coords, vels = defineCoordinates(2)
    base = _base_lagrangian(coords, vels)
    q0 = coords[0]
    constructed = base + sp.diff(q0 * sp.diff(q0, TIME), TIME)

    result = verifyEquivalenceClass(base, constructed, coords, vels)

    assert result.verdict == "equivalent-by-total-derivative"
    assert sp.simplify(result.scale - 1) == 0
    assert sp.expand(result.boundaryFunction - q0 * sp.diff(q0, TIME)) == 0


def test_verify_scale_and_total_derivative_equivalence():
    _t, coords, vels = defineCoordinates(1)
    base = _base_lagrangian(coords, vels)
    q = coords[0]
    constructed = 2 * base + sp.diff(q * sp.diff(q, TIME), TIME)

    result = verifyEquivalenceClass(base, constructed, coords, vels)

    assert result.verdict == "equivalent-by-scale-and-total-derivative"
    assert sp.simplify(result.scale - 2) == 0
    assert sp.expand(result.boundaryFunction - q * sp.diff(q, TIME)) == 0
    assert all(component == 0 for component in result.eulerLagrangeResidual)


def test_verify_numerically_close_near_miss_is_not_equivalent():
    _t, coords, vels = defineCoordinates(2)
    base = _base_lagrangian(coords, vels)
    q0 = coords[0]
    nearMiss = base + sp.Rational(1, 1000) * q0 ** 4

    result = verifyEquivalenceClass(base, nearMiss, coords, vels)

    assert result.verdict == "not-equivalent"
    assert result.scale is None


def test_classify_pair_accepts_scale_equivalent_candidates():
    _t, coords, vels = defineCoordinates(3)
    base = _base_lagrangian(coords, vels)

    verdict = classifyLagrangianPair(5 * base, base, coords, vels)

    assert verdict.equivalent
    assert sp.simplify(verdict.scale - 5) == 0
    assert "equivalent-by-scale" in verdict.detail


def test_classify_pair_rejects_close_but_distinct_candidates():
    _t, coords, vels = defineCoordinates(2)
    base = _base_lagrangian(coords, vels)
    nearMiss = base + sp.Rational(1, 50) * coords[0] ** 2

    verdict = classifyLagrangianPair(nearMiss, base, coords, vels)

    assert not verdict.equivalent
