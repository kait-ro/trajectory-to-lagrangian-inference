import sympy as sp
from experiments.ghost_detection_validation import _ghostBattery, rocReport
from generation.eqnofmotion import TIME, defineCoordinates
from generation.ghost_detection import characteristicRoots, detectGhost, dynamicalStability


def test_battery_has_both_labels():
    labels = {label for _name, label, _lagrangian in _ghostBattery()}
    assert labels == {"healthy", "ghost"}


def test_no_false_positives_or_negatives_on_clean_data():
    _text, records = rocReport(noiseLevels=(0.0,), seeds=(0,))
    stats = records["perNoise"]["0.0"]
    assert stats["falsePositiveRate"] == 0.0
    assert stats["falseNegativeRate"] == 0.0


def test_fully_second_class_degenerate_system_gets_a_real_verdict():
    _t, coords, _v = defineCoordinates(2)
    q1, q2 = coords
    degenerate = sp.diff(q1, TIME) * q2 - sp.Rational(1, 2) * q2 ** 2 - sp.Rational(1, 2) * q1 ** 2
    verdict = detectGhost(degenerate, coords)
    assert verdict["degenerate"] is True
    assert verdict["chainClosed"] is True
    assert verdict["ghost"] is False
    assert verdict["reducedHamiltonian"] is not None


def test_healthy_quadratic_oscillator_is_not_a_ghost():
    _t, coords, _v = defineCoordinates(1)
    q = coords[0]
    sho = sp.Rational(1, 2) * sp.diff(q, TIME) ** 2 - sp.Rational(1, 2) * 4 * q ** 2
    verdict = detectGhost(sho, coords)
    assert verdict["ghost"] is False


def _pu_lagrangian(coordinate, omega1, omega2):
    velocity = sp.diff(coordinate, TIME)
    acceleration = sp.diff(coordinate, TIME, 2)
    return sp.Rational(1, 2) * (
        acceleration ** 2 - (omega1 ** 2 + omega2 ** 2) * velocity ** 2 + omega1 ** 2 * omega2 ** 2 * coordinate ** 2
    )


def _frequencies(roots):
    return sorted(round(abs(root.imag), 3) for root in roots if abs(root.real) < 1e-3)


def test_characteristic_roots_single_field_matches_expected_frequencies():
    _t, coords, _v = defineCoordinates(1)
    roots = characteristicRoots(_pu_lagrangian(coords[0], 1.0, 2.0), coords, order=2)
    assert _frequencies(roots) == [1.0, 1.0, 2.0, 2.0]
    assert dynamicalStability(roots) == "oscillatory"


def test_characteristic_roots_decoupled_multi_field_matches_union_of_single_field():
    _t, coords, _v = defineCoordinates(2)
    q0, q1 = coords
    decoupled = _pu_lagrangian(q0, 1.0, 2.0) + _pu_lagrangian(q1, 1.0, 2.0)
    roots = characteristicRoots(decoupled, coords, order=2)
    assert _frequencies(roots) == [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
    assert dynamicalStability(roots) == "oscillatory"


def test_characteristic_roots_handles_genuinely_coupled_multi_field_system():
    _t, coords, _v = defineCoordinates(2)
    q0, q1 = coords
    coupled = _pu_lagrangian(q0, 1.0, 2.0) + _pu_lagrangian(q1, 1.0, 2.0) + sp.Rational(3, 10) * q0 * q1
    roots = characteristicRoots(coupled, coords, order=2)
    assert len(roots) == 8
    assert dynamicalStability(roots) == "oscillatory"


def _pu_lagrangian_twisted(coordinate, omega1, omega2):
    coupling = omega1 ** 2 + omega2 ** 2
    acceleration = sp.diff(coordinate, TIME, 2)
    return sp.Rational(1, 2) * (
        acceleration ** 2 + coupling * coordinate * acceleration + omega1 ** 2 * omega2 ** 2 * coordinate ** 2
    )


def test_ghost_verdict_invariant_under_null_lagrangian_representative_on_coupled_multi_field_system():
    _t, coords, _v = defineCoordinates(2)
    q0, q1 = coords
    canonical = _pu_lagrangian(q0, 1.0, 2.0) + _pu_lagrangian(q1, 1.0, 2.0) + sp.Rational(3, 10) * q0 * q1
    twisted = (
        _pu_lagrangian_twisted(q0, 1.0, 2.0) + _pu_lagrangian_twisted(q1, 1.0, 2.0) + sp.Rational(3, 10) * q0 * q1
    )

    canonicalVerdict = detectGhost(canonical, coords, order=2)
    twistedVerdict = detectGhost(twisted, coords, order=2)
    assert canonicalVerdict["ghost"] is True
    assert twistedVerdict["ghost"] is True
