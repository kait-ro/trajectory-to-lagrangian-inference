import sympy as sp

from experiments.ghost_detection_validation import _ghostBattery, rocReport
from generation.eqnofmotion import TIME, defineCoordinates
from generation.ghost_detection import detectGhost


def test_battery_has_both_labels():
    labels = {label for _name, label, _lagrangian in _ghostBattery()}
    assert labels == {"healthy", "ghost"}


def test_no_false_positives_or_negatives_on_clean_data():
    _text, records = rocReport(noiseLevels=(0.0,), seeds=(0,))
    stats = records["perNoise"]["0.0"]
    assert stats["falsePositiveRate"] == 0.0
    assert stats["falseNegativeRate"] == 0.0


def test_degenerate_system_is_flagged_not_scored():
    _t, coords, _v = defineCoordinates(2)
    q1, q2 = coords
    degenerate = sp.diff(q1, TIME) * q2 - sp.Rational(1, 2) * q2 ** 2 - sp.Rational(1, 2) * q1 ** 2
    verdict = detectGhost(degenerate, coords)
    assert verdict["degenerate"] is True
    assert verdict["ghost"] is None


def test_healthy_quadratic_oscillator_is_not_a_ghost():
    _t, coords, _v = defineCoordinates(1)
    q = coords[0]
    sho = sp.Rational(1, 2) * sp.diff(q, TIME) ** 2 - sp.Rational(1, 2) * 4 * q ** 2
    verdict = detectGhost(sho, coords)
    assert verdict["ghost"] is False
