"""Automatic Lagrangian-order inference from trajectory data."""

import numpy as np

from experiments.order_inference_validation import _anharmonicOscillatorColumns
from experiments.pu_system import groundTruthColumns
from finding_L.higher_order_discovery import inferLagrangianOrder


def test_pais_uhlenbeck_is_inferred_as_order_two():
    _dt, _position, columns = groundTruthColumns(6, dt=0.004, steps=12000)
    order, perOrder = inferLagrangianOrder([np.asarray(c, dtype=float) for c in columns], maxOrder=3)
    assert order == 2
    assert perOrder[0]["scaledResidual"] > 0.1   # order 1 cannot explain PU
    assert perOrder[1]["converged"]              # order 2 satisfies an EL equation


def test_anharmonic_oscillator_is_inferred_as_order_one():
    columns = _anharmonicOscillatorColumns(steps=7000, noTrajectories=6)
    order, perOrder = inferLagrangianOrder(columns, maxOrder=2, libraryMaxDegree=4)
    assert order == 1
    assert perOrder[0]["converged"]


def test_not_enough_derivative_levels_raises():
    columns = [np.random.default_rng(0).standard_normal(100) for _ in range(2)]  # only q, q'
    try:
        inferLagrangianOrder(columns, maxOrder=3)
    except ValueError as error:
        assert "derivative levels" in str(error)
    else:
        raise AssertionError("expected ValueError for insufficient derivative levels")
