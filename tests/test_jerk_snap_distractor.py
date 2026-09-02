import numpy as np
from experiments.jerk_snap_distractor_study import _scenario
from experiments.pu_system import groundTruthColumns
from finding_L.higher_order_discovery import reduceOrderToPrior


def _cleanColumns():
    _dt, _position, columns = groundTruthColumns(6, dt=0.004, steps=15000)
    return [np.asarray(c, dtype=float) for c in columns]


def test_order_three_library_fails_on_shell_without_prior():
    record = _scenario("ground_truth", _cleanColumns(), orderPrior=False)
    assert record["spuriousJerkTerms"]
    assert not record["equivalentUpToNullLagrangian"]


def test_order_prior_recovers_true_pais_uhlenbeck():
    record = _scenario("ground_truth+prior", _cleanColumns(), orderPrior=True)
    assert record["spuriousJerkTerms"] == []
    assert record["equivalentUpToNullLagrangian"]


def test_reduce_order_to_prior_collapses_over_specified_order():
    assert reduceOrderToPrior(_cleanColumns(), lagrangianOrder=3) == 2
