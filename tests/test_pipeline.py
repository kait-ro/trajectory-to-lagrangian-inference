import numpy as np
import sympy as sp
from experiments.pu_system import (
    groundTruthColumns,
    multiFieldGroundTruthColumns,
    multiFieldPaisUhlenbeckStateLagrangian,
)
from finding_L.equivalence_class import isNullLagrangian
from finding_L.higher_order_discovery import multiFieldStateToCoordinates
from finding_L.pipeline import endToEndPipeline
from generation.eqnofmotion import defineCoordinates


def _pu_positions(noiseLevel, seed):
    dt, _position, columns = groundTruthColumns(6, dt=0.004, steps=13000)
    clean = np.asarray(columns[0], dtype=float)
    rng = np.random.default_rng(seed)
    return dt, clean + rng.normal(0.0, noiseLevel * clean.std(), clean.shape)


def test_pipeline_recovers_pu_order_and_ghost_from_noisy_positions():
    dt, noisy = _pu_positions(0.002, seed=0)
    result = endToEndPipeline(noisy, dt, maxOrder=2)

    assert result.lagrangianOrder == 2
    assert result.ghost is True
    assert result.orderConfidence == 1.0
    assert result.ghostConfidence == 1.0
    assert len(result.perMethod) == 3


def test_pipeline_gets_the_pu_coefficients_at_zero_noise():
    dt, clean = _pu_positions(0.0, seed=0)
    result = endToEndPipeline(clean, dt, maxOrder=2)

    coefficients = sp.expand(result.discoveredLagrangian).as_coefficients_dict()
    s0, s2 = sp.Symbol("s0"), sp.Symbol("s2")
    assert abs(float(coefficients[s0 ** 2]) - 4.0) < 0.3
    assert abs(float(coefficients[s2 ** 2]) - 1.0) < 1e-6


def test_multi_field_pipeline_infers_the_coupled_pu_order_from_a_single_trajectory():
    dt, columns = multiFieldGroundTruthColumns(2, 4, coupling=0.3, dt=0.004, steps=13000, noTrajectories=1)
    clean = np.asarray(columns[0], dtype=float)

    result = endToEndPipeline(clean, dt, maxOrder=2, noFields=2)

    assert result.lagrangianOrder == 2
    assert result.orderConfidence == 1.0
    assert len(result.perMethod) == 3
    assert result.ghost in (True, False, None)


def test_multi_field_pipeline_recovers_equivalent_lagrangian_with_segmented_differentiation():
    steps = 9000
    dt, columns = multiFieldGroundTruthColumns(2, 4, coupling=0.3, steps=steps, noTrajectories=5)
    exact = np.asarray(columns[0], dtype=float)
    rng = np.random.default_rng(303)
    noisy = exact + rng.normal(0.0, 3e-4 * exact.std(), exact.shape)

    result = endToEndPipeline(noisy, dt, maxOrder=2, noFields=2, segmentLength=steps)

    assert result.lagrangianOrder == 2
    assert result.orderConfidence == 1.0
    assert result.ghost is True

    _t, coords, vels = defineCoordinates(2)
    expected = multiFieldPaisUhlenbeckStateLagrangian(2, 2, coupling=0.3)
    difference = multiFieldStateToCoordinates(result.discoveredLagrangian - expected, 2, 2, coords)
    equivalent, _residual = isNullLagrangian(difference, coords, vels, order=2)
    assert equivalent


def test_pipeline_never_feeds_ground_truth_derivatives():
    import inspect

    signature = inspect.signature(endToEndPipeline)
    assert list(signature.parameters)[:2] == ["noisyPositions", "dt"]
    assert "derivative" not in " ".join(signature.parameters).lower()
