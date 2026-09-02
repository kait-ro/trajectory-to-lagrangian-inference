import numpy as np
import sympy as sp
from experiments.pu_system import groundTruthColumns
from finding_L.pipeline import endToEndPipeline


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


def test_pipeline_never_feeds_ground_truth_derivatives():
    import inspect

    signature = inspect.signature(endToEndPipeline)
    assert list(signature.parameters)[:2] == ["noisyPositions", "dt"]
    assert "derivative" not in " ".join(signature.parameters).lower()
