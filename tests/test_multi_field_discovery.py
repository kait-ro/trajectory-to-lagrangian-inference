import numpy as np
import pytest
import sympy as sp
from experiments.pu_system import (
    multiFieldGroundTruthColumns,
    multiFieldPaisUhlenbeckStateLagrangian,
)
from finding_L.equivalence_class import isNullLagrangian
from finding_L.higher_order_candidates import (
    buildMultiFieldElMatrix,
    multiFieldLibrary,
    stateGridSymbols,
)
from finding_L.higher_order_discovery import (
    multiFieldStateToCoordinates,
    recoverMultiFieldHigherOrderLagrangian,
)
from generation.eqnofmotion import defineCoordinates
from generation.numerical_diff import segmentedDerivatives, smoothingSplineDerivatives


def test_multi_field_el_matrix_stacks_one_block_per_field():
    library = multiFieldLibrary(2, 1, 2)
    rows = 50
    data = [np.random.default_rng(k).standard_normal((rows, 2)) for k in range(3)]
    matrix, elExpressions = buildMultiFieldElMatrix(library, 2, 1, data)
    assert matrix.shape == (rows * 2, len(library))
    assert len(elExpressions) == len(library)
    assert len(elExpressions[0]) == 2


def _recovers_coupled_chain(noFields, coupling):
    _dt, columns = multiFieldGroundTruthColumns(
        noFields, 4, coupling=coupling, steps=9000, noTrajectories=5
    )
    recovered, selected = recoverMultiFieldHigherOrderLagrangian(
        [np.asarray(c, dtype=float) for c in columns], noFields, 2
    )
    expected = multiFieldPaisUhlenbeckStateLagrangian(noFields, 2, coupling=coupling)

    _t, coords, vels = defineCoordinates(noFields)
    difference = multiFieldStateToCoordinates(recovered - expected, noFields, 2, coords)
    isNull, _residual = isNullLagrangian(difference, coords, vels, order=2)
    return recovered, selected, isNull


def test_two_field_coupled_pu_recovered_up_to_null_lagrangian():
    _recovered, selected, isNull = _recovers_coupled_chain(2, 0.3)
    assert isNull

    grid = stateGridSymbols(2, 2)
    crossTerm = sp.expand(grid[0][0] * grid[1][0])
    coefficient = next(c for monomial, c in selected if sp.expand(monomial - crossTerm) == 0)
    assert abs(coefficient - 2 * 0.3) < 1e-3


def test_three_field_coupled_pu_recovered_up_to_null_lagrangian():
    _recovered, _selected, isNull = _recovers_coupled_chain(3, 0.2)
    assert isNull


def test_segmented_derivatives_differentiate_each_segment_and_can_trim_edges():
    dt = 0.01
    local = np.arange(1000) * dt
    coefficients = [0.7, -1.3, 2.1]
    rng = np.random.default_rng(0)
    signal = np.concatenate(
        [c * local ** 2 + 5.0 * i + rng.normal(0.0, 1e-4, local.shape) for i, c in enumerate(coefficients)]
    )

    plain = segmentedDerivatives(signal, dt, 2, 1000)
    assert [len(level) for level in plain] == [3000, 3000, 3000]
    for i, c in enumerate(coefficients):
        block = slice(i * 1000 + 30, i * 1000 + 970)
        assert np.allclose(plain[1][block], 2.0 * c * local[30:970], atol=1e-2)
        assert np.allclose(plain[2][block], 2.0 * c, atol=1e-2)

    trimmed = segmentedDerivatives(signal, dt, 2, 1000, edgeTrim=0.1)
    assert [len(level) for level in trimmed] == [2400, 2400, 2400]

    with pytest.raises(ValueError):
        segmentedDerivatives(signal, dt, 2, 700)


def _splinePositionColumns(noisyPositions, dt, noFields, steps, segmented):
    if segmented:
        perField = [
            segmentedDerivatives(noisyPositions[:, f], dt, 4, steps, edgeTrim=0.05)
            for f in range(noFields)
        ]
    else:
        perField = [smoothingSplineDerivatives(noisyPositions[:, f], dt, 4) for f in range(noFields)]
    return [np.column_stack([perField[f][level] for f in range(noFields)]) for level in range(5)]


def test_segmented_differentiation_recovers_coupled_chain_from_noisy_positions():
    steps = 9000
    dt, columns = multiFieldGroundTruthColumns(2, 4, coupling=0.3, steps=steps, noTrajectories=5)
    exact = [np.asarray(c, dtype=float) for c in columns]
    rng = np.random.default_rng(303)
    noisyPositions = exact[0] + rng.normal(0.0, 3e-4 * exact[0].std(), exact[0].shape)

    _t, coords, vels = defineCoordinates(2)
    expected = multiFieldPaisUhlenbeckStateLagrangian(2, 2, coupling=0.3)

    def _isNull(columnsForRecovery):
        recovered, _selected = recoverMultiFieldHigherOrderLagrangian(columnsForRecovery, 2, 2)
        difference = multiFieldStateToCoordinates(recovered - expected, 2, 2, coords)
        return isNullLagrangian(difference, coords, vels, order=2)[0]

    assert _isNull(_splinePositionColumns(noisyPositions, dt, 2, steps, segmented=True))
    assert not _isNull(_splinePositionColumns(noisyPositions, dt, 2, steps, segmented=False))


def test_column_perturbation_does_not_break_multi_field_recovery():
    _dt, columns = multiFieldGroundTruthColumns(2, 4, coupling=0.3, steps=9000, noTrajectories=5)
    exact = [np.asarray(c, dtype=float) for c in columns]
    rng = np.random.default_rng(0)
    perturbed = [c + rng.normal(0.0, 1e-3 * c.std(), c.shape) for c in exact]
    recovered, _selected = recoverMultiFieldHigherOrderLagrangian(perturbed, 2, 2)

    _t, coords, vels = defineCoordinates(2)
    expected = multiFieldPaisUhlenbeckStateLagrangian(2, 2, coupling=0.3)
    difference = multiFieldStateToCoordinates(recovered - expected, 2, 2, coords)
    isNull, _residual = isNullLagrangian(difference, coords, vels, order=2)
    assert isNull
