"""Multi-coordinate higher-derivative recovery: a coupled Pais-Uhlenbeck chain."""

import numpy as np
import sympy as sp

from experiments.pu_system import (
    multiFieldGroundTruthColumns,
    multiFieldPaisUhlenbeckStateLagrangian,
)
from finding_L.equivalence_class import isNullLagrangian
from finding_L.higher_order_candidates import buildMultiFieldElMatrix, multiFieldLibrary, stateGridSymbols
from finding_L.higher_order_discovery import (
    multiFieldStateToCoordinates,
    recoverMultiFieldHigherOrderLagrangian,
)
from generation.eqnofmotion import defineCoordinates


def test_multi_field_el_matrix_stacks_one_block_per_field():
    library = multiFieldLibrary(2, 1, 2)
    rows = 50
    data = [np.random.default_rng(k).standard_normal((rows, 2)) for k in range(3)]
    matrix, elExpressions = buildMultiFieldElMatrix(library, 2, 1, data)
    assert matrix.shape == (rows * 2, len(library))
    assert len(elExpressions) == len(library)
    assert len(elExpressions[0]) == 2  # one EL expression per field


def _recovers_coupled_chain(noFields, coupling):
    dt, columns = multiFieldGroundTruthColumns(
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
    recovered, selected, isNull = _recovers_coupled_chain(2, 0.3)
    assert isNull

    grid = stateGridSymbols(2, 2)
    crossTerm = sp.expand(grid[0][0] * grid[1][0])
    coefficient = next(c for monomial, c in selected if sp.expand(monomial - crossTerm) == 0)
    assert abs(coefficient - 2 * 0.3) < 1e-3  # coeff(q0 q1) == 2 * coupling


def test_three_field_coupled_pu_recovered_up_to_null_lagrangian():
    _recovered, _selected, isNull = _recovers_coupled_chain(3, 0.2)
    assert isNull


def test_column_perturbation_does_not_break_multi_field_recovery():
    dt, columns = multiFieldGroundTruthColumns(2, 4, coupling=0.3, steps=9000, noTrajectories=5)
    exact = [np.asarray(c, dtype=float) for c in columns]
    rng = np.random.default_rng(0)
    perturbed = [c + rng.normal(0.0, 1e-3 * c.std(), c.shape) for c in exact]
    recovered, _selected = recoverMultiFieldHigherOrderLagrangian(perturbed, 2, 2)

    _t, coords, vels = defineCoordinates(2)
    expected = multiFieldPaisUhlenbeckStateLagrangian(2, 2, coupling=0.3)
    difference = multiFieldStateToCoordinates(recovered - expected, 2, 2, coords)
    isNull, _residual = isNullLagrangian(difference, coords, vels, order=2)
    assert isNull
