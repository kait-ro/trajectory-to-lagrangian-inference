import numpy as np

from finding_L.higher_order_discovery import forwardSelectFromGram
from finding_L.regularized_select import lassoSelect, sequentialThresholdedLeastSquares


def _synthetic(nRows=800, nCols=10, kineticIndex=9, trueCoeffs=((0, 2.0), (3, -3.5), (6, 1.25)), noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    columns = rng.standard_normal((nRows, nCols))
    combination = sum(coefficient * columns[:, index] for index, coefficient in trueCoeffs)
    columns[:, kineticIndex] = -combination + rng.normal(0.0, noise, nRows)
    gram = columns.T @ columns
    return gram, kineticIndex, dict(trueCoeffs)


def test_stlsq_recovers_known_sparse_coefficients():
    gram, kineticIndex, trueCoeffs = _synthetic()
    b = -gram[:, kineticIndex]
    active, coefficients = sequentialThresholdedLeastSquares(gram, b, kineticIndex, relativeThreshold=1e-2)
    recovered = dict(zip(active, (float(c) for c in coefficients)))
    assert set(active) == set(trueCoeffs)
    for index, expected in trueCoeffs.items():
        assert abs(recovered[index] - expected) < 1e-6


def test_lasso_recovers_known_sparse_coefficients():
    gram, kineticIndex, trueCoeffs = _synthetic()
    b = -gram[:, kineticIndex]
    active, coefficients = lassoSelect(gram, b, kineticIndex, relativeThreshold=1e-2)
    recovered = dict(zip(active, (float(c) for c in coefficients)))
    assert set(trueCoeffs).issubset(set(active))
    for index, expected in trueCoeffs.items():
        assert abs(recovered[index] - expected) < 1e-3


def test_all_three_selectors_agree_on_clean_data():
    gram, kineticIndex, trueCoeffs = _synthetic()
    b = -gram[:, kineticIndex]
    greedy, _c = forwardSelectFromGram(gram, kineticIndex)
    stlsq, _c = sequentialThresholdedLeastSquares(gram, b, kineticIndex)
    lasso, _c = lassoSelect(gram, b, kineticIndex)
    assert set(greedy) == set(trueCoeffs)
    assert set(stlsq) == set(trueCoeffs)
    assert set(trueCoeffs).issubset(set(lasso))


def test_stlsq_is_more_robust_than_greedy_with_a_correlated_distractor():
    rng = np.random.default_rng(1)
    n = 1500
    columns = rng.standard_normal((n, 6))
    columns[:, 4] = columns[:, 0] + rng.normal(0.0, 0.05, n)
    columns[:, 5] = -(1.7 * columns[:, 0] + 0.9 * columns[:, 2]) + rng.normal(0.0, 0.02, n)
    gram = columns.T @ columns
    b = -gram[:, 5]

    stlsqActive, _c = sequentialThresholdedLeastSquares(gram, b, 5, relativeThreshold=1e-2)
    assert 4 not in stlsqActive
    assert {0, 2}.issubset(set(stlsqActive))
