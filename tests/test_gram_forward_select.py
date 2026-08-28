import numpy as np

from finding_L.gram_forward_select import fitActiveCoefficientsFromGram
from finding_L.higher_order_discovery import forwardSelectFromGram


def _synthetic_gram(nRows=600, nCandidates=7, kineticIndex=6, trueCoeffs=((0, 2.0), (3, -3.5)), seed=0):
    rng = np.random.default_rng(seed)
    columns = rng.standard_normal((nRows, nCandidates))
    combination = np.zeros(nRows)
    for index, coefficient in trueCoeffs:
        combination += coefficient * columns[:, index]
    columns[:, kineticIndex] = -combination
    gram = columns.T @ columns
    return gram, kineticIndex, dict(trueCoeffs)


def test_forward_selection_recovers_known_sparse_coefficients():
    gram, kineticIndex, trueCoeffs = _synthetic_gram()
    activeIndices, coefficients = forwardSelectFromGram(gram, kineticIndex)

    recovered = dict(zip(activeIndices, (float(c) for c in coefficients)))

    assert set(trueCoeffs).issubset(recovered), f"missing true terms: got {recovered}"
    assert len(activeIndices) <= 3, f"selected spurious terms: {activeIndices}"
    for index, expected in trueCoeffs.items():
        assert abs(recovered[index] - expected) < 1e-6


def test_fit_active_coefficients_matches_direct_solve():
    gram, kineticIndex, _ = _synthetic_gram()
    b = -gram[:, kineticIndex]
    active = [0, 3]
    coefficients = fitActiveCoefficientsFromGram(gram, b, active)
    expected = np.linalg.solve(gram[np.ix_(active, active)], b[active])
    assert np.allclose(coefficients, expected)


def test_empty_active_set_returns_empty():
    gram, kineticIndex, _ = _synthetic_gram()
    b = -gram[:, kineticIndex]
    assert fitActiveCoefficientsFromGram(gram, b, []).size == 0


def test_singular_active_block_warns_and_falls_back_to_lstsq(recwarn):
    gram = np.array([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [0.5, 0.5, 2.0]])
    b = np.array([1.0, 1.0, 0.3])
    coefficients = fitActiveCoefficientsFromGram(gram, b, [0, 1])
    assert coefficients.shape == (2,)
    assert any(issubclass(w.category, RuntimeWarning) for w in recwarn.list)
