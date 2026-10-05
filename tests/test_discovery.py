import sympy as sp
from experiments.discovery import _inferNoCoords


def test_infers_no_coords_from_state_symbols():
    q0, v0, q1, v1, q3 = sp.symbols("q0 v0 q1 v1 q3")
    assert _inferNoCoords(q0 ** 2 + v0 * q1) == 2
    assert _inferNoCoords(q3 ** 2) == 4


def test_infers_one_field_from_bare_higher_derivative_state_symbols():
    s0, s1, s2 = sp.symbols("s0 s1 s2")
    assert _inferNoCoords(s2 ** 2 - s1 ** 2 + s0 ** 2) == 1


def test_infers_field_count_from_multi_field_grid_symbols():
    s0_0, s0_1, s1_0, s1_1 = sp.symbols("s0_0 s0_1 s1_0 s1_1")
    assert _inferNoCoords(s0_0 * s1_0 + s0_1 ** 2) == 2
    assert _inferNoCoords(s0_0 ** 2) == 1


def test_infers_zero_from_an_expression_with_no_state_symbols():
    m, k = sp.symbols("m k")
    assert _inferNoCoords(m * k) == 0
