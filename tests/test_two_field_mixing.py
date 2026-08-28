import sympy as sp

from experiments.two_field_mixing import normalModeSpectrum, recoverTwoFieldPotential, simulateTwoField
from finding_L.higher_order_candidates import stateGridSymbols


def test_mass_matrix_mixing_drives_the_spectrum_complex():
    stiffness = sp.Matrix([[sp.Rational(3, 2), 0], [0, 1]])

    _v, _c, real = normalModeSpectrum(sp.eye(2), stiffness)
    assert real

    strongMix = sp.Matrix([[1, sp.Rational(6, 5)], [sp.Rational(6, 5), 1]])
    assert min(complex(e).real for e in strongMix.eigenvals()) < 0
    _v, _c, realStrong = normalModeSpectrum(strongMix, stiffness)
    assert not realStrong


def test_two_field_el_matrix_recovers_the_off_diagonal_potential():
    stiffness = sp.Matrix([[sp.Rational(5, 4), sp.Rational(1, 5)], [sp.Rational(1, 5), sp.Rational(9, 10)]])
    derivativeData, _lagrangian = simulateTwoField(sp.eye(2), stiffness, steps=8000, noTrajectories=25)
    coefficients = recoverTwoFieldPotential(derivativeData)

    symbols = stateGridSymbols(2, 1)
    q0, q1 = symbols[0][0], symbols[1][0]

    assert abs(coefficients[q0 ** 2] - (-float(stiffness[0, 0]))) < 0.02
    assert abs(coefficients[q1 ** 2] - (-float(stiffness[1, 1]))) < 0.02
    assert abs(coefficients[sp.expand(q0 * q1)] - (-2 * float(stiffness[0, 1]))) < 0.02
