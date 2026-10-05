import pytest
import sympy as sp
from generation.eqnofmotion import TIME, defineCoordinates
from generation.ostrogradski import (
    canonicalizeLagrangian,
    eulerLagrangeExpression,
    isNullLagrangianLocal,
    lagrangianOrder,
)
from generation.ostrogradski_hamiltonian import (
    NonUniqueTopDerivativeError,
    ostrogradskiHamiltonian,
)

OMEGA = sp.Symbol("omega", positive=True)


def _sho_lagrangian(coordinate):
    velocity = sp.diff(coordinate, TIME)
    return sp.Rational(1, 2) * velocity ** 2 - sp.Rational(1, 2) * OMEGA ** 2 * coordinate ** 2


def test_simple_harmonic_oscillator_euler_lagrange():
    _t, coords, _vels = defineCoordinates(1)
    q = coords[0]
    lagrangian = _sho_lagrangian(q)

    assert lagrangianOrder(lagrangian, coords) == 1

    residual = eulerLagrangeExpression(lagrangian, q, order=1)
    acceleration = sp.diff(q, TIME, 2)

    assert sp.simplify(residual + acceleration + OMEGA ** 2 * q) == 0


def test_simple_harmonic_oscillator_ostrogradski_hamiltonian():
    _t, coords, _vels = defineCoordinates(1)
    q = coords[0]
    lagrangian = _sho_lagrangian(q)

    data = ostrogradskiHamiltonian(lagrangian, coords)
    assert data["order"] == 1

    position = data["positionSymbols"][0][0]
    momentum = data["momentumSymbols"][0][0]

    expected = sp.Rational(1, 2) * momentum ** 2 + sp.Rational(1, 2) * OMEGA ** 2 * position ** 2
    assert sp.expand(data["hamiltonian"] - expected) == 0


def test_free_particle_hamiltonian_is_kinetic_only():
    _t, coords, _vels = defineCoordinates(1)
    q = coords[0]
    lagrangian = sp.Rational(1, 2) * sp.diff(q, TIME) ** 2

    data = ostrogradskiHamiltonian(lagrangian, coords)
    momentum = data["momentumSymbols"][0][0]
    assert sp.expand(data["hamiltonian"] - sp.Rational(1, 2) * momentum ** 2) == 0


def test_nonlinear_top_derivative_raises_rather_than_guessing_a_branch():
    _t, coords, _vels = defineCoordinates(1)
    q = coords[0]
    acceleration = sp.diff(q, TIME, 2)
    lagrangian = sp.Rational(1, 2) * acceleration ** 2 + sp.Rational(1, 4) * acceleration ** 4 - sp.Rational(1, 2) * q ** 2

    with pytest.raises(NonUniqueTopDerivativeError) as excinfo:
        ostrogradskiHamiltonian(lagrangian, coords)
    assert len(excinfo.value.branches) > 1


def test_canonicalize_reduces_even_gap_cross_term_to_signed_diagonal():
    _t, coords, _vels = defineCoordinates(1)
    q = coords[0]
    qdot = sp.diff(q, TIME)
    qddot = sp.diff(q, TIME, 2)

    canonical = canonicalizeLagrangian(q * qddot, coords, 2)
    assert sp.expand(canonical - (-(qdot ** 2))) == 0


def test_canonicalize_two_step_even_gap_flips_sign_back_positive():
    _t, coords, _vels = defineCoordinates(1)
    q = coords[0]
    qddot = sp.diff(q, TIME, 2)
    q4 = sp.diff(q, TIME, 4)

    canonical = canonicalizeLagrangian(q * q4, coords, 4)
    assert sp.expand(canonical - qddot ** 2) == 0


def test_canonicalize_odd_gap_terms_vanish_entirely():
    _t, coords, _vels = defineCoordinates(1)
    q = coords[0]
    qdot = sp.diff(q, TIME)
    qddot = sp.diff(q, TIME, 2)
    q3 = sp.diff(q, TIME, 3)

    assert canonicalizeLagrangian(q * qdot, coords, 1) == 0
    assert canonicalizeLagrangian(qdot * qddot, coords, 2) == 0
    assert canonicalizeLagrangian(q * q3, coords, 3) == 0


def test_canonicalize_leaves_already_diagonal_lagrangian_unchanged():
    _t, coords, _vels = defineCoordinates(1)
    q = coords[0]
    qdot = sp.diff(q, TIME)
    lagrangian = qdot ** 2 + sp.Integer(3) * q ** 2

    assert sp.expand(canonicalizeLagrangian(lagrangian, coords, 2) - lagrangian) == 0


def test_canonicalize_leaves_cross_field_terms_untouched():
    _t, coords, _vels = defineCoordinates(2)
    q0, q1 = coords
    coupling = q0 * sp.diff(q1, TIME, 2)

    assert sp.expand(canonicalizeLagrangian(coupling, coords, 2) - coupling) == 0


def test_canonicalize_reduction_is_verified_null_by_construction():
    _t, coords, _vels = defineCoordinates(1)
    q = coords[0]
    lagrangian = sp.Integer(5) * q * sp.diff(q, TIME, 2)

    canonical = canonicalizeLagrangian(lagrangian, coords, 2)
    assert isNullLagrangianLocal(canonical - lagrangian, coords, 2) is True
