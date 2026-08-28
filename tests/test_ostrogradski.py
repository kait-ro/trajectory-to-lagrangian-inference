"""Ostrogradski / Euler-Lagrange machinery against closed-form results."""

import pytest
import sympy as sp

from generation.eqnofmotion import TIME, defineCoordinates
from generation.ostrogradski import eulerLagrangeExpression, lagrangianOrder
from generation.ostrogradski_hamiltonian import NonUniqueTopDerivativeError, ostrogradskiHamiltonian

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

    # eulerLagrangeExpression (pipelineSign=False) returns dL/dq - d/dt dL/dqdot
    # = -omega**2 q - qddot, i.e. -(qddot + omega**2 q). The equation of motion
    # qddot + omega**2 q = 0 is recovered up to the overall sign.
    assert sp.simplify(residual + acceleration + OMEGA ** 2 * q) == 0


def test_simple_harmonic_oscillator_ostrogradski_hamiltonian():
    _t, coords, _vels = defineCoordinates(1)
    q = coords[0]
    lagrangian = _sho_lagrangian(q)

    data = ostrogradskiHamiltonian(lagrangian, coords)
    assert data["order"] == 1

    position = data["positionSymbols"][0][0]   # Q0_0  (== q)
    momentum = data["momentumSymbols"][0][0]   # P0_1  (== qdot)

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
    # L nonlinear in qddot -> the top-momentum relation p = qddot + qddot**3 has
    # three roots, so the Legendre transform is multi-valued (item 4).
    _t, coords, _vels = defineCoordinates(1)
    q = coords[0]
    acceleration = sp.diff(q, TIME, 2)
    lagrangian = sp.Rational(1, 2) * acceleration ** 2 + sp.Rational(1, 4) * acceleration ** 4 - sp.Rational(1, 2) * q ** 2

    with pytest.raises(NonUniqueTopDerivativeError) as excinfo:
        ostrogradskiHamiltonian(lagrangian, coords)
    assert len(excinfo.value.branches) > 1
