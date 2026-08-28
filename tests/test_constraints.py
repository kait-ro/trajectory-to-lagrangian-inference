"""Degenerate-Lagrangian constraint detection and first/second-class classification."""

import sympy as sp

from generation.constraints import DegenerateLagrangianResult, poissonBracket
from generation.eqnofmotion import TIME, defineCoordinates
from generation.ghost_detection import detectGhost
from generation.ostrogradski_hamiltonian import analyzeDegenerateLagrangian, ostrogradskiHamiltonian


def test_poisson_bracket_canonical_relations():
    q, p = sp.symbols("q p")
    assert poissonBracket(q, p, [q], [p]) == 1
    assert poissonBracket(p, q, [q], [p]) == -1
    assert poissonBracket(q, q, [q], [p]) == 0
    # {q^2/2 + p^2/2, ...} of itself is zero; with q it gives -p
    H = q ** 2 / 2 + p ** 2 / 2
    assert poissonBracket(q, H, [q], [p]) == p


def test_linear_in_highest_velocity_gives_second_class_pair():
    # L = q1' q2 - q2^2/2 - q1^2/2 : first order, linear in q1', independent of q2'
    _t, coords, _vels = defineCoordinates(2)
    q1, q2 = coords
    lagrangian = sp.diff(q1, TIME) * q2 - sp.Rational(1, 2) * q2 ** 2 - sp.Rational(1, 2) * q1 ** 2

    result = ostrogradskiHamiltonian(lagrangian, coords)
    assert isinstance(result, DegenerateLagrangianResult)
    assert result.degenerate

    assert len(result.primaryConstraints) == 2
    assert result.secondClassCount == 2
    assert result.firstClassCount == 0
    assert not result.secondaryConstraintsExpected

    # bracket matrix is the canonical symplectic 2x2 => full rank => second-class
    assert result.poissonBracketMatrix.rank() == 2

    # H_c = q1^2/2 + q2^2/2 in canonical symbols, with the undetermined velocities dropped
    assert result.canonicalHamiltonian.free_symbols.isdisjoint(set(result.multiplierSymbols))


def test_partial_degeneracy_flags_a_secondary_constraint():
    # L = q1'^2/2 + q1 q2' - (q1^2 + q2^2)/2 : q2' does not appear with a q2' kinetic term
    _t, coords, _vels = defineCoordinates(2)
    q1, q2 = coords
    lagrangian = (
        sp.Rational(1, 2) * sp.diff(q1, TIME) ** 2
        + q1 * sp.diff(q2, TIME)
        - sp.Rational(1, 2) * (q1 ** 2 + q2 ** 2)
    )

    result = analyzeDegenerateLagrangian(lagrangian, coords)
    assert len(result.primaryConstraints) == 1
    # the single constraint p2 - q1 Poisson-commutes with itself, but not with H
    assert result.poissonBracketMatrix.rank() == 0
    assert result.secondaryConstraintsExpected
    assert "pending secondary" in result.constraintClass[0]


def test_detect_ghost_reports_degeneracy_rather_than_crashing():
    _t, coords, _vels = defineCoordinates(2)
    q1, q2 = coords
    lagrangian = sp.diff(q1, TIME) * q2 - sp.Rational(1, 2) * q2 ** 2 - sp.Rational(1, 2) * q1 ** 2

    verdict = detectGhost(lagrangian, coords)
    assert verdict["ghost"] is None
    assert verdict["degenerate"] is True
    assert isinstance(verdict["constraintAnalysis"], DegenerateLagrangianResult)
