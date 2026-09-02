import sympy as sp
from generation.constraints import DegenerateLagrangianResult, poissonBracket
from generation.eqnofmotion import TIME, defineCoordinates
from generation.ghost_detection import detectGhost
from generation.ostrogradski_hamiltonian import (
    analyzeDegenerateLagrangian,
    ostrogradskiHamiltonian,
)


def test_poisson_bracket_canonical_relations():
    q, p = sp.symbols("q p")
    assert poissonBracket(q, p, [q], [p]) == 1
    assert poissonBracket(p, q, [q], [p]) == -1
    assert poissonBracket(q, q, [q], [p]) == 0
    H = q ** 2 / 2 + p ** 2 / 2
    assert poissonBracket(q, H, [q], [p]) == p


def test_linear_in_highest_velocity_gives_second_class_pair():
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

    assert result.poissonBracketMatrix.rank() == 2

    assert result.canonicalHamiltonian.free_symbols.isdisjoint(set(result.multiplierSymbols))


def test_partial_degeneracy_flags_a_secondary_constraint():
    _t, coords, _vels = defineCoordinates(2)
    q1, q2 = coords
    lagrangian = (
        sp.Rational(1, 2) * sp.diff(q1, TIME) ** 2
        + q1 * sp.diff(q2, TIME)
        - sp.Rational(1, 2) * (q1 ** 2 + q2 ** 2)
    )

    result = analyzeDegenerateLagrangian(lagrangian, coords)
    assert len(result.primaryConstraints) == 1
    assert result.poissonBracketMatrix.rank() == 0
    assert result.secondaryConstraintsExpected
    assert "pending secondary" in result.constraintClass[0]


def test_detect_ghost_reports_degeneracy_rather_than_crashing():
    _t, coords, _vels = defineCoordinates(2)
    q1, q2 = coords
    lagrangian = sp.diff(q1, TIME) * q2 - sp.Rational(1, 2) * q2 ** 2 - sp.Rational(1, 2) * q1 ** 2

    verdict = detectGhost(lagrangian, coords)
    assert verdict["degenerate"] is True
    assert verdict["ghost"] is False
    assert verdict["chainClosed"] is True
    assert verdict["reducedHamiltonian"] is not None
    assert isinstance(verdict["constraintAnalysis"], DegenerateLagrangianResult)
