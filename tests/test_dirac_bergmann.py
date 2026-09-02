import sympy as sp
from generation.constraints import (
    PrimaryConstraint,
    diracBergmannIteration,
    diracBracket,
    diracBracketMatrix,
)
from generation.eqnofmotion import TIME, defineCoordinates
from generation.ghost_detection import detectGhost
from generation.ostrogradski_hamiltonian import analyzeDegenerateLagrangian


def _coords():
    _t, coords, _v = defineCoordinates(2)
    return coords


def test_secondary_constraint_is_computed_and_chain_closes():
    q1, q2 = _coords()
    lagrangian = (
        sp.Rational(1, 2) * sp.diff(q1, TIME) ** 2
        + q1 * sp.diff(q2, TIME)
        - sp.Rational(1, 2) * (q1 ** 2 + q2 ** 2)
    )

    result = analyzeDegenerateLagrangian(lagrangian, _coords())

    assert len(result.primaryConstraints) == 1
    assert len(result.secondaryConstraints) == 1
    assert result.chainClosed
    assert result.constraintGenerations == [1, 2]
    assert result.totalSecondClassCount == 2
    assert result.totalFirstClassCount == 0
    assert result.physicalPhaseSpaceDimension == 2

    secondary = result.secondaryConstraints[0].expression
    positions, momenta = result.positionSymbols, result.momentumSymbols
    assert secondary.free_symbols <= set(positions) | set(momenta)


def test_two_second_class_primaries_build_a_dirac_bracket_matrix():
    q1, q2 = _coords()
    lagrangian = sp.diff(q1, TIME) * q2 - sp.Rational(1, 2) * q2 ** 2 - sp.Rational(1, 2) * q1 ** 2

    result = analyzeDegenerateLagrangian(lagrangian, _coords())

    assert result.secondaryConstraints == []
    assert result.chainClosed
    assert result.diracBracketMatrix is not None
    assert result.diracBracketMatrix.det() != 0
    assert result.allConstraintClasses == ["second-class", "second-class"]


def test_dirac_bracket_removes_the_second_class_directions():
    q, p, x, y = sp.symbols("q p x y")
    secondClass = [p, q]
    matrix = diracBracketMatrix(secondClass, [q, x], [p, y])
    assert matrix.tolist() == [[0, -1], [1, 0]]

    assert diracBracket(q, p, secondClass, [q, x], [p, y]) == 0
    assert diracBracket(x, y, secondClass, [q, x], [p, y]) == 1


def test_degenerate_lagrangian_gets_a_real_ghost_verdict():
    q1, q2 = _coords()
    lagrangian = (
        sp.Rational(1, 2) * sp.diff(q1, TIME) ** 2
        + q1 * sp.diff(q2, TIME)
        - sp.Rational(1, 2) * (q1 ** 2 + q2 ** 2)
    )

    verdict = detectGhost(lagrangian, _coords())

    assert verdict["degenerate"] is True
    assert verdict["ghost"] is False
    assert verdict["chainClosed"] is True
    assert verdict["reducedHamiltonian"] is not None


def test_first_class_constraints_leave_the_ghost_verdict_undetermined():
    q1, q2 = _coords()
    lagrangian = sp.Rational(1, 2) * (sp.diff(q1, TIME) - q2) ** 2

    verdict = detectGhost(lagrangian, _coords())

    assert verdict["degenerate"] is True
    assert verdict["ghost"] is None
    assert verdict["constraintAnalysis"].totalFirstClassCount >= 1


def test_iteration_stops_at_the_round_budget_without_closing():
    q1, q2, p1, p2 = sp.symbols("q1 q2 p1 p2")
    primaries = [PrimaryConstraint(p1, origin="test")]
    hamiltonian = p2 ** 2 / 2 + q1 * q2

    truncated = diracBergmannIteration(
        primaries, hamiltonian, [q1, q2], [p1, p2], maxRounds=3
    )
    assert truncated["chainClosed"] is False
    assert len(truncated["constraints"]) > len(primaries)

    closed = diracBergmannIteration(
        primaries, hamiltonian, [q1, q2], [p1, p2], maxRounds=8
    )
    assert closed["chainClosed"] is True
