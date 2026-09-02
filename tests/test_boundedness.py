import sympy as sp
from generation.boundedness import polynomialBoundedBelow
from generation.eqnofmotion import TIME, defineCoordinates
from generation.ghost_detection import detectGhost


def _verdict(expression, positions, momenta):
    return polynomialBoundedBelow(expression, positions, momenta)["verdict"]


def test_separable_quartic_potential_is_bounded_below():
    q, p = sp.symbols("q p")
    hamiltonian = p ** 2 / 2 + q ** 2 / 2 + q ** 4 / 4
    assert _verdict(hamiltonian, [q], [p]) == "bounded_below"


def test_inverted_quartic_is_unbounded_below():
    q, p = sp.symbols("q p")
    hamiltonian = p ** 2 / 2 + q ** 2 / 2 - q ** 4 / 4
    assert _verdict(hamiltonian, [q], [p]) == "unbounded_below"


def test_sextic_potential_is_bounded_below():
    q, p = sp.symbols("q p")
    hamiltonian = p ** 2 / 2 - q ** 2 + q ** 6
    assert _verdict(hamiltonian, [q], [p]) == "bounded_below"


def test_odd_degree_is_unbounded_below():
    q, p = sp.symbols("q p")
    hamiltonian = p ** 2 / 2 + q ** 3
    assert _verdict(hamiltonian, [q], [p]) == "unbounded_below"


def test_linear_momentum_term_is_unbounded_below():
    q0, q1, p0, p1 = sp.symbols("q0 q1 p0 p1")
    hamiltonian = p0 * q1 + p1 ** 2 / 4 - 4 * q0 ** 2 + 5 * q1 ** 2 + q0 ** 2 * q1 ** 2
    assert _verdict(hamiltonian, [q0, q1], [p0, p1]) == "unbounded_below"


def test_indefinite_quadratic_is_unbounded_below():
    q, p = sp.symbols("q p")
    assert _verdict(p ** 2 / 2 - q ** 2 / 2, [q], [p]) == "unbounded_below"


def test_anharmonic_oscillator_ghost_verdict_is_now_resolved():
    _t, coords, _v = defineCoordinates(1)
    q = coords[0]
    velocity = sp.diff(q, TIME)
    anharmonic = sp.Rational(1, 2) * velocity ** 2 - sp.Rational(1, 2) * q ** 2 - sp.Rational(1, 4) * q ** 4
    verdict = detectGhost(anharmonic, coords)
    assert verdict["ghost"] is False
    assert verdict["boundedness"] == "bounded_below"


def test_pais_uhlenbeck_with_spurious_quartic_still_reads_as_ghost():
    _t, coords, _v = defineCoordinates(1)
    q = coords[0]
    velocity = sp.diff(q, TIME)
    acceleration = sp.diff(q, TIME, 2)
    pais_uhlenbeck = sp.Rational(1, 2) * (
        acceleration ** 2 - 5 * velocity ** 2 + 4 * q ** 2
    )
    spurious = pais_uhlenbeck + sp.Rational(1, 10) * q ** 2 * velocity ** 2
    verdict = detectGhost(spurious, coords, order=2)
    assert verdict["ghost"] is True
    assert verdict["boundedness"] == "unbounded_below"
