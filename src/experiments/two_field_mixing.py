"""Two coupled fields with an explicit mass matrix -- a minimal worked example.

Shows (a) how the eigenvalues of M^-1 K decide whether the normal-mode spectrum
is real (oscillatory) or complex (runaway), and (b) how the single-coordinate
Euler-Lagrange-matrix recovery of `finding_L/higher_order_candidates.py`
generalises to two coupled coordinates -- the off-diagonal mass term is recovered
from data.

This is deliberately a small script, not a framework. The N-field generalisation
lives in `finding_L/higher_order_candidates.py` (see multiField* helpers there).
"""

import itertools

import numpy as np
import sympy as sp

from finding_L.higher_order_candidates import buildMultiFieldElMatrix, multiFieldLibrary, stateGridSymbols
from generation.eqnofmotion import TIME, defineCoordinates
from generation.integrator import GetAccelFunctions, simulateTrajectory

NO_FIELDS = 2
LAGRANGIAN_ORDER = 1  # L = L(q, qdot); EOM is 2nd order


def twoFieldLagrangian(coords, massMatrix, stiffnessMatrix):
    velocities = [sp.diff(c, TIME) for c in coords]
    kinetic = sp.Rational(1, 2) * sum(
        massMatrix[a, b] * velocities[a] * velocities[b] for a in range(2) for b in range(2)
    )
    potential = sp.Rational(1, 2) * sum(
        stiffnessMatrix[a, b] * coords[a] * coords[b] for a in range(2) for b in range(2)
    )
    return sp.expand(kinetic - potential)


def normalModeSpectrum(massMatrix, stiffnessMatrix):
    """omega^2 = eigenvalues(M^-1 K). Real & positive => oscillatory normal mode;
    negative => runaway; complex => spiral instability (mixed damping/growth)."""
    dynamical = sp.Matrix(massMatrix).inv() * sp.Matrix(stiffnessMatrix)
    omegaSquared = [complex(value) for value in dynamical.eigenvals(multiple=True)]
    classification = []
    for value in omegaSquared:
        if abs(value.imag) > 1e-9:
            classification.append("complex (spiral instability)")
        elif value.real < -1e-9:
            classification.append("negative (runaway)")
        else:
            classification.append("positive (oscillatory)")
    spectrumIsReal = all("oscillatory" in item for item in classification)
    return omegaSquared, classification, spectrumIsReal


def simulateTwoField(massMatrix, stiffnessMatrix, dt=0.004, steps=12000, noTrajectories=40, seed=7):
    _t, coords, vels = defineCoordinates(NO_FIELDS)
    lagrangian = twoFieldLagrangian(coords, massMatrix, stiffnessMatrix)
    accelFunctions = GetAccelFunctions(lagrangian, coords, vels, TIME, None)

    rng = np.random.default_rng(seed)
    perLevel = [[], [], []]  # q, qdot, qddot  -- each (rows, NO_FIELDS)
    for _ in range(noTrajectories):
        initialState = rng.uniform(-1.0, 1.0, size=2 * NO_FIELDS)
        _times, q, qdot, qddot = simulateTrajectory(initialState, accelFunctions, dt, steps)
        perLevel[0].append(q)
        perLevel[1].append(qdot)
        perLevel[2].append(qddot)
    return [np.vstack(level) for level in perLevel], lagrangian


def recoverTwoFieldPotential(derivativeData):
    """Recover the (possibly off-diagonal) stiffness matrix from two-field trajectory
    data, with the kinetic term taken as identity (v0^2 + v1^2, fixed).

    The 2-field Euler-Lagrange matrix stacks one row-block per coordinate; the
    position monomials q0, q1 give independent equations across the two blocks, so
    the potential -- including the mixing term q0*q1 -- is fully determined.

    The velocity sector (a non-trivial mass matrix) is NOT recoverable from
    on-shell data alone: on a trajectory q'' is a linear function of q, so the
    Euler-Lagrange column of any q'-quadratic term aliases a q-quadratic one.
    Same mechanism as the noise ceiling in PROJECT.md problem A.
    """
    symbols = stateGridSymbols(NO_FIELDS, LAGRANGIAN_ORDER)
    q0, q1 = symbols[0][0], symbols[1][0]
    v0, v1 = symbols[0][1], symbols[1][1]

    library = [v0 ** 2, v1 ** 2, q0 ** 2, q1 ** 2, sp.expand(q0 * q1)]
    matrix, _elExpressions = buildMultiFieldElMatrix(library, NO_FIELDS, LAGRANGIAN_ORDER, derivativeData)

    kineticColumns = matrix[:, 0] + matrix[:, 1]  # EL(v0^2 + v1^2), identity kinetic
    potentialLibrary = [q0 ** 2, q1 ** 2, sp.expand(q0 * q1)]
    design = matrix[:, 2:5]
    coefficients, *_ = np.linalg.lstsq(design, -kineticColumns, rcond=None)

    return {monomial: float(c) for monomial, c in zip(potentialLibrary, coefficients)}


def run():
    lines = ["Two coupled fields with an explicit mass matrix", ""]

    stiffness = sp.Matrix([[sp.Rational(3, 2), 0], [0, sp.Rational(1, 1)]])
    for label, mixing in [("no mixing (mu=0)", sp.Integer(0)), ("mixing mu=2/5", sp.Rational(2, 5)), ("strong mixing mu=6/5 (M indefinite)", sp.Rational(6, 5))]:
        mass = sp.Matrix([[1, mixing], [mixing, 1]])
        eigenvalues = list(sp.Matrix(mass).eigenvals().keys())
        omegaSquared, classification, real = normalModeSpectrum(mass, stiffness)
        lines.append(f"[{label}]")
        lines.append(f"  M eigenvalues: {[complex(v).real for v in (complex(e) for e in eigenvalues)]}")
        lines.append(f"  omega^2 = eig(M^-1 K): {[round(v.real, 3) + (round(v.imag, 3) * 1j if abs(v.imag) > 1e-9 else 0) for v in omegaSquared]}")
        lines.append(f"  spectrum: {classification}  ->  {'all real & oscillatory' if real else 'instability present'}")
        lines.append("")

    # Recovery of the coupled potential (identity kinetic) from simulated data.
    mass = sp.eye(2)
    stiffness = sp.Matrix([[sp.Rational(5, 4), sp.Rational(1, 5)], [sp.Rational(1, 5), sp.Rational(9, 10)]])
    derivativeData, _trueLagrangian = simulateTwoField(mass, stiffness)
    coefficients = recoverTwoFieldPotential(derivativeData)

    symbols = stateGridSymbols(NO_FIELDS, LAGRANGIAN_ORDER)
    q0, q1 = symbols[0][0], symbols[1][0]
    # L = 1/2(v0^2 + v1^2) - 1/2 q^T K q, scaled so the kinetic coefficient is 1:
    #   coeff(q_a q_b) = -K_ab   (and -2 K_12 on the q0 q1 cross term)
    expected = {
        q0 ** 2: -float(stiffness[0, 0]),
        q1 ** 2: -float(stiffness[1, 1]),
        sp.expand(q0 * q1): -2 * float(stiffness[0, 1]),
    }
    lines.append("[recovery of the coupled potential from data, kinetic = v0^2 + v1^2]")
    lines.append(f"  true off-diagonal stiffness K_12 = {float(stiffness[0, 1]):.3f}  ->  coeff(q0*q1) = {expected[sp.expand(q0 * q1)]:+.3f}")
    allOk = True
    for monomial in [q0 ** 2, q1 ** 2, sp.expand(q0 * q1)]:
        got = coefficients.get(monomial, 0.0)
        want = expected[monomial]
        ok = abs(got - want) < 0.02
        allOk = allOk and ok
        lines.append(f"    {str(monomial):>10}: recovered {got:+.4f}   expected {want:+.4f}   {'ok' if ok else 'MISMATCH'}")
    lines.append(f"  mixing term recovered: {allOk}")
    lines.append("")
    lines.append("Note: the velocity sector (a non-trivial mass matrix) is not recoverable from")
    lines.append("on-shell data -- q'' is a linear function of q on a trajectory, so a q'-quadratic")
    lines.append("EL column aliases a q-quadratic one. Off-manifold data would be needed.")

    return "\n".join(lines)


if __name__ == "__main__":
    print(run())
