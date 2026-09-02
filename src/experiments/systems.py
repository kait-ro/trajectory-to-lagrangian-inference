from collections.abc import Callable
from dataclasses import dataclass

import sympy as sp


def stateSymbols(noCoords):
    positions = [sp.Symbol(f"q{index}") for index in range(noCoords)]
    velocities = [sp.Symbol(f"v{index}") for index in range(noCoords)]
    return positions, velocities


@dataclass
class PhysicalSystem:
    name: str
    noCoords: int
    buildLagrangian: Callable
    expectedScaledLagrangian: Callable
    description: str
    dt: float = 0.01
    noSteps: int = 1000
    noTrajectories: int = 150
    initialStateScale: float = 1.0
    noiseLevels: tuple = (0.0, 0.05, 0.10, 0.25, 0.50)
    startingMaxDegree: int = 2
    maxRounds: int = 150

    def datasetStem(self):
        return f"{self.name}_n{self.noCoords}"


def _isotropicQuarticLagrangian(coords, vels):
    m, k, eps = sp.symbols("m k epsilon")
    rSquared = sum(q ** 2 for q in coords)
    vSquared = sum(v ** 2 for v in vels)
    lagrangian = sp.Rational(1, 2) * m * vSquared - sp.Rational(1, 2) * k * rSquared - sp.Rational(1, 4) * eps * rSquared ** 2
    constants = {m: 1.0, k: 1.0, eps: 0.3}
    return lagrangian, constants


def _isotropicQuarticExpected(noCoords):
    positions, velocities = stateSymbols(noCoords)
    kinetic = sum(v ** 2 for v in velocities)
    rSquared = sum(q ** 2 for q in positions)
    potential = sp.Rational(1, 1) * rSquared + sp.Rational(3, 20) * rSquared ** 2
    return sp.expand(kinetic - potential)


IsotropicQuartic = PhysicalSystem(
    name="isotropic_quartic_calibration",
    noCoords=6,
    buildLagrangian=_isotropicQuarticLagrangian,
    expectedScaledLagrangian=_isotropicQuarticExpected,
    description="Isotropic quartic oscillator L = m/2 v^2 - k/2 r^2 - eps/4 (r^2)^2. Calibration system.",
)


def _anharmonicChainConstants(noCoords):
    stiffnessCycle = [sp.Rational(4, 5), sp.Rational(1, 1), sp.Rational(6, 5), sp.Rational(9, 10), sp.Rational(11, 10), sp.Rational(1, 1)]
    return {
        "stiffness": [stiffnessCycle[index % len(stiffnessCycle)] for index in range(noCoords)],
        "coupling": sp.Rational(1, 10),
        "cubic": sp.Rational(1, 25),
        "quartic": sp.Rational(2, 25),
    }


def _anharmonicChainLagrangian(coords, vels):
    noCoords = len(coords)
    parameters = _anharmonicChainConstants(noCoords)

    kinetic = sp.Rational(1, 2) * sum(v ** 2 for v in vels)
    harmonic = sum(sp.Rational(1, 2) * parameters["stiffness"][index] * coords[index] ** 2 for index in range(noCoords))
    coupling = parameters["coupling"] * sum(coords[index] * coords[index + 1] for index in range(noCoords - 1))
    cubic = parameters["cubic"] * sum(q ** 3 for q in coords)
    quartic = parameters["quartic"] * sum(q ** 4 for q in coords)

    lagrangian = kinetic - harmonic - coupling - cubic - quartic
    return lagrangian, {}


def _anharmonicChainExpected(noCoords):
    positions, velocities = stateSymbols(noCoords)
    parameters = _anharmonicChainConstants(noCoords)

    kinetic = sum(v ** 2 for v in velocities)
    harmonic = sum(parameters["stiffness"][index] * positions[index] ** 2 for index in range(noCoords))
    coupling = 2 * parameters["coupling"] * sum(positions[index] * positions[index + 1] for index in range(noCoords - 1))
    cubic = 2 * parameters["cubic"] * sum(q ** 3 for q in positions)
    quartic = 2 * parameters["quartic"] * sum(q ** 4 for q in positions)

    return sp.expand(kinetic - harmonic - coupling - cubic - quartic)


AnharmonicChain = PhysicalSystem(
    name="anharmonic_chain_blind",
    noCoords=6,
    buildLagrangian=_anharmonicChainLagrangian,
    expectedScaledLagrangian=_anharmonicChainExpected,
    description=(
        "Anisotropic anharmonic chain: per-site stiffness, open nearest-neighbour bilinear coupling, "
        "cubic asymmetry, quartic confinement. Blind holdout - structurally disjoint from the isotropic "
        "(r^2)^2 calibration system (distinct per-coordinate coefficients, sparse coupling, odd-power term, "
        "no quartic cross terms)."
    ),
)


def _coupledQuarticConstants(noCoords):
    stiffnessCycle = [sp.Rational(1, 1), sp.Rational(6, 5), sp.Rational(4, 5), sp.Rational(11, 10)]
    selfQuarticCycle = [sp.Rational(3, 10), sp.Rational(1, 5), sp.Rational(2, 5), sp.Rational(1, 4)]
    return {
        "stiffness": [stiffnessCycle[index % len(stiffnessCycle)] for index in range(noCoords)],
        "selfQuartic": [selfQuarticCycle[index % len(selfQuarticCycle)] for index in range(noCoords)],
        "crossQuartic": sp.Rational(1, 5),
    }


def _coupledQuarticLagrangian(coords, vels):
    noCoords = len(coords)
    parameters = _coupledQuarticConstants(noCoords)

    kinetic = sp.Rational(1, 2) * sum(v ** 2 for v in vels)
    harmonic = sum(sp.Rational(1, 2) * parameters["stiffness"][index] * coords[index] ** 2 for index in range(noCoords))
    selfQuartic = sum(sp.Rational(1, 4) * parameters["selfQuartic"][index] * coords[index] ** 4 for index in range(noCoords))
    crossQuartic = sp.Rational(1, 2) * parameters["crossQuartic"] * coords[0] ** 2 * coords[1] ** 2

    return kinetic - harmonic - selfQuartic - crossQuartic, {}


def _coupledQuarticExpected(noCoords):
    positions, velocities = stateSymbols(noCoords)
    parameters = _coupledQuarticConstants(noCoords)

    kinetic = sum(v ** 2 for v in velocities)
    harmonic = sum(parameters["stiffness"][index] * positions[index] ** 2 for index in range(noCoords))
    selfQuartic = sum(sp.Rational(1, 2) * parameters["selfQuartic"][index] * positions[index] ** 4 for index in range(noCoords))
    crossQuartic = parameters["crossQuartic"] * positions[0] ** 2 * positions[1] ** 2

    return sp.expand(kinetic - harmonic - selfQuartic - crossQuartic)


CoupledQuartic = PhysicalSystem(
    name="coupled_quartic_blind",
    noCoords=4,
    buildLagrangian=_coupledQuarticLagrangian,
    expectedScaledLagrangian=_coupledQuarticExpected,
    description=(
        "Anisotropic quartic oscillator with an explicit off-diagonal quartic coupling "
        "(1/2 lambda q0^2 q1^2): per-site stiffness, per-site quartic confinement, one cross-quartic "
        "term. Blind holdout - the cross-quartic is exactly the term family the greedy selector "
        "hallucinates under noise, here present as ground truth."
    ),
    noiseLevels=(0.0, 0.01, 0.02, 0.05, 0.10),
)


SYSTEMS = {system.name: system for system in [IsotropicQuartic, AnharmonicChain, CoupledQuartic]}
