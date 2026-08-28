from dataclasses import dataclass
from typing import Callable

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
    # Search budgets, NOT tolerances. startingMaxDegree is where the candidate
    # library starts before on-demand expansion; maxRounds caps the number of
    # forward-selection rounds. maxRounds is deliberately large so that one of
    # the stopping conditions (A converged / B stalled / C stagnated), never the
    # round cap, decides whether a recovery succeeds. All actual tolerances live
    # in experiments.discovery.FROZEN_TOLERANCES and are identical for every
    # system -- there are no per-system tolerance fields.
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


ISOTROPIC_QUARTIC = PhysicalSystem(
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


ANHARMONIC_CHAIN = PhysicalSystem(
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
    dt=0.01,
    noSteps=1000,
    noTrajectories=150,
    initialStateScale=1.0,
    startingMaxDegree=2,
)


SYSTEMS = {system.name: system for system in [ISOTROPIC_QUARTIC, ANHARMONIC_CHAIN]}
