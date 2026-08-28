from dataclasses import dataclass

import sympy as sp

from finding_L.report import stateExpressionToFunctional
from generation.eqnofmotion import TIME
from generation.ostrogradski import eulerLagrangeExpression, lagrangianOrder


def _reduceToZero(expression):
    expanded = sp.expand(expression)
    if expanded == 0:
        return sp.Integer(0)
    simplified = sp.simplify(expanded)
    return sp.Integer(0) if simplified == 0 else simplified


def eulerLagrangeResidual(lagrangianFunctional, coords, vels, order=None):
    """Euler-Lagrange residual of a Lagrangian *functional* (q_i(t) form).

    Uses the full Ostrogradski operator sum_k (-1)^k d^k/dt^k dL/dq^(k), so it is
    valid for higher-derivative Lagrangians, not only L(q, q'). `order` defaults
    to the highest time-derivative order appearing in the expression.
    """
    lagrangian = sp.expand(lagrangianFunctional)
    resolvedOrder = lagrangianOrder(lagrangian, list(coords)) if order is None else order
    residual = [
        eulerLagrangeExpression(lagrangian, coordinate, resolvedOrder) for coordinate in coords
    ]
    return [_reduceToZero(component) for component in residual]


def isNullLagrangian(lagrangianFunctional, coords, vels, order=None):
    residual = eulerLagrangeResidual(lagrangianFunctional, coords, vels, order)
    return all(component == 0 for component in residual), residual


def reconstructBoundaryPotential(lagrangianFunctional, coords, vels):
    """Recover F with dL = dF/dt, for a first-order dL affine in the velocities.

    Returns None when dL is higher-order or not a genuine total time derivative;
    the null-Lagrangian test in `isNullLagrangian` is the authoritative check,
    this is only a convenience that names the boundary term when it can.
    """
    expanded = sp.expand(lagrangianFunctional)

    if lagrangianOrder(expanded, list(coords)) > 1:
        return None

    velocityCoefficients = []
    remainder = expanded
    for velocity in vels:
        coefficient = expanded.coeff(velocity, 1)
        velocityCoefficients.append(coefficient)
        remainder = remainder - coefficient * velocity
    remainder = sp.expand(remainder)

    positionProxies = [sp.Symbol(f"_p{index}") for index in range(len(coords))]
    toProxies = {coord: positionProxies[index] for index, coord in enumerate(coords)}
    fromProxies = {positionProxies[index]: coords[index] for index in range(len(coords))}

    allowedSymbols = set(positionProxies) | {TIME}
    coefficientsInProxies = []
    for coefficient in velocityCoefficients:
        inProxies = sp.expand(coefficient.subs(toProxies))
        if not inProxies.free_symbols <= allowedSymbols:
            return None
        coefficientsInProxies.append(inProxies)

    for i in range(len(positionProxies)):
        for j in range(i + 1, len(positionProxies)):
            crossCurl = sp.diff(coefficientsInProxies[i], positionProxies[j]) - sp.diff(coefficientsInProxies[j], positionProxies[i])
            if sp.expand(crossCurl) != 0:
                return None

    potential = sp.Integer(0)
    for index, coefficient in enumerate(coefficientsInProxies):
        gap = sp.expand(coefficient - sp.diff(potential, positionProxies[index]))
        potential = sp.expand(potential + sp.integrate(gap, positionProxies[index]))

    if sp.expand(remainder - sp.diff(potential, TIME)) != 0:
        return None

    return sp.expand(potential.subs(fromProxies))


@dataclass
class EquivalenceVerdict:
    equivalent: bool
    difference: sp.Expr
    eulerLagrangeResidual: list
    boundaryPotential: sp.Expr
    detail: str


def classifyLagrangianPair(lagrangianFunctionalA, lagrangianFunctionalB, coords, vels, order=None):
    """Decide whether two Lagrangian functionals are the same physical theory.

    They are equivalent iff their difference is a null Lagrangian: the
    Euler-Lagrange operator annihilates it identically (not just numerically
    small). A nonzero EL residual means the two produce different equations of
    motion, so treating them as "the same" would reflect loose acceptance
    tolerances rather than a real total-derivative degeneracy.
    """
    difference = sp.expand(lagrangianFunctionalA - lagrangianFunctionalB)

    if difference == 0:
        return EquivalenceVerdict(
            True,
            difference,
            [sp.Integer(0) for _ in coords],
            sp.Integer(0),
            "candidates are identical",
        )

    residualIsNull, residual = isNullLagrangian(difference, coords, vels, order)
    boundaryPotential = reconstructBoundaryPotential(difference, coords, vels) if residualIsNull else None

    if residualIsNull:
        detail = (
            "difference is a null Lagrangian (Euler-Lagrange operator identically zero): "
            "genuine equivalence-class member differing by a total time derivative"
        )
    else:
        detail = (
            "difference produces nonzero equations of motion: the two candidates are physically distinct, "
            "so acceptance of both reflects tolerances that are too loose, not a real degeneracy"
        )

    return EquivalenceVerdict(residualIsNull, difference, residual, boundaryPotential, detail)


def classifyDiscoveredPair(discoveredA, discoveredB, coords, vels, order=None):
    functionalA = stateExpressionToFunctional(discoveredA.expression, coords, vels)
    functionalB = stateExpressionToFunctional(discoveredB.expression, coords, vels)
    return classifyLagrangianPair(functionalA, functionalB, coords, vels, order)


def formatVerdict(verdict):
    lines = [
        f"Equivalent: {verdict.equivalent}",
        f"Candidate difference dL = {verdict.difference}",
        f"Euler-Lagrange residual of dL: {verdict.eulerLagrangeResidual}",
    ]
    if verdict.boundaryPotential is not None:
        lines.append(f"dL = d/dt F  with  F = {verdict.boundaryPotential}")
    lines.append(verdict.detail)
    return "\n".join(lines)


def _demo():
    from generation.eqnofmotion import defineCoordinates

    noCoords = 4
    t, coords, vels = defineCoordinates(noCoords)

    baseLagrangian = sp.Rational(1, 2) * sum(v ** 2 for v in vels) - sp.Rational(1, 2) * sum(q ** 2 for q in coords)

    nullAddition = sp.Rational(3, 10) * (coords[0] * vels[0] + coords[1] * vels[1])
    genuinePair = classifyLagrangianPair(baseLagrangian + nullAddition, baseLagrangian, coords, vels)
    print("=== constructed total-derivative difference ===")
    print(formatVerdict(genuinePair))
    print()

    physicalAddition = sp.Rational(1, 5) * coords[0] ** 2 + sp.Rational(1, 10) * coords[0] * coords[1]
    slopPair = classifyLagrangianPair(baseLagrangian + physicalAddition, baseLagrangian, coords, vels)
    print("=== constructed genuine (EOM-changing) difference ===")
    print(formatVerdict(slopPair))


if __name__ == "__main__":
    _demo()
