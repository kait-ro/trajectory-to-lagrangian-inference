import itertools
from dataclasses import dataclass

import sympy as sp
from generation.eqnofmotion import TIME, EulerLagrangeEqn
from generation.ostrogradski import eulerLagrangeExpression, lagrangianOrder


def _reduceToZero(expression):
    expanded = sp.expand(expression)
    if expanded == 0:
        return sp.Integer(0)
    simplified = sp.simplify(expanded)
    return sp.Integer(0) if simplified == 0 else simplified


def eulerLagrangeResidual(lagrangianFunctional, coords, vels, order=None):
    lagrangian = sp.expand(lagrangianFunctional)
    resolvedOrder = lagrangianOrder(lagrangian, list(coords)) if order is None else order

    if resolvedOrder <= 1:
        residual = list(EulerLagrangeEqn(lagrangian, list(coords), list(vels)))
    else:
        residual = [
            eulerLagrangeExpression(lagrangian, coordinate, resolvedOrder) for coordinate in coords
        ]
    return [_reduceToZero(component) for component in residual]


def isNullLagrangian(lagrangianFunctional, coords, vels, order=None):
    residual = eulerLagrangeResidual(lagrangianFunctional, coords, vels, order)
    return all(component == 0 for component in residual), residual


def reconstructBoundaryPotential(lagrangianFunctional, coords, vels):
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


VERDICT_SCALE = "equivalent-by-scale"
VERDICT_TOTAL_DERIVATIVE = "equivalent-by-total-derivative"
VERDICT_SCALE_AND_TOTAL_DERIVATIVE = "equivalent-by-scale-and-total-derivative"
VERDICT_NOT_EQUIVALENT = "not-equivalent"


@dataclass
class EquivalenceClassResult:
    verdict: str
    scale: sp.Expr
    boundaryFunction: sp.Expr
    eulerLagrangeResidual: list
    detail: str


def _jetVariables(expressions, coords):
    coordSet = set(coords)
    variables = set()
    for expression in expressions:
        expression = sp.sympify(expression)
        for function in expression.atoms(sp.core.function.AppliedUndef):
            if function in coordSet:
                variables.add(function)
        for derivative in expression.atoms(sp.Derivative):
            if derivative.expr in coordSet:
                variables.add(derivative)
    return sorted(variables, key=sp.default_sort_key)


def _polynomialDegree(expression, generators):
    if not generators:
        return 0
    try:
        return sp.Poly(sp.expand(expression), *generators).total_degree()
    except (sp.PolynomialError, sp.GeneratorsNeeded):
        return 0


def _monomialBasis(variables, maxDegree):
    basis = []
    seen = set()
    for totalDegree in range(1, maxDegree + 1):
        for combination in itertools.combinations_with_replacement(range(len(variables)), totalDegree):
            key = tuple(sorted(combination))
            if key in seen:
                continue
            seen.add(key)
            monomial = sp.Integer(1)
            for index in combination:
                monomial = monomial * variables[index]
            basis.append(sp.expand(monomial))
    return basis


def _notEquivalent(detail):
    return EquivalenceClassResult(VERDICT_NOT_EQUIVALENT, None, None, None, detail)


def verifyEquivalenceClass(L1, L2, coords, vels, t=TIME, order=None):
    lagrangian1 = sp.expand(sp.sympify(L1))
    lagrangian2 = sp.expand(sp.sympify(L2))
    coordList = list(coords)

    if sp.expand(lagrangian2 - lagrangian1) == 0:
        return EquivalenceClassResult(
            VERDICT_SCALE,
            sp.Integer(1),
            sp.Integer(0),
            [sp.Integer(0) for _ in coordList],
            "candidates are identical (scale factor 1, no boundary term)",
        )

    differenceOrder = lagrangianOrder(lagrangian1 + lagrangian2, coordList) if order is None else order

    boundaryVariables = []
    for coordinate in coordList:
        for k in range(max(differenceOrder, 1)):
            boundaryVariables.append(coordinate if k == 0 else sp.diff(coordinate, t, k))

    degreeJet = _jetVariables([lagrangian1, lagrangian2], coordList)
    degreeSubstitution = {variable: sp.Symbol(f"_d{index}") for index, variable in enumerate(degreeJet)}
    degreeSymbols = list(degreeSubstitution.values())
    ansatzDegree = max(
        _polynomialDegree(lagrangian1.subs(degreeSubstitution), degreeSymbols),
        _polynomialDegree(lagrangian2.subs(degreeSubstitution), degreeSymbols),
        1,
    )

    monomials = _monomialBasis(boundaryVariables, ansatzDegree)
    coefficientSymbols = list(sp.symbols(f"_a0:{len(monomials)}")) if monomials else []
    scaleSymbol = sp.Symbol("_c")

    boundaryFunction = sp.Integer(0)
    for symbol, monomial in zip(coefficientSymbols, monomials):
        boundaryFunction = boundaryFunction + symbol * monomial
    residual = sp.expand(lagrangian2 - scaleSymbol * lagrangian1 - sp.diff(boundaryFunction, t))

    unknowns = [scaleSymbol] + coefficientSymbols
    jetVariables = _jetVariables([residual], coordList)
    substitution = {variable: sp.Symbol(f"_j{index}") for index, variable in enumerate(jetVariables)}
    if t in residual.free_symbols:
        substitution[t] = sp.Symbol("_jt")
    generators = list(substitution.values())
    polynomialResidual = sp.expand(residual.subs(substitution))

    try:
        equations = sp.Poly(polynomialResidual, *generators).coeffs() if generators else [polynomialResidual]
    except (sp.PolynomialError, sp.GeneratorsNeeded):
        return _notEquivalent("candidate difference is not polynomial in the jet variables")

    try:
        solutionSet = sp.linsolve(equations, unknowns)
    except sp.SympifyError:
        return _notEquivalent("candidate difference does not reduce to a linear system for c and F")

    if not solutionSet:
        return _notEquivalent(
            "no constant scale factor and boundary function reproduce the candidate difference"
        )

    solution = next(iter(solutionSet))
    freeParameters = set()
    for value in solution:
        freeParameters |= value.free_symbols & set(unknowns)
    zeroMap = {parameter: (sp.Integer(1) if parameter == scaleSymbol else sp.Integer(0)) for parameter in freeParameters}
    resolved = {symbol: sp.expand(value.subs(zeroMap)) for symbol, value in zip(unknowns, solution)}

    scaleValue = sp.nsimplify(resolved[scaleSymbol]) if resolved[scaleSymbol].is_number else resolved[scaleSymbol]
    if scaleValue == 0:
        return _notEquivalent("the only linear solution sets the scale factor to zero")

    resolvedBoundary = sp.Integer(0)
    for symbol, monomial in zip(coefficientSymbols, monomials):
        resolvedBoundary = resolvedBoundary + resolved[symbol] * monomial
    resolvedBoundary = sp.expand(resolvedBoundary)

    crossCheck = eulerLagrangeResidual(sp.expand(lagrangian2 - scaleValue * lagrangian1), coordList, vels, order)
    if any(component != 0 for component in crossCheck):
        return _notEquivalent("Euler-Lagrange cross-check on L2 - c*L1 did not vanish identically")

    boundaryIsZero = resolvedBoundary == 0
    scaleIsOne = sp.simplify(scaleValue - 1) == 0

    if boundaryIsZero:
        return EquivalenceClassResult(
            VERDICT_SCALE,
            scaleValue,
            sp.Integer(0),
            crossCheck,
            f"L2 = c*L1 with c = {scaleValue}; no boundary term required",
        )
    if scaleIsOne:
        return EquivalenceClassResult(
            VERDICT_TOTAL_DERIVATIVE,
            sp.Integer(1),
            resolvedBoundary,
            crossCheck,
            f"L2 = L1 + dF/dt with F = {resolvedBoundary}",
        )
    return EquivalenceClassResult(
        VERDICT_SCALE_AND_TOTAL_DERIVATIVE,
        scaleValue,
        resolvedBoundary,
        crossCheck,
        f"L2 = c*L1 + dF/dt with c = {scaleValue}, F = {resolvedBoundary}",
    )


@dataclass
class EquivalenceVerdict:
    equivalent: bool
    difference: sp.Expr
    eulerLagrangeResidual: list
    boundaryPotential: sp.Expr
    detail: str
    scale: sp.Expr = None


def classifyLagrangianPair(lagrangianFunctionalA, lagrangianFunctionalB, coords, vels, order=None):
    difference = sp.expand(lagrangianFunctionalA - lagrangianFunctionalB)

    if difference == 0:
        return EquivalenceVerdict(
            True,
            difference,
            [sp.Integer(0) for _ in coords],
            sp.Integer(0),
            "candidates are identical",
            sp.Integer(1),
        )

    residualIsNull, residual = isNullLagrangian(difference, coords, vels, order)

    if residualIsNull:
        boundaryPotential = reconstructBoundaryPotential(difference, coords, vels)
        detail = (
            "difference is a null Lagrangian (Euler-Lagrange operator identically zero): "
            "genuine equivalence-class member differing by a total time derivative"
        )
        return EquivalenceVerdict(True, difference, residual, boundaryPotential, detail, sp.Integer(1))

    classResult = verifyEquivalenceClass(
        lagrangianFunctionalB, lagrangianFunctionalA, coords, vels, TIME, order
    )
    if classResult.verdict != VERDICT_NOT_EQUIVALENT:
        detail = (
            f"candidates are equivalent under the action symmetry ({classResult.verdict}): {classResult.detail}"
        )
        boundaryPotential = classResult.boundaryFunction if classResult.boundaryFunction != 0 else None
        return EquivalenceVerdict(
            True,
            difference,
            classResult.eulerLagrangeResidual,
            boundaryPotential,
            detail,
            classResult.scale,
        )

    detail = (
        "difference produces nonzero equations of motion and no constant rescaling turns it into a total "
        "time derivative: the two candidates are physically distinct, so acceptance of both reflects "
        "tolerances that are too loose, not a real degeneracy"
    )
    return EquivalenceVerdict(False, difference, residual, None, detail)


def formatVerdict(verdict):
    scaled = verdict.scale is not None and sp.simplify(verdict.scale - 1) != 0
    residualLabel = "L2 - c*L1" if scaled else "dL"
    lines = [
        f"Equivalent: {verdict.equivalent}",
        f"Candidate difference dL = {verdict.difference}",
        f"Euler-Lagrange residual of {residualLabel}: {verdict.eulerLagrangeResidual}",
    ]
    if scaled:
        lines.append(f"scale factor c = {verdict.scale}")
    if verdict.boundaryPotential is not None:
        lines.append(f"dL = d/dt F  with  F = {verdict.boundaryPotential}")
    lines.append(verdict.detail)
    return "\n".join(lines)


def _demo():
    from generation.eqnofmotion import defineCoordinates

    noCoords = 4
    _t, coords, vels = defineCoordinates(noCoords)

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
    print()

    scaledPair = classifyLagrangianPair(
        sp.Rational(5, 2) * baseLagrangian + nullAddition, baseLagrangian, coords, vels
    )
    print("=== constructed scale factor + total-derivative difference ===")
    print(formatVerdict(scaledPair))


if __name__ == "__main__":
    _demo()
