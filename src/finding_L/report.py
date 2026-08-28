from dataclasses import dataclass, field

import sympy as sp


def buildStateSymbolMap(coords, vels):
    coordinateSymbols = {coordinate: sp.Symbol(f"q{index}") for index, coordinate in enumerate(coords)}
    velocitySymbols = {velocity: sp.Symbol(f"v{index}") for index, velocity in enumerate(list(vels))}
    return {**velocitySymbols, **coordinateSymbols}


def buildInverseStateSymbolMap(coords, vels):
    inverse = {sp.Symbol(f"q{index}"): coordinate for index, coordinate in enumerate(coords)}
    inverse.update({sp.Symbol(f"v{index}"): velocity for index, velocity in enumerate(list(vels))})
    return inverse


def toStateExpression(expression, stateSymbolMap):
    substituted = sp.sympify(expression).subs(stateSymbolMap, simultaneous=True)
    return sp.expand(substituted)


def stateExpressionToFunctional(expression, coords, vels):
    return sp.expand(sp.sympify(expression).subs(buildInverseStateSymbolMap(coords, vels), simultaneous=True))


CLEAN_DENOMINATORS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 32, 40)


def snapCoefficient(value, relativeTolerance=0.01, zeroThreshold=1e-7):
    numericValue = float(value)
    if abs(numericValue) <= zeroThreshold:
        return sp.Integer(0)

    tolerance = max(relativeTolerance * abs(numericValue), zeroThreshold)
    for denominator in CLEAN_DENOMINATORS:
        numerator = round(numericValue * denominator)
        if numerator == 0:
            continue
        candidate = sp.Rational(numerator, denominator)
        if abs(float(candidate) - numericValue) <= tolerance:
            return candidate

    return sp.Float(round(numericValue, 6))


def expressionGenerators(*expressions):
    generators = set()
    for expression in expressions:
        generators |= sp.sympify(expression).free_symbols
    return sorted(generators, key=lambda symbol: symbol.name)


def monomialDegree(monomial, generators):
    monomial = sp.sympify(monomial)
    if monomial.is_Number or not generators:
        return 0
    try:
        polynomial = sp.Poly(monomial, *generators)
    except sp.PolynomialError:
        return len(generators) + 1
    return sum(polynomial.degree_list())


def orderedAdditiveTerms(expression, generators):
    terms = sp.Add.make_args(sp.expand(expression))
    return sorted(
        terms,
        key=lambda term: (monomialDegree(term.as_coeff_Mul()[1], generators), sp.default_sort_key(term)),
    )


def formatAdditiveExpression(expression, generators):
    terms = orderedAdditiveTerms(expression, generators)
    if not terms:
        return "0"
    rendered = str(terms[0])
    for term in terms[1:]:
        text = str(term)
        if text.startswith("-"):
            rendered += f" - {text[1:]}"
        else:
            rendered += f" + {text}"
    return rendered


NICE_DECIMAL_DENOMINATORS = {1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 100}


def formatCoefficientValue(value):
    if value.is_Integer:
        return str(int(value))
    if value.is_Rational and value.q in NICE_DECIMAL_DENOMINATORS:
        return f"{float(value):g}"
    if value.is_Rational:
        return f"{value.p}/{value.q}"
    return f"{float(value):g}"


def formatMagnitude(value):
    return formatCoefficientValue(abs(value))


def formatSignedValue(value):
    sign = "-" if value < 0 else "+"
    return f"{sign}{formatCoefficientValue(abs(value))}"


@dataclass
class TermContribution:
    monomial: sp.Expr
    rawCoefficient: float
    snappedCoefficient: sp.Expr


@dataclass
class DiscoveredLagrangian:
    expression: sp.Expr
    rawExpression: sp.Expr
    kineticTerm: sp.Expr
    contributions: list = field(default_factory=list)
    text: str = ""

    @property
    def activeContributions(self):
        return [contribution for contribution in self.contributions if contribution.snappedCoefficient != 0]

    def functionalExpression(self, coords, vels):
        return stateExpressionToFunctional(self.expression, coords, vels)


def groupContributionsByCoefficient(contributions, generators):
    grouped = {}
    for contribution in contributions:
        if contribution.snappedCoefficient == 0:
            continue
        degree = monomialDegree(contribution.monomial, generators)
        grouped.setdefault((degree, contribution.snappedCoefficient), []).append(contribution.monomial)
    return grouped


def formatLagrangian(kineticState, snappedExpression, contributions):
    generators = expressionGenerators(snappedExpression, kineticState)

    lines = ["Discovered Lagrangian (kinetic coefficient fixed at 1):"]
    lines.append(f"  L = {formatAdditiveExpression(kineticState, generators)}")

    grouped = groupContributionsByCoefficient(contributions, generators)
    for degree, coefficient in sorted(grouped, key=lambda key: (key[0], float(key[1]))):
        monomials = sorted(grouped[(degree, coefficient)], key=sp.default_sort_key)
        body = formatAdditiveExpression(sp.Add(*monomials), generators)
        sign = "-" if coefficient < 0 else "+"
        magnitude = formatMagnitude(abs(coefficient))
        wrapped = f"({body})" if len(monomials) > 1 else body
        lines.append(f"    {sign} {magnitude} * {wrapped}")

    dropped = [contribution for contribution in contributions if contribution.snappedCoefficient == 0]
    if dropped:
        lines.append("")
        lines.append("  Dropped (coefficient snapped to 0):")
        for contribution in dropped:
            lines.append(f"    {contribution.monomial}  (raw {contribution.rawCoefficient:+.2e})")

    lines.append("")
    lines.append("  Coefficient snapping (raw -> clean):")
    orderedContributions = sorted(
        contributions,
        key=lambda contribution: (monomialDegree(contribution.monomial, generators), sp.default_sort_key(contribution.monomial)),
    )
    for contribution in orderedContributions:
        lines.append(
            f"    {contribution.monomial}: {contribution.rawCoefficient:+.6f} -> {formatSignedValue(contribution.snappedCoefficient)}"
        )

    return "\n".join(lines)


def assembleDiscoveredLagrangian(
    kineticTerm,
    discoveredTerms,
    coords,
    vels,
    snapRelativeTolerance=0.01,
    relativeZeroFloor=1e-3,
):
    stateSymbolMap = buildStateSymbolMap(coords, vels)
    kineticState = toStateExpression(kineticTerm, stateSymbolMap)

    magnitudes = [abs(float(coefficient)) for _, coefficient in discoveredTerms]
    zeroThreshold = relativeZeroFloor * max(magnitudes, default=0.0)

    contributions = []
    rawExpression = kineticState
    snappedExpression = kineticState
    for term, coefficient in discoveredTerms:
        monomial = toStateExpression(term, stateSymbolMap)
        rawCoefficient = float(coefficient)
        snappedCoefficient = snapCoefficient(rawCoefficient, snapRelativeTolerance, max(zeroThreshold, 1e-7))

        rawExpression = rawExpression + sp.Float(rawCoefficient) * monomial
        if snappedCoefficient != 0:
            snappedExpression = snappedExpression + snappedCoefficient * monomial

        contributions.append(TermContribution(monomial, rawCoefficient, snappedCoefficient))

    rawExpression = sp.expand(rawExpression)
    snappedExpression = sp.expand(snappedExpression)
    text = formatLagrangian(kineticState, snappedExpression, contributions)

    return DiscoveredLagrangian(snappedExpression, rawExpression, kineticState, contributions, text)
