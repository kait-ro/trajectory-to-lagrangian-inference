import itertools

import numpy as np
import sympy as sp

BOUNDED_BELOW = "bounded_below"
UNBOUNDED_BELOW = "unbounded_below"
INCONCLUSIVE = "inconclusive"


def _homogeneousPart(polynomial, variables, degree):
    # polynomial.terms() yields (exponents, coefficient) pairs, one per monomial.
    # Keep only the monomials whose exponents sum to the target degree, and
    # rebuild each one as coefficient * variable_1**exponent_1 * variable_2**exponent_2 * ...
    termsAtThisDegree = [
        coefficient
        * sp.prod([variable**exponent for variable, exponent in zip(variables, exponents)])
        for exponents, coefficient in polynomial.terms()
        if sum(exponents) == degree
    ]
    return sp.Add(*termsAtThisDegree) if termsAtThisDegree else sp.Integer(0)


def _unitVectorAlongAxes(dimension, axisIndices, signPattern):
    vector = np.zeros(dimension)
    for axisIndex, sign in zip(axisIndices, signPattern):
        vector[axisIndex] = sign
    return vector


def _axisAlignedDirections(dimension):
    # Random directions alone can miss the fact that a positive-semidefinite
    # leading form touches zero along an axis or a sub-diagonal (e.g. q0 = q1).
    # So explicitly probe every combination of up to 3 axes, with every
    # combination of + and - signs on those axes.
    maximumNonzeroAxes = min(3, dimension)
    directions = []
    for numberOfNonzeroAxes in range(1, maximumNonzeroAxes + 1):
        axisIndexGroups = itertools.combinations(range(dimension), numberOfNonzeroAxes)
        signPatterns = itertools.product((1.0, -1.0), repeat=numberOfNonzeroAxes)
        for axisIndices, signPattern in itertools.product(axisIndexGroups, signPatterns):
            directions.append(_unitVectorAlongAxes(dimension, axisIndices, signPattern))
    return directions


def _sampleDirections(dimension, seed, randomDirectionCount=8000):
    randomNumberGenerator = np.random.default_rng(seed)
    randomDirections = randomNumberGenerator.normal(size=(randomDirectionCount, dimension))
    axisDirections = np.array(_axisAlignedDirections(dimension), dtype=float)

    directions = np.vstack([randomDirections, axisDirections])
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return directions / lengths


def _minimumValueOnUnitSphere(polynomialForm, variables, seed):
    if not variables:
        return float(polynomialForm), 1.0

    evaluateForm = sp.lambdify(variables, polynomialForm, "numpy")
    directions = _sampleDirections(len(variables), seed)

    # lambdify can return a single bare number instead of an array when the
    # form happens not to vary across these directions, so broadcast to get
    # exactly one value per sampled direction either way.
    rawValues = np.asarray(evaluateForm(*directions.T), dtype=float)
    values = np.broadcast_to(rawValues, (len(directions),))
    # A NaN sample (an indeterminate 0/0 along some direction) is treated as
    # +infinity so it can never masquerade as a negative, unbounded-below minimum.
    values = np.nan_to_num(values, nan=np.inf)

    minimumValue = float(np.min(values))
    magnitudeScale = float(np.max(np.abs(values))) or 1.0
    return minimumValue, magnitudeScale


def _boundedBelowRecursive(expression, variables, tolerance, seed, recursionDepth):
    expression = sp.expand(expression)
    if not variables:
        return BOUNDED_BELOW, "constant"

    polynomial = sp.Poly(expression, *variables)
    degree = polynomial.total_degree()
    if degree <= 0:
        return BOUNDED_BELOW, "constant"
    if degree % 2 == 1:
        # An odd-degree homogeneous form always takes both signs (flip the
        # sign of the direction to flip the sign of the value), so the
        # expression is unbounded below no matter what the lower-degree terms do.
        return UNBOUNDED_BELOW, f"odd total degree {degree}"

    leadingForm = _homogeneousPart(polynomial, variables, degree)
    minimumValue, magnitudeScale = _minimumValueOnUnitSphere(leadingForm, variables, seed + recursionDepth)
    margin = tolerance * magnitudeScale + tolerance

    if minimumValue < -margin:
        return UNBOUNDED_BELOW, (
            f"degree-{degree} leading form is negative along a direction "
            f"(min {minimumValue:.3e}); H -> -infinity along that ray"
        )
    if minimumValue > margin:
        return BOUNDED_BELOW, (
            f"degree-{degree} leading form is positive definite; H is coercive"
        )

    # The leading form is only positive *semidefinite*: it can still touch zero.
    # Split the variables into the ones the leading form actually depends on
    # and the ones it doesn't (the "flat" directions where it stays zero).
    variablesInLeadingForm = [symbol for symbol in variables if symbol in leadingForm.free_symbols]
    omittedVariables = [symbol for symbol in variables if symbol not in leadingForm.free_symbols]
    if not omittedVariables or recursionDepth >= len(variables):
        return INCONCLUSIVE, (
            f"degree-{degree} leading form is positive semidefinite and involves every "
            "variable; coercivity not established"
        )

    presentFormMinimum, _presentFormScale = _minimumValueOnUnitSphere(
        leadingForm, variablesInLeadingForm, seed + recursionDepth + 101
    )
    if presentFormMinimum <= margin:
        return INCONCLUSIVE, (
            f"degree-{degree} leading form stays semidefinite within the variables it "
            "involves; boundedness not resolved"
        )

    # The leading form is coercive (strictly positive away from zero) in the
    # variables it involves, so any way the whole expression could run off to
    # -infinity has to happen on the subspace where those variables are zero.
    # Set them to zero and recurse on what's left.
    restrictedExpression = sp.expand(expression.subs({symbol: 0 for symbol in variablesInLeadingForm}))
    verdict, detail = _boundedBelowRecursive(restrictedExpression, omittedVariables, tolerance, seed, recursionDepth + 1)
    presentVariableNames = ", ".join(str(symbol) for symbol in variablesInLeadingForm)
    if verdict == UNBOUNDED_BELOW:
        return UNBOUNDED_BELOW, (
            f"H is unbounded below on the {presentVariableNames}=0 subspace ({detail})"
        )
    if verdict == BOUNDED_BELOW:
        return BOUNDED_BELOW, (
            f"degree-{degree} leading form is coercive in ({presentVariableNames}) and H is bounded "
            f"below on {presentVariableNames}=0 ({detail})"
        )
    return INCONCLUSIVE, (
        f"degree-{degree} leading form coercive in ({presentVariableNames}); "
        f"residual analysis inconclusive ({detail})"
    )


def polynomialBoundedBelow(expression, positionSymbols, momentumSymbols, tolerance=1e-9, seed=0):
    expression = sp.expand(sp.sympify(expression))
    variables = list(positionSymbols) + list(momentumSymbols)

    if not expression.free_symbols <= set(variables) or not expression.is_polynomial(*variables):
        return {
            "verdict": INCONCLUSIVE,
            "degree": None,
            "detail": "H is not a polynomial in the phase-space variables; boundedness test does not apply",
        }

    degree = sp.Poly(expression, *variables).total_degree() if variables else 0

    # A Hamiltonian that is exactly linear in some momentum is the Ostrogradski
    # signature: for fixed values of every other variable, that momentum alone
    # can be driven to +-infinity, taking H to -infinity. This is always
    # unbounded below, independent of the sphere-sampling test below.
    for momentum in momentumSymbols:
        if sp.Poly(expression, momentum).degree() == 1:
            coefficient = sp.expand(expression.coeff(momentum, 1))
            if coefficient != 0:
                return {
                    "verdict": UNBOUNDED_BELOW,
                    "degree": degree,
                    "detail": (
                        f"H is exactly linear in momentum {momentum} (coefficient "
                        f"{sp.nsimplify(coefficient)}): Ostrogradski linear-momentum term"
                    ),
                }

    verdict, detail = _boundedBelowRecursive(expression, variables, tolerance, seed, 0)
    return {"verdict": verdict, "degree": degree, "detail": detail}
