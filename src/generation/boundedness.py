import itertools

import numpy as np
import sympy as sp


def _homogeneousPart(poly, gens, degree):
    terms = [
        coeff * sp.prod([gen ** exponent for gen, exponent in zip(gens, monom)])
        for monom, coeff in poly.terms()
        if sum(monom) == degree
    ]
    return sp.Add(*terms) if terms else sp.Integer(0)


def _sampleDirections(dimension, seed, randomCount=8000):
    rng = np.random.default_rng(seed)
    random = rng.normal(size=(randomCount, dimension))

    structured = []
    for axisCount in range(1, min(3, dimension) + 1):
        for axes in itertools.combinations(range(dimension), axisCount):
            for signs in itertools.product((1.0, -1.0), repeat=axisCount):
                vector = np.zeros(dimension)
                for axis, sign in zip(axes, signs):
                    vector[axis] = sign
                structured.append(vector)

    directions = np.vstack([random, np.array(structured, dtype=float)])
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return directions / norms


def _sphereMinimum(form, variables, seed):
    if not variables:
        return float(form), 1.0
    evaluate = sp.lambdify(variables, form, "numpy")
    directions = _sampleDirections(len(variables), seed)
    values = np.nan_to_num(
        np.asarray(evaluate(*directions.T), dtype=float) * np.ones(len(directions)),
        nan=np.inf,
    )
    return float(np.min(values)), (float(np.max(np.abs(values))) or 1.0)


def _boundedBelowRecursive(expression, variables, tolerance, seed, depth):
    expression = sp.expand(expression)
    if not variables:
        return "bounded_below", "constant"

    poly = sp.Poly(expression, *variables)
    degree = poly.total_degree()
    if degree <= 0:
        return "bounded_below", "constant"
    if degree % 2 == 1:
        return "unbounded_below", f"odd total degree {degree}"

    leadingForm = _homogeneousPart(poly, variables, degree)
    minimum, scale = _sphereMinimum(leadingForm, variables, seed + depth)
    margin = tolerance * scale + tolerance

    if minimum < -margin:
        return "unbounded_below", (
            f"degree-{degree} leading form is negative along a direction "
            f"(min {minimum:.3e}); H -> -infinity along that ray"
        )
    if minimum > margin:
        return "bounded_below", (
            f"degree-{degree} leading form is positive definite; H is coercive"
        )

    appearing = set(leadingForm.free_symbols)
    presentVars = [symbol for symbol in variables if symbol in appearing]
    absentVars = [symbol for symbol in variables if symbol not in appearing]
    if not absentVars or depth >= len(variables):
        return "inconclusive", (
            f"degree-{degree} leading form is positive semidefinite and involves every "
            "variable; coercivity not established"
        )

    presentMinimum, _presentScale = _sphereMinimum(leadingForm, presentVars, seed + depth + 101)
    if presentMinimum <= margin:
        return "inconclusive", (
            f"degree-{degree} leading form stays semidefinite within the variables it "
            "involves; boundedness not resolved"
        )

    restricted = sp.expand(expression.subs({symbol: 0 for symbol in presentVars}))
    verdict, detail = _boundedBelowRecursive(restricted, absentVars, tolerance, seed, depth + 1)
    presentText = ", ".join(str(symbol) for symbol in presentVars)
    if verdict == "unbounded_below":
        return "unbounded_below", (
            f"H is unbounded below on the {presentText}=0 subspace ({detail})"
        )
    if verdict == "bounded_below":
        return "bounded_below", (
            f"degree-{degree} leading form is coercive in ({presentText}) and H is bounded "
            f"below on {presentText}=0 ({detail})"
        )
    return "inconclusive", (
        f"degree-{degree} leading form coercive in ({presentText}); "
        f"residual analysis inconclusive ({detail})"
    )


def polynomialBoundedBelow(expression, positionSymbols, momentumSymbols, tolerance=1e-9, seed=0):
    expression = sp.expand(sp.sympify(expression))
    variables = list(positionSymbols) + list(momentumSymbols)

    if not expression.free_symbols <= set(variables) or not expression.is_polynomial(*variables):
        return {
            "verdict": "inconclusive",
            "degree": None,
            "detail": "H is not a polynomial in the phase-space variables; boundedness test does not apply",
        }

    degree = sp.Poly(expression, *variables).total_degree() if variables else 0

    for momentum in momentumSymbols:
        if sp.Poly(expression, momentum).degree() == 1:
            coefficient = sp.expand(expression.coeff(momentum, 1))
            if coefficient != 0:
                return {
                    "verdict": "unbounded_below",
                    "degree": degree,
                    "detail": (
                        f"H is exactly linear in momentum {momentum} (coefficient "
                        f"{sp.nsimplify(coefficient)}): Ostrogradski linear-momentum term"
                    ),
                }

    verdict, detail = _boundedBelowRecursive(expression, variables, tolerance, seed, 0)
    return {"verdict": verdict, "degree": degree, "detail": detail}
