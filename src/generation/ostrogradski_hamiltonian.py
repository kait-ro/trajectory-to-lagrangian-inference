
import sympy as sp
from generation.constraints import (
    DegenerateLagrangianResult,
    PrimaryConstraint,
    classifyConstraints,
)
from generation.ostrogradski import TIME, lagrangianOrder, timeDerivative


class NonUniqueTopDerivativeError(ValueError):
    """Raised when the Ostrogradski top-momentum relation has more than one branch.

    p_top = dL/dq^(n) can be inverted for q^(n) uniquely only when L is (at most)
    quadratic in its highest derivative -- then the relation is affine and the
    Legendre transform is single-valued. If L is nonlinear in q^(n) (e.g. a
    (q^(n))^4 term, or q^(n) appearing inside a non-polynomial function), the
    inversion has several roots, each giving a different Hamiltonian on a
    different sheet of phase space.

    There is no purely local rule for which branch is "physical": the standard
    choice is the sheet that connects continuously to the non-degenerate /
    weak-coupling limit, or the one consistent with a reference trajectory. That
    choice belongs to the caller, so this is surfaced rather than silently
    resolved by taking the first root.
    """

    def __init__(self, branches):
        self.branches = branches
        super().__init__(
            f"Ostrogradski top-derivative inversion is not unique ({len(branches)} branches); "
            f"L is nonlinear in its highest derivative. Reduce L, lower the order, or pick a "
            f"branch explicitly. Branches: {branches}"
        )


def ostrogradskiMomentumExpressions(lagrangian, coords, order):
    lagrangian = sp.sympify(lagrangian)
    momentaPerCoord = []
    for coordinate in coords:
        perCoord = []
        for lowestOrder in range(1, order + 1):
            total = sp.Integer(0)
            for extra in range(0, order - lowestOrder + 1):
                partial = sp.diff(lagrangian, timeDerivative(coordinate, lowestOrder + extra))
                total = total + sp.Integer(-1) ** extra * sp.diff(partial, TIME, extra)
            perCoord.append(sp.expand(total))
        momentaPerCoord.append(perCoord)
    return momentaPerCoord


def _canonicalSymbols(noCoords, order):
    positions = [[sp.Symbol(f"Q{i}_{s}") for s in range(order)] for i in range(noCoords)]
    momenta = [[sp.Symbol(f"P{i}_{s}") for s in range(1, order + 1)] for i in range(noCoords)]
    return positions, momenta


def ostrogradskiHamiltonian(lagrangian, coords, order=None, constants=None):
    lagrangian = sp.sympify(lagrangian)
    resolvedOrder = lagrangianOrder(lagrangian, coords) if order is None else order
    noCoords = len(coords)

    momentumExpressions = ostrogradskiMomentumExpressions(lagrangian, coords, resolvedOrder)
    positionSymbols, momentumSymbols = _canonicalSymbols(noCoords, resolvedOrder)

    topDerivativeSymbols = [sp.Symbol(f"topDerivative{i}") for i in range(noCoords)]
    topAsSymbol = {sp.diff(coords[j], TIME, resolvedOrder): topDerivativeSymbols[j] for j in range(noCoords)}

    topEquations = []
    for coordinateIndex, coordinate in enumerate(coords):
        highestPartial = sp.diff(lagrangian, timeDerivative(coordinate, resolvedOrder)).subs(topAsSymbol)
        topEquations.append(sp.Eq(momentumSymbols[coordinateIndex][resolvedOrder - 1], highestPartial))

    solutions = sp.solve(topEquations, topDerivativeSymbols, dict=True)
    if not solutions:
        # Degenerate: the top-momentum relation is not invertible for q^(n).
        # Return the constraint structure rather than raising -- see item 8.
        return analyzeDegenerateLagrangian(lagrangian, coords, resolvedOrder, constants)
    if len(solutions) > 1:
        # Multiple branches => L is nonlinear in q^(n) and the Legendre transform
        # is multi-valued. Refuse to guess; see NonUniqueTopDerivativeError.
        raise NonUniqueTopDerivativeError(
            [{str(k): v for k, v in solution.items()} for solution in solutions]
        )
    topSolution = solutions[0]

    canonicalSubstitution = {}
    for coordinateIndex in range(noCoords):
        for derivativeOrder in range(resolvedOrder):
            canonicalSubstitution[timeDerivative(coords[coordinateIndex], derivativeOrder)] = positionSymbols[coordinateIndex][derivativeOrder]
        canonicalSubstitution[sp.diff(coords[coordinateIndex], TIME, resolvedOrder)] = topSolution[topDerivativeSymbols[coordinateIndex]]

    lagrangianCanonical = sp.expand(lagrangian.subs(canonicalSubstitution, simultaneous=True))

    hamiltonian = sp.Integer(0)
    for coordinateIndex in range(noCoords):
        for derivativeOrder in range(1, resolvedOrder + 1):
            if derivativeOrder < resolvedOrder:
                phaseVelocity = positionSymbols[coordinateIndex][derivativeOrder]
            else:
                phaseVelocity = topSolution[topDerivativeSymbols[coordinateIndex]]
            hamiltonian = hamiltonian + momentumSymbols[coordinateIndex][derivativeOrder - 1] * phaseVelocity

    hamiltonian = sp.expand(hamiltonian - lagrangianCanonical)
    if constants:
        hamiltonian = hamiltonian.subs(constants)

    return {
        "hamiltonian": hamiltonian,
        "order": resolvedOrder,
        "positionSymbols": positionSymbols,
        "momentumSymbols": momentumSymbols,
        "momentumTimeExpressions": momentumExpressions,
        "topDerivativeSolution": {topDerivativeSymbols[i]: topSolution[topDerivativeSymbols[i]] for i in range(noCoords)},
    }


def _flatCanonicalPairs(positionSymbols, momentumSymbols):
    """Flatten the per-coordinate canonical symbols into paired flat lists so that
    {positions[k], momenta[k]} = 1 for every k."""
    positions, momenta = [], []
    for perCoordPositions, perCoordMomenta in zip(positionSymbols, momentumSymbols):
        positions.extend(perCoordPositions)
        momenta.extend(perCoordMomenta)
    return positions, momenta


def analyzeDegenerateLagrangian(lagrangian, coords, order=None, constants=None):
    """Primary-constraint detection + first/second-class classification for a
    Lagrangian whose top-momentum relation p_n = dL/dq^(n) cannot be inverted.

    Scope: primary constraints from the top-momentum Hessian null space, the
    Poisson-bracket matrix among them, and the {phi, H} consistency check.
    Secondary constraints are flagged, not computed; no Dirac bracket.
    """
    lagrangian = sp.expand(sp.sympify(lagrangian))
    resolvedOrder = lagrangianOrder(lagrangian, coords) if order is None else order
    noCoords = len(coords)

    positionSymbols, momentumSymbols = _canonicalSymbols(noCoords, resolvedOrder)
    multiplierSymbols = [sp.Symbol(f"u{i}") for i in range(noCoords)]  # undetermined q_i^(n)

    canonicalMap = {}
    for i in range(noCoords):
        for k in range(resolvedOrder):
            canonicalMap[timeDerivative(coords[i], k)] = positionSymbols[i][k]
        canonicalMap[sp.diff(coords[i], TIME, resolvedOrder)] = multiplierSymbols[i]

    lagrangianCanonical = sp.expand(lagrangian.subs(canonicalMap, simultaneous=True))

    # top-momentum relations  P_i^n - dL/dq_i^(n)  in canonical variables
    topPartials = [sp.diff(lagrangianCanonical, multiplierSymbols[i]) for i in range(noCoords)]
    topMomenta = [momentumSymbols[i][resolvedOrder - 1] for i in range(noCoords)]
    topRelations = [sp.expand(topMomenta[i] - topPartials[i]) for i in range(noCoords)]

    # Hessian W_ab = d^2 L / dq_a^(n) dq_b^(n); its null space gives the primary constraints
    hessian = sp.Matrix(
        noCoords, noCoords,
        lambda a, b: sp.diff(topPartials[a], multiplierSymbols[b]),
    )

    constraints = []
    for nullVector in hessian.nullspace():
        denominators = [sp.fraction(component)[1] for component in nullVector]
        scale = sp.ilcm(*[int(d) for d in denominators]) if denominators else 1
        integerVector = [sp.nsimplify(scale * component) for component in nullVector]
        expression = sp.expand(sum(integerVector[a] * topRelations[a] for a in range(noCoords)))
        # the multiplier parts cancel by construction (W . nullVector = 0); drop any residue
        expression = sp.expand(expression.subs({u: 0 for u in multiplierSymbols}))
        if expression != 0:
            constraints.append(
                PrimaryConstraint(expression, origin="top-momentum relation not invertible")
            )

    # canonical H_c = sum P * velocity - L_c, with solvable multipliers eliminated
    # and the remaining (undetermined) ones set to zero on the primary surface.
    solvableForMultipliers = sp.solve(
        [relation for relation in topRelations if relation.free_symbols & set(multiplierSymbols)],
        multiplierSymbols,
        dict=True,
    )
    multiplierValues = solvableForMultipliers[0] if solvableForMultipliers else {}
    multiplierValues = {u: multiplierValues.get(u, sp.Integer(0)) for u in multiplierSymbols}

    hamiltonian = sp.Integer(0)
    for i in range(noCoords):
        for k in range(1, resolvedOrder + 1):
            velocity = positionSymbols[i][k] if k < resolvedOrder else multiplierSymbols[i]
            hamiltonian += momentumSymbols[i][k - 1] * velocity
    hamiltonian = sp.expand(hamiltonian - lagrangianCanonical)
    hamiltonian = sp.expand(hamiltonian.subs(multiplierValues))
    if constants:
        hamiltonian = hamiltonian.subs(constants)
        constraints = [
            PrimaryConstraint(sp.expand(c.expression.subs(constants)), c.origin) for c in constraints
        ]

    flatPositions, flatMomenta = _flatCanonicalPairs(positionSymbols, momentumSymbols)
    classes, bracket, firstCount, secondCount, secondaryExpected = classifyConstraints(
        constraints, hamiltonian, flatPositions, flatMomenta
    )

    detail = (
        "Primary constraints found from the non-invertible top-momentum relation. "
        "First-class => generates a gauge transformation; second-class => reduces the "
        "physical phase space in pairs. Dirac-bracket construction and secondary-constraint "
        "iteration are out of scope."
    )
    return DegenerateLagrangianResult(
        order=resolvedOrder,
        positionSymbols=flatPositions,
        momentumSymbols=flatMomenta,
        canonicalHamiltonian=hamiltonian,
        primaryConstraints=constraints,
        poissonBracketMatrix=bracket,
        constraintClass=classes,
        firstClassCount=firstCount,
        secondClassCount=secondCount,
        secondaryConstraintsExpected=secondaryExpected,
        detail=detail,
        multiplierSymbols=multiplierSymbols,
    )


def hamiltonianOnTrajectory(hamiltonianData, coords, constants, derivativeArrays):
    order = hamiltonianData["order"]
    noCoords = len(coords)
    positionSymbols = hamiltonianData["positionSymbols"]
    momentumSymbols = hamiltonianData["momentumSymbols"]

    substitution = {}
    for coordinateIndex in range(noCoords):
        for derivativeOrder in range(order):
            substitution[positionSymbols[coordinateIndex][derivativeOrder]] = derivativeArrays[derivativeOrder][:, coordinateIndex]

    momentumExpressions = hamiltonianData["momentumTimeExpressions"]
    for coordinateIndex in range(noCoords):
        for momentumIndex in range(order):
            expression = momentumExpressions[coordinateIndex][momentumIndex]
            if constants:
                expression = expression.subs(constants)
            timeToArray = {}
            for level in range(len(derivativeArrays)):
                timeToArray[timeDerivative(coords[coordinateIndex], level)] = derivativeArrays[level][:, coordinateIndex]
            function = sp.lambdify(sorted(timeToArray, key=sp.default_sort_key), expression, modules="numpy")
            substitution[momentumSymbols[coordinateIndex][momentumIndex]] = function(
                *[timeToArray[symbol] for symbol in sorted(timeToArray, key=sp.default_sort_key)]
            )

    hamiltonian = hamiltonianData["hamiltonian"]
    if constants:
        hamiltonian = hamiltonian.subs(constants)
    orderedSymbols = sorted(substitution, key=sp.default_sort_key)
    hamiltonianFunction = sp.lambdify(orderedSymbols, hamiltonian, modules="numpy")
    return hamiltonianFunction(*[substitution[symbol] for symbol in orderedSymbols])
