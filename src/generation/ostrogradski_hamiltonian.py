import sympy as sp

from generation.constraints import (
    DegenerateLagrangianResult,
    PrimaryConstraint,
    classifyConstraints,
    diracBergmannIteration,
    diracBracketMatrix,
)
from generation.ostrogradski import TIME, lagrangianOrder, timeDerivative


class NonUniqueTopDerivativeError(ValueError):
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
            for extra in range(order - lowestOrder + 1):
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
    if not solutions or any(symbol not in solutions[0] for symbol in topDerivativeSymbols):
        return analyzeDegenerateLagrangian(lagrangian, coords, resolvedOrder, constants)
    if len(solutions) > 1:
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
    positions, momenta = [], []
    for perCoordPositions, perCoordMomenta in zip(positionSymbols, momentumSymbols):
        positions.extend(perCoordPositions)
        momenta.extend(perCoordMomenta)
    return positions, momenta


def analyzeDegenerateLagrangian(lagrangian, coords, order=None, constants=None):
    lagrangian = sp.expand(sp.sympify(lagrangian))
    resolvedOrder = lagrangianOrder(lagrangian, coords) if order is None else order
    noCoords = len(coords)

    positionSymbols, momentumSymbols = _canonicalSymbols(noCoords, resolvedOrder)
    multiplierSymbols = [sp.Symbol(f"u{i}") for i in range(noCoords)]

    canonicalMap = {}
    for i in range(noCoords):
        for k in range(resolvedOrder):
            canonicalMap[timeDerivative(coords[i], k)] = positionSymbols[i][k]
        canonicalMap[sp.diff(coords[i], TIME, resolvedOrder)] = multiplierSymbols[i]

    lagrangianCanonical = sp.expand(lagrangian.subs(canonicalMap, simultaneous=True))

    topPartials = [sp.diff(lagrangianCanonical, multiplierSymbols[i]) for i in range(noCoords)]
    topMomenta = [momentumSymbols[i][resolvedOrder - 1] for i in range(noCoords)]
    topRelations = [sp.expand(topMomenta[i] - topPartials[i]) for i in range(noCoords)]

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
        expression = sp.expand(expression.subs({u: 0 for u in multiplierSymbols}))
        if expression != 0:
            constraints.append(
                PrimaryConstraint(expression, origin="top-momentum relation not invertible")
            )

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
    primaryClasses, primaryBracket, _fc, _sc, primarySecondaryExpected = classifyConstraints(
        constraints, hamiltonian, flatPositions, flatMomenta
    )

    iteration = diracBergmannIteration(constraints, hamiltonian, flatPositions, flatMomenta)
    allConstraints = iteration["constraints"]
    allClasses = iteration["classes"]
    generations = iteration["generations"]
    primaryCount = iteration["primaryCount"]
    secondaryConstraints = allConstraints[primaryCount:]
    chainClosed = iteration["chainClosed"]
    totalFirstCount = iteration["firstClassCount"]
    totalSecondCount = iteration["secondClassCount"]

    physicalPhaseSpaceDimension = None
    if chainClosed:
        physicalPhaseSpaceDimension = (
            2 * len(flatPositions) - 2 * totalFirstCount - totalSecondCount
        )

    secondClassExpressions = [
        constraint.expression
        for constraint, klass in zip(allConstraints, allClasses)
        if klass == "second-class"
    ]
    diracMatrix = None
    if chainClosed and secondClassExpressions:
        candidate = diracBracketMatrix(secondClassExpressions, flatPositions, flatMomenta)
        if candidate.det() != 0:
            diracMatrix = candidate

    if not chainClosed:
        detail = (
            "Dirac-Bergmann iteration did not close within the round budget; the constraint "
            "structure below is incomplete."
        )
    elif totalFirstCount:
        detail = (
            "Dirac-Bergmann chain closed. First-class constraints present => residual gauge "
            "freedom; physical phase space is what remains after gauge fixing."
        )
    else:
        detail = (
            "Dirac-Bergmann chain closed; all constraints second-class. Physical phase space "
            "is the primary surface reduced by the constraint pairs."
        )

    return DegenerateLagrangianResult(
        order=resolvedOrder,
        positionSymbols=flatPositions,
        momentumSymbols=flatMomenta,
        canonicalHamiltonian=hamiltonian,
        primaryConstraints=constraints,
        poissonBracketMatrix=primaryBracket,
        constraintClass=primaryClasses,
        firstClassCount=sum(1 for klass in primaryClasses if klass.startswith("first-class")),
        secondClassCount=sum(1 for klass in primaryClasses if klass == "second-class"),
        secondaryConstraintsExpected=primarySecondaryExpected,
        detail=detail,
        multiplierSymbols=multiplierSymbols,
        secondaryConstraints=secondaryConstraints,
        allConstraints=allConstraints,
        allConstraintClasses=allClasses,
        constraintGenerations=generations,
        fullPoissonBracketMatrix=iteration["bracket"],
        totalFirstClassCount=totalFirstCount,
        totalSecondClassCount=totalSecondCount,
        chainClosed=chainClosed,
        physicalPhaseSpaceDimension=physicalPhaseSpaceDimension,
        diracBracketMatrix=diracMatrix,
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
