import sympy as sp

from generation.ostrogradski import TIME, lagrangianOrder, timeDerivative


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
        raise ValueError(
            "Lagrangian is degenerate in its highest derivative (cannot invert the highest momentum); "
            "reduce it to non-degenerate form or lower the order before Ostrogradski analysis"
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
