from dataclasses import dataclass, field

import sympy as sp
from sympy.polys.polyerrors import CoercionFailed


def _groebnerRemainder(expression, constraintExpressions, variables):
    polys = [sp.expand(c) for c in constraintExpressions]
    try:
        basis = sp.groebner(polys, *variables, order="lex")
        return sp.expand(basis.reduce(expression)[1])
    except CoercionFailed:
        basis = sp.groebner(polys, *variables, order="lex", domain="QQ")
        return sp.expand(basis.reduce(expression)[1])


@dataclass
class PrimaryConstraint:
    expression: sp.Expr
    origin: str


@dataclass
class DegenerateLagrangianResult:
    order: int
    positionSymbols: list
    momentumSymbols: list
    canonicalHamiltonian: sp.Expr
    primaryConstraints: list
    poissonBracketMatrix: sp.Matrix
    constraintClass: list
    firstClassCount: int
    secondClassCount: int
    secondaryConstraintsExpected: bool
    detail: str
    degenerate: bool = True
    multiplierSymbols: list = field(default_factory=list)
    secondaryConstraints: list = field(default_factory=list)
    allConstraints: list = field(default_factory=list)
    allConstraintClasses: list = field(default_factory=list)
    constraintGenerations: list = field(default_factory=list)
    fullPoissonBracketMatrix: sp.Matrix = None
    totalFirstClassCount: int = 0
    totalSecondClassCount: int = 0
    chainClosed: bool = True
    physicalPhaseSpaceDimension: int = None
    diracBracketMatrix: sp.Matrix = None

    def summary(self):
        lines = [
            (f"Degenerate Lagrangian (order {self.order}). "
            f"{len(self.primaryConstraints)} primary + {len(self.secondaryConstraints)} secondary "
            f"constraint(s): {self.totalFirstClassCount} first-class, "
            f"{self.totalSecondClassCount} second-class.")
        ]
        constraints = self.allConstraints or self.primaryConstraints
        classes = self.allConstraintClasses or self.constraintClass
        generations = self.constraintGenerations or [1] * len(constraints)
        for constraint, klass, generation in zip(constraints, classes, generations):
            lines.append(
                f"  gen {generation}  {klass:>28}:  {constraint.expression} = 0   ({constraint.origin})"
            )
        matrix = self.fullPoissonBracketMatrix if self.fullPoissonBracketMatrix is not None else self.poissonBracketMatrix
        lines.append("  Poisson-bracket matrix C_ab = {phi_a, phi_b}:")
        lines.append(f"    {matrix.tolist()}")
        lines.append(f"  canonical H (primary surface) = {self.canonicalHamiltonian}")
        if self.chainClosed:
            lines.append("  Dirac-Bergmann chain closed.")
            if self.physicalPhaseSpaceDimension is not None:
                lines.append(f"  physical phase-space dimension = {self.physicalPhaseSpaceDimension}")
        else:
            lines.append(
                "  Dirac-Bergmann iteration did not close within the round budget "
                "-> constraint structure incomplete."
            )
        if self.diracBracketMatrix is not None:
            lines.append(f"  second-class Dirac-bracket matrix = {self.diracBracketMatrix.tolist()}")
        lines.append(f"  {self.detail}")
        return "\n".join(lines)


def poissonBracket(f, g, positions, momenta):
    if len(positions) != len(momenta):
        raise ValueError("positions and momenta must pair up one-to-one")
    f = sp.sympify(f)
    g = sp.sympify(g)
    total = sp.Integer(0)
    for q, p in zip(positions, momenta):
        total += sp.diff(f, q) * sp.diff(g, p) - sp.diff(f, p) * sp.diff(g, q)
    return sp.expand(total)


def weaklyVanishes(expression, constraintExpressions, variables):
    expression = sp.expand(sp.sympify(expression))
    if expression == 0:
        return True
    if not constraintExpressions:
        return sp.simplify(expression) == 0

    try:
        return _groebnerRemainder(expression, constraintExpressions, variables) == 0
    except (sp.PolynomialError, sp.GeneratorsError, TypeError, CoercionFailed):
        substituted = expression
        for constraint in constraintExpressions:
            for variable in variables:
                solved = sp.solve(constraint, variable, dict=True)
                if solved:
                    substituted = sp.expand(substituted.subs(solved[0]))
        return sp.simplify(substituted) == 0


def reduceModulo(expression, constraintExpressions, variables):
    expression = sp.expand(sp.sympify(expression))
    if expression == 0 or not constraintExpressions:
        return expression
    try:
        return _groebnerRemainder(expression, constraintExpressions, variables)
    except (sp.PolynomialError, sp.GeneratorsError, TypeError, CoercionFailed):
        return expression


def _independentOfExisting(candidate, existingExpressions, variables):
    candidate = sp.expand(candidate)
    if candidate == 0:
        return False
    if reduceModulo(candidate, existingExpressions, variables) == 0:
        return False
    for existing in existingExpressions:
        existing = sp.expand(existing)
        if existing == 0:
            continue
        ratio = sp.simplify(candidate / existing)
        if ratio != 0 and not ratio.free_symbols:
            return False
    return True


def diracBergmannIteration(primaryConstraints, hamiltonian, positions, momenta, maxRounds=8):
    variables = list(positions) + list(momenta)
    allConstraints = list(primaryConstraints)
    generations = [1] * len(primaryConstraints)
    primaryCount = len(primaryConstraints)
    frontier = list(range(primaryCount))
    chainClosed = primaryCount == 0

    for roundIndex in range(1, maxRounds + 1):
        if not frontier:
            chainClosed = True
            break
        expressions = [c.expression for c in allConstraints]
        pending = []
        for a in frontier:
            determinesMultiplier = any(
                not weaklyVanishes(
                    poissonBracket(expressions[a], expressions[p], positions, momenta),
                    expressions,
                    variables,
                )
                for p in range(primaryCount)
            )
            if determinesMultiplier:
                continue
            consistency = poissonBracket(expressions[a], hamiltonian, positions, momenta)
            reduced = reduceModulo(consistency, expressions, variables)
            if _independentOfExisting(reduced, expressions + pending, variables):
                pending.append(sp.expand(reduced))
        if not pending:
            chainClosed = True
            break
        firstNew = len(allConstraints)
        for expression in pending:
            allConstraints.append(
                PrimaryConstraint(
                    expression,
                    origin=f"Dirac-Bergmann consistency (generation {roundIndex + 1})",
                )
            )
        generations.extend([roundIndex + 1] * len(pending))
        frontier = list(range(firstNew, len(allConstraints)))

    classes, bracket, firstCount, secondCount, secondaryExpected = classifyConstraints(
        allConstraints, hamiltonian, positions, momenta
    )
    return {
        "constraints": allConstraints,
        "generations": generations,
        "classes": classes,
        "bracket": bracket,
        "firstClassCount": firstCount,
        "secondClassCount": secondCount,
        "primaryCount": primaryCount,
        "chainClosed": bool(chainClosed and not secondaryExpected),
    }


def diracBracketMatrix(secondClassExpressions, positions, momenta):
    n = len(secondClassExpressions)
    return sp.Matrix(
        n,
        n,
        lambda a, b: poissonBracket(
            secondClassExpressions[a], secondClassExpressions[b], positions, momenta
        ),
    )


def diracBracket(f, g, secondClassExpressions, positions, momenta):
    canonical = poissonBracket(f, g, positions, momenta)
    if not secondClassExpressions:
        return canonical
    matrix = diracBracketMatrix(secondClassExpressions, positions, momenta)
    if matrix.det() == 0:
        raise ValueError("second-class constraint matrix is singular; Dirac bracket undefined")
    inverse = matrix.inv()
    n = len(secondClassExpressions)
    correction = sp.Integer(0)
    for a in range(n):
        left = poissonBracket(f, secondClassExpressions[a], positions, momenta)
        if left == 0:
            continue
        for b in range(n):
            right = poissonBracket(secondClassExpressions[b], g, positions, momenta)
            if right == 0:
                continue
            correction += left * inverse[a, b] * right
    return sp.expand(canonical - correction)


def classifyConstraints(constraints, hamiltonian, positions, momenta):
    expressions = [c.expression for c in constraints]
    variables = list(positions) + list(momenta)
    n = len(constraints)

    bracket = sp.zeros(n, n)
    for a in range(n):
        for b in range(n):
            bracket[a, b] = poissonBracket(expressions[a], expressions[b], positions, momenta)

    classes = []
    secondaryExpected = False
    firstClassCount = 0
    secondClassCount = 0
    for a in range(n):
        rowWeaklyZero = all(
            weaklyVanishes(bracket[a, b], expressions, variables) for b in range(n)
        )
        if not rowWeaklyZero:
            classes.append("second-class")
            secondClassCount += 1
            continue
        consistency = poissonBracket(expressions[a], hamiltonian, positions, momenta)
        if weaklyVanishes(consistency, expressions, variables):
            classes.append("first-class")
        else:
            classes.append("first-class (pending secondary)")
            secondaryExpected = True
        firstClassCount += 1

    return classes, bracket, firstClassCount, secondClassCount, secondaryExpected
