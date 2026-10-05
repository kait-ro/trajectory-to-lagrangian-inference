from dataclasses import dataclass, field

import sympy as sp
from sympy.polys.polyerrors import CoercionFailed


def _groebnerRemainder(expression, constraintExpressions, variables):
    constraintPolynomials = [sp.expand(constraint) for constraint in constraintExpressions]
    try:
        groebnerBasis = sp.groebner(constraintPolynomials, *variables, order="lex")
        return sp.expand(groebnerBasis.reduce(expression)[1])
    except CoercionFailed:
        # sympy picks the coefficient domain from the inputs and that guess fails
        # on mixed rational coefficients; force the rational field and retry.
        groebnerBasis = sp.groebner(
            constraintPolynomials, *variables, order="lex", domain="QQ"
        )
        return sp.expand(groebnerBasis.reduce(expression)[1])


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
        for constraint, constraintClassLabel, generation in zip(constraints, classes, generations):
            lines.append(
                f"  gen {generation}  {constraintClassLabel:>28}:  {constraint.expression} = 0   ({constraint.origin})"
            )
        bracketMatrix = (
            self.fullPoissonBracketMatrix
            if self.fullPoissonBracketMatrix is not None
            else self.poissonBracketMatrix
        )
        lines.append("  Poisson-bracket matrix C_ab = {phi_a, phi_b}:")
        lines.append(f"    {bracketMatrix.tolist()}")
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


def poissonBracket(leftObservable, rightObservable, positions, momenta):
    if len(positions) != len(momenta):
        raise ValueError("positions and momenta must pair up one-to-one")
    leftObservable = sp.sympify(leftObservable)
    rightObservable = sp.sympify(rightObservable)
    bracketSum = sp.Integer(0)
    for position, momentum in zip(positions, momenta):
        bracketSum += (
            sp.diff(leftObservable, position) * sp.diff(rightObservable, momentum)
            - sp.diff(leftObservable, momentum) * sp.diff(rightObservable, position)
        )
    return sp.expand(bracketSum)


def weaklyVanishes(expression, constraintExpressions, variables):
    # "Weakly" means zero on the constraint surface (modulo the ideal the
    # constraints generate), not necessarily zero as a free expression.
    expression = sp.expand(sp.sympify(expression))
    if expression == 0:
        return True
    if not constraintExpressions:
        return sp.simplify(expression) == 0

    try:
        return _groebnerRemainder(expression, constraintExpressions, variables) == 0
    except (sp.PolynomialError, sp.GeneratorsError, TypeError, CoercionFailed):
        # The constraints do not form an ideal Groebner reduction can handle here,
        # so fall back to solving each one for a single variable and substituting
        # it away, taking the constraints in turn.
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
    # A newly derived relation is only a genuinely new constraint if the current
    # set does not already imply it: it must neither reduce to zero modulo them
    # nor be a constant multiple of one of them.
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


def _fixesAMultiplier(
    constraintExpression, allConstraintExpressions, primaryCount, positions, momenta, variables
):
    # If {phi_a, phi_primary} fails to vanish weakly for some primary constraint,
    # the time-consistency equation for phi_a pins down a Lagrange multiplier
    # rather than yielding a fresh constraint.
    return any(
        not weaklyVanishes(
            poissonBracket(
                constraintExpression,
                allConstraintExpressions[primaryIndex],
                positions,
                momenta,
            ),
            allConstraintExpressions,
            variables,
        )
        for primaryIndex in range(primaryCount)
    )


def _nextGenerationConstraints(
    frontierIndices, allConstraints, hamiltonian, primaryCount, positions, momenta, variables
):
    constraintExpressions = [constraint.expression for constraint in allConstraints]
    newConstraintExpressions = []
    for constraintIndex in frontierIndices:
        if _fixesAMultiplier(
            constraintExpressions[constraintIndex],
            constraintExpressions,
            primaryCount,
            positions,
            momenta,
            variables,
        ):
            continue
        # Time-consistency: {phi_a, H} must also vanish on the constraint surface.
        consistency = poissonBracket(
            constraintExpressions[constraintIndex], hamiltonian, positions, momenta
        )
        reduced = reduceModulo(consistency, constraintExpressions, variables)
        if _independentOfExisting(
            reduced, constraintExpressions + newConstraintExpressions, variables
        ):
            newConstraintExpressions.append(sp.expand(reduced))
    return newConstraintExpressions


def diracBergmannIteration(primaryConstraints, hamiltonian, positions, momenta, maxRounds=8):
    variables = list(positions) + list(momenta)
    allConstraints = list(primaryConstraints)
    generations = [1] * len(primaryConstraints)
    primaryCount = len(primaryConstraints)
    # The frontier holds only the newest generation of constraints whose
    # time-consistency has not been checked yet.
    frontierIndices = list(range(primaryCount))
    chainClosed = primaryCount == 0

    for roundIndex in range(1, maxRounds + 1):
        if not frontierIndices:
            chainClosed = True
            break
        newConstraintExpressions = _nextGenerationConstraints(
            frontierIndices,
            allConstraints,
            hamiltonian,
            primaryCount,
            positions,
            momenta,
            variables,
        )
        if not newConstraintExpressions:
            chainClosed = True
            break
        firstNewConstraintIndex = len(allConstraints)
        # Primaries are generation 1, so constraints found in round `roundIndex`
        # form generation `roundIndex + 1`.
        nextGeneration = roundIndex + 1
        for expression in newConstraintExpressions:
            allConstraints.append(
                PrimaryConstraint(
                    expression,
                    origin=f"Dirac-Bergmann consistency (generation {nextGeneration})",
                )
            )
        generations.extend([nextGeneration] * len(newConstraintExpressions))
        frontierIndices = list(range(firstNewConstraintIndex, len(allConstraints)))

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
    count = len(secondClassExpressions)
    return sp.Matrix(
        count,
        count,
        lambda rowIndex, columnIndex: poissonBracket(
            secondClassExpressions[rowIndex],
            secondClassExpressions[columnIndex],
            positions,
            momenta,
        ),
    )


def diracBracket(leftObservable, rightObservable, secondClassExpressions, positions, momenta):
    canonical = poissonBracket(leftObservable, rightObservable, positions, momenta)
    if not secondClassExpressions:
        return canonical
    secondClassBracketMatrix = diracBracketMatrix(secondClassExpressions, positions, momenta)
    if secondClassBracketMatrix.det() == 0:
        raise ValueError("second-class constraint matrix is singular; Dirac bracket undefined")
    inverseBracketMatrix = secondClassBracketMatrix.inv()
    secondClassCount = len(secondClassExpressions)
    # The correction subtracts the second-class directions so those constraints
    # can be imposed strongly: {f, g}* = {f, g} - {f, phi_a} (C^-1)_ab {phi_b, g},
    # with C the second-class Poisson-bracket matrix.
    diracCorrection = sp.Integer(0)
    for rowIndex in range(secondClassCount):
        bracketWithLeft = poissonBracket(
            leftObservable, secondClassExpressions[rowIndex], positions, momenta
        )
        if bracketWithLeft == 0:
            continue
        for columnIndex in range(secondClassCount):
            bracketWithRight = poissonBracket(
                secondClassExpressions[columnIndex], rightObservable, positions, momenta
            )
            if bracketWithRight == 0:
                continue
            diracCorrection += (
                bracketWithLeft
                * inverseBracketMatrix[rowIndex, columnIndex]
                * bracketWithRight
            )
    return sp.expand(canonical - diracCorrection)


def _poissonBracketMatrix(constraintExpressions, positions, momenta):
    constraintCount = len(constraintExpressions)
    bracket = sp.zeros(constraintCount, constraintCount)
    for rowIndex in range(constraintCount):
        for columnIndex in range(constraintCount):
            bracket[rowIndex, columnIndex] = poissonBracket(
                constraintExpressions[rowIndex],
                constraintExpressions[columnIndex],
                positions,
                momenta,
            )
    return bracket


def _rowVanishesWeakly(bracketMatrix, rowIndex, constraintExpressions, variables):
    return all(
        weaklyVanishes(bracketMatrix[rowIndex, columnIndex], constraintExpressions, variables)
        for columnIndex in range(len(constraintExpressions))
    )


def classifyConstraints(constraints, hamiltonian, positions, momenta):
    constraintExpressions = [constraint.expression for constraint in constraints]
    variables = list(positions) + list(momenta)
    constraintCount = len(constraints)

    bracket = _poissonBracketMatrix(constraintExpressions, positions, momenta)

    classes = []
    secondaryExpected = False
    firstClassCount = 0
    secondClassCount = 0
    for rowIndex in range(constraintCount):
        # First-class: the Poisson bracket with every constraint vanishes on the
        # constraint surface. Any surviving bracket makes it second-class.
        if not _rowVanishesWeakly(bracket, rowIndex, constraintExpressions, variables):
            classes.append("second-class")
            secondClassCount += 1
            continue
        # A first-class constraint whose {phi, H} does not weakly vanish means a
        # secondary constraint is still pending further down the chain.
        consistency = poissonBracket(
            constraintExpressions[rowIndex], hamiltonian, positions, momenta
        )
        if weaklyVanishes(consistency, constraintExpressions, variables):
            classes.append("first-class")
        else:
            classes.append("first-class (pending secondary)")
            secondaryExpected = True
        firstClassCount += 1

    return classes, bracket, firstClassCount, secondClassCount, secondaryExpected
