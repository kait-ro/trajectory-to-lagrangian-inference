from dataclasses import dataclass, field

import sympy as sp


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

    def summary(self):
        lines = [
            f"Degenerate Lagrangian (order {self.order}). "
            f"{len(self.primaryConstraints)} primary constraint(s): "
            f"{self.firstClassCount} first-class, {self.secondClassCount} second-class."
        ]
        for constraint, klass in zip(self.primaryConstraints, self.constraintClass):
            lines.append(f"  {klass:>28}:  {constraint.expression} = 0   ({constraint.origin})")
        lines.append(f"  Poisson-bracket matrix C_ab = {{phi_a, phi_b}}:")
        lines.append(f"    {self.poissonBracketMatrix.tolist()}")
        lines.append(f"  canonical H (primary surface) = {self.canonicalHamiltonian}")
        if self.secondaryConstraintsExpected:
            lines.append(
                "  consistency {phi, H} does not vanish weakly for at least one constraint "
                "-> a secondary constraint is required (Dirac-Bergmann iteration not performed)."
            )
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
        basis = sp.groebner(
            [sp.expand(c) for c in constraintExpressions], *variables, order="lex"
        )
        remainder = basis.reduce(expression)[1]
        return sp.expand(remainder) == 0
    except (sp.PolynomialError, sp.GeneratorsError, TypeError):
        substituted = expression
        for constraint in constraintExpressions:
            for variable in variables:
                solved = sp.solve(constraint, variable, dict=True)
                if solved:
                    substituted = sp.expand(substituted.subs(solved[0]))
        return sp.simplify(substituted) == 0


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
