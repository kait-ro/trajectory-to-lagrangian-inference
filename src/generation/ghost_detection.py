import numpy as np
import sympy as sp

from generation.boundedness import polynomialBoundedBelow
from generation.constraints import DegenerateLagrangianResult
from generation.ostrogradski import TIME, eulerLagrangeSystem
from generation.ostrogradski_hamiltonian import ostrogradskiHamiltonian


def characteristicRoots(lagrangian, coords, order=None):
    elSystem, resolvedOrder = eulerLagrangeSystem(lagrangian, coords, order)
    growthRate = sp.Symbol("s")
    roots = []
    for coordinate, expression in zip(coords, elSystem):
        substitution = {}
        for k in range(2 * resolvedOrder + 1):
            derivative = coordinate if k == 0 else sp.diff(coordinate, TIME, k)
            substitution[derivative] = growthRate ** k
        polynomial = sp.expand(expression.subs(substitution))
        polynomial = sp.Poly(polynomial, growthRate)
        for root in np.roots([complex(coefficient) for coefficient in polynomial.all_coeffs()]):
            roots.append(complex(root))
    return roots


def dynamicalStability(roots, tolerance=1e-6):
    if not roots:
        return "undetermined"
    if all(abs(root.real) < tolerance for root in roots):
        return "oscillatory"
    if any(root.real > tolerance for root in roots):
        return "runaway"
    return "damped"


def _reduceHamiltonianByConstraints(hamiltonian, constraintExpressions, positions, momenta):
    remainingPositions = list(positions)
    remainingMomenta = list(momenta)
    substitutions = {}
    for expression in constraintExpressions:
        expression = sp.expand(sp.sympify(expression).subs(substitutions))
        if expression == 0:
            continue
        target = None
        for symbol in list(remainingMomenta) + list(remainingPositions):
            try:
                polynomial = sp.Poly(expression, symbol)
            except sp.PolynomialError:
                continue
            if polynomial.degree() == 1 and not polynomial.LC().free_symbols:
                target = symbol
                break
        if target is None:
            free = expression.free_symbols & (set(remainingMomenta) | set(remainingPositions))
            solved = sp.solve(expression, list(free), dict=True)
            if not solved:
                return None
            for symbol, value in solved[0].items():
                substitutions[symbol] = value
                (remainingMomenta if symbol in remainingMomenta else remainingPositions).remove(symbol)
            continue
        solved = sp.solve(expression, target, dict=True)
        if not solved:
            return None
        substitutions[target] = solved[0][target]
        (remainingMomenta if target in remainingMomenta else remainingPositions).remove(target)
    for symbol in list(substitutions):
        substitutions[symbol] = sp.expand(sp.sympify(substitutions[symbol]).subs(substitutions))
    reduced = sp.expand(sp.sympify(hamiltonian).subs(substitutions))
    if reduced.free_symbols - (set(remainingPositions) | set(remainingMomenta)):
        return None
    return reduced, remainingPositions, remainingMomenta


def _degenerateGhostVerdict(constraintAnalysis, stability, eigenvalueTolerance):
    if not constraintAnalysis.chainClosed:
        return None, None, "Dirac-Bergmann chain did not close; constraint structure incomplete -> undetermined"
    if constraintAnalysis.totalFirstClassCount:
        return None, None, (
            f"{constraintAnalysis.totalFirstClassCount} first-class constraint(s) -> residual gauge "
            "freedom; ghost verdict needs gauge fixing -> undetermined"
        )
    secondClass = [
        constraint.expression
        for constraint, klass in zip(
            constraintAnalysis.allConstraints, constraintAnalysis.allConstraintClasses
        )
        if klass == "second-class"
    ]
    if not secondClass:
        return None, None, "no second-class constraint structure to reduce onto -> undetermined"
    reduction = _reduceHamiltonianByConstraints(
        constraintAnalysis.canonicalHamiltonian,
        secondClass,
        constraintAnalysis.positionSymbols,
        constraintAnalysis.momentumSymbols,
    )
    if reduction is None:
        return None, None, "could not solve the second-class constraints to reduce H -> undetermined"
    reducedHamiltonian, reducedPositions, reducedMomenta = reduction
    reducedVariables = reducedPositions + reducedMomenta
    if not reducedVariables:
        return False, reducedHamiltonian, (
            f"reduced H is the constant {reducedHamiltonian}; bounded below -> no ghost"
        )
    try:
        isQuadratic = sp.Poly(reducedHamiltonian, *reducedVariables).total_degree() <= 2
    except sp.PolynomialError:
        isQuadratic = False
    if isQuadratic:
        size = len(reducedVariables)
        hessian = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                hessian[i, j] = float(sp.diff(reducedHamiltonian, reducedVariables[i], reducedVariables[j]))
        symmetric = 0.5 * (hessian + hessian.T)
        eigenvalues = np.linalg.eigvalsh(symmetric)
        indefinite = bool(np.any(eigenvalues < -eigenvalueTolerance))
        eigenvalueText = [round(float(value), 6) for value in eigenvalues]
        if indefinite and stability == "oscillatory":
            return True, reducedHamiltonian, (
                f"reduced H is indefinite (eigenvalues {eigenvalueText}) while dynamics are "
                "oscillatory -> negative-energy mode on the constraint surface -> Ostrogradski ghost"
            )
        if indefinite:
            return False, reducedHamiltonian, (
                f"reduced H is indefinite (eigenvalues {eigenvalueText}) but dynamics are "
                f"'{stability}': runaway/tachyonic instability rather than a ghost"
            )
        return False, reducedHamiltonian, (
            f"reduced H is positive semidefinite (eigenvalues {eigenvalueText}) -> bounded below -> no ghost"
        )
    boundedness = polynomialBoundedBelow(reducedHamiltonian, reducedPositions, reducedMomenta)
    if boundedness["verdict"] == "unbounded_below" and stability == "oscillatory":
        return True, reducedHamiltonian, (
            f"reduced H is non-quadratic and unbounded below ({boundedness['detail']}); "
            "dynamics oscillatory -> Ostrogradski ghost"
        )
    if boundedness["verdict"] == "bounded_below":
        return False, reducedHamiltonian, (
            f"reduced H is non-quadratic but bounded below ({boundedness['detail']}) -> no ghost"
        )
    return None, reducedHamiltonian, (
        f"reduced H is non-quadratic and boundedness is '{boundedness['verdict']}' "
        f"({boundedness['detail']}) -> undetermined"
    )


def _phaseSpaceSymbols(hamiltonianData):
    positions = [symbol for group in hamiltonianData["positionSymbols"] for symbol in group]
    momenta = [symbol for group in hamiltonianData["momentumSymbols"] for symbol in group]
    return positions, momenta


def hamiltonianHessian(hamiltonian, variables):
    size = len(variables)
    hessian = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            hessian[i, j] = float(sp.diff(hamiltonian, variables[i], variables[j]))
    return hessian


def detectGhost(lagrangian, coords, order=None, constants=None, eigenvalueTolerance=1e-8):
    hamiltonianData = ostrogradskiHamiltonian(lagrangian, coords, order, constants)

    if isinstance(hamiltonianData, DegenerateLagrangianResult):
        try:
            roots = characteristicRoots(lagrangian, coords, order)
            stability = dynamicalStability(roots)
        except (ValueError, TypeError, sp.PolynomialError):
            roots, stability = [], "undetermined"
        try:
            ghost, reducedHamiltonian, verdictDetail = _degenerateGhostVerdict(
                hamiltonianData, stability, eigenvalueTolerance
            )
        except (ValueError, TypeError, sp.PolynomialError) as error:
            ghost, reducedHamiltonian = None, None
            verdictDetail = f"constrained-Hamiltonian reduction failed ({error}) -> undetermined"
        return {
            "ghost": ghost,
            "degenerate": True,
            "order": hamiltonianData.order,
            "constraintAnalysis": hamiltonianData,
            "chainClosed": hamiltonianData.chainClosed,
            "physicalPhaseSpaceDimension": hamiltonianData.physicalPhaseSpaceDimension,
            "reducedHamiltonian": reducedHamiltonian,
            "characteristicRoots": [complex(root) for root in roots],
            "dynamicalStability": stability,
            "detail": (
                f"Lagrangian is degenerate ({hamiltonianData.totalFirstClassCount} first-class, "
                f"{hamiltonianData.totalSecondClassCount} second-class constraint(s) after "
                f"Dirac-Bergmann). {verdictDetail}"
            ),
        }

    hamiltonian = sp.expand(hamiltonianData["hamiltonian"])

    positions, momenta = _phaseSpaceSymbols(hamiltonianData)
    variables = positions + momenta

    try:
        isQuadratic = sp.Poly(hamiltonian, *variables).total_degree() <= 2
    except sp.PolynomialError:
        isQuadratic = False

    roots = characteristicRoots(lagrangian, coords, order)
    stability = dynamicalStability(roots)

    result = {
        "order": hamiltonianData["order"],
        "hamiltonian": hamiltonian,
        "quadratic": isQuadratic,
        "characteristicRoots": [complex(root) for root in roots],
        "dynamicalStability": stability,
    }

    if not isQuadratic:
        boundedness = polynomialBoundedBelow(hamiltonian, positions, momenta)
        result["boundedness"] = boundedness["verdict"]
        result["boundednessDetail"] = boundedness["detail"]
        if boundedness["verdict"] == "unbounded_below":
            if stability == "oscillatory":
                result["ghost"] = True
                result["detail"] = (
                    f"H is non-quadratic and unbounded below ({boundedness['detail']}); "
                    "dynamics are oscillatory -> negative-energy mode present -> Ostrogradski ghost"
                )
            else:
                result["ghost"] = None
                result["detail"] = (
                    f"H is non-quadratic and unbounded below ({boundedness['detail']}) but "
                    f"dynamics are '{stability}': the tachyon/ghost split needs a trustworthy linear "
                    "dynamics classification, which a non-quadratic EOM does not give -> undetermined"
                )
        elif boundedness["verdict"] == "bounded_below":
            result["ghost"] = False
            result["detail"] = f"H is non-quadratic but bounded below ({boundedness['detail']}) -> no ghost"
        else:
            result["ghost"] = None
            result["detail"] = (
                f"H is non-quadratic and boundedness is undetermined ({boundedness['detail']}); "
                "full nonlinear boundedness analysis required"
            )
        return result

    hessian = hamiltonianHessian(hamiltonian, variables)
    symmetric = 0.5 * (hessian + hessian.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)

    momentumIndices = list(range(len(positions), len(variables)))
    momentumBlock = symmetric[np.ix_(momentumIndices, momentumIndices)]
    momentumEigenvalues = np.linalg.eigvalsh(momentumBlock)

    negativeDirections = int(np.sum(eigenvalues < -eigenvalueTolerance))
    nullMomentumDirections = int(np.sum(np.abs(momentumEigenvalues) < eigenvalueTolerance))
    indefinite = negativeDirections > 0

    if indefinite and stability == "oscillatory":
        ghost = True
        detail = (
            f"H is indefinite ({negativeDirections} negative direction(s)) while the dynamics are oscillatory: "
            f"negative-energy mode present -> Ostrogradski ghost. Linear-in-momentum sector: "
            f"{nullMomentumDirections} null momentum direction(s)."
        )
    elif indefinite:
        ghost = False
        detail = f"H is indefinite but dynamics are '{stability}': runaway/tachyonic instability rather than a ghost mode"
    else:
        ghost = False
        detail = "H is positive semidefinite -> bounded below -> no ghost"

    result.update(
        {
            "ghost": ghost,
            "hamiltonianEigenvalues": [float(value) for value in eigenvalues],
            "negativeDirections": negativeDirections,
            "momentumSectorEigenvalues": [float(value) for value in momentumEigenvalues],
            "nullMomentumDirections": nullMomentumDirections,
            "detail": detail,
        }
    )
    return result
