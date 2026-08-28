import numpy as np
import sympy as sp
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
    hamiltonian = sp.expand(hamiltonianData["hamiltonian"])

    positions, momenta = _phaseSpaceSymbols(hamiltonianData)
    variables = positions + momenta

    polynomial = sp.Poly(hamiltonian, *variables)
    isQuadratic = polynomial.total_degree() <= 2

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
        result["ghost"] = None
        result["detail"] = "Hamiltonian not quadratic; use full nonlinear boundedness analysis"
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
