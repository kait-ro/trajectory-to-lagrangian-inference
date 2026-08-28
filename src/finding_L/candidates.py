import itertools

import sympy as sp


def buildCandidateLibrary(coords: list, vels: list, maxDegree: int) -> list[sp.Expr]:
    variables = list(coords) + list(vels)

    candidateTerms = []
    seenExponents = set()

    for degree in range(1, maxDegree + 1):
        for combo in itertools.combinations_with_replacement(range(len(variables)), degree):
            exponents = tuple(sorted(combo))
            if exponents in seenExponents:
                continue
            seenExponents.add(exponents)

            term = sp.Integer(1)
            for index in combo:
                term = term * variables[index]
            candidateTerms.append(sp.expand(term))

    return candidateTerms

def filterPureVelocityTerms(candidateTerms: list, coords: list) -> list:
    filteredTerms = [
        term for term in candidateTerms
        if any(sp.diff(term, q) != 0 for q in coords)
    ]
    return filteredTerms