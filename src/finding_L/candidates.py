import itertools

import sympy as sp


def monomialLibrary(symbols, maxDegree):
    library = []
    seen = set()
    for degree in range(1, maxDegree + 1):
        for combo in itertools.combinations_with_replacement(range(len(symbols)), degree):
            key = tuple(sorted(combo))
            if key in seen:
                continue
            seen.add(key)
            monomial = sp.Integer(1)
            for index in combo:
                monomial = monomial * symbols[index]
            library.append(sp.expand(monomial))
    return library


def buildCandidateLibrary(coords: list, vels: list, maxDegree: int) -> list[sp.Expr]:
    return monomialLibrary(list(coords) + list(vels), maxDegree)


def filterPureVelocityTerms(candidateTerms: list, coords: list) -> list:
    filteredTerms = [
        term for term in candidateTerms
        if any(sp.diff(term, q) != 0 for q in coords)
    ]
    return filteredTerms
