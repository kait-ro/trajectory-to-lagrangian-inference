def checkCorrelationCutoff(bestScore: float, cutoff: float = 0.1):
    scoreMagnitude = abs(bestScore)
    hasStalled = scoreMagnitude < cutoff
    return hasStalled, scoreMagnitude


def checkDegreeExpansionNeeded(
    residualConverged: bool,
    correlationStalled: bool,
    currentMaxDegree: int,
    degreeCap: int = 6,
):
    libraryTooNarrow = correlationStalled and not residualConverged
    if not libraryTooNarrow:
        return False, currentMaxDegree
    if currentMaxDegree >= degreeCap:
        return False, currentMaxDegree
    return True, currentMaxDegree + 1


def checkResidualStagnation(residualHistory: list, tolerance: float = 0.01, patience: int = 3) -> bool:
    if len(residualHistory) <= patience:
        return False
    window = residualHistory[-(patience + 1):]
    improvements = [
        (earlier - later) / earlier
        for earlier, later in zip(window[:-1], window[1:])
        if earlier > 0
    ]
    return len(improvements) == patience and all(step < tolerance for step in improvements)
