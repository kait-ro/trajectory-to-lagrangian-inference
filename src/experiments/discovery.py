from dataclasses import dataclass

import sympy as sp

from finding_L import gram_forward_select, stopping_conditions
from finding_L.equivalence_class import classifyLagrangianPair, formatVerdict
from finding_L.main_streaming import runDiscoveryStreaming
from finding_L.report import stateExpressionToFunctional
from generation.eqnofmotion import defineCoordinates

# The system the recovery pipeline was originally shaped around. NOTHING is
# calibrated against it here -- no tuning search is run. It is only the reference
# point that the blind holdout (any other system) is contrasted with, and the
# "calibration" in its name is historical (it is the dataset stem too).
REFERENCE_SYSTEM = "isotropic_quartic_calibration"

# --- Frozen tolerances -----------------------------------------------------
# The library default tolerances, frozen verbatim as the SINGLE set used for
# every system. These are NOT the output of a calibration / parameter search --
# they are the defaults that already ship in finding_L, written out here so the
# value reported as "the tolerance" is provably the value the pipeline uses.
#
# Every entry is threaded explicitly into runDiscoveryStreaming (below), so
# changing a value here changes behaviour; none of them fall back to a function
# default. PhysicalSystem carries no tolerance fields -- the holdout cannot
# diverge.
#
#   key                    source (file / function / default)
#   ---------------------  --------------------------------------------------------------
#   correlationCutoff      finding_L/stopping_conditions.py checkCorrelationCutoff  (0.1)
#   residualRmsTolerance   finding_L/gram_forward_select.py checkResidualToleranceFromGram  (0.01)
#   pruneRelativeThreshold finding_L/gram_forward_select.py pruneNearZeroCoefficients  (1e-2)
#   stagnationTolerance    finding_L/stopping_conditions.py checkResidualStagnation  (0.01)
#   stagnationPatience     finding_L/stopping_conditions.py checkResidualStagnation  (3)
#   degreeCap              max monomial degree the library may expand to. 4 == the degree
#                          of both benchmark Lagrangians (quartic); this is the value the
#                          sweep has always used. The checkDegreeExpansionNeeded signature
#                          default (6) is only a generic-API fallback and is never used here.
FROZEN_TOLERANCES = {
    "correlationCutoff": stopping_conditions.checkCorrelationCutoff.__defaults__[0],
    "residualRmsTolerance": gram_forward_select.checkResidualToleranceFromGram.__defaults__[0],
    "pruneRelativeThreshold": gram_forward_select.pruneNearZeroCoefficients.__defaults__[0],
    "stagnationTolerance": stopping_conditions.checkResidualStagnation.__defaults__[0],
    "stagnationPatience": stopping_conditions.checkResidualStagnation.__defaults__[1],
    "degreeCap": 4,
}


def lockedDiscoveryTolerances(system=None):
    """The exact tolerance kwargs runSystemDiscovery feeds the pipeline.

    `system` is accepted and ignored: the point is that the frozen set does not
    depend on which system is being run, so `lockedDiscoveryTolerances(a) ==
    lockedDiscoveryTolerances(b)` for any two systems.
    """
    return dict(FROZEN_TOLERANCES)


def frozenTolerancesReport():
    lines = [
        "Frozen tolerances (finding_L library defaults, identical for every system; "
        "no tuning search was run):"
    ]
    for name, value in FROZEN_TOLERANCES.items():
        lines.append(f"  {name:>22} = {value}")
    return "\n".join(lines)


@dataclass
class RecoveryComparison:
    success: bool
    missingMonomials: list
    spuriousMonomials: list
    coefficientErrors: dict
    maxAbsoluteError: float
    # Verdict from finding_L.equivalence_class on (discovered - expected): is the
    # difference a genuine null Lagrangian (EL operator identically zero), or are
    # the two Lagrangians physically distinct and only numerically close?
    equivalenceVerdict: object = None

    @property
    def structurallyEquivalent(self):
        return self.equivalenceVerdict is not None and self.equivalenceVerdict.equivalent


def _coefficientDict(expression):
    return {monomial: float(coefficient) for monomial, coefficient in sp.expand(expression).as_coefficients_dict().items()}


def _inferNoCoords(*expressions):
    maxIndex = -1
    for expression in expressions:
        for symbol in sp.sympify(expression).free_symbols:
            name = symbol.name
            if len(name) >= 2 and name[0] in "qv" and name[1:].isdigit():
                maxIndex = max(maxIndex, int(name[1:]))
    return maxIndex + 1


def compareToExpected(
    discovered,
    expectedExpression,
    coefficientAbsoluteTolerance=0.05,
    coefficientRelativeTolerance=0.1,
    noCoords=None,
):
    expected = _coefficientDict(expectedExpression)
    recoveredRaw = _coefficientDict(discovered.rawExpression)
    recoveredSnapped = _coefficientDict(discovered.expression)

    expectedMonomials = {monomial for monomial in expected if monomial != 1}
    recoveredMonomials = {monomial for monomial, value in recoveredSnapped.items() if monomial != 1 and value != 0.0}

    missing = sorted(expectedMonomials - recoveredMonomials, key=sp.default_sort_key)
    spurious = sorted(recoveredMonomials - expectedMonomials, key=sp.default_sort_key)
    shared = sorted(expectedMonomials & recoveredMonomials, key=sp.default_sort_key)

    coefficientErrors = {}
    maxAbsoluteError = 0.0
    coefficientsWithinTolerance = True
    for monomial in shared:
        expectedValue = expected[monomial]
        recoveredValue = recoveredRaw.get(monomial, 0.0)
        absoluteError = abs(recoveredValue - expectedValue)
        tolerance = coefficientAbsoluteTolerance + coefficientRelativeTolerance * abs(expectedValue)
        passed = absoluteError <= tolerance
        coefficientErrors[monomial] = {
            "expected": expectedValue,
            "recovered": recoveredValue,
            "absoluteError": absoluteError,
            "passed": passed,
        }
        maxAbsoluteError = max(maxAbsoluteError, absoluteError)
        coefficientsWithinTolerance = coefficientsWithinTolerance and passed

    success = not missing and not spurious and coefficientsWithinTolerance

    # Equivalence-class check: rather than trusting "same monomials, close
    # coefficients", verify that (discovered - expected) is annihilated by the
    # Euler-Lagrange operator. difference == 0 => exact structural match;
    # null but nonzero => the two differ only by a total time derivative;
    # non-null => physically distinct theories that merely scored similarly.
    if noCoords is None:
        noCoords = _inferNoCoords(expectedExpression, discovered.expression)
    _t, coords, vels = defineCoordinates(noCoords)
    discoveredFunctional = stateExpressionToFunctional(discovered.expression, coords, vels)
    expectedFunctional = stateExpressionToFunctional(expectedExpression, coords, vels)
    equivalenceVerdict = classifyLagrangianPair(discoveredFunctional, expectedFunctional, coords, vels)

    return RecoveryComparison(
        success, missing, spurious, coefficientErrors, maxAbsoluteError, equivalenceVerdict
    )


def runSystemDiscovery(system, csvPath, chunkRows=200_000):
    """Run streaming discovery on `system` with the frozen tolerance set.

    Every tolerance comes from FROZEN_TOLERANCES; only search budgets
    (noCoords, startingMaxDegree, maxRounds) come from the PhysicalSystem.

    Returns (discovered, logFrame, tolerancesUsed). `tolerancesUsed` is the exact
    dict passed to the pipeline, so a caller can assert the holdout ran on the
    same set as the reference system.
    """
    tolerances = lockedDiscoveryTolerances(system)
    # Provably identical to the frozen set for any system:
    assert tolerances == FROZEN_TOLERANCES, "discovery tolerances diverged from FROZEN_TOLERANCES"

    discovered, logFrame = runDiscoveryStreaming(
        csvPath,
        noCoords=system.noCoords,
        startingMaxDegree=system.startingMaxDegree,   # search budget
        maxRounds=system.maxRounds,                    # search budget
        chunkRows=chunkRows,
        degreeCap=tolerances["degreeCap"],
        residualRmsTolerance=tolerances["residualRmsTolerance"],
        correlationCutoff=tolerances["correlationCutoff"],
        stagnationTolerance=tolerances["stagnationTolerance"],
        stagnationPatience=tolerances["stagnationPatience"],
        pruneRelativeThreshold=tolerances["pruneRelativeThreshold"],
    )
    return discovered, logFrame, tolerances


def formatComparison(comparison, discovered):
    lines = []
    lines.append(frozenTolerancesReport())
    lines.append(f"Recovery success: {comparison.success}")
    lines.append(f"Max |coefficient error| on shared terms: {comparison.maxAbsoluteError:.4f}")

    if comparison.missingMonomials:
        lines.append(f"Missing terms ({len(comparison.missingMonomials)}): {comparison.missingMonomials}")
    else:
        lines.append("Missing terms: none")

    if comparison.spuriousMonomials:
        lines.append(f"Spurious terms ({len(comparison.spuriousMonomials)}): {comparison.spuriousMonomials}")
    else:
        lines.append("Spurious terms: none")

    lines.append("Coefficient comparison (monomial: expected -> recovered, |err|, pass):")
    for monomial, record in comparison.coefficientErrors.items():
        lines.append(
            f"  {monomial}: {record['expected']:+.4f} -> {record['recovered']:+.4f}  "
            f"|err|={record['absoluteError']:.4f}  {'ok' if record['passed'] else 'FAIL'}"
        )

    if comparison.equivalenceVerdict is not None:
        lines.append("")
        lines.append("Equivalence-class check on (discovered - expected):")
        for verdictLine in formatVerdict(comparison.equivalenceVerdict).splitlines():
            lines.append(f"  {verdictLine}")

    lines.append("")
    lines.append(discovered.text)
    return "\n".join(lines)
