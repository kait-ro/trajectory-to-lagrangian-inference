from dataclasses import dataclass

import sympy as sp

from finding_L import gram_forward_select, stopping_conditions
from finding_L.main_streaming import runDiscoveryStreaming

LOCKED_TOLERANCES = {
    "correlationCutoff": stopping_conditions.checkCorrelationCutoff.__defaults__[0],
    "residualRmsTolerance": gram_forward_select.checkResidualToleranceFromGram.__defaults__[0],
    "degreeCap": stopping_conditions.checkDegreeExpansionNeeded.__defaults__[0],
}


@dataclass
class RecoveryComparison:
    success: bool
    missingMonomials: list
    spuriousMonomials: list
    coefficientErrors: dict
    maxAbsoluteError: float


def _coefficientDict(expression):
    return {monomial: float(coefficient) for monomial, coefficient in sp.expand(expression).as_coefficients_dict().items()}


def compareToExpected(
    discovered,
    expectedExpression,
    coefficientAbsoluteTolerance=0.05,
    coefficientRelativeTolerance=0.1,
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
    return RecoveryComparison(success, missing, spurious, coefficientErrors, maxAbsoluteError)


def runSystemDiscovery(system, csvPath, chunkRows=200_000):
    return runDiscoveryStreaming(
        csvPath,
        noCoords=system.noCoords,
        startingMaxDegree=system.startingMaxDegree,
        maxRounds=system.maxRounds,
        chunkRows=chunkRows,
        degreeCap=system.degreeCap,
        residualRmsTolerance=system.residualRmsTolerance,
    )


def formatComparison(comparison, discovered):
    lines = []
    lines.append(f"Locked tolerances: {LOCKED_TOLERANCES}")
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

    lines.append("")
    lines.append(discovered.text)
    return "\n".join(lines)
