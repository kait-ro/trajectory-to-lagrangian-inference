from dataclasses import dataclass

import sympy as sp

from finding_L import gram_forward_select, stopping_conditions
from finding_L.equivalence_class import classifyLagrangianPair, formatVerdict
from finding_L.main_streaming import runDiscoveryStreaming
from finding_L.report import stateExpressionToFunctional
from generation.eqnofmotion import defineCoordinates

# The one system tolerances are allowed to be calibrated against. Everything else
# is a blind holdout.
CALIBRATION_SYSTEM = "isotropic_quartic_calibration"

# --- Locked tolerances ------------------------------------------------------
# These knobs were calibrated once, on the ISOTROPIC_QUARTIC calibration system,
# and are then FROZEN. Every other system (currently only ANHARMONIC_CHAIN) is a
# blind holdout: it is scored with exactly these values and no per-system
# retuning. That discipline is what makes the holdout result meaningful, so
# runSystemDiscovery(..., enforceLocked=True) refuses to run a non-calibration
# system whose PhysicalSystem overrides any locked value, and threads these
# values explicitly into runDiscoveryStreaming rather than relying on the
# per-system dataclass fields.
LOCKED_TOLERANCES = {
    # forward-selection stall cutoff (finding_L/stopping_conditions.checkCorrelationCutoff)
    "correlationCutoff": stopping_conditions.checkCorrelationCutoff.__defaults__[0],
    # RMS residual convergence (finding_L/gram_forward_select.checkResidualToleranceFromGram)
    "residualRmsTolerance": gram_forward_select.checkResidualToleranceFromGram.__defaults__[0],
    # relative drop threshold in pruneNearZeroCoefficients (finding_L/gram_forward_select)
    "pruneRelativeThreshold": gram_forward_select.pruneNearZeroCoefficients.__defaults__[0],
    # residual-stagnation stop (finding_L/stopping_conditions.checkResidualStagnation)
    "stagnationTolerance": stopping_conditions.checkResidualStagnation.__defaults__[0],
    "stagnationPatience": stopping_conditions.checkResidualStagnation.__defaults__[1],
    # library degree ceiling -- structural (both benchmark systems are quartic)
    "degreeCap": 4,
}


def _assertNoToleranceOverride(system):
    violations = []
    if system.residualRmsTolerance != LOCKED_TOLERANCES["residualRmsTolerance"]:
        violations.append(
            f"residualRmsTolerance={system.residualRmsTolerance} "
            f"!= locked {LOCKED_TOLERANCES['residualRmsTolerance']}"
        )
    if system.degreeCap != LOCKED_TOLERANCES["degreeCap"]:
        violations.append(f"degreeCap={system.degreeCap} != locked {LOCKED_TOLERANCES['degreeCap']}")
    if violations:
        raise ValueError(
            f"holdout discipline violation: system '{system.name}' overrides locked tolerances "
            f"({'; '.join(violations)}). Recalibrate on '{CALIBRATION_SYSTEM}' or drop the override."
        )


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


def runSystemDiscovery(system, csvPath, chunkRows=200_000, enforceLocked=True):
    """Run streaming discovery on `system`.

    With enforceLocked=True (the default), every tolerance comes from
    LOCKED_TOLERANCES -- calibrated on CALIBRATION_SYSTEM -- and a non-calibration
    system that tries to override one raises. Only structural choices
    (noCoords, startingMaxDegree, maxRounds) are taken from the PhysicalSystem.
    """
    if enforceLocked and system.name != CALIBRATION_SYSTEM:
        _assertNoToleranceOverride(system)

    if enforceLocked:
        degreeCap = LOCKED_TOLERANCES["degreeCap"]
        residualRmsTolerance = LOCKED_TOLERANCES["residualRmsTolerance"]
        correlationCutoff = LOCKED_TOLERANCES["correlationCutoff"]
        stagnationTolerance = LOCKED_TOLERANCES["stagnationTolerance"]
        stagnationPatience = LOCKED_TOLERANCES["stagnationPatience"]
    else:
        degreeCap = system.degreeCap
        residualRmsTolerance = system.residualRmsTolerance
        correlationCutoff = stopping_conditions.checkCorrelationCutoff.__defaults__[0]
        stagnationTolerance = stopping_conditions.checkResidualStagnation.__defaults__[0]
        stagnationPatience = stopping_conditions.checkResidualStagnation.__defaults__[1]

    return runDiscoveryStreaming(
        csvPath,
        noCoords=system.noCoords,
        startingMaxDegree=system.startingMaxDegree,
        maxRounds=system.maxRounds,
        chunkRows=chunkRows,
        degreeCap=degreeCap,
        residualRmsTolerance=residualRmsTolerance,
        correlationCutoff=correlationCutoff,
        stagnationTolerance=stagnationTolerance,
        stagnationPatience=stagnationPatience,
    )


def lockedTolerancesReport():
    """Human-readable banner of the frozen tolerances, for experiment output."""
    lines = [
        f"Locked tolerances (calibrated on '{CALIBRATION_SYSTEM}', frozen for all holdout systems):"
    ]
    for name, value in LOCKED_TOLERANCES.items():
        lines.append(f"  {name:>22} = {value}")
    return "\n".join(lines)


def formatComparison(comparison, discovered):
    lines = []
    lines.append(lockedTolerancesReport())
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
