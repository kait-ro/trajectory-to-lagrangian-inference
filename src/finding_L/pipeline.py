"""End-to-end recovery from noisy position data only.

noisy q(t)  ->  differentiation-method selection (unsupervised)
            ->  Lagrangian-order inference
            ->  higher-derivative Lagrangian recovery
            ->  Ostrogradski ghost verdict
            ->  confidence estimate (agreement across differentiation methods)

No step is given ground-truth derivatives or a pre-chosen differentiation method.
Single coordinate (multi-field noisy differentiation is not yet good enough --
see experiments/multi_field_discovery_validation.py).
"""

from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from finding_L.higher_order_discovery import (
    inferLagrangianOrder,
    recoverHigherOrderLagrangian,
    stateToCoordinate,
)
from generation.eqnofmotion import TIME, defineCoordinates
from generation.ghost_detection import detectGhost
from generation.ostrogradski import eulerLagrangeExpression
from generation.numerical_diff import (
    savitzkyGolayDerivatives,
    smoothingSplineDerivatives,
)

# differentiation methods to grid-search. finite differences are excluded: the
# study in experiments/differentiation_method_study.py shows they are unusable
# above order 1 under any noise.
_METHODS = {
    "savitzky_golay": savitzkyGolayDerivatives,
    "smoothing_spline": smoothingSplineDerivatives,
    "savitzky_golay_poly8": lambda signal, dt, maxOrder: savitzkyGolayDerivatives(signal, dt, maxOrder, polyOrder=8),
}


@dataclass
class PipelineResult:
    differentiationMethod: str
    lagrangianOrder: int
    discoveredLagrangian: sp.Expr        # state-symbol form (s0, s1, ...)
    discoveredLagrangianInCoords: sp.Expr
    ghost: object                        # True / False / None
    ghostDetail: str
    # Separate confidences: order and ghost are robust to the differentiation
    # step; the detailed coefficients are not (see PROJECT.md problem A).
    orderConfidence: float
    ghostConfidence: float
    coefficientConfidence: float
    confidence: float                    # overall = the weakest link
    perMethod: list = field(default_factory=list)
    coefficientSpread: dict = field(default_factory=dict)
    detail: str = ""

    def summary(self):
        lines = [
            f"differentiation method selected: {self.differentiationMethod}",
            f"inferred Lagrangian order:       {self.lagrangianOrder}   (confidence {self.orderConfidence:.2f})",
            f"discovered L:                    {self.discoveredLagrangian}",
            f"ghost:                           {self.ghost}   (confidence {self.ghostConfidence:.2f})",
            f"                                 {self.ghostDetail}",
            f"coefficient confidence:          {self.coefficientConfidence:.2f}",
            f"OVERALL confidence:              {self.confidence:.2f}",
            "per differentiation method:",
        ]
        for record in self.perMethod:
            lines.append(
                f"  {record['method']:>20}: order {record['order']}, ghost {record['ghost']}, "
                f"own-EL residual {record['orderResidual']:.4f}"
            )
        if self.coefficientSpread:
            lines.append("coefficient spread across methods (std):")
            for monomial, spread in sorted(self.coefficientSpread.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {str(monomial):>14}: {spread:.4f}")
        lines.append(self.detail)
        return "\n".join(lines)


_METHOD_MAX_LEVEL = {"smoothing_spline": 4}  # the quintic spline caps at 4th derivative


def _derivativeColumns(name, method, signal, dt, maxLevel):
    level = min(maxLevel, _METHOD_MAX_LEVEL.get(name, maxLevel))
    derivatives = method(np.asarray(signal, dtype=float), dt, level)
    return [np.asarray(component, dtype=float) for component in derivatives]


def _lagrangianElResidual(recoveredStateExpression, noStateVars, order, columns):
    """||EL(recovered L)|| / ||EL(kinetic)|| on the data -- how well the *recovered*
    (sparse) Lagrangian, not a dense projection, satisfies its own EL equation."""
    _t, coords, _v = defineCoordinates(1)
    coordinate = coords[0]
    kineticLevel = min(2, order)
    lagrangian = stateToCoordinate(recoveredStateExpression, noStateVars, coordinate)
    kinetic = stateToCoordinate(sp.Symbol(f"s{kineticLevel}") ** 2, noStateVars, coordinate)

    columnOrder = 2 * order
    dataSymbols = [sp.Symbol(f"d{k}") for k in range(columnOrder + 1)]
    toData = {(coordinate if k == 0 else sp.diff(coordinate, TIME, k)): dataSymbols[k] for k in range(columnOrder + 1)}

    def evaluate(expression):
        substituted = eulerLagrangeExpression(expression, coordinate, order, pipelineSign=True).subs(toData)
        if substituted.atoms(sp.Derivative):
            return None
        function = sp.lambdify(dataSymbols, substituted, modules="numpy")
        return np.broadcast_to(np.asarray(function(*columns[: columnOrder + 1]), dtype=float), (len(columns[0]),))

    lagrangianColumn = evaluate(lagrangian)
    kineticColumn = evaluate(kinetic)
    if lagrangianColumn is None or kineticColumn is None:
        return float("inf")
    denominator = np.linalg.norm(kineticColumn)
    return float(np.linalg.norm(lagrangianColumn) / max(denominator, 1e-30))


def _recoverAndDiagnose(columns, maxOrder, libraryMaxDegree):
    order, _perOrder = inferLagrangianOrder(columns, maxOrder=maxOrder, libraryMaxDegree=libraryMaxDegree)
    noStateVars = order + 1
    recovered, selected = recoverHigherOrderLagrangian(
        columns[: 2 * order + 1], noStateVars, order, libraryMaxDegree=libraryMaxDegree
    )
    orderResidual = _lagrangianElResidual(recovered, noStateVars, order, columns[: 2 * order + 1])

    _t, coords, _v = defineCoordinates(1)
    lagrangianInCoords = stateToCoordinate(recovered, noStateVars, coords[0])
    try:
        verdict = detectGhost(lagrangianInCoords, coords, order=order)
        ghost = verdict.get("ghost")
        ghostDetail = verdict.get("detail", "")
    except Exception as error:  # ghost analysis can fail on a badly-recovered L
        ghost, ghostDetail = None, f"ghost analysis failed: {error}"

    return {
        "order": order,
        "recovered": recovered,
        "selected": selected,
        "lagrangianInCoords": lagrangianInCoords,
        "ghost": ghost,
        "ghostDetail": ghostDetail,
        "orderResidual": float(orderResidual),
    }


def _coefficientDict(expression):
    return {
        monomial: float(coefficient)
        for monomial, coefficient in sp.expand(expression).as_coefficients_dict().items()
        if monomial != 1
    }


def endToEndPipeline(noisyPositions, dt, maxOrder=3, libraryMaxDegree=2):
    """Run the whole chain on a single noisy position signal. Returns PipelineResult."""
    signal = np.asarray(noisyPositions, dtype=float).reshape(-1)
    maxLevel = 2 * maxOrder

    perMethod = []
    for name, method in _METHODS.items():
        try:
            columns = _derivativeColumns(name, method, signal, dt, maxLevel)
            diagnosis = _recoverAndDiagnose(columns, maxOrder, libraryMaxDegree)
            diagnosis["method"] = name
            perMethod.append(diagnosis)
        except Exception as error:
            perMethod.append({"method": name, "order": None, "ghost": None, "orderResidual": float("inf"), "error": str(error)})

    usable = [record for record in perMethod if record.get("order") is not None]
    if not usable:
        raise RuntimeError("every differentiation method failed on this signal")

    # order by majority vote across methods (robust); ghost by majority among
    # methods that agree on that order.
    orders = [record["order"] for record in usable]
    consensusOrder = max(set(orders), key=orders.count)
    atConsensus = [record for record in usable if record["order"] == consensusOrder]

    # --- differentiation-method selection --------------------------------------
    # A recovered L with a huge coefficient is a near-total-derivative artifact of
    # bad higher derivatives; prefer plausible recoveries, then lowest own-EL
    # residual.
    def _maxAbsCoefficient(record):
        return max((abs(v) for v in _coefficientDict(record["recovered"]).values()), default=0.0)

    plausible = [record for record in atConsensus if _maxAbsCoefficient(record) < 20.0]
    pool = plausible if plausible else atConsensus
    best = min(pool, key=lambda record: record["orderResidual"])

    # --- confidences ----------------------------------------------------------
    ghosts = [record["ghost"] for record in atConsensus]
    orderConfidence = orders.count(consensusOrder) / len(orders)
    ghostConfidence = ghosts.count(best["ghost"]) / len(ghosts)

    # coefficient spread across the *plausible* recoveries only (an implausible
    # one is a known artifact, not a genuine disagreement)
    coefficientSpread = {}
    spreadPool = plausible if len(plausible) > 1 else atConsensus
    coefficientDicts = [_coefficientDict(record["recovered"]) for record in spreadPool]
    if len(coefficientDicts) > 1:
        for monomial in set().union(*[set(d) for d in coefficientDicts]):
            values = [d.get(monomial, 0.0) for d in coefficientDicts]
            coefficientSpread[monomial] = float(np.std(values))
    maxSpread = max(coefficientSpread.values(), default=0.0)
    implausibleCount = len(atConsensus) - len(plausible)
    plausibleSelection = _maxAbsCoefficient(best) < 20.0
    coefficientConfidence = float(
        np.clip(
            (1.0 / (1.0 + 3.0 * maxSpread))
            * (1.0 if plausibleSelection else 0.2)
            * (1.0 - 0.3 * implausibleCount / max(len(atConsensus), 1)),
            0.0,
            1.0,
        )
    )

    confidence = float(min(orderConfidence, ghostConfidence))  # overall = the robust part
    if not plausibleSelection or maxSpread > 1.0:
        detailSuffix = " Coefficients are unreliable (differentiation-limited); trust the order and ghost verdict, not the numbers."
    else:
        detailSuffix = ""

    detail = (
        f"{orderConfidence:.0%} of methods agree on order {consensusOrder}, "
        f"{ghostConfidence:.0%} agree on the ghost verdict; "
        f"max coefficient spread {maxSpread:.3f}.{detailSuffix}"
    )

    return PipelineResult(
        differentiationMethod=best["method"],
        lagrangianOrder=consensusOrder,
        discoveredLagrangian=best["recovered"],
        discoveredLagrangianInCoords=best["lagrangianInCoords"],
        ghost=best["ghost"],
        ghostDetail=best["ghostDetail"],
        orderConfidence=orderConfidence,
        ghostConfidence=ghostConfidence,
        coefficientConfidence=coefficientConfidence,
        confidence=confidence,
        perMethod=[
            {
                "method": record["method"],
                "order": record.get("order"),
                "ghost": record.get("ghost"),
                "orderResidual": record.get("orderResidual", float("inf")),
            }
            for record in perMethod
        ],
        coefficientSpread=coefficientSpread,
        detail=detail,
    )
