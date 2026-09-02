import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from experiments.discovery import FROZEN_TOLERANCES, compareToExpected
from finding_L.build_matrix import buildGramMatrixChunked
from finding_L.candidates import buildCandidateLibrary, filterPureVelocityTerms
from finding_L.higher_order_discovery import forwardSelectFromGram
from finding_L.main_streaming import runDiscoveryStreaming
from finding_L.regularized_select import lassoSelect, sequentialThresholdedLeastSquares
from finding_L.report import assembleDiscoveredLagrangian
from generation.eqnofmotion import TIME, defineCoordinates
from generation.ghost_detection import detectGhost

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "assets"
CACHE = REPO / "src" / "experiments" / "results" / "model_selection_comparison.json"
DASHBOARD_PNG = ASSETS / "recovery_dashboard.png"
NOISE_PNG = ASSETS / "noise_ceiling.png"

OK = "#2e7d32"
BAD = "#c62828"
UNDET = "#8a8a8a"
GREEDY_C = "#c62828"
LASSO_C = "#1565c0"
STLSQ_C = "#f9a825"
DEGREE = FROZEN_TOLERANCES["degreeCap"]
PRUNE = FROZEN_TOLERANCES["pruneRelativeThreshold"]


def expectedLagrangian():
    q = sp.symbols("q0 q1 q2")
    v = sp.symbols("v0 v1 v2")
    rSquared = sum(s**2 for s in q)
    vSquared = sum(s**2 for s in v)
    return sp.expand(vSquared - rSquared - sp.Rational(3, 20) * rSquared**2)


def coefficientDict(expression):
    return {
        monomial: float(coefficient)
        for monomial, coefficient in sp.expand(expression)
        .as_coefficients_dict()
        .items()
        if monomial != 1
    }


def runDiscovery():
    frames = []
    discovered, logFrame = runDiscoveryStreaming(
        str(ASSETS / "isotropic_quartic_small_n3_noise1.csv"),
        noCoords=3,
        startingMaxDegree=2,
        maxRounds=60,
        chunkRows=120_000,
        degreeCap=FROZEN_TOLERANCES["degreeCap"],
        residualRmsTolerance=FROZEN_TOLERANCES["residualRmsTolerance"],
        correlationCutoff=FROZEN_TOLERANCES["correlationCutoff"],
        stagnationTolerance=FROZEN_TOLERANCES["stagnationTolerance"],
        stagnationPatience=FROZEN_TOLERANCES["stagnationPatience"],
        pruneRelativeThreshold=FROZEN_TOLERANCES["pruneRelativeThreshold"],
        roundCallback=frames.append,
        selector="greedy",
    )
    return discovered, logFrame, frames


def buildGram(csvPath, noCoords=3, chunkRows=150_000):
    _t, coords, vels = defineCoordinates(noCoords)
    kineticTerm = sum(v**2 for v in vels)
    candidateTerms = filterPureVelocityTerms(
        buildCandidateLibrary(coords, vels, DEGREE), coords
    )
    if kineticTerm not in candidateTerms:
        candidateTerms.append(kineticTerm)
    n, colSum, gram = buildGramMatrixChunked(
        candidateTerms, coords, vels, TIME, csvPath, noCoords, chunkRows
    )
    kineticIndex = candidateTerms.index(kineticTerm)
    variance = gram.diagonal() / n - (colSum / n) ** 2
    admissible = [
        i
        for i in range(len(candidateTerms))
        if variance[i] > 1e-12 or i == kineticIndex
    ]
    subGram = gram[np.ix_(admissible, admissible)]
    subTerms = [candidateTerms[i] for i in admissible]
    subKinetic = admissible.index(kineticIndex)
    return subGram, subTerms, subKinetic, coords, vels, kineticTerm


def selectorScores(csvPath):
    gram, terms, kineticIndex, coords, vels, kineticTerm = buildGram(csvPath)
    b = -gram[:, kineticIndex]
    expected = expectedLagrangian()

    greedyActive, greedyCoef = forwardSelectFromGram(gram, kineticIndex, maxRounds=150)
    stlsqActive, stlsqCoef = sequentialThresholdedLeastSquares(
        gram, b, kineticIndex, relativeThreshold=PRUNE
    )
    lassoActive, lassoCoef = lassoSelect(gram, b, kineticIndex, relativeThreshold=PRUNE)

    methods = {
        "greedy": (greedyActive, greedyCoef),
        "stlsq": (stlsqActive, stlsqCoef),
        "lasso": (lassoActive, lassoCoef),
    }
    out = {}
    for name, (active, coefficients) in methods.items():
        discoveredTerms = [(terms[i], float(c)) for i, c in zip(active, coefficients)]
        discovered = assembleDiscoveredLagrangian(
            kineticTerm, discoveredTerms, coords, vels
        )
        comparison = compareToExpected(discovered, expected, noCoords=3)
        out[name] = {
            "maxAbsError": comparison.maxAbsoluteError,
            "missing": len(comparison.missingMonomials),
            "spurious": len(comparison.spuriousMonomials),
            "success": comparison.success,
        }
    return out


def cachedCalibration():
    data = json.loads(CACHE.read_text())
    rows = data["isotropic_quartic_calibration"]
    noise = [row["noiseLevel"] * 100 for row in rows]
    greedy = [row["greedy"]["maxAbsError"] for row in rows]
    stlsq = [row["stlsq"]["maxAbsError"] for row in rows]
    lasso = [row["lasso"]["maxAbsError"] for row in rows]
    greedySuccess = [row["greedy"]["success"] for row in rows]
    return noise, greedy, stlsq, lasso, greedySuccess


def ghostBattery():
    _t, coords, _v = defineCoordinates(1)
    q = coords[0]
    v = sp.diff(q, TIME)
    a = sp.diff(q, TIME, 2)

    systems = [
        ("SHO w=1", "healthy", sp.Rational(1, 2) * v**2 - sp.Rational(1, 2) * q**2),
        ("SHO w=2", "healthy", sp.Rational(1, 2) * v**2 - sp.Rational(1, 2) * 4 * q**2),
        (
            "anharmonic",
            "healthy",
            sp.Rational(1, 2) * v**2
            - sp.Rational(1, 2) * q**2
            - sp.Rational(1, 4) * q**4,
        ),
    ]
    for w1, w2 in [(1.0, 2.0), (1.0, 3.0), (0.7, 1.6)]:
        pu = sp.Rational(1, 2) * (a**2 - (w1**2 + w2**2) * v**2 + w1**2 * w2**2 * q**2)
        systems.append((f"PU {w1},{w2}", "ghost", sp.expand(pu)))

    results = []
    for name, label, lagrangian in systems:
        verdict = detectGhost(lagrangian, coords)
        eigenvalues = verdict.get("hamiltonianEigenvalues")
        results.append(
            {
                "name": name,
                "label": label,
                "ghost": verdict["ghost"],
                "minEig": min(eigenvalues) if eigenvalues else None,
                "maxAbsEig": max(abs(e) for e in eigenvalues) if eigenvalues else None,
                "detail": verdict.get("detail", ""),
            }
        )
    return results


def ghostColour(ghost):
    if ghost is True:
        return BAD
    if ghost is False:
        return OK
    return UNDET


def panelParity(ax, discovered, comparison, expected):
    expectedCoefficients = coefficientDict(expected)
    recoveredCoefficients = coefficientDict(discovered.rawExpression)
    monomials = sorted(
        set(expectedCoefficients) | set(recoveredCoefficients), key=sp.default_sort_key
    )
    xs, ys, colours = [], [], []
    for monomial in monomials:
        trueValue = expectedCoefficients.get(monomial, 0.0)
        recoveredValue = recoveredCoefficients.get(monomial, 0.0)
        xs.append(trueValue)
        ys.append(recoveredValue)
        record = comparison.coefficientErrors.get(monomial)
        if record is not None:
            colours.append(OK if record["passed"] else BAD)
        elif abs(recoveredValue) > 1e-6 and trueValue == 0.0:
            colours.append(BAD)
        else:
            colours.append(OK)
    span = xs + ys
    limits = [min(span) - 0.15, max(span) + 0.15]
    ax.plot(limits, limits, "--", color="#999999", lw=1, zorder=1)
    ax.scatter(xs, ys, c=colours, s=48, edgecolors="k", linewidths=0.4, zorder=3)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("true coefficient")
    ax.set_ylabel("recovered (raw, unsnapped) coefficient")
    ax.set_title("1. Coefficient parity")
    ax.grid(alpha=0.3)


def panelResidual(ax, frames):
    rounds = [frame["round"] for frame in frames]
    residual = [frame["scaledResidual"] for frame in frames]
    degree = [frame["currentMaxDegree"] for frame in frames]
    ax.semilogy(rounds, residual, "o-", color=LASSO_C, ms=4)
    top = max(residual)
    bottom = min(residual)
    for i in range(1, len(frames)):
        if degree[i] != degree[i - 1]:
            ax.axvline(rounds[i], color="#6a1b9a", ls="--", lw=1)
            ax.text(
                rounds[i],
                top,
                f" degree -> {degree[i]}",
                rotation=90,
                va="top",
                fontsize=7,
                color="#6a1b9a",
            )
    converged = [frame["round"] for frame in frames if frame.get("converged")]
    if converged:
        ax.axvline(converged[0], color=OK, ls="-", lw=1.5)
        ax.text(
            converged[0],
            bottom,
            " converged",
            rotation=90,
            va="bottom",
            fontsize=7,
            color=OK,
        )
    ax.set_xlabel("round")
    ax.set_ylabel("scaled residual (log)")
    ax.set_title("2. Residual vs round")
    ax.grid(alpha=0.3, which="both")


def panelTermError(ax, comparison):
    items = sorted(
        comparison.coefficientErrors.items(), key=lambda kv: kv[1]["absoluteError"]
    )
    labels = [str(monomial) for monomial, _ in items]
    errors = [record["absoluteError"] for _, record in items]
    colours = [OK if record["passed"] else BAD for _, record in items]
    positions = np.arange(len(labels))
    ax.barh(positions, errors, color=colours)
    ax.set_xscale("log")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0.05, color="k", ls="--", lw=1, label="abs tolerance 0.05")
    ax.set_xlabel("|coefficient error| on shared terms (log)")
    ax.set_title("3. Per-term coefficient error")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="x")


def panelSelectors(ax, freshRows, cached):
    noise, greedy, stlsq, lasso, _greedySuccess = cached
    ax.semilogy(noise, greedy, "o-", color=GREEDY_C, label="greedy (6-DOF cache)")
    ax.semilogy(noise, stlsq, "s-", color=STLSQ_C, label="STLSQ (6-DOF cache)")
    ax.semilogy(noise, lasso, "^-", color=LASSO_C, label="LASSO (6-DOF cache)")
    freshNoise = sorted(freshRows)
    ax.semilogy(
        freshNoise,
        [freshRows[x]["greedy"]["maxAbsError"] for x in freshNoise],
        "x",
        color=GREEDY_C,
        ms=11,
        mew=2.5,
        label="greedy (fresh 3-DOF)",
    )
    ax.semilogy(
        freshNoise,
        [freshRows[x]["lasso"]["maxAbsError"] for x in freshNoise],
        "+",
        color=LASSO_C,
        ms=13,
        mew=2.5,
        label="LASSO (fresh 3-DOF)",
    )
    ax.set_xlabel("position noise (%)")
    ax.set_ylabel("max |coefficient error| (log)")
    ax.set_title(
        "4. Model selection vs noise\n(line: cached 6-DOF; x/+: fresh 3-DOF, 0/1% only)",
        fontsize=10,
    )
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3, which="both")


def panelGhost(ax, battery):
    for index, record in enumerate(battery):
        colour = ghostColour(record["ghost"])
        x = record["minEig"] if record["minEig"] is not None else 0.0
        ax.scatter([x], [index], c=colour, s=90, edgecolors="k", zorder=3)
        suffix = "" if record["minEig"] is not None else "  (non-quadratic H)"
        ax.text(x, index + 0.18, record["name"] + suffix, fontsize=7, ha="center")
    ax.axvline(0.0, color="k", ls="--", lw=1)
    ax.set_xscale("symlog")
    ax.set_ylim(-0.8, len(battery) - 0.2)
    ax.set_xlabel("min Hessian(H) eigenvalue (symlog); x=0 is the decision boundary")
    ax.set_ylabel("system index")
    ax.set_title(
        "5. Ghost map\n(green healthy, red ghost, grey undetermined)", fontsize=10
    )
    ax.grid(alpha=0.3)


def wrapExpression(expression):
    return "\n  ".join(textwrap.wrap(str(expression), width=64))


def panelText(ax, discovered, comparison, frames):
    ax.axis("off")
    lines = [
        "RECOVERED L (snapped):",
        "  " + wrapExpression(discovered.expression),
        "",
        f"structurallyEquivalent : {comparison.structurallyEquivalent}",
        f"success                : {comparison.success}",
        f"maxAbsoluteError       : {comparison.maxAbsoluteError:.6f}",
        f"rounds                 : {frames[-1]['round']}  ({len(frames)} frames)",
        f"final scaledResidual   : {frames[-1]['scaledResidual']:.6e}",
        f"missing monomials      : {[str(m) for m in comparison.missingMonomials] or 'none'}",
        f"spurious monomials     : {[str(m) for m in comparison.spuriousMonomials] or 'none'}",
        "",
        "FROZEN_TOLERANCES:",
    ]
    for name, value in FROZEN_TOLERANCES.items():
        lines.append(f"  {name:>22} = {value}")
    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        transform=ax.transAxes,
    )
    ax.set_title("6. Verdict / metrics")


def buildDashboard(discovered, comparison, frames, freshRows, cached, battery):
    expected = expectedLagrangian()
    figure, axes = plt.subplots(2, 3, figsize=(18, 11))
    panelParity(axes[0, 0], discovered, comparison, expected)
    panelResidual(axes[0, 1], frames)
    panelTermError(axes[0, 2], comparison)
    panelSelectors(axes[1, 0], freshRows, cached)
    panelGhost(axes[1, 1], battery)
    panelText(axes[1, 2], discovered, comparison, frames)
    figure.suptitle(
        "Lagrangian recovery diagnostic dashboard -- isotropic quartic 3-DOF, 1% position noise",
        fontsize=13,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96], w_pad=2.5, h_pad=3.0)
    figure.savefig(DASHBOARD_PNG, dpi=130)
    plt.close(figure)


def buildNoiseCeiling(cached):
    noise, greedy, _stlsq, lasso, greedySuccess = cached
    figure, ax = plt.subplots(figsize=(9, 5.6))
    ax.semilogy(
        noise,
        greedy,
        "o-",
        color=GREEDY_C,
        lw=2,
        ms=7,
        label="greedy forward-selection",
    )
    ax.semilogy(noise, lasso, "s-", color=LASSO_C, lw=2, ms=7, label="debiased LASSO")

    ceiling = next(
        (noise[i] for i, ok in enumerate(greedySuccess) if not ok), noise[-1]
    )
    ax.axvspan(ceiling, noise[-1], color=GREEDY_C, alpha=0.08)
    ax.axvline(ceiling, color=GREEDY_C, ls="--", lw=1)
    ax.text(
        ceiling + 0.1,
        greedy[-1] * 0.5,
        f"greedy fails from ~{ceiling:g}%\nLASSO still recovers through {noise[-1]:g}%",
        fontsize=8,
        color=GREEDY_C,
        va="top",
    )
    ax.set_xlabel("position noise (%)")
    ax.set_ylabel("max |coefficient error| on shared terms (log)")
    ax.set_title("Noise ceiling: greedy forward-selection vs debiased LASSO")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    ax.text(
        1.0,
        -0.16,
        "source: src/experiments/results/model_selection_comparison.json "
        "(isotropic quartic, degree-4 library, one seed per noise level)",
        transform=ax.transAxes,
        ha="right",
        fontsize=6.5,
        color="#666666",
    )
    figure.tight_layout()
    figure.savefig(NOISE_PNG, dpi=150)
    plt.close(figure)


def main():
    discovered, _logFrame, frames = runDiscovery()
    expected = expectedLagrangian()
    comparison = compareToExpected(discovered, expected, noCoords=3)

    freshRows = {}
    for percentage, filename in [
        (0.0, "isotropic_quartic_small_n3_noise0.csv"),
        (1.0, "isotropic_quartic_small_n3_noise1.csv"),
    ]:
        freshRows[percentage] = selectorScores(str(ASSETS / filename))

    cached = cachedCalibration()
    battery = ghostBattery()

    buildDashboard(discovered, comparison, frames, freshRows, cached, battery)
    buildNoiseCeiling(cached)

    print(f"recovered L        : {discovered.expression}")
    print(f"structurallyEquiv. : {comparison.structurallyEquivalent}")
    print(f"success            : {comparison.success}")
    print(f"maxAbsoluteError   : {comparison.maxAbsoluteError:.6f}")
    print(f"rounds             : {frames[-1]['round']} ({len(frames)} frames)")
    print(f"final scaledResid. : {frames[-1]['scaledResidual']:.6e}")
    print(
        f"missing / spurious : {comparison.missingMonomials} / {comparison.spuriousMonomials}"
    )
    print(f"fresh 3-DOF rows   : {freshRows}")
    print(
        f"dashboard          : {DASHBOARD_PNG} ({DASHBOARD_PNG.stat().st_size} bytes)"
    )
    print(f"noise ceiling      : {NOISE_PNG} ({NOISE_PNG.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
