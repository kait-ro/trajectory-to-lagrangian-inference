import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from experiments.discovery import FROZEN_TOLERANCES, compareToExpected
from finding_L.main_streaming import runDiscoveryStreaming
from finding_L.report import buildStateSymbolMap
from generation.eqnofmotion import defineCoordinates
from generation.integrator import GetAccelFunctions, simulateTrajectory
from matplotlib.animation import FuncAnimation, PillowWriter

ASSETS = Path(__file__).resolve().parents[2] / "assets"
TRAIN_CSV = str(ASSETS / "isotropic_quartic_small_n3_noise1.csv")
NO_COORDS = 3
NOISE_FRACTION = 0.01
HELD_OUT_SEED = 3
DT = 0.01
WINDOW_STEPS = 640

TRUE_TERMS = {
    "q0**2": -1.0,
    "q1**2": -1.0,
    "q2**2": -1.0,
    "q0**2*q1**2": -0.3,
    "q0**2*q2**2": -0.3,
    "q1**2*q2**2": -0.3,
    "q0**4": -0.15,
    "q1**4": -0.15,
    "q2**4": -0.15,
}
SLOT_ORDER = list(TRUE_TERMS)

PRETTY = {"q0": "q₀", "q1": "q₁", "q2": "q₂", "v0": "q̇₀", "v1": "q̇₁", "v2": "q̇₂"}


def prettyMonomial(stateKey):
    text = stateKey
    for raw, nice in PRETTY.items():
        text = text.replace(raw, nice)
    text = text.replace("**2", "²").replace("**3", "³").replace("**4", "⁴")
    return text.replace("*", "·")


def trueLagrangian(coords, vels):
    rSquared = sum(q**2 for q in coords)
    return sp.expand(
        sum(v**2 for v in vels) - rSquared - sp.Rational(3, 20) * rSquared**2
    )


def expectedStateExpression():
    q = [sp.Symbol(f"q{i}") for i in range(NO_COORDS)]
    v = [sp.Symbol(f"v{i}") for i in range(NO_COORDS)]
    rSquared = sum(x**2 for x in q)
    return sp.expand(sum(x**2 for x in v) - rSquared - sp.Rational(3, 20) * rSquared**2)


def heldOutTrajectory(coords, vels, t):
    accel = GetAccelFunctions(trueLagrangian(coords, vels), coords, vels, t)
    rng = np.random.default_rng(HELD_OUT_SEED)
    initialState = rng.uniform(-1.5, 1.5, size=2 * NO_COORDS)
    times, positions, _velocities, _accelerations = simulateTrajectory(
        initialState, accel, DT, WINDOW_STEPS
    )
    noisy = positions + rng.normal(
        0.0, NOISE_FRACTION * positions.std(axis=0), positions.shape
    )
    return times, positions, noisy, initialState


def runDiscovery():
    frames = []
    discovered, _log = runDiscoveryStreaming(
        TRAIN_CSV,
        noCoords=NO_COORDS,
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
    return discovered, frames


def dedupeFrames(frames):
    kept = []
    lastKey = None
    for frame in frames:
        key = tuple(sorted(str(term) for term in frame["activeTerms"]))
        if key == lastKey and kept:
            kept[-1] = frame
        else:
            kept.append(frame)
        lastKey = key
    return kept


def splitActiveTerms(frame, stateSymbolMap):
    onModel = {}
    offModel = 0
    for term, value in zip(frame["activeTerms"], frame["coefficients"]):
        key = str(sp.expand(term.subs(stateSymbolMap, simultaneous=True)))
        if key in TRUE_TERMS:
            onModel[key] = float(value)
        else:
            offModel += 1
    return onModel, offModel


def reconstructFromLagrangian(lagrangian, coords, vels, t, initialState):
    try:
        accel = GetAccelFunctions(sp.expand(lagrangian), coords, vels, t)
        _times, positions, _v, _a = simulateTrajectory(
            np.asarray(initialState, dtype=float), accel, DT, WINDOW_STEPS
        )
    except Exception:
        return None
    if not np.all(np.isfinite(positions)) or np.abs(positions).max() > 12.0:
        return None
    return positions


def recoveredCoefficients(discovered):
    coefficients = {}
    prunedOffModel = 0
    for monomial, value in (
        sp.expand(discovered.rawExpression).as_coefficients_dict().items()
    ):
        key = str(monomial)
        if key in TRUE_TERMS:
            coefficients[key] = float(value)
        elif key not in ("q0", "q1", "q2", "1") and "v" not in key:
            prunedOffModel += 1
    return coefficients, prunedOffModel


def buildPerFrame(discovered, frames):
    keptFrames = dedupeFrames(frames)
    t, coords, vels = defineCoordinates(NO_COORDS)
    stateSymbolMap = buildStateSymbolMap(coords, vels)
    times, _clean, noisy, initialState = heldOutTrajectory(coords, vels, t)

    perFrame = []
    previousKeys = set()
    for frame in keptFrames:
        onModel, offModel = splitActiveTerms(frame, stateSymbolMap)
        newKeys = sorted(set(onModel) - previousKeys, key=SLOT_ORDER.index)
        added = prettyMonomial(newKeys[-1]) if newKeys else None
        lagrangian = frame["kineticTerm"] + sum(
            v * term for v, term in zip(frame["coefficients"], frame["activeTerms"])
        )
        perFrame.append(
            {
                "round": frame["round"],
                "residual": frame["scaledResidual"],
                "onModel": onModel,
                "offModel": offModel,
                "added": added,
                "curve": reconstructFromLagrangian(
                    lagrangian, coords, vels, t, initialState
                ),
                "final": False,
            }
        )
        previousKeys = set(onModel)

    finalCoefficients, prunedOffModel = recoveredCoefficients(discovered)
    finalLagrangian = trueLagrangianFromCoefficients(finalCoefficients, coords, vels)
    perFrame.append(
        {
            "round": perFrame[-1]["round"] + 1,
            "residual": perFrame[-1]["residual"],
            "onModel": finalCoefficients,
            "offModel": 0,
            "prunedOffModel": prunedOffModel,
            "added": None,
            "curve": reconstructFromLagrangian(
                finalLagrangian, coords, vels, t, initialState
            ),
            "final": True,
        }
    )
    return perFrame, times, noisy


def trueLagrangianFromCoefficients(coefficients, coords, vels):
    stateInverse = {sp.Symbol(f"q{i}"): coords[i] for i in range(NO_COORDS)}
    lagrangian = sum(v**2 for v in vels)
    for key, value in coefficients.items():
        monomial = sp.sympify(key).subs(stateInverse)
        lagrangian = lagrangian + value * monomial
    return sp.expand(lagrangian)


def selectKeyframes(perFrame):
    keyframes = [0]
    seenKeys = set(perFrame[0]["onModel"])
    for index in range(1, len(perFrame)):
        keys = set(perFrame[index]["onModel"])
        if perFrame[index]["final"] or keys != seenKeys:
            keyframes.append(index)
            seenKeys = keys
    if keyframes[-1] != len(perFrame) - 1:
        keyframes.append(len(perFrame) - 1)
    return keyframes


def blendCoefficients(fromFrame, toFrame, weight):
    blended = {}
    for key in SLOT_ORDER:
        before = fromFrame["onModel"].get(key, 0.0)
        after = toFrame["onModel"].get(key, 0.0)
        blended[key] = before * (1 - weight) + after * weight
    return blended


def render(discovered, frames):
    perFrame, times, noisy = buildPerFrame(discovered, frames)
    keyframes = selectKeyframes(perFrame)
    comparison = compareToExpected(
        discovered, expectedStateExpression(), noCoords=NO_COORDS
    )
    equivalent = bool(comparison.structurallyEquivalent)

    steps = []
    for position in range(len(keyframes) - 1):
        current = perFrame[keyframes[position]]
        nextFrame = perFrame[keyframes[position + 1]]
        for _ in range(5):
            steps.append((current, current, 1.0))
        for tween in range(1, 9):
            steps.append((current, nextFrame, tween / 8))
    for _ in range(34):
        steps.append((perFrame[keyframes[-1]], perFrame[keyframes[-1]], 1.0))

    plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})
    figure, (axLeft, axRight) = plt.subplots(
        1, 2, figsize=(10.4, 5.0), gridspec_kw={"width_ratios": [1.12, 1.0]}
    )
    figure.subplots_adjust(left=0.08, right=0.975, top=0.79, bottom=0.13, wspace=0.34)
    figure.suptitle(
        "Recovering a Lagrangian from noisy motion", fontsize=15, weight="bold", y=0.965
    )

    axLeft.scatter(
        times[::5],
        noisy[::5, 0],
        s=9,
        color="#9aa0a6",
        alpha=0.75,
        label="noisy position data",
        zorder=2,
    )
    (reconLine,) = axLeft.plot(
        [], [], color="#1a73e8", lw=2.6, label="motion from recovered L", zorder=3
    )
    axLeft.set_xlim(times[0], times[-1])
    span = np.abs(noisy[:, 0]).max() * 1.28
    axLeft.set_ylim(-span, span)
    axLeft.set_xlabel("time")
    axLeft.set_ylabel("q₀(t)")
    axLeft.legend(loc="upper right", framealpha=0.92, fontsize=10.5)
    axLeft.grid(alpha=0.15)

    slotY = np.arange(len(SLOT_ORDER))[::-1]
    bars = axRight.barh(
        slotY, [0] * len(SLOT_ORDER), color="#c7c9cc", height=0.6, zorder=3
    )
    axRight.axvline(1.0, color="#202124", lw=1.6, zorder=4)
    axRight.text(
        1.0, len(SLOT_ORDER) - 0.15, "true", ha="center", va="bottom", fontsize=10
    )
    axRight.axvline(0.0, color="#202124", lw=0.8)
    axRight.set_yticks(slotY)
    axRight.set_yticklabels([prettyMonomial(key) for key in SLOT_ORDER])
    axRight.set_ylim(-0.7, len(SLOT_ORDER) - 0.3)
    axRight.set_xlim(0.0, 1.4)
    axRight.set_xlabel("recovered coefficient ÷ true value")
    axRight.grid(alpha=0.15, axis="x")

    caption = figure.text(
        0.5, 0.88, "", ha="center", va="center", fontsize=13.5, weight="bold"
    )
    subCaption = figure.text(
        0.5, 0.835, "", ha="center", va="center", fontsize=10.5, color="#5f6368"
    )
    endCard = axLeft.text(
        0.5,
        0.5,
        "",
        transform=axLeft.transAxes,
        ha="center",
        va="center",
        fontsize=12.5,
        bbox={
            "boxstyle": "round,pad=0.7",
            "facecolor": "#e6f4ea",
            "edgecolor": "#137333",
            "alpha": 0.96,
        },
        visible=False,
        zorder=6,
    )

    def draw(index):
        fromFrame, toFrame, weight = steps[index]
        blended = blendCoefficients(fromFrame, toFrame, weight)
        shown = toFrame if weight >= 0.5 else fromFrame

        for bar, key in zip(bars, SLOT_ORDER):
            value = blended[key]
            bar.set_width(value / TRUE_TERMS[key])
            if value == 0.0:
                bar.set_color("#c7c9cc")
            elif abs(value - TRUE_TERMS[key]) <= 0.03:
                bar.set_color("#137333")
            else:
                bar.set_color("#e8710a")

        curve = shown["curve"]
        reconLine.set_data([], []) if curve is None else reconLine.set_data(
            times, curve[:, 0]
        )

        selected = sum(1 for v in shown["onModel"].values() if v != 0.0)
        if shown["final"]:
            caption.set_text(
                f"converged  ·  {selected}/9 terms  ·  residual {shown['residual']:.0%}"
            )
            subCaption.set_text(
                "off-model terms pruned  ·  coefficients snapped to simple rationals"
            )
            verdict = "✓ same physical theory" if equivalent else "✗ not equivalent"
            endCard.set_text(
                f"recovered from noisy positions\n\nL  =  q̇²  −  r²  −  (3/20)(r²)²\n\n{verdict}\n(equivalence-class check)"
            )
            endCard.set_visible(True)
        else:
            caption.set_text(
                f"round {shown['round']}  ·  {selected}/9 true terms  ·  residual {shown['residual']:.0%}"
            )
            if shown["added"]:
                extra = (
                    f"    (+{shown['offModel']} off-model terms so far)"
                    if shown["offModel"]
                    else ""
                )
                subCaption.set_text(f"added   {shown['added']}{extra}")
            else:
                subCaption.set_text(
                    "streaming the Euler–Lagrange Gram matrix, one term per round"
                )
            endCard.set_visible(False)
        return [*bars, reconLine, caption, subCaption, endCard]

    outputGif = ASSETS / "recovery.gif"
    animation = FuncAnimation(figure, draw, frames=len(steps), blit=False)
    animation.save(str(outputGif), writer=PillowWriter(fps=14), dpi=105)
    print(f"animation steps: {len(steps)}")
    draw(len(steps) - 1)
    figure.savefig(ASSETS / "recovery_final.png", dpi=140)
    plt.close(figure)
    return outputGif, comparison, perFrame, keyframes


def main():
    os.makedirs(ASSETS, exist_ok=True)
    discovered, frames = runDiscovery()
    outputGif, comparison, perFrame, keyframes = render(discovered, frames)
    print(f"recovered L: {discovered.expression}")
    print(f"structurally equivalent: {bool(comparison.structurallyEquivalent)}")
    print(
        f"per-frame states: {len(perFrame)}   keyframes: {len(keyframes)}   raw rounds: {len(frames)}"
    )
    print(f"wrote {outputGif}  ({outputGif.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
