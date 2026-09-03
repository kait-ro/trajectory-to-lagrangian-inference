import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.recovery_animation import (
    NO_COORDS,
    SLOT_ORDER,
    TRUE_TERMS,
    blendCoefficients,
    buildPerFrame,
    compareToExpected,
    expectedStateExpression,
    prettyMonomial,
    runDiscovery,
    selectKeyframes,
)
from matplotlib.animation import FuncAnimation, PillowWriter

OUT = REPO / "assets" / "reports" / "gifs" / "social_recovery_greedy.gif"

INK = "#1a1a1a"
DATA = "#8a8a8a"
LINE = "#1a1a1a"


def labStyle():
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.grid": True,
            "grid.color": "#c8c8c8",
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.edgecolor": "#1a1a1a",
        }
    )


def buildSteps(perFrame, keyframes):
    steps = []
    for position in range(len(keyframes) - 1):
        current = perFrame[keyframes[position]]
        nextFrame = perFrame[keyframes[position + 1]]
        for _ in range(5):
            steps.append((current, current, 1.0))
        for tween in range(1, 9):
            steps.append((current, nextFrame, tween / 8))
    for _ in range(36):
        steps.append((perFrame[keyframes[-1]], perFrame[keyframes[-1]], 1.0))
    return steps


def render(discovered, frames):
    perFrame, times, noisy = buildPerFrame(discovered, frames)
    keyframes = selectKeyframes(perFrame)
    comparison = compareToExpected(
        discovered, expectedStateExpression(), noCoords=NO_COORDS
    )
    equivalent = bool(comparison.structurallyEquivalent)
    steps = buildSteps(perFrame, keyframes)

    labStyle()
    figure, (axLeft, axRight) = plt.subplots(
        1, 2, figsize=(10.0, 5.0), gridspec_kw={"width_ratios": [1.1, 1.0]}
    )
    figure.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.12, wspace=0.30)
    figure.suptitle(
        "Recovering a Lagrangian from noisy motion (greedy forward-selection)",
        fontsize=12,
        y=0.965,
    )

    axLeft.scatter(
        times[::5], noisy[::5, 0], s=8, color=DATA, alpha=0.8,
        label="noisy measured $q_0(t)$", zorder=2,
    )
    (reconLine,) = axLeft.plot(
        [], [], color=LINE, lw=1.8, label="motion from current $L$", zorder=3
    )
    axLeft.set_xlim(times[0], times[-1])
    span = np.abs(noisy[:, 0]).max() * 1.3
    axLeft.set_ylim(-span, span)
    axLeft.set_xlabel("time")
    axLeft.set_ylabel("$q_0(t)$")
    axLeft.legend(loc="upper right", fontsize=9)

    slotY = np.arange(len(SLOT_ORDER))[::-1]
    bars = axRight.barh(
        slotY, [0.0] * len(SLOT_ORDER), height=0.62, color=INK, zorder=3
    )
    axRight.axvline(1.0, color=INK, lw=1.0, ls="--", zorder=4)
    axRight.text(1.0, len(SLOT_ORDER) - 0.2, " true value", ha="left", va="bottom",
                 fontsize=8.5, color=INK)
    axRight.axvline(0.0, color=INK, lw=0.8)
    axRight.set_yticks(slotY)
    axRight.set_yticklabels([prettyMonomial(key) for key in SLOT_ORDER], fontsize=10)
    axRight.set_ylim(-0.7, len(SLOT_ORDER) - 0.3)
    axRight.set_xlim(0.0, 1.45)
    axRight.set_xlabel(r"recovered coefficient $\div$ true value")
    axRight.grid(True, axis="x")
    axRight.grid(False, axis="y")

    status = figure.text(0.5, 0.885, "", ha="center", va="center", fontsize=11)
    detail = figure.text(0.5, 0.845, "", ha="center", va="center", fontsize=9,
                         color="#555555")
    endText = axLeft.text(
        0.5, 0.06, "", transform=axLeft.transAxes, ha="center", va="bottom",
        fontsize=9.5, family="monospace",
        bbox={"boxstyle": "square,pad=0.4", "facecolor": "white", "edgecolor": INK},
        visible=False, zorder=6,
    )

    def draw(index):
        fromFrame, toFrame, weight = steps[index]
        blended = blendCoefficients(fromFrame, toFrame, weight)
        shown = toFrame if weight >= 0.5 else fromFrame

        for bar, key in zip(bars, SLOT_ORDER):
            bar.set_width(blended[key] / TRUE_TERMS[key])

        curve = shown["curve"]
        if curve is None:
            reconLine.set_data([], [])
        else:
            reconLine.set_data(times, curve[:, 0])

        selected = sum(1 for value in shown["onModel"].values() if value != 0.0)
        if shown["final"]:
            status.set_text(
                f"converged   ·   {selected}/9 target terms   ·   "
                f"scaled residual {shown['residual']:.0%}"
            )
            pruned = shown.get("prunedOffModel", 0)
            prunedText = (
                f"{pruned} off-model terms pruned   ·   " if pruned else ""
            )
            detail.set_text(
                prunedText + "coefficients snapped to simple rationals"
            )
            verdict = "dL == 0" if equivalent else "not equivalent"
            endText.set_text(
                "L  =  q̇²  −  r²  −  (3/20)(r²)²\n"
                f"equivalence-class check:  {verdict}"
            )
            endText.set_visible(True)
        else:
            status.set_text(
                f"round {shown['round']}   ·   {selected}/9 target terms   ·   "
                f"scaled residual {shown['residual']:.0%}"
            )
            carried = shown["offModel"]
            if shown["added"]:
                tail = f"   (off-model terms carried: {carried})" if carried else ""
                detail.set_text(f"added term:  {shown['added']}{tail}")
            else:
                detail.set_text(
                    "streaming the Euler–Lagrange Gram matrix, one term per round"
                )
            endText.set_visible(False)
        return [*bars, reconLine, status, detail, endText]

    animation = FuncAnimation(figure, draw, frames=len(steps), blit=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    animation.save(str(OUT), writer=PillowWriter(fps=14), dpi=100)
    plt.close(figure)
    return len(steps), len(keyframes), equivalent


def main():
    discovered, frames = runDiscovery()
    nSteps, nKey, equivalent = render(discovered, frames)
    size = OUT.stat().st_size / 1024
    print(f"recovered L: {discovered.expression}")
    print(f"structurally equivalent: {equivalent}")
    print(f"raw rounds: {len(frames)}   keyframes: {nKey}   animation steps: {nSteps}")
    print(f"wrote {OUT.relative_to(REPO)}  ({size:.0f} KiB)")


if __name__ == "__main__":
    main()
