import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from finding_L.build_matrix import buildGramMatrixChunked
from finding_L.candidates import buildCandidateLibrary, filterPureVelocityTerms
from finding_L.regularized_select import _lstsqOnActive, lassoPathFromGram
from generation.eqnofmotion import TIME, defineCoordinates
from matplotlib.animation import FuncAnimation, PillowWriter

CSV = REPO / "assets" / "isotropic_quartic_small_n3_noise1.csv"
OUT = REPO / "assets" / "reports" / "gifs" / "social_lasso_path.gif"
NO_COORDS = 3
DEGREE = 4
PRUNE = 1e-2

INK = "#1a1a1a"
MOVER = "#6a6a6a"
SURV = "#1a1a1a"
SWEEP = "#c62828"

_SUP = str.maketrans({"0": "₀", "1": "₁", "2": "₂", "3": "₃",
                      "4": "₄"})


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
        }
    )


def prettyTerm(term):
    text = str(term).replace("(t)", "")
    text = text.replace("**2", "^2").replace("**3", "^3").replace("**4", "^4")
    text = text.replace("*", "")
    for i in range(NO_COORDS):
        text = text.replace(f"q{i}", "q" + str(i).translate(_SUP))
    text = text.replace("^2", "²").replace("^3", "³").replace("^4", "⁴")
    return text


def buildAdmissibleGram():
    _t, coords, vels = defineCoordinates(NO_COORDS)
    kinetic = sp.expand(sum(v**2 for v in vels))
    library = filterPureVelocityTerms(
        buildCandidateLibrary(coords, vels, DEGREE), coords
    )
    if kinetic not in library:
        library.append(kinetic)
    n, colSum, gram = buildGramMatrixChunked(
        library, coords, vels, TIME, str(CSV), NO_COORDS, 120_000
    )
    kineticIndex = library.index(kinetic)
    variance = gram.diagonal() / n - (colSum / n) ** 2
    admissible = [
        i for i in range(len(library)) if variance[i] > 1e-12 or i == kineticIndex
    ]
    subGram = gram[np.ix_(admissible, admissible)]
    subTerms = [library[i] for i in admissible]
    return subGram, subTerms, admissible.index(kineticIndex)


def chooseLambdaRow(gram, b, kineticIndex, paths, candidates):
    targetNormSq = gram[kineticIndex, kineticIndex]

    def refitResidual(active):
        coefficients = _lstsqOnActive(gram, b, active)
        if coefficients.size == 0:
            return targetNormSq
        return float(targetNormSq - coefficients @ b[active])

    densest = [candidates[j] for j in np.nonzero(np.abs(paths[-1]) > 0)[0]]
    best = refitResidual(densest)
    for row, values in enumerate(paths):
        active = [candidates[j] for j, v in enumerate(values) if v != 0.0]
        if not active:
            continue
        if refitResidual(active) <= best * 1.05 + 1e-12:
            return row, active
    return len(paths) - 1, densest


def main():
    gram, terms, kineticIndex = buildAdmissibleGram()
    b = -gram[:, kineticIndex]
    lambdas, paths, candidates = lassoPathFromGram(gram, b, kineticIndex)
    nSteps = len(lambdas)

    chosenRow, chosenActive = chooseLambdaRow(gram, b, kineticIndex, paths, candidates)
    coefficients = _lstsqOnActive(gram, b, chosenActive)
    cutoff = PRUNE * np.max(np.abs(coefficients))
    survivors = [i for i, v in zip(chosenActive, coefficients) if abs(v) >= cutoff]
    survCoef = _lstsqOnActive(gram, b, survivors)
    survLookup = dict(zip(survivors, survCoef))

    moverCols = [j for j in range(len(candidates))
                 if np.abs(paths[:, j]).max() > 1e-4]
    survCols = {j for j in range(len(candidates)) if candidates[j] in survLookup}
    zeroCount = len(candidates) - len(moverCols)

    print(f"admissible candidate terms : {len(candidates)}")
    print(f"terms that ever move        : {len(moverCols)}")
    print(f"lambda* row                 : {chosenRow}  (lambda={lambdas[chosenRow]:.3g})")
    print(f"survivors after threshold   : {len(survivors)}")
    for index in sorted(survivors, key=lambda i: str(terms[i])):
        print(f"    {prettyTerm(terms[index]):14s} {survLookup[index]:+.4f}")

    xs = np.arange(nSteps)
    labStyle()
    figure, ax = plt.subplots(figsize=(10.0, 5.6))
    figure.subplots_adjust(left=0.10, right=0.80, top=0.86, bottom=0.13)
    figure.suptitle(
        "Sparse selection by regularization path (debiased LASSO)",
        fontsize=12, y=0.965,
    )

    lines = {}
    for j in moverCols:
        (line,) = ax.plot([], [], lw=0.9, color=MOVER, zorder=2)
        lines[j] = line

    sweep = ax.axvline(xs[0], color=SWEEP, lw=1.2, zorder=4)
    ax.axhline(0.0, color=INK, lw=0.8, zorder=1)

    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(paths.min() * 1.08 - 0.03, 0.10)
    tickPos = [0, nSteps // 4, nSteps // 2, 3 * nSteps // 4, nSteps - 1]
    ax.set_xticks(tickPos)
    ax.set_xticklabels([f"{lambdas[p]:.0f}" for p in tickPos])
    ax.set_xlabel(r"regularization strength  $\lambda$   (decreasing $\rightarrow$ weaker)")
    ax.set_ylabel("coefficient in $L$")

    counter = figure.text(0.10, 0.90, "", ha="left", va="center", fontsize=10)
    zeroNote = ax.text(
        0.66, 0.46, f"{zeroCount} of {len(candidates)} candidate terms\nstay exactly at zero",
        transform=ax.transAxes, ha="left", va="center", fontsize=9,
        color=MOVER, style="italic",
    )
    phaseNote = figure.text(0.80, 0.90, "", ha="right", va="center", fontsize=9.5,
                            color=SWEEP)

    groups = {}
    for index in survivors:
        key = round(float(survLookup[index]), 2)
        groups.setdefault(key, []).append(index)
    def shapeOf(term):
        text = prettyTerm(term)
        subs = []
        for ch in "₀₁₂₃₄":
            if ch in text and ch not in subs:
                subs.append(ch)
        for ch, repl in zip(subs, ["ᵢ", "ⱼ", "ₖ"]):
            text = text.replace(ch, repl)
        return text

    groupLabels = []
    for value, members in sorted(groups.items()):
        groupLabels.append((float(np.mean([survLookup[m] for m in members])),
                            f"{shapeOf(terms[members[0]])}  (×{len(members)})   {value:+.3f}"))

    labelHandles = []

    sweepFrames = [("grow", s) for s in range(nSteps) for _ in range(2)]
    holdFull = [("grow", nSteps - 1)] * 6
    markChosen = [("chosen", 0)] * 10
    threshold = [("thresh", k) for k in range(1, 9)]
    finalHold = [("final", 0)] * 32
    plan = sweepFrames + holdFull + markChosen + threshold + finalHold

    def clearLabels():
        while labelHandles:
            labelHandles.pop().remove()

    def draw(frameIndex):
        phase, value = plan[frameIndex]

        if phase == "grow":
            s = value
            for j, line in lines.items():
                line.set_data(xs[: s + 1], paths[: s + 1, j])
                line.set_color(MOVER)
                line.set_linewidth(0.9)
            sweep.set_xdata([xs[s], xs[s]])
            sweep.set_alpha(1.0)
            active = int(np.sum(np.abs(paths[s]) > 1e-3))
            counter.set_text(f"nonzero coefficients: {active}")
            phaseNote.set_text("")
            clearLabels()

        elif phase == "chosen":
            for j, line in lines.items():
                line.set_data(xs, paths[:, j])
            sweep.set_xdata([xs[chosenRow], xs[chosenRow]])
            counter.set_text(
                f"nonzero coefficients: {int(np.sum(np.abs(paths[chosenRow]) > 1e-3))}"
            )
            phaseNote.set_text(r"$\lambda^\ast$: sparsest fit within 5% of the full residual")

        elif phase == "thresh":
            k = value / 8.0
            for j, line in lines.items():
                if j in survCols:
                    line.set_data(xs, paths[:, j])
                    line.set_color(SURV)
                    line.set_linewidth(1.8)
                else:
                    faded = paths[:, j] * (1 - k)
                    line.set_data(xs, faded)
                    line.set_color(MOVER)
                    line.set_linewidth(0.9)
                    line.set_alpha(1 - 0.75 * k)
            sweep.set_xdata([xs[chosenRow], xs[chosenRow]])
            counter.set_text(f"nonzero coefficients: {len(survivors)}")
            phaseNote.set_text("threshold: drop |coef| < 1% of the largest")

        elif phase == "final":
            clearLabels()
            for j, line in lines.items():
                if j in survCols:
                    line.set_data(xs, paths[:, j])
                    line.set_color(SURV)
                    line.set_linewidth(1.8)
                    line.set_alpha(1.0)
                else:
                    line.set_data([], [])
            for yValue, text in groupLabels:
                handle = ax.annotate(
                    text, xy=(xs[-1], yValue), xytext=(8, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=9, color=SURV, annotation_clip=False,
                )
                labelHandles.append(handle)
            sweep.set_alpha(0.0)
            counter.set_text(f"recovered: {len(survivors)} terms")
            phaseNote.set_text("debiased on the surviving support")

        return list(lines.values()) + [sweep, counter, phaseNote, zeroNote]

    animation = FuncAnimation(figure, draw, frames=len(plan), blit=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    animation.save(str(OUT), writer=PillowWriter(fps=14), dpi=100)
    plt.close(figure)
    size = OUT.stat().st_size / 1024
    print(f"animation frames: {len(plan)}")
    print(f"wrote {OUT.relative_to(REPO)}  ({size:.0f} KiB)")


if __name__ == "__main__":
    main()
