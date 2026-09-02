import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from experiments import vizstyle
from finding_L.equivalence_class import classifyLagrangianPair, eulerLagrangeResidual
from generation.eqnofmotion import defineCoordinates

ASSETS = vizstyle.ASSETS
OUTPUT = ASSETS / "equivalence_class_explainer.png"
GRID = np.linspace(-1.6, 1.6, 220)


def buildPairs():
    _t, coords, vels = defineCoordinates(2)
    q0, q1 = coords
    base = sp.Rational(1, 2) * sum(v**2 for v in vels) - sp.Rational(1, 2) * sum(q**2 for q in coords)

    nullDelta = q0 * vels[1] + q1 * vels[0]
    genuineDelta = sp.Rational(1, 2) * q0**2 * q1

    nullPair = ("total time derivative", base + nullDelta, base, nullDelta, "d/dt( q0 q1 )")
    genuinePair = ("genuine coefficient change", base + genuineDelta, base, genuineDelta, None)
    return coords, vels, nullPair, genuinePair


def residualField(deltaExpression, coords, vels):
    residual = eulerLagrangeResidual(deltaExpression, coords, vels)
    stateSymbols = {coords[0]: sp.Symbol("x0"), coords[1]: sp.Symbol("x1")}
    fields = []
    meshX, meshY = np.meshgrid(GRID, GRID)
    for component in residual:
        substituted = sp.expand(component.subs(stateSymbols))
        function = sp.lambdify((sp.Symbol("x0"), sp.Symbol("x1")), substituted, "numpy")
        value = function(meshX, meshY)
        fields.append(np.broadcast_to(np.asarray(value, dtype=float), meshX.shape).copy())
    return fields, residual


def prettyLagrangian(expression):
    text = sp.sstr(sp.expand(expression))
    for raw, nice in [
        ("Derivative(q0(t), t)", "q0'"),
        ("Derivative(q1(t), t)", "q1'"),
        ("q0(t)", "q0"),
        ("q1(t)", "q1"),
        ("**", "^"),
        ("*", " "),
    ]:
        text = text.replace(raw, nice)
    return text


def drawColumn(figure, gridSpec, column, pairData, coords, vels, colourLimit):
    kind, lagA, lagB, delta, boundary = pairData
    fields, residual = residualField(delta, coords, vels)
    verdict = classifyLagrangianPair(lagA, lagB, coords, vels)

    headAx = figure.add_subplot(gridSpec[0, column])
    headAx.axis("off")
    headText = [
        f"{kind}",
        "",
        f"L_A  =  {prettyLagrangian(lagA)}",
        f"L_B  =  {prettyLagrangian(lagB)}",
        f"dL   =  {prettyLagrangian(delta)}",
    ]
    if boundary:
        headText.append(f"       =  {boundary}")
    headAx.text(
        0.5,
        1.0,
        "\n".join(headText),
        transform=headAx.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        family="monospace",
    )

    for row, (component, field) in enumerate(zip(residual, fields), start=1):
        ax = figure.add_subplot(gridSpec[row, column])
        mesh = ax.imshow(
            field,
            extent=(GRID[0], GRID[-1], GRID[0], GRID[-1]),
            origin="lower",
            cmap="RdBu_r",
            vmin=-colourLimit,
            vmax=colourLimit,
            aspect="equal",
        )
        ax.set_xlabel("q0")
        ax.set_ylabel("q1")
        peak = np.abs(field).max()
        if peak < 1e-9:
            label = f"EL(dL) component {row - 1}   ==  0   (max |.| = {peak:.0e})"
            ax.text(
                0.5,
                0.5,
                "EL(dL)  =  0\nexactly, everywhere",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="#137333",
            )
        else:
            label = f"EL(dL) component {row - 1}   =  {prettyLagrangian(component)}"
        ax.set_title(label, fontsize=10)
        figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)

    verdictAx = figure.add_subplot(gridSpec[3, column])
    verdictAx.axis("off")
    if verdict.equivalent:
        vizstyle.badge(
            verdictAx,
            "SAME physical theory\nEL(dL) identically zero  ->  differ by a total time derivative",
            xy=(0.5, 0.55),
            facecolor="#e6f4ea",
            edgecolor="#137333",
        )
    else:
        vizstyle.badge(
            verdictAx,
            "DIFFERENT theory\nEL(dL) is a nonzero field  ->  the equations of motion changed",
            xy=(0.5, 0.55),
            facecolor="#fdecec",
            edgecolor=vizstyle.BAD_RED,
        )
    verdictAx.text(
        0.5,
        0.05,
        "coefficient vectors of L_A and L_B differ in BOTH columns - proximity cannot tell these apart",
        transform=verdictAx.transAxes,
        ha="center",
        fontsize=8.4,
        style="italic",
        color="#5f6368",
    )


def main():
    vizstyle.applyStyle()
    coords, vels, nullPair, genuinePair = buildPairs()

    genuineFields, _ = residualField(genuinePair[3], coords, vels)
    colourLimit = max(np.abs(f).max() for f in genuineFields) or 1.0

    figure = plt.figure(figsize=(15, 12))
    gridSpec = figure.add_gridspec(
        4, 2, height_ratios=[0.26, 1.0, 1.0, 0.3], hspace=0.28, wspace=0.24
    )
    drawColumn(figure, gridSpec, 0, nullPair, coords, vels, colourLimit)
    drawColumn(figure, gridSpec, 1, genuinePair, coords, vels, colourLimit)

    figure.suptitle(
        "\"Same theory?\" is an Euler-Lagrange question, not a coefficient question",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.945,
        "left: L_A and L_B differ by d/dt(q0 q1) - EL(dL) vanishes as a field over phase space.   "
        "right: L_A and L_B differ by a genuine monomial - EL(dL) is structured and nonzero.",
        ha="center",
        fontsize=10,
        style="italic",
        color="#4a515b",
    )
    vizstyle.provenance(
        figure,
        "computed live from finding_L.equivalence_class.eulerLagrangeResidual  (2-coordinate base L, symbolic EL operator)",
    )
    figure.subplots_adjust(top=0.92, bottom=0.04)
    size = vizstyle.savefig(figure, OUTPUT)
    print(f"wrote {OUTPUT}  ({size} bytes)")


if __name__ == "__main__":
    main()
