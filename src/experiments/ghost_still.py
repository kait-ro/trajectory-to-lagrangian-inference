import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from generation.eqnofmotion import TIME, defineCoordinates
from generation.ghost_detection import detectGhost
from generation.higher_order_integrator import simulateHigherOrderTrajectory
from generation.ostrogradski import buildStateDerivative

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "ghost_still.png"
)

HEALTHY_BLUE = "#1f6feb"
GHOST_RED = "#d1495b"
POS_GREEN = "#2e8b57"
NEG_RED = "#c62828"


def buildSystems():
    _t, coords, _vels = defineCoordinates(1)
    q = coords[0]
    v = sp.diff(q, TIME)
    a = sp.diff(q, TIME, 2)

    healthy = sp.Rational(1, 2) * v**2 - sp.Rational(1, 2) * 4 * q**2
    paisUhlenbeck = sp.Rational(1, 2) * (a**2 - 5 * v**2 + 4 * q**2)
    healthyQuartic = sp.Rational(1, 2) * v**2 - 2 * q**2 - sp.Rational(1, 4) * q**4
    paisUhlenbeckQuartic = paisUhlenbeck - sp.Rational(1, 4) * q**4

    return {
        "coords": coords,
        "healthy": healthy,
        "paisUhlenbeck": paisUhlenbeck,
        "healthyQuartic": healthyQuartic,
        "paisUhlenbeckQuartic": paisUhlenbeckQuartic,
    }


def integrateTrajectory(lagrangian, coords, initialState, dt, steps):
    stateDerivative, equationOrder, noCoords = buildStateDerivative(lagrangian, coords)
    times, perDerivative = simulateHigherOrderTrajectory(
        list(initialState), stateDerivative, dt, steps, equationOrder, noCoords
    )
    return times, perDerivative[0][:, 0]


def prettyHamiltonian(hamiltonian):
    text = str(sp.expand(hamiltonian))
    for source, target in [
        ("Q0_0", "q"),
        ("Q0_1", "q'"),
        ("P0_1", "p1"),
        ("P0_2", "p2"),
        ("**2", "^2"),
        ("*", ""),
    ]:
        text = text.replace(source, target)
    return text


def clipRunaway(values, ceiling):
    finite = np.where(np.isfinite(values), np.abs(values), ceiling)
    clipped = np.clip(finite, 1e-12, ceiling)
    blowIndex = int(np.argmax(clipped >= ceiling)) if np.any(clipped >= ceiling) else -1
    return clipped, blowIndex


def leftPanel(axis, systems):
    times, healthyPos = integrateTrajectory(
        systems["healthy"], systems["coords"], [1.0, 0.4], 0.003, 10000
    )
    _times, ghostPos = integrateTrajectory(
        systems["paisUhlenbeck"], systems["coords"], [1.0, 0.4, 0.0, 0.0], 0.003, 10000
    )

    axis.plot(
        times,
        healthyPos,
        color=HEALTHY_BLUE,
        lw=2.2,
        label="healthy 2nd-order oscillator",
    )
    axis.plot(
        times, ghostPos, color=GHOST_RED, lw=2.2, label="Pais-Uhlenbeck (hidden ghost)"
    )
    axis.axhline(0.0, color="#9aa0a6", lw=0.8, zorder=0)

    axis.set_title(
        "Same bounded-looking motion", fontsize=15, fontweight="bold", pad=10
    )
    axis.set_xlabel("time", fontsize=13)
    axis.set_ylabel("q(t)", fontsize=13)
    axis.legend(loc="lower center", fontsize=10, framealpha=0.92)
    axis.set_ylim(-2.0, 1.85)
    axis.margins(x=0.01)


def rightPanel(axis, verdicts):
    healthyEig = np.array(verdicts["healthy"]["hamiltonianEigenvalues"])
    ghostEig = np.array(verdicts["paisUhlenbeck"]["hamiltonianEigenvalues"])

    healthyX = np.arange(len(healthyEig))
    ghostX = np.arange(len(ghostEig)) + len(healthyEig) + 1.4
    allX = np.concatenate([healthyX, ghostX])
    allEig = np.concatenate([healthyEig, ghostEig])
    colors = [POS_GREEN if value > 0 else NEG_RED for value in allEig]

    axis.bar(allX, allEig, width=0.72, color=colors, edgecolor="#2f2f2f", lw=0.6)
    axis.axhline(0.0, color="#2f2f2f", lw=1.0)
    axis.set_xticks([healthyX.mean(), ghostX.mean()])
    axis.set_xticklabels(
        [
            f"healthy oscillator\nghost = {verdicts['healthy']['ghost']}",
            f"Pais-Uhlenbeck\nghost = {verdicts['paisUhlenbeck']['ghost']}",
        ],
        fontsize=12.5,
    )
    axis.set_ylabel("eigenvalues of Hessian(H)", fontsize=13)
    axis.set_title(
        "The Lagrangian tells them apart", fontsize=16, fontweight="bold", pad=10
    )

    for x, value in zip(allX, allEig):
        offset = 0.4 if value >= 0 else -0.4
        axis.text(
            x,
            value + offset,
            f"{value:+.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=10,
            color="#2f2f2f",
        )

    top = allEig.max() + 4.6
    bottom = allEig.min() - 2.2
    axis.set_ylim(bottom, top)

    axis.text(
        healthyX.mean(),
        top - 0.5,
        "H bounded below\nno ghost",
        ha="center",
        va="top",
        fontsize=11.5,
        color=POS_GREEN,
        fontweight="bold",
    )
    axis.text(
        ghostX.mean(),
        top - 0.5,
        "H unbounded below\nOstrogradski ghost",
        ha="center",
        va="top",
        fontsize=11.5,
        color=NEG_RED,
        fontweight="bold",
    )
    axis.text(
        0.0,
        -0.42,
        "PU Hamiltonian   H = "
        + prettyHamiltonian(verdicts["paisUhlenbeck"]["hamiltonian"])
        + "\np1q' is linear in momentum  ->  H has no lower bound",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.8,
        color="#3c4043",
    )


def runawayStrip(axis, systems):
    np.seterr(all="ignore")
    ceiling = 1e3

    _t, ghostPos = integrateTrajectory(
        systems["paisUhlenbeckQuartic"],
        systems["coords"],
        [1.0, 0.0, 0.0, 0.0],
        0.001,
        40000,
    )
    times, healthyPos = integrateTrajectory(
        systems["healthyQuartic"], systems["coords"], [1.0, 0.0], 0.001, 40000
    )

    ghostMag, blowIndex = clipRunaway(ghostPos, ceiling)
    healthyMag, _ = clipRunaway(healthyPos, ceiling)

    axis.semilogy(
        times,
        healthyMag,
        color=HEALTHY_BLUE,
        lw=2.0,
        label="healthy quartic oscillator: bounded",
    )
    axis.semilogy(
        times[: blowIndex + 1],
        ghostMag[: blowIndex + 1],
        color=GHOST_RED,
        lw=2.0,
        label="nonlinear Pais-Uhlenbeck: finite-time blow-up",
    )

    if blowIndex > 0:
        axis.axvline(times[blowIndex], color=GHOST_RED, ls="--", lw=1.3)
        axis.text(
            times[blowIndex],
            ceiling,
            f"  blow-up at t = {times[blowIndex]:.1f}",
            va="top",
            ha="left",
            fontsize=11,
            color=GHOST_RED,
            fontweight="bold",
        )

    axis.set_title(
        "Add any nonlinearity (- q^4/4) and the ghost mode runs away",
        fontsize=14,
        fontweight="bold",
        pad=8,
    )
    axis.set_xlabel("time", fontsize=12)
    axis.set_ylabel("|q(t)|", fontsize=12)
    axis.set_ylim(7e-2, ceiling * 2)
    axis.legend(loc="upper left", fontsize=10.5, framealpha=0.9)
    axis.margins(x=0.01)


def buildFigure():
    systems = buildSystems()
    verdicts = {
        "healthy": detectGhost(systems["healthy"], systems["coords"]),
        "paisUhlenbeck": detectGhost(systems["paisUhlenbeck"], systems["coords"]),
    }

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfd",
            "axes.edgecolor": "#c7ccd1",
            "axes.grid": True,
            "grid.color": "#e6e8eb",
            "grid.linewidth": 0.8,
            "font.size": 12,
            "text.color": "#202124",
            "axes.labelcolor": "#202124",
            "xtick.color": "#3c4043",
            "ytick.color": "#3c4043",
        }
    )

    figure = plt.figure(figsize=(1400 / 150, 940 / 150), dpi=150)
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[3.0, 1.25],
        hspace=0.95,
        wspace=0.24,
        left=0.085,
        right=0.975,
        top=0.80,
        bottom=0.09,
    )
    axisLeft = figure.add_subplot(grid[0, 0])
    axisRight = figure.add_subplot(grid[0, 1])
    axisStrip = figure.add_subplot(grid[1, :])

    figure.suptitle(
        "A bounded trajectory can still hide an Ostrogradski ghost",
        fontsize=19,
        fontweight="bold",
        y=0.965,
    )
    figure.text(
        0.5,
        0.905,
        "from the trajectory alone you cannot tell which theory is pathological",
        ha="center",
        fontsize=11.5,
        style="italic",
        color="#3c4043",
    )

    leftPanel(axisLeft, systems)
    rightPanel(axisRight, verdicts)
    runawayStrip(axisStrip, systems)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150, facecolor="white")
    plt.close(figure)
    return verdicts


if __name__ == "__main__":
    finalVerdicts = buildFigure()
    for key, verdict in finalVerdicts.items():
        eigenvalues = np.round(verdict["hamiltonianEigenvalues"], 6).tolist()
        print(
            f"{key}: ghost={verdict['ghost']} order={verdict['order']} eig(H)={eigenvalues}"
        )
    print(f"wrote {os.path.abspath(OUTPUT_PATH)}")
