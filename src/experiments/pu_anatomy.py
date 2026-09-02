import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from experiments import vizstyle

ASSETS = vizstyle.ASSETS
OUTPUT = ASSETS / "pu_anatomy.png"
OMEGA1, OMEGA2 = 1.0, 2.0


def dispersion(omega):
    return omega**4 - (OMEGA1**2 + OMEGA2**2) * omega**2 + OMEGA1**2 * OMEGA2**2


def panelDispersion(ax):
    omega = np.linspace(0, 2.7, 600)
    ax.plot(omega, dispersion(omega), color=vizstyle.ACCENT, lw=2.4)
    ax.axhline(0, color="#1b1f24", lw=1.0)
    for value, colour, name in [(OMEGA1, vizstyle.MODE_LOW, "omega_1"), (OMEGA2, vizstyle.MODE_HIGH, "omega_2")]:
        ax.scatter([value], [0], s=90, color=colour, zorder=5)
        ax.annotate(
            f"{name} = {value:g}",
            xy=(value, 0),
            xytext=(value, dispersion(0) * 0.42),
            ha="center",
            fontsize=10,
            color=colour,
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": colour},
        )
    ax.set_xlabel("mode frequency  omega")
    ax.set_ylabel("dispersion polynomial  D(omega)")
    ax.set_title("4th-order dispersion: two real normal modes")
    ax.annotate(
        "(omega^2 - w1^2)(omega^2 - w2^2) = 0",
        xy=(0.03, 0.06),
        xycoords="axes fraction",
        fontsize=9.5,
        family="monospace",
    )


def panelModes(ax):
    time = np.linspace(0, 4 * np.pi, 1400)
    lowMode = np.cos(OMEGA1 * time)
    highMode = 0.55 * np.cos(OMEGA2 * time)
    ax.plot(time, lowMode, color=vizstyle.MODE_LOW, lw=1.8, label=f"healthy mode  (omega_1 = {OMEGA1:g})")
    ax.plot(time, highMode, color=vizstyle.MODE_HIGH, lw=1.8, label=f"ghost mode  (omega_2 = {OMEGA2:g}, negative energy)")
    ax.plot(time, lowMode + highMode, color="#3b424b", lw=1.1, alpha=0.6, label="observed q(t) = superposition")
    ax.set_xlabel("time")
    ax.set_ylabel("q(t)")
    ax.set_title("Both modes oscillate - the pathology is not in the motion")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_xlim(time[0], time[-1])


def panelEnergy(ax, eigenvalues):
    j1 = np.linspace(0, 3, 200)
    j2 = np.linspace(0, 3, 200)
    meshJ1, meshJ2 = np.meshgrid(j1, j2)
    hamiltonian = OMEGA1 * meshJ1 - OMEGA2 * meshJ2
    limit = np.abs(hamiltonian).max()
    mesh = ax.pcolormesh(meshJ1, meshJ2, hamiltonian, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto")
    contour = ax.contour(meshJ1, meshJ2, hamiltonian, levels=[-4, -2, 0, 2], colors="#1b1f24", linewidths=0.7)
    ax.clabel(contour, fmt="%d", fontsize=7)
    ax.set_xlabel("action of healthy mode  J_1")
    ax.set_ylabel("action of ghost mode  J_2")
    ax.set_title("Ostrogradski H = w1 J1 - w2 J2:  unbounded below along J_2")
    ax.annotate(
        "no ground state:\ndrive J_2 up, H -> -inf",
        xy=(0.4, 2.4),
        fontsize=9,
        color="white",
        fontweight="bold",
    )
    figure = ax.get_figure()
    figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03, label="conserved energy H")


def panelRecovered(ax, discovery, boundary):
    terms = ["s0**2", "s1**2", "s2**2", "s0*s2"]
    pretty = ["q^2", "q'^2", "q''^2", "q q''"]
    truth = {"s0**2": 4.0, "s1**2": -5.0, "s2**2": 1.0, "s0*s2": 0.0}
    groundTruth = next(row for row in discovery if row["scenario"] == "ground_truth_derivatives")
    recovered = {row[0]: row[2] for row in groundTruth["coefficientRows"]}

    x = np.arange(len(terms))
    ax.bar(x - 0.2, [truth[t] for t in terms], 0.38, color="#3b424b", label="true L")
    ax.bar(x + 0.2, [recovered.get(t, 0.0) for t in terms], 0.38, color=vizstyle.MODE_HIGH, label="recovered from clean data")
    ax.axhline(0, color="#1b1f24", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(pretty)
    ax.set_ylim(-6.6, 6.8)
    ax.set_ylabel("coefficient  (kinetic q''^2 locked to 1)")
    ax.set_title("Recovered L differs by a null Lagrangian - same theory, still ghost")
    ax.legend(loc="lower left", fontsize=9)
    ax.text(
        0.5,
        0.94,
        "dL = 5 ( q q'' + q'^2 ) = 5 d/dt( q q' )   ->   EL(dL) = 0",
        transform=ax.transAxes,
        ha="center",
        fontsize=9.5,
        family="monospace",
        color=vizstyle.OK_GREEN,
        fontweight="bold",
    )

    inset = ax.inset_axes([0.61, 0.10, 0.35, 0.34])
    noise = []
    s0sq = []
    for row in boundary:
        scenario = row["scenario"]
        level = 0.0 if scenario == "ground_truth" else float(scenario.replace("pct", ""))
        expression = sp.sympify(row["recovered"].replace("s0", "x0").replace("s2", "x2"))
        coefficient = float(expression.coeff(sp.Symbol("x0") ** 2))
        noise.append(level)
        s0sq.append(coefficient)
    order = np.argsort(noise)
    inset.plot(np.array(noise)[order], np.array(s0sq)[order], "o-", color=vizstyle.GHOST, ms=4)
    inset.axhline(4.0, color="#3b424b", ls="--", lw=1)
    inset.set_title("q^2 coeff vs noise\n(verdict stays ghost=True)", fontsize=7.5)
    inset.set_xlabel("noise %", fontsize=7)
    inset.tick_params(labelsize=6.5)


def main():
    vizstyle.applyStyle()
    ghost = vizstyle.loadResult("ghost_detection.json")
    discovery = vizstyle.loadResult("higher_order_discovery.json")
    eigenvalues = ghost["reference"]["pais_uhlenbeck_ghost"]["hamiltonianEigenvalues"]

    figure = plt.figure(figsize=(17, 12))
    grid = figure.add_gridspec(2, 2, hspace=0.34, wspace=0.24)
    panelDispersion(figure.add_subplot(grid[0, 0]))
    panelModes(figure.add_subplot(grid[0, 1]))
    panelEnergy(figure.add_subplot(grid[1, 0]), eigenvalues)
    panelRecovered(figure.add_subplot(grid[1, 1]), discovery, ghost["noiseBoundary"])

    figure.suptitle(
        "Anatomy of the Pais-Uhlenbeck oscillator:  L = 1/2 ( q''^2 - (w1^2+w2^2) q'^2 + w1^2 w2^2 q^2 ),   w1=1, w2=2",
        fontsize=14.5,
        fontweight="bold",
    )
    vizstyle.provenance(
        figure,
        "modes/dispersion/energy analytic;  recovered coefficients from "
        "src/experiments/results/higher_order_discovery.json + ghost_detection.json",
    )
    figure.subplots_adjust(top=0.92, bottom=0.06)
    size = vizstyle.savefig(figure, OUTPUT)
    print(f"wrote {OUTPUT}  ({size} bytes)")


if __name__ == "__main__":
    main()
