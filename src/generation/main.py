import os
from pathlib import Path

import sympy as sp

from generation.eqnofmotion import defineCoordinates
from generation.generate_data import generateDatasetStreaming
from generation.integrator import GetAccelFunctions


def main():
    noCoords = 15
    noTrajectories = 150

    noSteps = 1000
    dt = 0.01
    noiseLevels = [0.0, 0.05, 0.10, 0.25, 0.50]

    t, coords, vels = defineCoordinates(noCoords)

    m, k, eps = sp.symbols("m k epsilon")
    rSquared = sum(q**2 for q in coords)
    vSquared = sum(v**2 for v in vels)
    L = sp.Rational(1, 2) * m * vSquared - sp.Rational(1, 2) * k * rSquared - sp.Rational(1, 4) * eps * rSquared**2

    constants = {m: 1.0, k: 1.0, eps: 0.3}
    print(f"Deriving equations of motion for {noCoords} coordinates")
    accelFuncs = GetAccelFunctions(L, coords, vels, t, constants)

    dataDir = str(Path(__file__).resolve().parents[2] / "assets")
    os.makedirs(dataDir, exist_ok=True)

    for noisePercentage in noiseLevels:
        outputPath = os.path.join(
            dataDir, f"trajectories_forwardselection_smallsteps_noise{int(noisePercentage*100)}.csv"
        )
        print(f"Generating {noTrajectories} trajectories, {noSteps} steps each")
        generateDatasetStreaming(
            outputPath=outputPath,
            noTrajectories=noTrajectories,
            noSteps=noSteps,
            dt=dt,
            noisePercentage=noisePercentage,
            accelFunctions=accelFuncs,
            noCoords=noCoords,
        )


if __name__ == "__main__":
    main()
