import argparse
import os
from pathlib import Path

import numpy as np
from experiments.systems import SYSTEMS
from generation.eqnofmotion import defineCoordinates
from generation.generate_data import generateDatasetStreaming
from generation.integrator import GetAccelFunctions

ASSETS_DIR = str(Path(__file__).resolve().parents[2] / "assets")
DEFAULT_SEED = 20260828


def datasetPath(system, noisePercentage, seed=None):
    suffix = f"_seed{seed}" if seed is not None and seed != DEFAULT_SEED else ""
    return os.path.join(ASSETS_DIR, f"{system.datasetStem()}_noise{round(noisePercentage * 100)}{suffix}.csv")


def generateSystemDatasets(systemName, noiseLevels=None, overwrite=False, seed=DEFAULT_SEED):
    system = SYSTEMS[systemName]
    requestedLevels = system.noiseLevels if noiseLevels is None else noiseLevels

    t, coords, vels = defineCoordinates(system.noCoords)
    lagrangian, constants = system.buildLagrangian(coords, vels)
    accelFunctions = GetAccelFunctions(lagrangian, coords, vels, t, constants)

    os.makedirs(ASSETS_DIR, exist_ok=True)
    producedPaths = []
    for noisePercentage in requestedLevels:
        outputPath = datasetPath(system, noisePercentage, seed)
        if os.path.exists(outputPath) and not overwrite:
            producedPaths.append(outputPath)
            continue
        np.random.seed(seed)
        generateDatasetStreaming(
            outputPath=outputPath,
            noTrajectories=system.noTrajectories,
            noSteps=system.noSteps,
            dt=system.dt,
            noisePercentage=noisePercentage,
            accelFunctions=accelFunctions,
            noCoords=system.noCoords,
        )
        producedPaths.append(outputPath)
    return producedPaths


def _parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("system", choices=sorted(SYSTEMS))
    parser.add_argument("--noise", type=float, nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parseArgs()
    for path in generateSystemDatasets(arguments.system, arguments.noise, arguments.overwrite, arguments.seed):
        print(path)
