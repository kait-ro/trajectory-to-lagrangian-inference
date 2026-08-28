import argparse
import os

import numpy as np

from experiments.systems import SYSTEMS
from generation.eqnofmotion import defineCoordinates
from generation.generate_data import generateDatasetStreaming
from generation.integrator import GetAccelFunctions

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_SEED = 20260828


def datasetPath(system, noisePercentage):
    return os.path.join(DATA_DIR, f"{system.datasetStem()}_noise{int(round(noisePercentage * 100))}.csv")


def generateSystemDatasets(systemName, noiseLevels=None, overwrite=False, seed=DEFAULT_SEED):
    system = SYSTEMS[systemName]
    requestedLevels = system.noiseLevels if noiseLevels is None else noiseLevels

    t, coords, vels = defineCoordinates(system.noCoords)
    lagrangian, constants = system.buildLagrangian(coords, vels)
    accelFunctions = GetAccelFunctions(lagrangian, coords, vels, t, constants)

    os.makedirs(DATA_DIR, exist_ok=True)
    producedPaths = []
    for noisePercentage in requestedLevels:
        outputPath = datasetPath(system, noisePercentage)
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
