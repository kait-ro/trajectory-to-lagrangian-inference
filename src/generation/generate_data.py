import csv

import numpy as np
from generation.integrator import simulateTrajectory
from generation.noise import addNoise


def _buildHeader(noCoords: int) -> list:
    header = ["trajectory_id", "t"]
    for c in range(noCoords):
        header += [f"q{c}", f"q{c}dot", f"q{c}ddot"]
    return header


def generateDatasetStreaming(
    outputPath: str,
    noTrajectories: int,
    noSteps: int,
    dt: float,
    noisePercentage: float,
    accelFunctions: list,
    noCoords: int = 3,
    flushEvery: int = 50,
):
    header = _buildHeader(noCoords)

    with open(outputPath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        rowsBuffer = []
        
        for trajectoryId in range(noTrajectories):
            initialState = np.random.uniform(-1, 1, size=2 * noCoords)
            t_arr, q_arr, qdot_arr, qddot_arr = simulateTrajectory(
                initialState, accelFunctions, dt, noSteps
            )

            q_arr = addNoise(q_arr, noisePercentage)
            qdot_arr = addNoise(qdot_arr, noisePercentage)
            qddot_arr = addNoise(qddot_arr, noisePercentage)

            for i in range(noSteps):
                row = [trajectoryId, t_arr[i]]
                for c in range(noCoords):
                    row += [q_arr[i, c], qdot_arr[i, c], qddot_arr[i, c]]
                rowsBuffer.append(row)

            if (trajectoryId + 1) % flushEvery == 0:
                writer.writerows(rowsBuffer)
                rowsBuffer = []

        if rowsBuffer:
            writer.writerows(rowsBuffer)

    print(f"Streamed {noTrajectories} trajectories ({noTrajectories * noSteps} rows) to {outputPath}")