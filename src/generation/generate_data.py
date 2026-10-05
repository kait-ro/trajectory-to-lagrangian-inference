import csv

import numpy as np
from generation.higher_order_integrator import rk4Step
from generation.integrator import simulateTrajectory
from generation.noise import addNoise


def _buildHeader(noCoords: int) -> list:
    header = ["trajectory_id", "t"]
    for c in range(noCoords):
        header += [f"q{c}", f"q{c}dot", f"q{c}ddot"]
    return header


def _higherOrderHeader(noCoords: int, equationOrder: int) -> list:
    header = ["trajectory_id", "t"]
    for c in range(noCoords):
        for level in range(equationOrder + 1):
            header.append(f"q{c}_d{level}")
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


def generateHigherOrderDatasetStreaming(
    outputPath: str,
    noTrajectories: int,
    noSteps: int,
    dt: float,
    noisePercentage: float,
    stateDerivative,
    equationOrder: int,
    noCoords: int,
    flushEvery: int = 50,
):
    header = _higherOrderHeader(noCoords, equationOrder)

    with open(outputPath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        rowsBuffer = []

        for trajectoryId in range(noTrajectories):
            state = np.random.uniform(-1, 1, size=equationOrder * noCoords)
            recorded = np.zeros((noSteps, equationOrder * noCoords))
            topDerivative = np.zeros((noSteps, noCoords))

            for stepIndex in range(noSteps):
                recorded[stepIndex] = state
                derivative = stateDerivative(state)
                topDerivative[stepIndex] = derivative[-noCoords:]
                state = rk4Step(state, dt, stateDerivative)

            perDerivative = [recorded[:, level * noCoords:(level + 1) * noCoords] for level in range(equationOrder)]
            perDerivative.append(topDerivative)

            noisyDerivatives = [addNoise(block, noisePercentage) for block in perDerivative]

            for i in range(noSteps):
                row = [trajectoryId, i * dt]
                for c in range(noCoords):
                    for level in range(equationOrder + 1):
                        row.append(noisyDerivatives[level][i, c])
                rowsBuffer.append(row)

            if (trajectoryId + 1) % flushEvery == 0:
                writer.writerows(rowsBuffer)
                rowsBuffer = []

        if rowsBuffer:
            writer.writerows(rowsBuffer)

    print(f"Streamed {noTrajectories} trajectories ({noTrajectories * noSteps} rows) to {outputPath}")
