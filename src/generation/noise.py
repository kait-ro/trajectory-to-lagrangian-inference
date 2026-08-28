import numpy as np


def addNoise(data, noisePercentage):
    std = noisePercentage * np.std(data, axis=0)
    noise = np.random.normal(0, std, data.shape)
    return data + noise