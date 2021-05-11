import numpy as np


def block_diag(arrays, numruns):

    rows, cols = arrays[0].shape
    result = np.zeros((numruns * rows, numruns * cols), dtype=arrays[0].dtype)
    for k in range(numruns):
        result[k * rows:(k + 1) * rows, k * cols:(k + 1) * cols] = arrays[k]
    return result
