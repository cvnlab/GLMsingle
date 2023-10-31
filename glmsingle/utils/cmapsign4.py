import numpy as np


def cmapsign4(n=64):
    """
    Returns a cyan-blue-black-red-yellow colormap.
    
    n: desired number of entries, default: 64

    Suitable for ranges like [-X X].
    """
    # Constants
    colors = np.array([
        [.8, 1, 1],   # cyan-white
        [0, 1, 1],    # cyan
        [0, 0, 1],    # blue
        [0, 0, 0],    # black
        [1, 0, 0],    # red
        [1, 1, 0],    # yellow
        [1, 1, .8]    # yellow-white
    ])

    # Interpolation
    x = np.linspace(0, 1, colors.shape[0])
    f = np.zeros((n, 3))
    for p in range(3):
        f[:, p] = np.interp(np.linspace(0, 1, n), x, colors[:, p])
    return f