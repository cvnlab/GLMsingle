import numpy as np
import matplotlib.pyplot as plt
from glmsingle.utils.normalizerange import normalizerange
from glmsingle.utils.cmapsign4 import cmapsign4


def cmaplookup(x, mn, mx, circulartype=0, cmap=None):
    """
    Map values in x to a colormap.
    
    x: matrix with values
    mn, mx: values to associate with the minimum and maximum of the colormap
    circulartype: 
        0 means normal colormap lookup (min is .5, max is n+.5)
        1 means centered colormap lookup (min is 1, max is n+1 (repeat first color))
    cmap: the colormap. Default is current figure colormap.
    """
    # Default values
    if cmap is None:
        cmap=cmapsign4(256)

    # also make sure x is an array
    if type(x) is not np.ndarray:
        # list or float provided
        x = np.array(x)


    ncols = cmap.shape[0]
    
    # Calculate indices into colormap
    if circulartype == 0:
        f = np.round(normalizerange(x, 0.5, ncols + 0.5, mn, mx))
        f[f == ncols + 1] = ncols  # Handle outliers at the top
        f[f == 0] = 1  # Handle rounding errors
    elif circulartype == 1:
        f = np.round(normalizerange(x, 1, ncols + 1, mn, mx))
        f[f == ncols + 1] = 1  # Top one is the same as the first
    
    f[np.isnan(f)] = 1


    # Reshape to return RGB values
    if np.ndim(x) == 1:
        sz = [len(x), 3]
    else:
        sz = list(x.shape) + [3]

    f = np.reshape(cmap[f.astype(int)-1], sz)

    return f

