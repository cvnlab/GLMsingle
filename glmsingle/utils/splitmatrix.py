import numpy as np
from math import floor, ceil

import numpy as np
import matplotlib.pyplot as plt

def splitmatrix(m, dim, splt=None):
    """
    m is a matrix
    dim is a dimension
    splt (optional) is a list of positive integers indicating
    how to perform the split.  default: [1]*m.shape[dim].
    you can also negate the entries to indicate that you
    do not want that entry returned.  special case is splt==0
    which means use splt equal to m.shape[dim].

    split m along dimension dim, returning a list of arrays.
    """

    # handle optional input
    if splt is None:
        splt = []
    if splt == 0:
        splt = [m.shape[dim]]

    # figure out the dimensions of m
    msize = [1]*len(m.shape)
    msize[:m.ndim] = list(m.shape)

    # hack it in
    if len(splt) == 0:
        splt = [1]*m.shape[dim]
    msize[dim] = [abs(s) for s in splt]

    # do it
    array_list = np.split(m, np.cumsum(msize[dim])[:-1], dim)
    return [f for f, s in zip(array_list, splt) if s > 0]