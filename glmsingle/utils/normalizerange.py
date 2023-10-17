import numpy as np


def normalizerange(m, targetmin, targetmax, sourcemin=None, sourcemax=None, chop=1, mode=0, fast=False):
    """
    m : a matrix
    targetmin : the minimum desired value.  can be a scalar or a matrix the same size as m.
    targetmax : the maximum desired value.  can be a scalar or a matrix the same size as m.
    sourcemin : (optional) sets the min value of m.  can be a scalar or a matrix the same size as m.
    sourcemax : (optional) sets the max value of m.  can be a scalar or a matrix the same size as m.
    chop : (optional) is whether to chop off the ends such that there are no values below targetmin nor above targetmax.
    mode : (optional) 
        0 means normal operation
        1 means interpret sourcemin and sourcemax as multipliers for the std of m. In this mode, sourcemin and sourcemax cannot be NaN.
    fast : (optional) means we have a guarantee that all inputs are fully specified and m is not empty.

    Return m scaled and translated such that [sourcemin,sourcemax] maps to [targetmin,targetmax]. 
    If chop, we also threshold values below targetmin and values above targetmax.
    """

    if fast:
        pass
    else:
        if m.size == 0:
            return m
        if sourcemin is None:
            sourcemin = np.nanmin(m)
        if sourcemax is None:
            sourcemax = np.nanmax(m)
        if chop is None:
            chop = 1
        if mode is None:
            mode = 0

    skipchop = (mode==0 and (sourcemin is None and sourcemax is None)) or (mode==0 and np.isnan(sourcemin) and np.isnan(sourcemax))

    if mode == 0:
        if sourcemin is None:
            sourcemin = np.nanmin(m)
        if sourcemax is None:
            sourcemax = np.nanmax(m)
        if np.isnan(sourcemin) or np.isnan(sourcemax):
            temp = np.nanmax(np.abs(m))
            if np.isnan(sourcemin):
                sourcemin = -temp
            if np.isnan(sourcemax):
                sourcemax = temp
    elif mode == 1:
        if sourcemin is None:
            sourcemin = -3
        if sourcemax is None:
            sourcemax = 3
        mn = np.nanmean(m)
        sd = np.nanstd(m)
        sourcemin = mn+sourcemin*sd
        sourcemax = mn+sourcemax*sd

    if np.any(sourcemin==sourcemax):
        raise ValueError('sourcemin and sourcemax are the same in at least one case')

    if chop and not skipchop:
        temp = np.isnan(m)
        m = np.maximum(np.minimum(m,sourcemax),sourcemin)
        m[temp] = np.nan

    val = (targetmax-targetmin)/(sourcemax-sourcemin)
    f = m*val - (sourcemin*val - targetmin)

    return f