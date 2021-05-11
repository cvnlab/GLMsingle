import numpy as np
from scipy.interpolate import pchip


def convolve_design(X, hrf, opt=None):
    """convolve each column of a 2d design matrix with hrf

    Args:
        X ([2D design matrix]): time by cond, or list of onsets
        hrf ([1D hrf function]): hrf
        opt: if onset case, provides n_times and tr for
             interpolation

    Returns:
        [convdes]: 2D: Samples by cond
    """

    # if onset-time case
    if type(X) is list:
        errmsg = 'n_times needs to be in opt'
        np.testing.assert_equal(
            'n_times' in opt,
            True,
            err_msg=errmsg)
        n_times = opt['n_times']
        tr = opt['tr']

        # calc
        n_conditions = len(X)
        convdes = np.zeros((n_times, n_conditions))

        all_times = np.linspace(0, tr*(n_times-1), n_times)
        hrf_times = np.linspace(0, tr*(len(hrf)-1), len(hrf))

        for q in range(n_conditions):
            # onset times for qth condition in run p
            otimes = X[q]

            # intialize
            yvals = np.zeros((n_times))

            # loop over onset times
            for r in otimes:
                # interpolate to find values at the
                # data sampling time points
                f = pchip(
                    r + hrf_times,
                    hrf,
                    extrapolate=False)(all_times)

                yvals = yvals + np.nan_to_num(f)

            # record
            convdes[:, q] = yvals

    # normal vector or matrix cases
    else:
        ndims = X.ndim
        if ndims == 1:
            ntime = X.shape[0]
            convdes = np.convolve(X, hrf)
            convdes = convdes[range(ntime)]
        else:
            ntime, ncond = X.shape
            convdes = np.asarray(
                [np.convolve(X[:, x], hrf, ) for x in range(ncond)]).T
            convdes = convdes[range(ntime), :]

    return convdes
