import numpy as np
from glmdenoise.utils.zerodiv import zerodiv


def calc_cod(x, y, dim=None, wantgain=0, wantmeansub=1):
    """Calculate the coefficient of determination

    Args:
        x ([type]): matrix with the same dimensions as y
        y ([type]): matrix with the same dimensions as x
        dim ([type]): is the dimension of interest
        wantgain (int, optional): Defaults to 0. 0 means normal
            1 means allow a gain to be applied to each case of <x>
            to minimize the squared error with respect to <y>.
            in this case, there cannot be any NaNs in <x> or <y>.
            2 is like 1 except that gains are restricted to be non-negative.
            so, if the gain that minimizes the squared error is negative,
            we simply set the gain to be applied to be 0.
            default: 0.
        wantmeansub (int, optional): Defaults to 1.
            0 means do not subtract any mean.  this makes it such that
            the variance quantification is relative to 0.
            1 means subtract the mean of each case of <y> from both
            <x> and <y> before performing the calculation.  this makes
            it such that the variance quantification
            is relative to the mean of each case of <y>.
            note that <wantgain> occurs before <wantmeansub>.
            default: 1.

    calculate the coefficient of determination (R^2) indicating
    the percent variance in <y> that is explained by <x>.  this is achieved
    by calculating 100*(1 - sum((y-x).^2) / sum(y.^2)).  note that
    by default, we subtract the mean of each case of <y> from both <x>
    and <y> before proceeding with the calculation.

    the quantity is at most 100 but can be 0 or negative (unbounded).
    note that this metric is sensitive to DC and scale and is not symmetric
    (i.e. if you swap <x> and <y>, you may obtain different results).
    it is therefore fundamentally different than Pearson's correlation
    coefficient (see calccorrelation.m).

    NaNs are handled gracefully (a NaN causes that data point to be ignored).

    if there are no valid data points (i.e. all data points are
    ignored because of NaNs), we return NaN for that case.

    note some weird cases:
    calccod([],[]) is []

    history:
    2013/08/18 - fix pernicious case where <x> is all zeros and <wantgain>
    is 1 or 2.
    2010/11/28 - add <wantgain>==2 case
    2010/11/23 - changed the output range to percentages.  thus, the range
    is (-Inf,100]. also, we removed the <wantr> input since
    it was dumb.

    example:
    x = np.random.random(100)
    calccod(x,x+0.1*np.random.random(100))
    """

    # input
    if dim is None:
        dim = np.argmax(x.shape)

    # handle gain
    if wantgain:
        # to get the residuals, we want to do something like y-x*inv(x'*x)*x'*y
        temp = 1/np.dot(x, x) * np.dot(x, y)
        if wantgain == 2:
            # if the gain was going to be negative, rectify it to 0.
            temp[temp < 0] = 0
        x = x * temp

    # propagate NaNs (i.e. ignore invalid data points)
    x[np.isnan(y)] = np.nan
    y[np.isnan(x)] = np.nan

    # handle mean subtraction
    if wantmeansub:
        mn = np.nanmean(y, axis=dim)
        y = y - mn
        x = x - mn

    # finally, compute it
    with np.errstate(divide='ignore', invalid='ignore'):
        nom = np.nansum((y-x) ** 2, axis=dim)
        denom = np.nansum((y**2), axis=dim)
        f = np.nan_to_num(1 - (nom / denom))
    return 100*f


def calc_cod_stack(yhat, y):
    """
    [summary]

    Args:
        data ([type]): [description]
        pred ([type]): [description]

    Returns:
        r2s: global r2s

    """

    numer = np.asarray(
        [np.sum((a-b)**2, axis=0) for a, b in zip(yhat, y)]
        ).sum(axis=0)
    denom = np.asarray(
        [np.sum(a**2, axis=0) for a in y]
        ).sum(axis=0)

    # calculate global R2

    r2s = 100*(1 - zerodiv(
        numer,
        denom,
        val=np.nan,
        wantcaution=0
        )
    )

    return r2s
