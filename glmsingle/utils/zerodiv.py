import numpy as np
import warnings
import pdb

def zerodiv(x, y, val=0, wantcaution=1):
    """zerodiv(data1,data2,val,wantcaution)
    Args:
        <x>,<y> are matrices of the same size or either
                        or both can be scalars.
        <val> (optional) is the value to use when <y> is 0.
                        default: 0.
        <wantcaution> (optional) is whether to perform special
                        handling of weird cases (see below).
                        default: 1.
        calculate x/y but use <val> when y is 0.
        if <wantcaution>, then if the absolute value of one or
                        more elements of y is less than 1e-5
                        (but not exactly 0), we issue a warning
                        and then treat these elements as if they
                        are exactly 0.
        if not <wantcaution>, then we do nothing special.

    note some weird cases:
    if either x or y is [], we return [].

    """

    # Check if either x or y is empty, return empty if so
    if x.size == 0 or y.size == 0:
        return np.array([])

    # handle special case of y being scalar
    if np.isscalar(y):
        if y == 0:
            return np.full(x.shape, val)
        else:
            if wantcaution and abs(y) < 1e-5:   # see allzero.m
                warnings.warn('abs value of divisor is less than 1e-5. we are treating the divisor as 0.')
                return np.full(x.shape, val)
            else:
                return x / y[:, np.newaxis]
    else:
        bad = y == 0
        bad2 = abs(y) < 1e-5  # see allzero.m
        if wantcaution and np.any(np.logical_and(bad2, ~bad)):
            warnings.warn('abs value of one or more divisors is less than 1e-5. we are treating these divisors as 0.')
        if wantcaution:
            tmp = y
            tmp[bad2] = 1
            if x.ndim == 1:
                f = x / tmp
            else:
                f = x / tmp[:, np.newaxis]
            f[bad2] = val
        else:
            tmp = y
            tmp[bad] = 1
            if x.ndim == 1:
                f = x / tmp
            else:
                f = x / tmp[:, np.newaxis]
            f[bad] = val
        return f