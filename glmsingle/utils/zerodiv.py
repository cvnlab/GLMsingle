import numpy as np


def zerodiv(data1, data2, val=0, wantcaution=1):
    """zerodiv(data1,data2,val,wantcaution)
    Args:
        <data1>,<data2> are matrices of the same size or either
                        or both can be scalars.
        <val> (optional) is the value to use when <data2> is 0.
                        default: 0.
        <wantcaution> (optional) is whether to perform special
                        handling of weird cases (see below).
                        default: 1.
        calculate data1./data2 but use <val> when data2 is 0.
        if <wantcaution>, then if the absolute value of one or
                        more elements of data2 is less than 1e-5
                        (but not exactly 0), we issue a warning
                        and then treat these elements as if they
                        are exactly 0.
        if not <wantcaution>, then we do nothing special.

    note some weird cases:
    if either data1 or data2 is [], we return [].
    NaNs in data1 and data2 are handled in the usual way.

    """

    # handle special case of data2 being scalar
    if np.isscalar(data2):
        if data2 == 0:
            f = np.tile(val, data1.shape)
        else:
            if wantcaution and abs(data2) < 1e-5:
                print(
                    'warning: abs value of divisor is less than 1e-5.'
                    'treating the divisor as 0.')
                f = np.tile(val, data1.shape)
            else:
                f = data1/data2

    else:
        # do it
        bad = data2 == 0
        bad2 = abs(data2) < 1e-5
        if wantcaution and np.any(bad2.ravel()) and ~bad.ravel():
            print(
                'warning: abs value of one or more divisors'
                'less than 1e-5.treating them as 0.')

        if wantcaution:
            data2[bad2] = 1
            f = data1/data2
            f[bad2] = val
        else:
            data2[bad] = 1
            f = data1/data2
            f[bad] = val

    return f
