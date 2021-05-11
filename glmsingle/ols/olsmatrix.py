import numpy as np
from sklearn.preprocessing import normalize
import scipy.linalg


def olsmatrix_ulen(X, mode=0):
    """
     olsmatrix(X, mode, verbose)

     <X> is samples x parameters
     <mode> (optional) is
       0 means normal operation
       1 means use inv instead of '\' and omit unit-length normalization.
         the point of this mode is to reduce memory usage (i think).
       default: 0.

     what we want to do is to perform OLS regression using <X>
     and obtain the parameter estimates.  this is accomplished
     by inv(X'*X)*X'*y = f*y where y is the data (samples x cases).

     what this function does is to return <f> which has dimensions
     parameters x samples.

     to ensure well-conditioning, we unit-length normalize each column
     of <X> before doing the inv.m operation.  also, we actually use
     left-slash ('\'), which apparently is equivalent to inv.m but faster
     and more accurate (see code for details).  if you pass <mode> as 1,
     we omit the normalization and use the inv method instead of the '\'
     method.

     also, in the case that one or more regressors in <X> are all zero, then
     without special handling, then this case will result in warnings and
     NaN results.  what we do is to explicitly ensure that all-zero regressors
     are ignored and that there are all-zero rows for these regressors
     in <f>.  this makes it such that the weights estimated for these
     regressors are simply zero.

     history:
     2011/06/27 - explicitly ignore 0
     2011/03/29 - mode 1 now omits unit length normalization.
     2011/03/28 - add <mode> input.
     2010/08/05: now, we do not try to detect weird cases with low variance.
                 instead, we just blindly unit-length normalize each column.
    """

    # input
    # deal with degenerate regressors
    good = np.invert(np.all(X == 0, axis=0))

    # initialize result
    f = np.zeros((X.shape[1], X.shape[0]))

    # do it
    if mode == 0:
        X, length = normalize(X[:, good], axis=0, return_norm=True)
        di = np.diag(1/length)
        XT = np.transpose(X)
        glm = scipy.linalg.solve(
                np.einsum('ij,jk', XT, X),
                XT)
        temp = np.einsum('ij,jk', di, glm)

    elif mode == 1:
        X = X[:, good]
        XT = np.transpose(X)
        temp = np.einsum('ij,jk', np.linalg.inv(np.einsum('ij,jk', XT, X)), XT)

    # return
    if np.any(good):
        f[good, :] = temp

    return f


def olsmatrix(X, lambd=0, verbose=True):
    """OLS regression

    what we want to do is to perform OLS regression using <X>
    and obtain the parameter estimates. this is accomplished
    by np.linalg.inv(X.T @ X) @ X.T @ y = f @ y where y is the
    data (samples x cases).

    what this function does is to return <f> which has dimensions
    parameters x samples.

    we check for a special case, namely, when one or more regressors
    are all zeros.  if we find that this is the case, we issue a warning
    and simply ignore these regressors when fitting.  thus, the weights
    associated with these regressors will be zeros.

    if any warning messages are produced by the inversion process, then we die.
    this is a conservative strategy that ensures that the regression is
    well-behaved (i.e. has a unique, finite solution).  (note that this does
    not cover the case of zero regressors, which is gracefully handled as
    described above.)

    note that no scale normalization of the regressor columns is performed.

    Args:
        X (ndarray): Samples by parameters

    Returns:
        (f): 2D parameters by Samples
    """

    bad = np.all(X == 0, axis=0)
    good = np.invert(bad)

    # report warning
    if not np.any(good) == 1:
        if verbose:
            print(
                "regressors are all zeros. \n"
                "we will estimate a 0 weight for those regressors."
            )
        f = np.zeros((X.shape[1], X.shape[0])).astype(np.float32)
        return f

    # do it
    if np.any(bad):
        if verbose:
            print(
                "One or more regressors are all zeros. \n"
                "we will estimate a 0 weight for those regressors."
            )
        f = np.zeros((X.shape[1], X.shape[0])).astype(np.float32)
        X = X[:, good]
        XT = np.transpose(X)
        XTX = np.einsum('ij,jk', XT, X)
        f[good, :] = np.einsum(
            'ij,jk',
            np.linalg.inv(XTX),
            XT)

    else:
        XT = np.transpose(X)
        XTX = np.einsum('ij,jk', XT, X)
        f = np.einsum('ij,jk', np.linalg.inv(XTX), XT)

    return f.astype(np.float32)
