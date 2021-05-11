from sklearn.preprocessing import normalize
import numpy as np


def projectionmatrix(X):
    """ Calculates a projection matrix

    Args:
        X (array): design matrix

    Returns:
        array: Projection matrix size of X.shape[0] x X.shape[0]
    """
    if X is None:
        return 1
    else:
        X = np.mat(X)
        # Note (mat)
        # It is no longer recommended to use this class, even for linear algebra.
        # Instead use regular arrays. The class may be removed in the future.
        return np.eye(X.shape[0]) - (X*(np.linalg.inv(X.T*X)*X.T))


def constructpolynomialmatrix(n, degrees):
    """Calculates a matrix of polynomials used to regress them out of your data

    Args:
        n (int): number of points
        degrees (array): vector of polynomial degrees

    Returns:
        array: array of n x len(degrees)
    """
    time_points = np.linspace(-1, 1, n)[np.newaxis].T
    polys = np.zeros((n, len(degrees)))

    # Loop over degrees
    for i, d in enumerate(degrees):
        polyvector = np.mat(time_points**d)

        if i > 0:  # project out the other polynomials
            polyvector = projectionmatrix(polys[:, :i]) * polyvector

        polys[:, i] = normalize(polyvector.T)
    return polys
