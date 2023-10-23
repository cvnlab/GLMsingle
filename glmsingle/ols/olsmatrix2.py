import numpy as np


def olsmatrix2(X, lambda_=0):
    """
    Returns the OLS matrix.

    Parameters:
        - X: A 2D numpy array with shape (samples, parameters).
        - lambda_: Ridge parameter. Default is 0.
    
    Returns:
        - f: A 2D numpy array with shape (parameters, samples).
    """
    
    # Check for NaN values
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values.")
    
    # bad regressors are those that are all zeros
    bad = np.all(X == 0, axis=0)
    good = ~bad
    
    # report warning
    if np.any(bad):
        print("Warning: One or more regressors are all zeros; we will estimate a 0 weight for those regressors.")
    
    # Initialize f
    f = np.zeros((X.shape[1], X.shape[0]))
    
    if np.any(bad):
        good_indices = np.where(good)[0]
        part1 = np.dot(X[:, good_indices].T, X[:, good_indices]) + lambda_ * np.eye(np.sum(good))
        f[good, :] = np.linalg.solve(part1, X[:, good_indices].T)
    else:
        part1 = np.dot(X.T, X) + lambda_ * np.eye(X.shape[1])
        f = np.linalg.solve(part1, X.T)

    return f

# # Test
# X = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
# result = olsmatrix2(X)
# print(result)