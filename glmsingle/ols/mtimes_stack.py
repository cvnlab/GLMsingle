import numpy as np


def mtimes_stack(X, data):
    """f = mtimes_stack(X, m2)

    simply return <m1>*np.vstack(m2) but do so in a way that doesn't cause
    too much memory usage.

    Args:
        m1 ([A x B]): is A x B
        m2 ([B x C]): is a stack of matrices such that np.vstack(m2) is B x C
    """
    betas = 0
    start_col = 0

    for run in range(len(data)):
        n_vols = data[run].shape[0]
        these_cols = np.arange(n_vols) + start_col
        betas += X[:, these_cols] @ data[run]
        start_col += data[run].shape[0]

    return betas
