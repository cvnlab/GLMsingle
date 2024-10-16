import numpy as np


def select_noise_regressors(r2_nrs, pcstop=1.05):
    """How many components to include

    Args:
        r2_nrs (ndarray): Model fit value per solution
        pcstop (float, optional): Defaults to 1.05.

    Returns:
        int: Number of noise regressors to include
    """
    numpcstotry = r2_nrs.size - 1

    # this is the performance curve that starts at 0 (corresponding to 0 PCs)
    curve = r2_nrs - r2_nrs[0]

    # initialize (this will hold the best performance observed thus far)
    chosen = 0
    best = -np.inf
    for p in range(numpcstotry+1):  # notice the +1

        # if better than best so far
        if curve[p] > best:

            # record this number of PCs as the best
            chosen = p
            best = curve[p]

            # if we are within opt.pcstop of the max, then we stop.
            if (best * pcstop) >= curve.max():
                break

    return chosen
