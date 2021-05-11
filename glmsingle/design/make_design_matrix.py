import numpy as np
from scipy.interpolate import pchip


def make_design(events, tr, n_times, hrf=None):
    """generate either a blip design or one convolved with an hrf

    Args:
        events ([type]): [description]
        tr ([type]): [description]
        n_times ([type]): [description]
        hrf ([type], optional): Defaults to None. [description]

    Returns:
        [type]: [description]
    """

    # loop over conditions
    conditions = np.unique(events.trial_type)
    n_conditions = len(set(events['trial_type'].values))

    dm = np.zeros((n_times, n_conditions))

    if hrf is None:
        for i, cond in enumerate(conditions):

            # onset times for qth condition in run p
            otimes = np.array(
                (events[events['trial_type'] == cond]['onset'].values//tr)).astype(int)
            yvals = np.zeros((n_times))
            for r in otimes:
                yvals[r] = 1
            dm[:, i] = yvals

    else:
        # calc
        all_times = np.linspace(0, tr*(n_times-1), n_times)
        hrf_times = np.linspace(0, tr*(len(hrf)-1), len(hrf))

        for i, cond in enumerate(conditions):
            # onset times for qth condition in run p
            otimes = events[events['trial_type'] == cond]['onset'].values

            # intialize
            yvals = np.zeros((n_times))

            # loop over onset times
            for r in otimes:
                # interpolate to find values at the data sampling time points
                f = pchip(r + hrf_times, hrf, extrapolate=False)(all_times)
                yvals = yvals + np.nan_to_num(f)

            # record
            dm[:, i] = yvals

    return dm
