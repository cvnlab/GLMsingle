import numpy as np


def check_inputs(data, design):
    """
    check that the data and design meet the required
    specifications for glm single trial estimates.

    Arguments
    _________

        data (list): could be x,y,z,t or XYZ,t
        design (list of runs or single run): design matrix

    Returns:
    ________

        data (list): X Y Z T or XYZ 1 1 T  data format
        design (list): design matrix with a list entry per run
    """
    # massage <design> and sanity-check it
    if type(design) is not list:
        # case sparse, not a list:
        if 'sparse' in str(type(design)):
            design = np.asarray(design.todense())
        design = [design]
    else:
        # it is a list and it list elements are sparse
        if 'sparse' in str(type(design[0])):
            design = [np.asarray(d.todense()) for d in design]

    numcond = design[0].shape[1]
    for p in range(len(design)):
        np.testing.assert_array_equal(
            np.unique(design[p]),
            [0, 1],
            err_msg='<design> must consist of 0s and 1s')
        condmsg = \
            'all runs in <design> should have equal number of conditions'
        np.testing.assert_equal(
            design[p].shape[1],
            numcond,
            err_msg=condmsg)
        # if the design happened to be a sparse?
        # design[p] = np.full(design[p])

    # massage <data> and sanity-check it
    if type(data) is not list:
        data = [data]

    # make sure it is single
    for p in range(len(data)):
        data[p] = data[p].astype(np.float32, copy=False)

    np.testing.assert_equal(
        np.all(np.isfinite(data[0].flatten())),
        True,
        err_msg='We checked the first run and '
        'found some non-finite values (e.g. NaN, Inf).'
        'Please fix and re-run.')
    np.testing.assert_equal(
        len(design),
        len(data),
        err_msg='<design> and <data> should have '
        'the same number of runs')

    # make sure design is in float 32
    for p in range(len(design)):
        design[p] = design[p].astype(np.float32, copy=False)


    # check whether data is 2d (e.g. surf) or 3d (e.g. vol)
    is3d = data[0].ndim > 2  # is this the X Y Z T case?

    if not is3d:
        xyz=False
        for p in range(len(data)):
            s = data[p].shape
            if len(s) == 2:  # If it's just X Y
                data[p] = data[p].reshape(s[0], 1, 1, s[1])
            elif len(s) == 3:  # If it's X Y Z
                data[p] = data[p].reshape(s[0], s[1], 1, s[2])
    else:
        xyz=data[0].shape[:3]


    # check number of time points and truncate if necessary
    for run_i in np.arange(len(data)):
        if data[run_i].shape[3] > design[run_i].shape[0]:
            print(
                f'WARNING: run {run_i} has more time points'
                'in <data> than <design>. We are truncating'
                '<data>.\n')
            data[run_i] = data[run_i][:, :, :, np.arange(
                design[run_i].shape[0])]

        if data[run_i].shape[3] < design[run_i].shape[0]:
            print(
                f'WARNING: run {run_i} has more time points in <design>'
                'than <data>. We are truncating <design>.\n')
            design[run_i] = design[run_i][np.arange(data[run_i].shape[-1]), :]

    return data, design, xyz
