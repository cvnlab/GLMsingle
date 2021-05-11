import numpy as np
from glmsingle.cod.calc_cod import calc_cod, calc_cod_stack
from glmsingle.design.construct_stim_matrices import construct_stim_matrices
from glmsingle.design.make_design_matrix import make_design
from glmsingle.ols.mtimes_stack import mtimes_stack
from glmsingle.ols.olsmatrix import olsmatrix


def optimise_hrf(
        design,
        data,
        tr,
        hrfknobs,
        combinedmatrix,
        numforhrf=50,
        hrfthresh=0.5,
        hrffitmask=1,
        ):
    """Optimise hrf from a selection of voxels.

    This uses an iterative fitting optimisation procedure,
    where we fit for betas and then fit for hrf using a fir
    like approach.

    Args:
        design (pandas dataframe): this is a pandas data frame with keys:
            ['trial_type']: stimulus condition index
            ['onset']: onsets in s for each event
            ['duration']: duration for each event
        data (2d array): data (time x vox). this data should already
            have polynomials projected out.
        tr (float): the sampling rate in seconds
        hrfknobs (1d array): should be time x 1 with the initial seed
            for the HRF.  The length of this vector indicates the
            number of time points that we will attempt to estimate
            in the HRF. Note on normalization: after fitting the HRF, we
            normalize the HRF to peak at 1 (and adjust amplitudes
            accordingly).
        combinedmatrix (stack of 2d arrays): projection matrix of the
                polynomials and extra regressors (if passed by user).
                This is used to whiten the design matrix.
        numforhrf (int, optional): Defaults to 50.
                is a positive integer indicating the number of voxels
                (with the best R^2 values) to consider in fitting the
                global HRF.  (If there are fewer than that number of
                voxels available, we just use the voxels that are
                available.)
        hrfthresh (float, optional): Defaults to .5.
                If the R^2 between the estimated HRF and the initial HRF
                is less than <hrfthresh>, we decide to just use the initial
                HRF. Set <hrfthresh> to -Inf if you never want to reject
                the estimated HRF.

    Returns:
        (Dict): we return a dictionary with kers:
            ["hrf"]: the optimised hrf (but see note above on hrfthresh)
            ["hrffitvoxels"]: the indices of the voxels used to fit.
            ["convdesign"]: the design convolved with the optimised hrf
            and polynomials projected out.
            ["seedhrf"]: we return the seed hrf for book keeping.

    """

    minR2 = 0.99

    # calc
    numinhrf = len(hrfknobs)

    numruns = len(design)

    postnumlag = numinhrf - 1

    # collect ntimes per run
    ntimes = []

    for p in range(numruns):
        ntimes.append(data[p].shape[0])

    # loop until convergence
    currenthrf = hrfknobs  # initialize
    cnt = 1
    while True:
        print('\t optimising hrf :{}\n'.format(cnt))

        # fix the HRF, estimate the amplitudes
        if cnt % 2 == 1:

            # prepare design matrix
            convdesign = []
            for p in range(numruns):

                # get design matrix with HRF
                # number of time points

                convdes = make_design(design[p], tr, ntimes[p], currenthrf)

                # project the polynomials out
                convdes = np.dot(combinedmatrix[p], convdes)
                # time x conditions

                convdesign.append(convdes)

            # stack design across runs
            stackdesign = np.vstack(convdesign)

            # estimate the amplitudes (output: conditions x voxels)
            currentbeta = mtimes_stack(olsmatrix(stackdesign), data)

            # calculate R^2
            modelfit = [np.dot(convdesign[p], currentbeta).astype(np.float32)
                        for p in range(numruns)]

            R2 = calc_cod_stack(data, modelfit)

            # figure out indices of good voxels
            if hrffitmask == 1:
                temp = R2
            else:  # if people provided a mask for hrf fitting
                temp = np.zeros((R2.shape))
                temp[np.invert(hrffitmask.ravel())] = -np.inf
                # shove -Inf in where invalid

            temp[np.isnan(temp)] = -np.inf
            ii = np.argsort(temp)
            nii = len(ii)
            iichosen = ii[np.max((1, nii - numforhrf)):nii]
            iichosen = np.setdiff1d(
                iichosen, iichosen[temp[iichosen] == -np.inf]
            ).tolist()
            hrffitvoxels = iichosen

        # fix the amplitudes, estimate the HRF
        else:

            nhrfvox = len(hrffitvoxels)

            # prepare design matrix
            convdesign = []
            for p in range(numruns):

                X = make_design(design[p], tr, ntimes[p])

                # expand design matrix using delta functions
                numcond = X.shape[1]
                # time x L*conditions
                stimmat = construct_stim_matrices(
                    X.T, prenumlag=0, postnumlag=postnumlag
                ).reshape(-1, numcond, order='F').astype(np.float32)

                # calc
                # weight and sum based on the current amplitude estimates.
                # only include the good voxels.
                # return shape time*L x voxels
                convdes = np.dot(
                    stimmat, currentbeta[:, hrffitvoxels]).astype(np.float32)

                # remove polynomials and extra regressors
                # time x L*voxels
                convdes = convdes.reshape(
                    (ntimes[p], -1), order='F')
                # time x L*voxels
                convdes = np.array(np.dot(combinedmatrix[p], convdes))
                # time x voxels x L
                convdes = convdes.reshape((ntimes[p], numinhrf, -1), order='F')
                convdesign.append(
                    np.transpose(convdes, (0, 2, 1))
                )

            # estimate the HRF
            previoushrf = currenthrf
            datasubset = np.array(np.vstack(
                [data[x][:, hrffitvoxels] for x in range(numruns)]
            ))

            stackdesign = np.vstack(convdesign)
            ntime = stackdesign.shape[0]

            stackdesign = stackdesign.reshape(
                (ntime * nhrfvox, numinhrf), order='F')
            stackdesign = olsmatrix(stackdesign)
            currenthrf = np.asarray(stackdesign.dot(
                datasubset.reshape((-1), order='F')))[0]

            # if HRF is all zeros (this can happen when the data are all zeros)
            # get out prematurely
            if np.all(currenthrf == 0):
                print('current hrf went all to 0 after {} attempts\n'.format(cnt))
                break

            # check for convergence
            # how much variance of the previous estimate does
            # the current one explain?
            hrfR2 = calc_cod(previoushrf, currenthrf, wantmeansub=0)

            if (hrfR2 >= minR2 and cnt > 2):
                break

        cnt += 1

    # sanity check
    # we want to see that we're not converging in a weird place
    # so we compute the coefficient of determination between the
    # current estimate and the seed hrf
    hrfR2 = calc_cod(hrfknobs, previoushrf, wantmeansub=0)

    # sanity check to make sure that we are not doing worse.
    if hrfR2 < hrfthresh:
        print(
            "Global HRF estimate is far from the initial seed,"
            "probably indicating low SNR.  We are just going to"
            "use the initial seed as the HRF estimate.\n"
        )
        # prepare design matrix
        convdesign = []
        whitedesign = []
        for p in range(numruns):
            # get design matrix with HRF
            # number of time points
            convdes = make_design(design[p], tr, ntimes[p], hrfknobs)

            # project the polynomials out
            whitedesign.append(np.dot(combinedmatrix[p], convdes))
            # time x conditions

            convdesign.append(convdes)
        f = dict()
        f["hrf"] = hrfknobs
        f["hrffitvoxels"] = None
        f["convdesign"] = convdesign
        f["whitedesign"] = whitedesign
        f["seedhrf"] = hrfknobs
        return f

    # normalize results
    mx = np.max(previoushrf)
    previoushrf = previoushrf / mx
    currentbeta = currentbeta * mx

    # prepare design matrix
    whitedesign = []
    convdesign = []
    for p in range(numruns):
        # get design matrix with HRF
        # number of time points
        convdes = make_design(design[p], tr, ntimes[p], previoushrf)

        # project the polynomials out
        whitedesign.append(np.dot(combinedmatrix[p], convdes))
        # time x conditions

        convdesign.append(convdes)

    # return
    f = dict()
    f["hrf"] = previoushrf
    f["hrffitvoxels"] = hrffitvoxels
    f["convdesign"] = convdesign
    f["whitedesign"] = whitedesign
    f["seedhrf"] = hrfknobs
    f["hrffitmask"] = hrffitmask
    return f
