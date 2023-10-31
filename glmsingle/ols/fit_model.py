
import numpy as np
from glmsingle.cod.calc_cod import calc_cod, calc_cod_stack
from glmsingle.design.construct_stim_matrices import construct_stim_matrices
from glmsingle.design.make_design_matrix import make_design
from glmsingle.design.convolve_design import convolve_design
from glmsingle.ols.mtimes_stack import mtimes_stack
from glmsingle.ols.olsmatrix2 import olsmatrix2
from fracridge import fracridge


def fit_model(des, data2, tr, hrfmodel, hrfknobs,
              opt, combinedmatrix, cache=None):
    """[summary]
    f, cache = fit_model(des, data2, tr, hrfmodel, hrfknobs,
              opt, combinedmatrix, cache)

    if hrfmodel is 'fir', then <f> will be voxels x conditions x time
    (flattened format)
    if hrfmodel is 'assume' or 'optimize', then <f> will be {A B}
    where A is time x 1 and B is voxels x conditions (flattened format).
    <hrffitvoxels> is [] unless hrfmodel is 'optimize', in which case it
    will be a column vector of voxel indices.

    note: cache.rawdesign will exist for 'fir' and 'assume' but not
        'optimize'.

    Args:
        des ([list]): [description]
        data2 ([list]): [description]
        tr ([int]): [description]
        hrfmodel ([str]): [description]
        hrfknobs ([array]): [description]
        opt ([dict]): [description]
        combinedmatrix ([array]): [description]
        cache ([dict]): [description]

    Returns:
        [tuple]: [f, cache]
    """

    # internal constants
    minR2 = 99
    # in 'optimize' mode, if R^2 between previous HRF and new HRF
    # is above this threshold (and we have at least gone through
    # one complete round of fitting (just so that we can change
    # a little from the initial seed)), then we stop fitting.

    # init
    hrffitvoxels = []

    # make sure data is np.float32
    for p in range(len(data2)):
        data2[p] = data2[p].astype(np.float32, copy=False)

    n_runs = len(des)

    if cache is None:
        cache = {}
        cache['design'] = [None for x in range(n_runs)]
        cache['rawdesign'] = [None for x in range(n_runs)]

    if hrfmodel == 'fir':
        # since 'fir', we can assume design is not the onset case, but check it
        des = [dest.astype(int) for dest in des]
        # np.testing.assert_equal(des[0].dtype is np.dtype(np.int64), True)

        # calc
        numconditions = des[0].shape[1]

        # prepare design matrix
        desw = []
        for p in range(len(des)):

            # expand original design matrix using delta basis functions.
            # the length of each timecourse is L.
            desw.append(construct_stim_matrices(
                des[p].T,
                0,
                hrfknobs,
                0).astype(np.float32))
            # time x L*conditions

            # save a record of the raw design matrix
            cache['rawdesign'][p] = desw[p]

            # remove polynomials and extra regressors
            desw[p] = combinedmatrix[p].astype(np.float32) @ desw[p]
            # time x L*conditions

            # save a record of the projected-out design matrix
            cache['design'][p] = desw[p]

        # fit model
        if opt['wantfracridge']:
            f = fracridge(
                np.concatenate(desw),
                np.concatenate(data2),
                opt['frac'],
            )[0]

        else:
            f = mtimes_stack(
                olsmatrix2(np.concatenate(desw)),
                data2)  # L*conditions x voxels

        # voxels x conditions x L
        f = np.transpose(np.reshape(f, [hrfknobs+1, numconditions, -1]), [2, 1, 0])

        fout = {}
        fout['betas'] = f
        fout['hrffitvoxels'] = hrffitvoxels

    elif hrfmodel == 'assume':

        # prepare design matrix
        desopts = {
            'n_times': data2[0].shape[0],
            'tr': tr
        }
        desw = []
        for p in range(n_runs):

            # convolve original design matrix with HRF
            # number of time points
            desw.append(
                convolve_design(
                    des[p],
                    hrfknobs,
                    desopts).astype(np.float32))

            # save a record of the raw design matrix
            cache['rawdesign'][p] = desw[p]

            # remove polynomials and extra regressors
            desw[p] = combinedmatrix[p].astype(np.float32) @ desw[p]
            # time x conditions

            # save a record of the projected-out design matrix
            cache['design'][p] = desw[p]

        # fit model
        if opt['wantfracridge']:
            f = fracridge(
                    np.concatenate(desw),
                    np.concatenate(data2),
                    opt['frac'])[0]
            # conditions x voxels
        else:
            f = mtimes_stack(
                olsmatrix2(np.concatenate(desw)),
                data2)  # conditions x voxels

        fout = {}
        fout['hrfknobs'] = hrfknobs
        fout['betas'] = f.T.astype(np.float32)
        fout['hrffitvoxels'] = hrffitvoxels

    elif hrfmodel == 'optimize':

        # since 'optimize', we can assume design is not the onset case,
        # but check it
        np.testing.assert_true(type(des[0]), list)

        # calc
        numinhrf = len(hrfknobs)
        numconds = des[0].shape[1]

        numruns = len(des)
        numconds = des[0].shape[1]
        postnumlag = numinhrf - 1

        if 'design2pre' not in cache:

            # precompute for speed
            design2pre = []
            for p in range(len(data2)):

                # expand design matrix using delta functions
                ntime = des[p].shape[0]  # number of time points
                design2pre.append(
                    construct_stim_matrices(
                        des[p].T,
                        prenumlag=0,
                        postnumlag=postnumlag
                    ).reshape(
                        -1,
                        numconds,
                        order='F').astype(np.float32))
                # time*L x conditions

            # record it
            cache['design2pre'] = design2pre
        else:
            if 'design2pre' in cache:
                design2pre = cache['design2pre']

        # collect ntimes per run
        ntimes = [data2[p].shape[0] for p in range(numruns)]

        # loop until convergence
        currenthrf = hrfknobs  # initialize
        cnt = 1
        while True:

            # fix the HRF, estimate the amplitudes
            if cnt % 2 == 1:

                # prepare design matrix
                design2 = []
                for p in range(numruns):

                    # get design matrix with HRF
                    # number of time points
                    design2.append(
                        make_design(
                            des[p],
                            tr,
                            ntimes[p],
                            currenthrf
                            )
                        )

                    # project the polynomials out
                    design2[p] = combinedmatrix[p] @ design2[p]
                    # time x conditions

                # estimate the amplitudes (output: conditions x voxels)
                currentbeta = mtimes_stack(
                    olsmatrix2(np.vstack(design2)),
                    data2
                    )

                # calculate R^2
                modelfit = [(design2[p] @ currentbeta).astype(
                    np.float32) for p in range(numruns)]

                R2 = calc_cod_stack(data2, modelfit)

                # figure out indices of good voxels
                if opt['hrffitmask'] == 1:
                    temp = R2
                else:  # if people provided a mask for hrf fitting
                    temp = np.zeros((R2.shape))
                    temp[np.invert(opt['hrffitmask'].ravel())] = -np.inf
                    # shove -Inf in where invalid

                temp[np.isnan(temp)] = -np.inf
                ii = np.argsort(temp)
                nii = len(ii)
                iichosen = ii[np.max((1, nii - opt['numforhrf'])):nii]
                iichosen = np.setdiff1d(
                    iichosen, iichosen[temp[iichosen] == -np.inf]
                ).tolist()
                hrffitvoxels = iichosen

            # fix the amplitudes, estimate the HRF
            else:

                nhrfvox = len(hrffitvoxels)

                # prepare design matrix
                design2 = []
                for p in range(numruns):

                    # calc
                    # weight and sum based on the current amplitude estimates.
                    # only include the good voxels.
                    # return shape time*L x voxels

                    design2.append((
                        design2pre[p] @ currentbeta[:, hrffitvoxels]
                        ).astype(np.float32))

                    # remove polynomials and extra regressors
                    # time x L*voxels
                    design2[p] = design2[p].reshape(
                        (ntimes[p], -1), order='F')

                    design2[p] = combinedmatrix[p] @ design2[p]
                    design2[p] = design2[p].reshape(
                        (ntimes[p], numinhrf, -1),
                        order='F')
                    design2[p] = np.transpose(design2[p], (0, 2, 1))

                # estimate the HRF
                previoushrf = currenthrf
                datasubset = np.array(np.vstack(
                    [data2[x][:, hrffitvoxels] for x in range(numruns)]
                ))

                stackdesign = np.vstack(design2)
                ntime = stackdesign.shape[0]

                stackdesign = stackdesign.reshape(
                    (ntime * nhrfvox, numinhrf), order='F')
                stackdesign = olsmatrix2(stackdesign)
                currenthrf = np.asarray(stackdesign.dot(
                    datasubset.reshape((-1), order='F')))[0]

                # if HRF is all zeros (this can happen when the data
                # are all zeros)
                # get out prematurely
                if np.all(currenthrf == 0):
                    print(
                        f'current hrf went all to 0 after {cnt} attempts\n')
                    break

                # check for convergence
                hrfR2 = calc_cod(previoushrf, currenthrf, wantmeansub=0)
                if hrfR2 >= minR2 and cnt > 2:
                    break

            cnt += 1

            # sanity check
            # we want to see that we're not converging in a weird place
            # so we compute the coefficient of determination between the
            # current estimate and the seed hrf
            hrfR2 = calc_cod(hrfknobs, previoushrf, wantmeansub=0)

            # sanity check to make sure that we are not doing worse.
            if hrfR2 < opt['hrfthresh']:
                print(
                    "Global HRF estimate is far from the initial seed,"
                    "probably indicating low SNR.  We are just going to"
                    "use the initial seed as the HRF estimate.\n"
                )
                fout = fit_model(
                    des,
                    data2,
                    tr,
                    'assume',
                    hrfknobs,
                    opt,
                    combinedmatrix,
                    )[0]

                return fout, cache

            # normalize results
            mx = np.max(previoushrf)
            previoushrf = previoushrf / mx
            currentbeta = currentbeta * mx

            # return
            fout = dict()
            fout["hrf"] = previoushrf
            fout['betas'] = currentbeta.T
            fout['hrffitvoxels'] = hrffitvoxels

    return fout, cache
