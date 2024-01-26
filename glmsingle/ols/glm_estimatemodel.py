import time
import numpy as np
from itertools import compress
from tqdm import tqdm
from glmsingle.check_inputs import check_inputs
from glmsingle.cod.calc_cod import calc_cod, calc_cod_stack
from glmsingle.hrf.gethrf import getcanonicalhrf
from glmsingle.hrf.normalisemax import normalisemax
from glmsingle.ols.fit_model import fit_model
from glmsingle.ols.glm_predictresponses import glm_predictresponses
from glmsingle.ols.make_poly_matrix import (make_polynomial_matrix,
                                             make_projection_matrix)
from glmsingle.utils.alt_round import alt_round
from glmsingle.utils.squish import squish


def glm_estimatemodel(design, data, stimdur, tr, hrfmodel, hrfknobs,
                      resampling, opt=None, cache=None, mode=0):

    """
    glm_estimatemodel

    [summary]

    Based on the experimental design (<design>, <stimdur>, <tr>) and the
    model specification (<hrfmodel>, <hrfknobs>), fit a GLM model to the
    data (<data>) using a certain resampling scheme (<resampling>).

    Arguments:
    __________

    <design>:

    is the experimental design.

    There are three possible cases:
    1. A where A is a matrix with dimensions time x conditions.
    Each column should be zeros except for ones indicating condition
    onsets.
    (Fractional values in the design matrix are also allowed.)

    2. [A1, A2, ...An] where each of the A's are like the previous case.
    The different A's correspond to different runs, and different
    runs can have different numbers of time points.

    3. [[[C1_1], [C2_1], [C3_1] ...],[[C1_2],[C2_2],[C3_2], ...] ...]
    where Ca_b is a vector of onset times (in seconds) for condition a
    in run b.
    Time starts at 0 and is coincident with the acquisition of the
    first volume. This case is compatible only with <hrfmodel> set to
    'assume'.

    <data>:

    is the time-series data with dimensions X x Y x Z x time or a
    list vector of elements that are each X x Y x Z x time.  XYZ can be
    collapsed such that the data are given as a 2D matrix (XYZ x time);
    however, if you do this, then several of the figures that are written
    out by this function will not be useful to look at. The dimensions of
    <data> should mirror that of <design>.
    (For example, <design> and <data> should have the same number
    of runs, the same number of time points, etc.)
    <data> should not contain any NaNs. We automatically convert <data>
    to single format (if necessary).

    <stimdur>:

    is the duration of a trial in seconds

    <tr>:

    is the sampling rate in seconds

    <hrfmodel>:

    indicates the type of model to use for the HRF:

    'fir': a finite impulse response model (a separate timecourse
    is estimated for every voxel and every condition)

    'assume': that the HRF is provided (see <hrfknobs>)

    'optimize': indicates that we should estimate a global HRF from
    the data

    <hrfknobs> (optional):

    if <hrfmodel> is 'fir', then <hrfknobs> should be the number of
    time points in the future to model (N >= 0).  For example, if N
    is 10, then timecourses will consist of 11 points, with the first
    point coinciding with condition onset.

    if <hrfmodel> is 'assume', then <hrfknobs> should be time x 1 with
    the HRF to assume.

    if <hrfmodel> is 'optimize', then <hrfknobs> should be time x 1 with
    the initial seed for the HRF.  The length of this vector indicates
    the number of time points that we will attempt to estimate in the
    HRF.

    Note on normalization:  In the case that <hrfmodel> is 'assume' or
    'optimize', we automatically divide <hrfknobs> by the maximum value
    so that the peak is equal to 1.  And if <hrfmodel> is 'optimize',
    then after fitting the HRF, we again normalize the HRF to peak at 1
    (and adjust amplitudes accordingly).  Default in the case of 'fir' is
    20.  Default in the case of 'assume' and 'optimize' is to use a
    canonical HRF that is calculated based on <stimdur> and <tr>.

    <resampling>

    specifies the resampling scheme:

    0 means fit fully (don't bootstrap or cross-validate)

    A means bootstrap A times (A >= 1)

    -1 means perform leave-one-run-out cross-validation (in this case,
    there must be at least two runs)

    <opt> (optional) is a struct with the following fields:

    <extra_regressors> (optional)

    is time x regressors or a cell vector
    of elements that are each time x regressors.  The dimensions of
    <extra_regressors> should mirror that of <design> (i.e. same number of
    runs, same number of time points).  The number of extra regressors
    does not have to be the same across runs, and each run can have zero
    or more extra regressors.  If None or not supplied, we do
    not use extra regressors in the model.

    <maxpolydeg> (optional):

    is a non-negative integer with the maximum
    polynomial degree to use for polynomial nuisance functions, which
    are used to capture low-frequency noise fluctuations in each run.
    Can be a vector with length equal to the number of runs (this
    allows you to specify different degrees for different runs).
    Default is to use round(L/2) for each run where L is the
    duration in minutes of a given run.

    <seed> (optional):

    is the random number seed to use (this affects
    the selection of bootstrap samples). Default: sum(100*clock).

    <bootgroups> (optional):

    is a vector of positive integers indicating
    the grouping of runs to use when bootstrapping.  For example,
    a grouping of [1 1 1 2 2 2] means that of the six samples that are
    drawn, three samples will be drawn (with replacement) from the first
    three runs and three samples will be drawn (with replacement) from
    the second three runs.  This functionality is useful in situations
    where different runs involve different conditions.
    Default: ones(1, D) where D is the number of runs.

    <numforhrf> (optional):

    is a positive integer indicating the number
    of voxels (with the best R^2 values) to consider in fitting the
    global HRF.  This input matters only when <hrfmodel> is 'optimize'.
    Default: 50.  (If there are fewer than that number of voxels
    available, we just use the voxels that are available.)

    <hrffitmask> (optional):

    is X x Y x Z with 1s indicating all possible
    voxels to consider for fitting the global HRF.  This input matters
    only when <hrfmodel> is 'optimize'.  Special case is 1 which means
    all voxels can be potentially chosen.  Default: 1.

    <wantpercentbold> (optional):

    is whether to convert the amplitude
    estimates in 'models', 'modelmd', 'modelse', and 'residstd*' to
    percent BOLD change.
    This is done as the very last step, and is accomplished by dividing
    by the absolute value of 'meanvol' and multiplying by 100.  (The
    absolute value prevents negative values in 'meanvol' from flipping
    the sign.)
    Default: 1.

    <hrfthresh> (optional):

    is an R^2 threshold.  If the R^2 between the
    estimated HRF and the initial HRF is less than <hrfthresh>, we
    decide to just use the initial HRF.  Set <hrfthresh> to -Inf if
    you never want to reject the estimated HRF.
    Default: 50.

    <suppressoutput> (optional):

    is whether to suppress fprintf statements.
    Default: 0.

    <lambda> (optional):

    is the lambda constant to use for ridge regression.
    This takes effect only for the 'fir' and 'assume' cases.  Default: 0.

    <frac> (optional):

    is the frac to use for fractional ridge regression.
    This takes effect only for the 'fir' and 'assume' cases.
    Default: [].
    If [], we use <lambda>. If not [], we use <frac>.

    <cache> (optional):

    is used for speeding up execution.

    If you are calling this function with identical inputs except
    potentially for different <data>, then if you can take the <cache>
    returned by the first call and re-use it for subsequent calls.

    <mode=0> (optional):

    1 means that only the 'R2' output is desired (to save compute time)
    2 means that hrfmodel is 'optimize', resampling is 0, we only care
        about the hrf and hrffitvoxels outputs (to save time and memory)
    Default: 0.



    Returns:
    __________

    <results>:

    as a dict containing the following keys:

    <betas>:

    contains the full set of model estimates (e.g. all bootstrap results)

    <betasmd>:

    contains the final model estimate (median of the estimates in <betas>)

    <modelse>:

    contains the standard error of the final model estimate (half
    of the 68% range of the estimates in <models>).  Note that <modelse>
    will be computed in all resampling schemes (full-fit, bootstrapping,
    and cross-validation) but can be interpreted as an estimate of
    standard error only in the bootstrapping scheme.

    <R2>:

    is XYZ with model accuracy expressed in terms of R^2 (percentage).
    In the full-fit and bootstrap cases, <R2> is an R^2 value indicating
    how well the final model estimate (<modelmd>) fits the data.
    In the cross-validation case, <R2> is an R^2 value indicating how
    well the cross-validated predictions of the model match the data.
    (The predictions and the data are each aggregated across runs before
    the computation of R^2.)

    <R2run>:

    is [XYZ] x runs with R^2 values calculated on a per-run basis.

    <signal>:

    is XYZ with the maximum absolute amplitude in <modelmd>
    (this is computed over all conditions and time points in the case of
    'fir' and over all conditions in the case of 'assume' and 'optimize')

    <noise>:

    is XYZ with the average amplitude error in <modelse>.

    <SNR>:

    is XYZ with <signal> divided by <noise>.

    <hrffitvoxels>:

    is XYZ with 1s indicating the voxels used for
    fitting the global HRF. This input is returned as [] if <hrfmodel>
    is not 'optimize'.
    In the bootstrap and cross-validation cases, <hrffitvoxels> indicates
    the voxels corresponding to the last iteration.

    <meanvol>:

    is XYZ with the mean of all volumes

    <inputs>:

    is a dict containing all inputs used in the call to this
    function, excluding <data>.  We additionally include a field called
    'datasize' which contains the size of each element of <data>.

    Additional details on the format of <models>, <modelmd>, and <modelse>:
    ______________________________________________________________________


    If <hrfmodel> is 'fir', then model estimates consist of timecourses:
    <betas> is XYZ x conditions x time x resamples
    <betasmd> is XYZ x conditions x time
    <betasse> is XYZ x conditions x time

    If <hrfmodel> is 'assume' or 'optimize', then model estimates consist
    of HRF estimates and amplitude estimates:
    <hrfknobs> is time x resamples (HRF estimates)
    <hrfknobsmd> is time x 1
    <hrfknobsse> is time x 1

    Notes on model accuracy (R^2):

    We quantify the accuracy of the GLM model as the amount of variance in
    the time-series data that is explained by the deterministic portion of
    the model, that is, the hemodynamic responses evoked by the various
    experimental conditions.

    Note that this does not include the nuisance components of the model,
    that is, the polynomial regressors and any extra regressors provided by
    the user (see opt.extraregressors).

    The metric that we use for accuracy is R^2.  Specifically:
    R^2 = 100 * (1-sum((data-model)^2)/sum(data^2))

    Before computing R^2 between the model and the data, we project out
    polynomial regressors from both the model and the data. The purpose of
    this is to reduce the influence of low-frequency fluctuations (which
    can be quite large in fMRI data) on the model accuracy metric.

    Notes on bootstrapping:

    Bootstrap samples are drawn from entire runs. (Bootstrapping individual
    data points would be inappropriate due to temporal correlations in fMRI
    noise.)

    For example, if there are 10 runs, each bootstrap sample consists of 10
    runs drawn with replacement from the 10 runs.

    In cases of unbalanced designs, it is possible that a bootstrap sample
    contains no occurrences of a given condition; in this case, a warning is
    reported and the beta weight estimated for that condition is set to zero.

    Notes on the estimation of a global HRF:

    When <hrfmodel> is 'optimize', we estimate a global HRF from the data.
    This is achieved using an iterative fitting strategy:  First, the HRF is
    fixed to the initial seed provided by <hrfknobs>, and we estimate the
    amplitudes using OLS.  Then, the amplitudes are fixed (to the estimates
    obtained in the previous step), and we estimate the HRF using OLS.  Next,
    the HRF is fixed
    (to the estimate obtained in the previous step), and we re-estimate the
    amplitudes using OLS.  This process is repeated until convergence.

    The reason for the iterative fitting strategy is that the entire model
    cannot be estimated at once using linear fitting techniques (and nonlinear
    techniques would be too costly).
    - At the HRF-estimation steps of the fitting process, the entire dataset
    can in theory be fit.  However, this is undesirable for two reasons.  One,
    fitting the entire dataset may have exorbitant memory requirements.
    Two, assuming that most voxels are unrelated to the experimental paradigm
    (as is typically the case in an fMRI experiment), fitting the entire
    dataset will result in a poor-quality (noisy) HRF.  To resolve these
    issues, we use a strategy in which we determine the best voxels in terms
    of R^2 at a given amplitude-estimation step and fit only these voxels in
    the subsequent HRF-estimation step.  The number of voxels that are chosen
    is controlled by opt.numforhrf, and the pool of chosen voxels is updated
    at each amplitude-estimation step.
    - In some cases, the fitted global HRF may diverge wildly from the initial
    seed.  This may indicate extremely low SNR and/or a problem with the
    coding of the experimental design and/or a poor initial seed for the HRF.
    If the R^2 between the initial seed and the fitted global HRF is less than
    opt.hrfthresh, we issue a warning and simply use the initial seed as the
    HRF (instead of relying on the fitted global HRF).  These cases should be
    inspected and troubleshooted on a case-by-case basis.
    (In GLMdenoisedata.m, a figure named 'HRF.png' is created --- if the
    initial and estimated HRF are exactly overlapping on the figure, this
    indicates that the exception case occured.)

    Additional information:
    - In some circumstances (e.g. using a FIR model with insufficient data),
    the design matrix may be singular and there is no unique solution.  Our
    strategy for these cases is as follows: If MATLAB issues a warning during
    the inversion of the autocorrelation matrix (i.e. X'*X), then program
    execution halts.

    History:
    [MATLAB]
    - 2020/05/09: add opt.frac
    - 2019/03/22: return design in cache, add opt.lambda
    - 2014/07/31: return rawdesign in cache; change cubic to pchip to avoid
                    warnings
    - 2013/12/11: now, after we are done using opt.seed, we reset the random
                    number seed to something random
                    (specifically, sum(100*clock)).
    - 2013/11/18: add cache input/output; update documentation; new default
                    for maxpolydeg (it used to always be 2); add opt.hrfthresh;
                    add opt.suppressoutput; some speed-ups
    - 2013/05/12: allow <design> to specify onset times
    - 2013/05/12: update to indicate fractional values in design matrix are
                    allowed.
    - 2013/05/12 - regressors that are all zero now receive a 0 weight
                    (instead of crashing)
    - 2013/05/12 - fixed a bug regarding how the extraregressors were being
                    handled. previously, the extraregressors and the polynomial
                    regressors were being regressed out sequentially, which is
                    improper.  now, the two regressors are being fit
                    simultaneously, which is the correct way to do it.
    - 2012/12/06: automatically convert data to single format
    - 2012/12/03: *** Tag: Version 1.02 ***. Use faster OLS computation (less
                    error-checking; program execution will halt if design
                    matrix is singular);
                    implement various speed-ups; minor bug fixes.
    - 2012/11/24:
        - INPUTS: add stimdur and tr; hrfknobs is optional now; add
                opt.hrffitmask; add opt.wantpercentbold
        - OUTPUTS: add signal,noise,SNR; add hrffitvoxels; add meanvol;
                    add inputs
        - add a speed-up (design2pre)
    - 2012/11/02 - Initial version.
    - 2012/10/30 - Automatic division of HRF. Ensure one complete round of
                    fitting in optimize case.
                    Add sanity check on HRF.

    """

    # DEAL WITH INPUTS, ETC.

    if type(design) is not list:
        design = [design]

    if type(data) is not list:
        data =[data]

    for datai in range(len(data)):
        data[datai] = data[datai].astype(np.float32)

    # data, design, xyz = check_inputs(data, design)
    datashape = data[0].shape
    is3d = False
    if len(datashape) > 2:
        is3d=True

    if is3d:
        dimdata = 2
        xyzsize = list(data[0].shape[:3])
    else:
        dimdata = 0
        xyzsize = [data[0].shape[0]]

    numruns = len(design)

    # calc
    # deal with defaults
    if opt is None:
        opt = {}

    if 'extra_regressors' not in opt or opt['extra_regressors'][0] is False or not np.any(opt['extra_regressors'][0]):
        opt['extra_regressors'] = [None for x in range(numruns)]

    if 'maxpolydeg' not in opt:
        opt['maxpolydeg'] = [
            np.arange(
                alt_round(
                    ((data[r].shape[-1]*tr)/60)/2) + 1
                ) for r in range(numruns)]

    if 'seed' not in opt:
        opt['seed'] = time.time()

    if 'wantfracridge' not in opt:
        opt['wantfracridge'] = False

    if 'bootgroups' not in opt:
        opt['bootgroups'] = np.zeros((1, numruns))

    if 'numforhrf' not in opt:
        opt['numforhrf'] = 50

    if 'hrffitmask' not in opt:
        opt['hrffitmask'] = 1

    if 'wantpercentbold' not in opt:
        opt['wantpercentbold'] = 1

    if 'hrfthresh' not in opt:
        opt['hrfthresh'] = 50

    if 'suppressoutput' not in opt:
        opt['suppressoutput'] = 0

    if 'lambda' not in opt:
        opt['lambda'] = 0

    if 'frac' not in opt:
        opt['frac'] = None

    if type(opt['maxpolydeg']) is int:
        opt['maxpolydeg'] = [opt['maxpolydeg'] for x in range(numruns)]

    # input
    if hrfknobs is None:
        if hrfmodel == 'fir':
            hrfknobs = 20
        else:
            hrfknobs = normalisemax(getcanonicalhrf(stimdur, tr))

    if resampling == 0:
        resamplecase = 'full'
    elif resampling >= 1:
        resamplecase = 'boot'
    else:
        resamplecase = 'xval'

    if len(opt['maxpolydeg']) == 1:
        opt['maxpolydeg'] = np.tile(
            opt['maxpolydeg'], numruns).tolist()

    if hrfmodel == 'assume' or hrfmodel == 'optimize':
        hrfknobs = normalisemax(hrfknobs, dim='global')

    # CALCULATE MEAN VOLUME

    volcnt = [x.shape[-1] for x in data]
    meanvol = np.concatenate(data, axis=-1).mean(axis=-1)

    # DEAL WITH NUISANCE COMPONENTS

    # construct projection matrices for the nuisance components
    polymatrix = []
    combinedmatrix = []
    for p in range(numruns):

        # this projects out polynomials
        pmatrix = make_polynomial_matrix(
            volcnt[p],
            opt['maxpolydeg'][p])
        polymatrix.append(
            make_projection_matrix(pmatrix))

        extra_regressors = opt['extra_regressors'][p]

        # this projects out both of them (second condition checks for empty matrix)
        if extra_regressors is not None and extra_regressors.any():
            combinedmatrix.append(
                make_projection_matrix(
                    np.c_[pmatrix, extra_regressors]
                )
            )
        else:
            combinedmatrix.append(
                make_projection_matrix(pmatrix))

    # project out nuisance components from the data.
    # after this step, data will have polynomials removed,
    # and data2 will have both polynomials and extra regressors removed.
    data2 = []  # NOTE: dataw and data2 are big be careful of MEMORY usage.
    dataw = []
    for run_i in range(numruns):
        dataw.append(np.transpose(squish(data[run_i], dimdata+1)))
        data2.append(
            combinedmatrix[run_i].astype(np.float32) @ dataw[run_i]
        )
        dataw[run_i] = polymatrix[run_i].astype(np.float32) @ dataw[run_i]

    # note that data and data2 are in flattened format (time x voxels)!!

    # FIT MODELS

    # NOTE: cache['rawdesign'] will exist for 'fir' and 'assume' but not
    # 'optimize' and for 'full' and 'boot' but not 'xval'.

    if resamplecase == 'full':

        # this is the full-fit case

        # fit the model to the entire dataset.  we obtain just one analysis
        # result.
        if opt['suppressoutput'] == 0:
            print('fitting model...')

        results, cache = fit_model(
            design,
            data2,
            tr,
            hrfmodel,
            hrfknobs,
            opt,
            combinedmatrix,
            cache)

        if opt['suppressoutput'] == 0:
            print('done.\n')

    elif resamplecase == 'boot':

        # this is the bootstrap case

        # set random seed
        np.random.seed(opt['seed'])

        # there are two reasons to call this line.  one is that in the
        # case of (bootstrap + optimize), we have to do a pre-call to
        # get some cache.
        # another is that we may need the cache.rawdesign output.
        # so, let's just call it.
        cache = fit_model(
            design,
            data2,
            tr,
            hrfmodel,
            hrfknobs,
            opt,
            combinedmatrix,
            cache)[1] # note we index cache here, and not fout

        # loop over bootstraps and collect up the analysis results.
        results = []
        if opt['suppressoutput'] == 0:
            print('bootstrapping model')

        for p in tqdm(range(resampling)):
            # figure out bootstrap sample
            ix = []
            boot_groups = np.unique(opt['bootgroups'])
            # loop over the boot groups
            for q in boot_groups:
                # find which run belongs to thew current boot_group
                indices_in_group_q = np.flatnonzero(opt['bootgroups'] == q)
                num = len(indices_in_group_q)
                # next we want to randomly sample with replacement the runs
                # belonging to this group
                bootstrapped_indices = np.random.choice(indices_in_group_q, num)
                # now that we have sampled this group, we extend ix
                # and move on to the other group if there is one.
                ix.extend(bootstrapped_indices)


            boot_design = [design[x] for x in ix]
            boot_data = [data2[x] for x in ix]
            boot_combinedmatrix = [combinedmatrix[x] for x in ix]

            # fit the model to the bootstrap sample
            if hrfmodel == 'optimize':
                boot_design2pre = [cache['design2pre'][x] for x in ix]
                cache2 = {'design2pre': boot_design2pre}
            else:
                cache2 = None

            results.append(
                fit_model(
                    boot_design,
                    boot_data,
                    tr,
                    hrfmodel,
                    hrfknobs,
                    opt,
                    boot_combinedmatrix,
                    cache2
                    )[0]
            ) # no need for cache here

        if opt['suppressoutput'] == 0:
            print('done.\n')

        # randomize the random seed
        np.random.seed(1)

    elif resamplecase == 'xval':

        # this is the cross-validation case
        # loop over cross-validation iterations.  in each iteration, we
        # record the analysis result and also record the time-series
        # predictions.
        modelfit = []
        results = []
        if opt['suppressoutput'] == 0:
            print('cross-validating model')

        for p in tqdm(range(numruns)):

            # figure out resampling scheme
            mask = np.arange(numruns) != p

            # fit the model
            results.append(
                fit_model(
                    list(compress(design, mask)),
                    list(compress(data2, mask)),
                    tr,
                    hrfmodel,
                    hrfknobs,
                    opt,
                    list(compress(combinedmatrix, mask))
                    )[0])  # NOTE: no cache

            # compute the prediction
            modelfit.append(
                glm_predictresponses(
                    results[p],
                    [design[p]],
                    tr,
                    data2[p].shape[0],
                    0)
                )  # 0 because results{p} is in flattened format

            # massage format. the data passed to fit_model and predict
            # responses were in XYZ flattened, we want to go back to
            # volume.
            xyzsizefull = xyzsize + [modelfit[p].shape[1]]
            modelfit[p] = np.reshape(
                modelfit[p], xyzsizefull)

        if opt['suppressoutput'] == 0:
            print('done.\n')

    # PREPARE MODEL ESTIMATES FOR OUTPUT

    # in this special case, we do not have to perform this section,
    # so let's skip it to save computational time.
    if resamplecase == 'xval' and mode == 1:
        results = []

    # otherwise, do it as usual
    else:

        if opt['suppressoutput'] == 0:
            print('preparing output...')

        if hrfmodel in ['fir']:
            # normal no bootstrap case.
            if len(results['betas'].shape) == 3:
                results['betasmd'] = results['betas']
                results['betasse'] = np.zeros(results['betas'].shape, dtype=results['betas'].dtype)
            else:
                # here, the FIR was resampled with bootstrap
                temp = np.zeros([3] + list(results['betas'].shape[:3]), dtype=results['betas'].dtype)
                # loop over the different FIR predictors
                for p in range(results['betas'].shape[2]):  # ugly to avoid memory usage
                    this_model = results['betas'][:,:,p,:]
                    temp[:,:, :, p] = np.percentile(this_model, [16, 50, 84],2)

                results['betasmd'] = temp[1, :, :, :]
                results['betasse'] = (temp[2, :, :, :] - temp[0, :, :, :])/2

            # massage format
            # the outputs of FIR will be in format XYZ x 1 x nhrfknobs

            # the betas could have boots in them at this point, catch this:
            if results['betas'].ndim==4: # bootstrap case
                sz = results['betas'].shape[1:]
                sz2 = results['betas'].shape[1:-1]
            else:
                sz = results['betas'].shape[1:]
                sz2 = results['betas'].shape[1:]

            results['betas'] = np.reshape(results['betas'],   xyzsize + list(sz))
            results['betasmd'] = np.reshape(results['betasmd'], xyzsize + list(sz2))
            results['betasse'] = np.reshape(results['betasse'], xyzsize + list(sz2))


        if hrfmodel in ['assume', 'optimize']:
            if type(results) is list:

                temp = [tt['hrfknobs'] for tt in results]
                betas = [tt['betas'] for tt in results]

                # TODO test this:
                results['betas'] = np.concatenate(
                    np.asarray(betas)[:, :, np.newaxis],
                    axis=1
                )
                # TODO test this
                results['hrfknobs'] = np.concatenate(
                    np.asarray(temp),
                    axis=1)

            # deal with {1}
            if results['hrfknobs'].ndim == 1:
                results['hrfknobsse'] = np.zeros(
                    results['hrfknobs'].shape, dtype=results['hrfknobs'].dtype)
            else:
                temp = np.percentile(results['hrfknobs'], [16, 50, 84], 1)
                results['hrfknobsmd'] = temp[1, :]
                results['hrfknobsse'] = (temp[2, :] - temp[0, :])/2

            # deal with {2}
            if results['betas'].ndim == 2 or results['betas'].ndim == 1:
                if results['betas'].ndim == 1:
                    results['betas'] = results['betas'].reshape(1, -1) # IC: not sure why this
                # XYZ by n_conditions
                results['betasmd'] = results['betas'].astype(np.float32)
                results['betasse'] = np.zeros(
                    results['betas'].shape, dtype=results['betas'].dtype)
            else:
                # XYZ by boot by n_conditions
                results['betasmd'] = np.median(
                    results['betas'], 1).astype(np.float32)
                temp = np.percentile(results['betas'], [16, 50, 84], 1)
                results['betasse'] = (temp[2, :, :] - temp[0, :, :])/2

            # massage format. the data passed to fit_model and predict
            # responses were in XYZ flattened, we want to go back to
            # volume.
            xyzsizefull = xyzsize + [results['betas'].shape[1]]

            results['betas'] = np.reshape(results['betas'], xyzsizefull)
            results['betasmd'] = np.reshape(results['betasmd'], xyzsizefull)
            results['betasse'] = np.reshape(results['betasse'], xyzsizefull)


        if opt['suppressoutput'] == 0:
            print('done.\n')

    # COMPUTE MODEL FITS (IF NECESSARY)

    # to save memory, perhaps construct modelfit in chunks??

    if mode not in [2]:
        if opt['suppressoutput'] == 0:
            print('computing model fits...')

        if resamplecase in ['full', 'boot']:

            # compute the time-series fit corresponding
            # to the final model estimate
            ntimes = [x.shape[0] for x in dataw]
            modelfit = glm_predictresponses(
                results,
                design,
                tr,
                ntimes,
                dimdata)
        if resamplecase == 'xval':
            pass
            # in the cross-validation case, we have already computed
            # the cross-validated
            # predictions of the model and stored them in the variable
            # 'modelfit'.

        if opt['suppressoutput'] == 0:
            print('done.\n')

    # COMPUTE R^2
    if mode not in [2, 3]:

        if opt['suppressoutput'] == 0:
            print('computing R^2...')

        # remove polynomials from the model fits (or predictions)
        modelfit = [polymatrix[x] @ squish(modelfit[x], dimdata+1).T for x in range(numruns)]

        # calculate overall R^2 [beware: MEMORY] XYZ x time
        results['R2'] = np.reshape(calc_cod_stack(modelfit, dataw, 0), xyzsize)

        # TODO: port this stuff
        # compute residual std
        # lowthresh = 0.1;  # 0.1 Hz is the upper limit [HARD-CODED]
        # the data already have polynomials removed.
        # above, we just removed polynomials from the model fits (or predictions).
        # so, we just need to subtract these two to get the residuals

        #results.residstd = reshape(sqrt(sum(catcell(1,cellfun(@(a,b) sum((a-b).^2,1),data,modelfit,'UniformOutput',0)),1) ./ (sum(volcnt)-1)),[xyzsize 1]);
        #results.residstdlowpass = reshape(sqrt(sum(catcell(1,cellfun(@(a,b) ...
        #sum( tsfilter((a-b)',constructbutterfilter1D(size(a,1),lowthresh*(size(a,1)*tr)))'.^2,1),data,modelfit,'UniformOutput',0)),1) ./ (sum(volcnt)-1)),[xyzsize 1]);

        # [XYZ by time] by n_runs
        results['R2run'] = [
            np.reshape(
                calc_cod(
                    mfit,
                    cdata,
                    dim=0,
                    wantgain=0,
                    wantmeansub=0),
                xyzsize) for mfit, cdata in zip(modelfit, dataw)]
        if opt['suppressoutput'] == 0:
            print('done.\n')

    # COMPUTE SNR

    if opt['suppressoutput'] == 0:
        print('computing SNR...')

    if not ((resamplecase == 'xval') and mode == 1) and mode != 2:

        if hrfmodel == 'fir':
            # results['betasmd'] shape xyz x n_cond x n_fir?
            results['signal'] = np.max(
                np.max(
                    np.abs(
                        results['betasmd']
                    ),
                    dimdata+1
                    ),
                dimdata+1
            )
            results['noise'] = np.mean(
                np.mean(
                    results['betasse'],
                    dimdata+1
                ),
                dimdata+1
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                results['SNR'] = results['signal'] / results['noise']

        if hrfmodel in ['assume', 'optimize']:
            # betasmd is a volume (most likely)
            # or a flattented volume
            results['signal'] = np.max(np.abs(results['betasmd']), axis=dimdata+1)
            results['noise'] = np.mean(results['betasse'], axis=dimdata+1)
            with np.errstate(divide="ignore", invalid="ignore"):
                results['SNR'] = results['signal'] / results['noise']

    if opt['suppressoutput'] == 0:
        print('done.\n')

    # PREPARE ADDITIONAL OUTPUTS

    # this is a special case
    if results['hrffitvoxels']:

        # TODO
        results['hrffitvoxels'] = np.tile(
            results['hrffitvoxels']
            (numvoxels, 1)
            ).astype(np.float32)

    results['meanvol'] = meanvol

    # return all the inputs (except for the data) in the output.
    # also, include a new field 'datasize'.
    results['inputs'] = {
        'design': design,
        'datasize': [x.shape for x in data],
        'stimdur': stimdur,
        'tr': tr,
        'hrfmodel': hrfmodel,
        'hrfknobs': hrfknobs,
        'resampling': resampling,
        'params': opt
    }

    # CONVERT TO % BOLD CHANGE

    if opt['wantpercentbold'] == 1:

        if not(resamplecase == 'xval' and mode == 1):

            # XYZ
            con = 1/np.abs(results['meanvol']) * 100

            numnewdims = results['betas'].ndim - con.ndim
            slicing = [slice(None)] * con.ndim + [np.newaxis] * numnewdims
            # Broadcast and multiply
            results['betas'] = results['betas'] * con[tuple(slicing)]

            numnewdims = results['betasmd'].ndim - con.ndim
            slicing = [slice(None)] * con.ndim + [np.newaxis] * numnewdims
            # Broadcast and multiply
            results['betasmd'] = results['betasmd'] * con[tuple(slicing)]
            numnewdims = results['betasse'].ndim - con.ndim
            slicing = [slice(None)] * con.ndim + [np.newaxis] * numnewdims

            results['betasse'] = results['betasse'] * con[tuple(slicing)]

    return results, cache
