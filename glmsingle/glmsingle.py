from __future__ import absolute_import, division, print_function
import os
import warnings
import numpy as np
import h5py
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
from glmsingle.check_inputs import check_inputs
from glmsingle.defaults import default_params
from glmsingle.gmm.findtailthreshold import findtailthreshold
from glmsingle.hrf.gethrf import getcanonicalhrf, getcanonicalhrflibrary
from glmsingle.hrf.normalisemax import normalisemax
from glmsingle.ols.glm_estimatemodel import glm_estimatemodel
from glmsingle.ols.make_poly_matrix import (make_polynomial_matrix,
                                            make_projection_matrix)
from glmsingle.ols.olsmatrix import olsmatrix
from glmsingle.utils.select_noise_regressors import select_noise_regressors
from glmsingle.ssq.calcbadness import calcbadness
from glmsingle.utils.chunking import chunking
from glmsingle.utils.make_image_stack import make_image_stack
from glmsingle.utils.alt_round import alt_round
from glmsingle.utils.squish import squish
from glmsingle.utils.cmapturbo import cmapturbo
from glmsingle.utils.cmapsign4 import cmapsign4
from glmsingle.utils.cmaplookup import cmaplookup
from glmsingle.utils.calcdmetric import calcdmetric


__all__ = ["GLM_single"]
dir0 = os.path.dirname(os.path.realpath(__file__))

warnings.simplefilter(action='ignore', category=RuntimeWarning)

class GLM_single():

    def __init__(self, params=None):
        """glm singletrial denoise constructor

        This function computes up to four model outputs (called type-A (ONOFF),
        type-B (FITHRF), type-C (FITHRF_GLMDENOISE), and type-D
        (FITHRF_GLMDENOISE_RR)),and either saves the model outputs to disk,
        or returns them in <results>, or both,depending on what the user
        specifies.

        There are a variety of cases that you can achieve. Here are some
        examples:

        - wantlibrary=1, wantglmdenoise=1, wantfracridge=1 [Default]
            A = simple ONOFF model
            B = single-trial estimates using a tailored HRF for every voxel
            C = like B but with GLMdenoise regressors added into the model
            D = like C but with ridge regression regularization (tailored to
                each voxel)

        - wantlibrary=0
            A fixed assumed HRF is used in all model types.

        - wantglmdenoise=0, wantfracridge=0
            Model types C and D are not computed.

        - wantglmdenoise=0, wantfracridge=1
            Model type C is not computed; model type D is computed using 0
            GLMdenoise regressors.

        - wantglmdenoise=1, wantfracridge=0
            Model type C is computed; model type D is not computed.

        - wantlss=1
            Model type B is computed, but using least-squares-separate instead
            of OLS. Other model types, if computed, use OLS.

        Note that if you set wantglmdenoise=1, you MUST have repeats of
        conditions and an associated cross-validation scheme <params.xvalscheme>
        UNLESS you specify params.pcstop = -B. In other words, you can perform
        wantglmdenoise without any cross-validation, but you need to provide
        params.pcstop = -B.

        Note that if you set wantfracridge=1, you MUST have repeats of
        conditions and an associated cross-validation scheme
        (<params.xvalscheme>), UNLESS you specify a single scalar params.fracs.
        In other words, you can perform wantfracridge without any
        cross-validation, but you need to provide params.fracs as a scalar.

        Arguments:
        __________

        params (dict): Dictionary of parameters. Optional

        *** MAJOR, HIGH-LEVEL FLAGS ***

        <wantlibrary> (optional) is
         0 means use an assumed HRF
         1 means determine the best HRF for each voxel using the
           library-of-HRFs approach
         Default: 1.

        <wantglmdenoise> (optional) is
         0 means do not perform GLMdenoise
         1 means perform GLMdenoise
         Default: 1.

        <wantfracridge> (optional) is
         0 means do not perform ridge regression
         1 means perform ridge regression
         Default: 1.

       <chunklen> (optional) is the number of voxels that we will process at
         the same time. This number should be large in order to speed
         computation, but should not be so large that you run out of RAM. 
         Note that the <chunklen> you choose does not affect any of the
         results or outputs; it merely affects execution time and RAM usage.
         Default: 50000.

        <xvalscheme> (optional) is a list of lists or list of run indices,
         indicating the cross-validation scheme. For example, if we have 8
         runs, we could use [[0, 1], [2, 3], [4, 5], [6, 7]] which indicates
         to do 4 folds of cross-validation, first holding out the 1st and 2nd
         runs, then the 3rd and 4th runs, etc.
         Default: [[0], [1], [2], ... [n-1]] where n is the number of runs.
         Notice the 0-based indexing here.

       <sessionindicator> (optional) is 1 x n (where n is the number of runs)
         with positive integers indicating the run groupings that are
         interpreted as "sessions". The purpose of this input is to allow for
         session-wise z-scoring of single-trial beta weights for the purposes of
         hyperparameter evaluation.
         For example, if you are analyzing data aggregated from multiple scan
         sessions, you may want beta weights to be z-scored per voxel within
         each session in order to compensate for any potential gross changes in
         betas across scan sessions.
         Note that the z-scoring has effect only INTERNALLY: it is used merely to
         calculate the cross-validation performance and the associated
         hyperparameter selection; the outputs of this function do not reflect
         z-scoring, and the user may wish to post-hoc apply z-scoring.
         Default: np.ones((1,n)).astype(int) which means to interpret
         all runs as coming from the same session. Here, we use 1-based
         indexing for the session indicator. e.g. [1, 2, 3, 4] for 4 sessions.

       *** I/O FLAGS ***

        <wantfileoutputs> (optional) is a logical vector [A, B, C, D]
         indicating which of the four model types to save to disk (assuming
         that they are computed).
         A = 0/1 for saving the results of the ONOFF model
         B = 0/1 for saving the results of the FITHRF model
         C = 0/1 for saving the results of the FITHRF_GLMDENOISE model
         D = 0/1 for saving the results of the FITHRF_GLMDENOISE_RR model
         Default: [1, 1, 1, 1] which means save all computed results to disk.

        <wantmemoryoutputs> (optional) is a logical vector [A, B, C, D]
         indicating which of the four model types to return in the output
         <results>. The user must be careful with this, as large datasets
         can require a lot of RAM.
         If you do not request the various model types, they will be
         cleared from memory (but still potentially saved to disk).
         Default: [0, 0, 0, 1] which means return only the final type-D model.

        <wanthdf5> (optional) is an optional flag that allows saving files in
         hdf5 format. This is useful if your output file is about to he huge
         (>4Gb). Default to false, which saves in a .npy format.

        *** GLM FLAGS ***

        <extra_regressors> (optional) is time x regressors or a list
         of elements that are each time x regressors. The dimensions of
         <extraregressors> should mirror that of <design> (i.e. same number of
         runs, same number of time points). The number of extra regressors
         does not have to be the same across runs, and each run can have zero
         or more extra regressors. If [] or not supplied, we do
         not use extra regressors in the model.

        <maxpolydeg> (optional) is a non-negative integer with the maximum
         polynomial degree to use for polynomial nuisance functions, which
         are used to capture low-frequency noise fluctuations in each run.
         Can be a vector with length equal to the number of runs (this
         allows you to specify different degrees for different runs).
         Default is to use round(L/2) for each run where L is the
         duration in minutes of a given run.

        <wantpercentbold> (optional) is whether to convert amplitude estimates
         to percent BOLD change. This is done as the very last step, and is
         accomplished by dividing by the absolute value of 'meanvol' and
         multiplying by 100. (The absolute value prevents negative values in
         'meanvol' from flipping the sign.) Default: 1.

        *** HRF FLAGS ***

        <hrftoassume> (optional) is time x 1 with an assumed HRF that
         characterizes the evoked response to each trial. We automatically
         divide by the maximum value so that the peak is equal to 1. Default
         is to generate a canonical HRF (see getcanonicalhrf in hrf/gethrf.py).
         Note that the HRF supplied in <hrftoassume> is used in only two
         instances:
         (1) it is used for the simple ONOFF type-A model, and (2) if the
             user sets <wantlibrary> to 0, it is also used for the type-B,
             type-C, and type-D models.

        <hrflibrary> (optional) is an np.array of shape time x H,
         with H different HRFs to choose from for the library-of-HRFs approach.
         We automatically normalize each HRF to peak at 1.
         Default is to generate a library of 20 HRFs (see
         getcanonicalhrflibrary).
         Note that if <wantlibrary> is 0, <hrflibrary> is clobbered with the
         contents of <hrftoassume>, which in effect causes a single assumed
         HRF to be used.

        *** MODEL TYPE A (ONOFF) FLAGS ***

        (none)

        *** MODEL TYPE B (FITHRF) FLAGS ***

        <wantlss> (optional) is 0/1 indicating whether 'least-squares-separate'
         estimates are desired. If 1, then the type-B model will be estimated
         using the least-squares-separate method (as opposed to ordinary
         least squares). Default: 0.

        *** MODEL TYPE C (FITHRF_GLMDENOISE) FLAGS ***

        <n_pcs> (optional) is a non-negative integer indicating the
         maximum number of PCs to enter into the model. Default: 10.

        <brainthresh> (optional) is [A, B] where A is a percentile for voxel
         intensity values and B is a fraction to apply to the percentile. These
         parameters are used in the selection of the noise pool.
         Default: [99, 0.1].

        <brainR2> (optional) is an R^2 value (percentage). After fitting the
         type-A model, voxels whose R^2 is below this value are allowed to
         enter the noise pool.
         Default is [] which means to automatically determine a good value.

        <brainexclude> (optional) is X x Y x Z (or XYZ x 1) with 1s indicating
         voxels to specifically exclude when selecting the noise pool. 0 means
         all voxels can be potentially chosen. Default: 0.

        <pcR2cutoff> (optional) is an R^2 value (percentage). To decide the
         number of PCs to include, we examine a subset of the available voxels.
         Specifically, we examine voxels whose type-A model R^2 is above
         <pcR2cutoff>. Default is []
         which means to automatically determine a good value.

        <pcR2cutoffmask> (optional) is X x Y x Z (or XYZ x 1) with 1s
         indicating all possible voxels to consider when selecting the subset
         of voxels. 1 means all voxels can be potentially selected. Default: 1.

        <pcstop> (optional) is
         A: a number greater than or equal to 1 indicating when to stop adding
            PCs into the model. For example, 1.05 means that if the
            cross-validation performance with the current number of PCs is
            within 5 of the maximum observed, then use that number of PCs.
            (Performance is measured relative to the case of 0 PCs.) When
            <pcstop> is 1, the selection strategy reduces to simply choosing
            the PC number that achieves the maximum. The advantage of stopping
            early is to achieve a selection strategy that is robust to noise
            and shallow performance curves and that avoids overfitting.
        -B: where B is the number of PCs to use for the final model. B can be
            any integer between 0 and params.n_pcs. Note that if -B case is
            used, cross-validation is NOT performed for the type-C model, and
            instead weblindly use B PCs.
         Default: 1.05.

       *** MODEL TYPE D (FITHRF_GLMDENOISE_RR) FLAGS ***

        <fracs> (optional) is a numpy vector of fractions that are greater
         than 0 and less than or equal to 1. We automatically sort in
         descending order and ensure the fractions are unique. These fractions
         indicate the regularization levels to evaluate using fractional ridge
         regression (fracridge) and cross-validation.
         Default: np.linspace(1, 0.05, 20).
         A special case is when <fracs> is specified as a single scalar value.
         In this case, cross-validation is NOT performed for the type-D model,
         and we instead blindly usethe supplied fractional value for the type-D
         model.

        <wantautoscale> (optional) is whether to automatically scale and offset
         the model estimates from the type-D model to best match the
         unregularized estimates. Default: 1.
        """

        params = params or dict()
        for key, _ in default_params.items():
            if key not in params.keys():
                params[key] = default_params[key]

        # Check if all opt arguments are allowed
        allowed = list(default_params.keys()) + [
            'xvalscheme',
            'sessionindicator',
            'hrflibrary',
            'hrftoassume',
            'maxpolydeg'
        ]
        for key in params.keys():
            if key not in allowed:
                raise ValueError(f"""
                Input parameter not recognized: '{key}'
                Possible input parameters are:\n{allowed}
                """)

        self.params = params

    def fit(self, design, data, stimdur, tr, outputdir=None, figuredir=None):
        """
        Arguments:
        __________

        <design> is the experimental design. There are two possible cases:
        1. A where A is a matrix with dimensions time x conditions.
            Each column should be zeros except for ones indicating condition
            onsets.
        2. [A1, A2, ... An] where each of the A's are like the previous case.
            The different A's correspond to different runs, and different runs
            can have different numbers of time points. However, all A's must
            have the same number of conditions.
        Note that we ultimately compute single-trial response estimates (one
        estimate for each condition onset), and these will be provided in
        chronological order. However, by specifying that a given condition
        occurs more than one time over the course of the experiment, this
        information can and will be used for cross-validation purposes.

        <data> is the time-series data with dimensions X x Y x Z x time or a
         list vector of elements that are each X x Y x Z x time. XYZ can be
         collapsed such that the data are given as a 2D matrix (units x time),
         which is useful for surface-format data.
         The dimensions of <data> should mirror that of <design>. For example,
         <design> and <data> should have the same number of runs, the same
         number of time points, etc.
         <data> should not contain any NaNs. We automatically convert <data> to
         single format if not already in single format.
         <stimdur> is the duration of a trial in seconds. For example, 3.5
         means that you expect the neural activity from a given trial to last
         for 3.5 s.

        <tr> is the sampling rate in seconds. For example, 1 means that we get
         a new time point every 1 s. Note that <tr> applies to both <design>
         and <data>.

        <outputdir> (optional) is a directory to which data will be written.
         (If the directory does not exist, we create it; if the directory
         already exists, we delete its contents so we can start fresh.) If you
         set <outputdir> to None, we will not create a directory and no files
         will be written.
         Default is 'GLMestimatesingletrialoutputs' (created in the current
         working directory).

        <figuredir> (optional) is a directory to which figures will be written.
         (If the directory does not exist, we create it; if the directory
         already exists, we delete its contents so we can start fresh.) If you
         set <figuredir> to None, we will not create a directory and no files
         will be written.
         Default is 'GLMestimatesingletrialfigures' (created in the current
         working directory).


        Returns:
        __________

        There are various outputs for each of the four model types:

        <modelmd> is either
         (1) the HRF (time x 1) and ON-OFF beta weights (X x Y x Z)
         (2) the full set of single-trial beta weights (X x Y x Z x TRIALS)

        <R2> is model accuracy expressed in terms of R^2 (percentage).

        <R2run> is R2 separated by run

        <meanvol> is the mean of all volumes

        <FitHRFR2> is the R2 for each of the different HRFs in the library

        <FitHRFR2run> is separated by run

        <HRFindex> is the 1-index of the best HRF

        <HRFindexrun> is HRFiniex separated by run

        <noisepool> indicates voxels selected for the noise pool

        <pcregressors> indicates the full set of candidate GLMdenoise
         regressors that were found

        <glmbadness> is cross-validation results for GLMdenoise

        <pcvoxels> is the set of voxels used to summarize GLMdenoise
         cross-validation results

        <xvaltrend> is the summary GLMdenoise cross-validation result on which
                    pcnum selection is done

        <pcnum> is the number of PCs that were selected for the final model

        <FRACvalue> is the fractional regularization level chosen for each
         voxel

        <scaleoffset> is the scale and offset applied to RR estimates to best
                    match the unregularized result

        """

        # DEAL WITH INPUTS
        params = self.params

        # initialise return
        results = {}

        # xyz can either be a tuple of dimensions x y z
        # or a boolean indicating that data was 2D
        data, design, xyz = check_inputs(data, design)

        # keep class bound data and design
        self.data = data
        self.design = design



        # calc
        numruns = len(design)
        numtimepoints = [data[run_i].shape[-1] for run_i in range(numruns)]

        numcond = design[0].shape[1]

        if xyz:
            numvoxels = np.prod(xyz)
            dimdata = 3
        else:
            numvoxels = self.data[0].shape[0]
            dimdata = 2

        # inputs
        if 'xvalscheme' not in params:
            params['xvalscheme'] = np.arange(numruns)

        # additional check for the file format
        if 'wanthdf5' not in params:
            params['wanthdf5'] = 0

        if 'sessionindicator' not in params:
            params['sessionindicator'] = np.ones((1, numruns)).astype(int)

        if 'maxpolydeg' not in params:
            params['maxpolydeg'] = [
                np.arange(
                    alt_round(
                        ((self.data[r].shape[-1]*tr)/60)/2) + 1
                    ) for r in np.arange(numruns)]

        if 'hrftoassume' not in params:
            params['hrftoassume'] = normalisemax(
                getcanonicalhrf(stimdur, tr),
                dim='global'
            )

        if 'hrflibrary' not in params:
            params['hrflibrary'] = getcanonicalhrflibrary(stimdur, tr).T

        if 'firdelay' not in params:
            params['firdelay'] = 30

        if 'firpct' not in params:
            params['firpct'] = 99

        # deal with length issues and other miscellaneous things
        if not isinstance(params['extra_regressors'], list):
            params['extra_regressors'] = [params['extra_regressors']]

        if type(params['maxpolydeg']) is int:
            params['maxpolydeg'] = np.tile(
                params['maxpolydeg'], numruns
            ).tolist()

        # normalise maximal amplitude on hrfs
        params['hrftoassume'] = normalisemax(
            params['hrftoassume'],
            dim='global'
        )

        params['hrflibrary'] = normalisemax(params['hrflibrary'], 0)
        params['fracs'] = np.unique(params['fracs'])[::-1]
        np.testing.assert_equal(
            np.all(params['fracs'] > 0),
            True,
            err_msg='fracs must be greater than 0')

        np.testing.assert_equal(
            np.all(params['fracs'] <= 1),
            True,
            err_msg='fracs must be less than or equal to 1')

        if params['extra_regressors']:
            if params['extra_regressors'][0] is not False:
                assert len(params['extra_regressors']) == numruns, '<extra_regressors> should match the number of runs'

        if figuredir is not None:
            wantfig = 1  # if outputdir is not None, we want figures
        else:
            wantfig = 0

        # deal with output directory
        if outputdir is None:
            cwd = os.getcwd()
            outputdir = os.path.join(cwd, 'GLMestimatesingletrialoutputs')

        if os.path.exists(outputdir):
            import shutil
            shutil.rmtree(outputdir)
            os.makedirs(outputdir)
        else:
            os.makedirs(outputdir)

        # deal with figure directory
        if figuredir is None:
            cwd = os.getcwd()
            figuredir = os.path.join(cwd, 'GLMestimatesingletrialfigures')

        if os.path.exists(figuredir):
            import shutil
            shutil.rmtree(figuredir)
            os.makedirs(figuredir)
        else:
            os.makedirs(figuredir)

        if np.any(params['wantfileoutputs']):
            errm = 'specify an <outputdir> in order to get file outputs'
            np.testing.assert_equal(
                type(outputdir),
                str,
                err_msg=errm)

        # deal with special library stuff
        if params['wantlibrary'] == 0:
            params['hrflibrary'] = params['hrftoassume'].reshape(-1, 1)

        # calc
        # if the data was passed as 3d, unpack xyz
        if xyz:
            nx, ny, nz = xyz
        else:
            nx = numvoxels
            ny = 1
            nz = 1

        nh = params['hrflibrary'].shape[1]

        # figure out chunking scheme
        chunks = chunking(
            np.arange(nx),
            int(np.ceil(nx/np.ceil(numvoxels/params['chunklen']))))

        # deal with special cases
        if params['wantglmdenoise'] == 1:
            errm = '<wantglmdenoise> is 1, but you didnt request type C nor D'
            test = np.any(
                    params['wantfileoutputs'][-2:]
                    ) or np.any(params['wantmemoryoutputs'][-2:])
            np.testing.assert_equal(
                test, True,
                err_msg=errm)

        if params['wantfracridge'] == 1:
            test = params['wantfileoutputs'][3] == 1 \
                or params['wantmemoryoutputs'][3] == 1
            np.testing.assert_equal(
                test, True,
                err_msg='<wantfracridge> is 1, but you did not request type D')

        if params['wantlss'] == 1:
            test = params['wantfileoutputs'][1] == 1 \
                    or params['wantmemoryoutputs'][1] == 1
            np.testing.assert_equal(
                test, True,
                err_msg='<wantlss> is 1, but you did not request type B')

        drng = None
        betavizmx = None

        # PRE-PROCESSING FOR THE EXPERIMENTAL DESIGN

        # calculate the number of trials
        # number of trials in each run
        numtrialrun = np.asarray(
            [np.sum(x.flatten()) for x in self.design]).astype(int).tolist()
        numtrials = np.sum(numtrialrun).astype(int)  # number of total trials

        # create a single-trial design matrix and calculate a bunch
        # of extra information
        designSINGLE = []
        cnt = 0

        # 1 x numtrials indicating which condition each trial belongs to
        stimorder = []

        # each element is the vector of trial indices associated with the run
        validcolumns = []

        # each element is the vector of actual condition numbers occurring
        # with a given run
        stimix = []

        # loop through runs
        for run_i in np.arange(len(self.design)):
            designSINGLE.append(
                np.zeros((self.design[run_i].shape[0], numtrials)))

            run_validcolumns = []
            # loop through the volumes for that run
            for cond_i in np.arange(self.design[run_i].shape[0]):
                # if a condition was presented on that volume
                # find which
                temp = np.where(self.design[run_i][cond_i, :])[0]
                assert len(temp) <= 1, 'two conditions have exactly the same trial onset! this is not allowed!'

                # if that volume had a condition shown
                if not np.size(temp) == 0:
                    # flip it on
                    designSINGLE[run_i][cond_i, cnt] = 1
                    # keep track of the order
                    stimorder.append(temp[0])
                    run_validcolumns.append(cnt)
                    cnt += 1
            validcolumns.append(np.asarray(run_validcolumns))

            stimix.append(np.asarray(stimorder)[np.asarray(run_validcolumns)])

        # Calculate number of trials for each condition
        condcounts = [np.sum(np.asarray(stimorder) == p) for p in range(0, numcond )]

        # Calculate for each condition, how many runs it shows up in
        condinruns = [np.sum([(p in run) for run in stimix]) for p in range(0, numcond)]

        # Calculate buffer at the end of each run
        endbuffers = []
        for run in self.design:
            temp = np.where(np.sum(run, axis=1))[0]  # Indices of when trials happen
            temp = run.shape[0] - temp[-1] - 1  # Number of volumes AFTER last trial onset
            endbuffers.append(temp * tr)  # Number of seconds AFTER last trial onset for which we have data

        # Diagnostics
        print('*** DIAGNOSTICS ***:')
        print(f'There are {len(design)} runs.')
        print(f'The number of conditions in this experiment is {numcond}.')
        print(f'The stimulus duration corresponding to each trial is {stimdur:.2f} seconds.')
        print(f'The TR (time between successive data points) is {tr:.2f} seconds.')
        print(f'The number of trials in each run is: {numtrialrun}.')
        print(f'The number of trials for each condition is: {condcounts}.')
        print(f'For each condition, the number of runs in which it appears: {condinruns}.')
        print(f'For each run, how much ending buffer do we have in seconds? {endbuffers}.')

        # Issue warning if trials get too close to the end
        if any(buffer < 8 for buffer in endbuffers):
            msg = 'Warning: You have specified trial onsets that occur less than 8 seconds' + \
                  ' from the end of at least one of the runs. This may cause estimation' + \
                  ' problems! As a solution, consider simply omitting specification of these' + \
                  ' ending trials from the original design matrix.'
            warnings.warn(msg)

        # Issue warning if no repeats
        if np.all(np.array(condinruns) <= 1):
            msg = 'None of your conditions occur in more than one run.' + \
                  ' Are you sure this is what you intend?'
            warnings.warn(msg)

            if params['wantglmdenoise']:
                if params['pcstop'] <= 0:
                    msg = 'pcstop is specified as the -B case. We will not be performing ' + \
                          'cross-validation, but will be performing glmdenoise using B number of PCs'
                    warnings.warn(msg)
                else:
                    msg = 'Since there are no repeats, standard cross-validation usage of ' + \
                          '<wantglmdenoise> cannot be performed. Setting <wantglmdenoise> to 0.'
                    warnings.warn(msg)
                    params['wantglmdenoise'] = 0

            if params['wantfracridge']:
                if len(params['fracs']) > 1:
                    msg = 'Since there are no repeats, standard cross-validation usage of' + \
                        ' <wantfracridge> cannot be performed. Setting <wantfracridge> to 0.'
                    warnings.warn(msg)
                    params['wantfracridge'] = 0
                else:
                    msg = 'fracs is specified as the single scalar case. We will not be' + \
                        'performing cross-validation, but will be performing ridge regression ' + \
                        'using the user-supplied fraction'
                    warnings.warn(msg)

        # Construct a nice output dictionary for this design-related stuff
        resultsdesign = {
            'design': self.design,
            'stimdur': stimdur,
            'tr': tr,
            'params': params,
            'designSINGLE': designSINGLE,
            'stimorder': stimorder,
            'numtrialrun': numtrialrun,
            'condcounts': condcounts,
            'condinruns': condinruns,
            'endbuffers': endbuffers
        }

        file0 = os.path.join(outputdir, 'DESIGNINFO.npy')
        print(f'*** Saving design-related results to {file0}. ***')
        np.save(file0, resultsdesign, allow_pickle=True)

        # FIT DIAGNOSTIC RUN-WISE FIR MODEL
        # The approach:
        # (1) Every stimulus is treated as the same.
        # (2) We fit an FIR model up to 30 s.
        # (3) Each run is fit completely separately.

        print('*** FITTING DIAGNOSTIC RUN-WISE FIR MODEL ***')

        opt0 = {
            'extra_regressors': params['extra_regressors'],
            'maxpolydeg': params['maxpolydeg'],
            'wantpercentbold': params['wantpercentbold'],
            'suppressoutput': 1
        }

        firR2 = []
        firtcs= []
        design0 = [np.sum(run, axis=1, keepdims=True, dtype=np.int64) for run in self.design]
        for p in range(len(self.data)):
            results0 = glm_estimatemodel(
                design0[p],
                data[p],
                stimdur,
                tr,
                'fir',
                np.floor(params['firdelay']/tr).astype(int),
                0,
                opt0
            )[0]

            firR2.append(results0['R2'])
            firtcs.append(results0['betasmd'])

        firR2 = np.array(firR2)
        firtcs = np.array(firtcs)

        # calc
        firR2mn = np.mean(firR2, axis=0)
        firthresh = np.percentile(firR2mn[np.isfinite(firR2mn)],params['firpct'])
        firix = np.flatnonzero(firR2mn > firthresh);  # we want to average the top 1st percentile

        # calc timecourse averages
        firavg = []  # time x runs
        for rr in range(len(data)):
            temp = squish(firtcs[rr, ...], 4)[firix, :]  # voxels x time
            firavg.append(np.median(temp, axis=0))

        firavg=np.array(firavg)

        firgrandavg = np.mean(firavg, axis=0)  # time x 1

        # figures
        if wantfig:
            colors = cmapturbo(len(data))
            # make the figure
            plt.figure(figsize=(11, 7.5))
            plt.subplot(2, 2, 1)
            legh = []
            legendlabs = []
            for rr in range(len(data)):
                line, = plt.plot(np.arange(0, firavg.shape[1]*tr, step=tr), firavg[rr, :], 'o-', color=colors[rr,:3])
                legh.append(line)
                legendlabs.append(f"Run {rr+1}")
            line, = plt.plot(np.arange(0, len(firgrandavg)*tr, step=tr), firgrandavg, 'r-', linewidth=2)
            legh.append(line)
            legendlabs.append('Run Avg')
            mxix = np.argmax(firgrandavg)
            ax = plt.axis()
            plt.axis([0, firavg.shape[1] * tr * 1.5, ax[2], ax[3]])
            ax = plt.axis()
            plt.axhline(0, color='k')
            plt.axvline((mxix) * tr, linestyle=':', color='k')
            plt.xlabel('Time from trial onset (s)')
            plt.ylabel('BOLD (%)')
            plt.legend(legh, legendlabs, loc='upper right')


            # Plot BOLD at peak time for each run number
            plt.subplot(2, 2, 2)
            plt.bar(np.arange(len(data)) + 1, firavg[:, mxix])
            plt.axhline(0, color='k')
            plt.axis([0, len(data) + 1, ax[2], ax[3]])
            plt.xticks(np.arange(len(data)) + 1)
            plt.xlabel('Run number')
            plt.ylabel('BOLD at peak time (%)')


            # Plot different HRFs and the assumed HRF
            plt.subplot(2, 2, 3)
            cmap0 = cmapturbo(nh)
            legh = []
            legendlabs = []

            for hh in range(nh):
                times = np.arange(0, params['hrflibrary'].shape[0] * tr, tr)
                line, = plt.plot(times, params['hrflibrary'][:, hh], '-', color=cmap0[hh])
                legh.append(line)
                legendlabs.append(f'HRFindex{hh + 1}')

            times_assumed = np.arange(0, len(params['hrftoassume']) * tr, tr)
            line, = plt.plot(times_assumed, params['hrftoassume'], 'k-', linewidth=2)
            legh.append(line)
            legendlabs.append('HRFassume')

            plt.xlim(ax[0], ax[1])
            plt.axhline(0, color='k')
            plt.xlabel('Time from trial onset (s)')
            plt.ylabel('BOLD (a.u.)')
            plt.legend(legh, legendlabs, loc='upper right', fontsize='xx-small')

            plt.tight_layout()
            plt.savefig(os.path.join(figuredir, 'runwiseFIR.png'))
            plt.clf()


            # more figures
            cmap = mpl.colormaps['hot'].resampled(256)

            if xyz:
                for rr in range(firR2.shape[0]):
                    filer = os.path.join(figuredir,f'runwiseFIR_R2_run{rr+1:02d}.png')
                    plt.imsave(
                        filer,
                        np.uint8(255*make_image_stack(firR2[rr,:,:,:],[0, 100])**0.5),
                        cmap=cmap, vmin=0, vmax=255)
                filer = os.path.join(figuredir,'runwiseFIR_R2_runavg.png')
                plt.imsave(
                    filer,
                    np.uint8(255*make_image_stack(firR2mn,[0, 100])**0.5),
                    cmap=cmap, vmin=0, vmax=255)

                cmap_gray = mpl.colormaps['gray'].resampled(256)
                filer = os.path.join(figuredir,'runwiseFIR_summaryvoxels.png')
                plt.imsave(
                    filer,
                    np.uint8(255*make_image_stack(firR2mn > firthresh, [0, 1])),
                    cmap=cmap_gray, vmin=0, vmax=255)



        # save
        if isinstance(outputdir, str):
            file0 = os.path.join(outputdir,'RUNWISEFIR.npy')

            resultsfir = {
            'firR2': firR2,
            'firtcs': firtcs,
            'firavg': firavg,
            'firgrandavg': firgrandavg
            }
            print(f'*** Saving FIR results to {file0}. ***\n')
            np.save(file0, resultsfir, allow_pickle='True')

        # FIT TYPE-A MODEL [ON-OFF]
        # The approach:
        # (1) Every stimulus is treated as the same.
        # (2) We assume the HRF.

        # define
        whmodel = 0

        # collapse all conditions and fit
        print('*** FITTING TYPE-A MODEL (ONOFF) ***\n')
        design0 = [np.sum(x, axis=1)[:, np.newaxis] for x in self.design]
        optB = {
            'extra_regressors': params['extra_regressors'],
            'maxpolydeg': params['maxpolydeg'],
            'wantpercentbold': params['wantpercentbold'],
            'suppressoutput': 0
        }
        results0 = glm_estimatemodel(
            design0,
            self.data,
            stimdur,
            tr,
            'assume',
            params['hrftoassume'],
            0,
            optB
            )[0]

        onoffR2 = results0['R2']
        meanvol = results0['meanvol']
        betasmd = results0['betasmd']


        # determine onoffvizix for beta inspection
        onoffvizix = np.argsort(np.where(np.isnan(onoffR2), -np.inf, onoffR2).reshape(-1), kind='stable')

        # Generate the sequence representing percentiles and index into onoffvizix
        onoffvizix = onoffvizix[np.floor(len(onoffvizix) * np.r_[np.arange(1, 0.75, -0.25/999), 0.75]).astype(int)-1]

        # save to disk if desired
        if params['wantfileoutputs'][whmodel] == 1:
            if params['wanthdf5'] == 1:
                file0 = os.path.join(outputdir, 'TYPEA_ONOFF.hdf5')
            else:
                file0 = os.path.join(outputdir, 'TYPEA_ONOFF.npy')

            print(f'\n*** Saving results to {file0}. ***\n')

            results_out = {
                'onoffR2': onoffR2,
                'meanvol': meanvol,
                'betasmd': betasmd
            }
            if params['wanthdf5'] == 1:
                hf = h5py.File(file0, 'w')
                for k, v in results_out.items():
                    if isinstance(v, list):
                        v = np.array(v)
                    if v is not None:
                        hf.create_dataset(k, data=v)
                    else:
                        hf.create_dataset(k, data=h5py.Empty("f"))
                hf.close()
            else:
                np.save(file0, results_out)

        # figures
        if wantfig:
            if xyz:
                plt.imsave(
                    os.path.join(figuredir, 'onoffR2.png'),
                    np.uint8(255*make_image_stack(onoffR2,[0, 100])**0.5),
                    cmap=cmap
                )
                cmap = mpl.colormaps['gray'].resampled(256)
                plt.imsave(
                    os.path.join(figuredir, 'meanvol.png'),
                    np.uint8(255*make_image_stack(meanvol,1)),
                    cmap=cmap, vmin=0, vmax=255
                )

        # preserve in memory if desired, and then clean up
        if params['wantmemoryoutputs'][whmodel] == 1:
            results['typea'] = {
                'onoffR2': onoffR2,
                'meanvol': meanvol,
                'betasmd': betasmd
            }

        # DETERMINE THRESHOLDS
        if wantfig:
            thresh = findtailthreshold(
                onoffR2.flatten(),
                os.path.join(figuredir, 'onoffR2hist.png'))[0]
        else:
            thresh = findtailthreshold(onoffR2.flatten())[0]

        if 'brainR2' not in params or not params['brainR2']:
            print(f'*** Setting brain R2 threshold to {thresh} ***\n')
            params['brainR2'] = thresh

        if 'pcR2cutoff' not in params or not params['pcR2cutoff']:
            params['pcR2cutoff'] = thresh

        # FIT TYPE-B MODEL [FITHRF]

        # The approach:
        # (1) Fit single trials.
        # (2) Evaluate the library of HRFs (or the single assumed HRF).
        #     Choose based on R2 for each voxel.

        # if the user does not want file output nor memory output AND
        # if the number of HRFs to choose
        # from is just 1, we can short-circuit this whole endeavor!
        if params['wantfileoutputs'][1] == 0 and \
                params['wantmemoryoutputs'][1] == 0 and \
                params['hrflibrary'].shape[1] == 1:

            # short-circuit all of the work
            HRFindex = np.ones(xyz)  # easy peasy

        else:

            # define
            whmodel = 1

            # initialize
            FitHRFR2 = np.zeros((nx,ny,nz,nh), dtype=np.float32)
            # X x Y x Z x HRFs with R2 values (all runs)
            FitHRFR2run = np.zeros((nx,ny,nz,numruns,nh), dtype=np.float32)

            # X x Y x Z x runs x HRFs with R2 separated by runs
            modelmd = np.zeros((nx,ny,nz,numtrials), dtype=np.float32)

            # X x Y x Z x trialbetas
            optC = {
                    'extra_regressors': params['extra_regressors'],
                    'maxpolydeg': params['maxpolydeg'],
                    'wantpercentbold': params['wantpercentbold'],
                    'suppressoutput': 1
            }

            # loop over chunks
            print('*** FITTING TYPE-B MODEL (FITHRF) ***\n')
            for zin in tqdm(np.arange(len(chunks)), desc='chunks'):

                this_chunk = chunks[zin]
                n_inchunk = len(this_chunk)

                data_chunk = [datx[this_chunk, :, :, :] for datx in self.data]
                # do the fitting and accumulate all the betas
                modelmd0 = np.zeros(
                    (n_inchunk, ny, nz, numtrials, nh),
                    dtype=np.float32)
                # someXYZ x trialbetas x HRFs
                for ph in np.arange(nh):
                    results0 = glm_estimatemodel(
                        designSINGLE,
                        data_chunk,
                        stimdur,
                        tr,
                        'assume',
                        params['hrflibrary'][:, ph],
                        0,
                        optC
                    )[0] # NOTE no cache

                    FitHRFR2[this_chunk, :, :, ph] = results0['R2']

                    FitHRFR2run[this_chunk, :, :, :, ph] = np.transpose(np.stack(
                        results0['R2run']), [1, 2, 3, 0])
                    modelmd0[:, :, :, :, ph] = results0['betasmd']

                # keep only the betas we want
                # find the hrf for which the R2 is largest
                ii = np.argmax(FitHRFR2[this_chunk, :, :, :], axis=-1)

                # flatten ii tile it as someXYZ x numtrials
                iiflat = np.tile(
                    ii.flatten()[:, np.newaxis], numtrials).flatten()

                # squish modelmd0
                modelmd0=squish(modelmd0,4)

                # index modelmd0 and reshape into modelmd
                modelmd[this_chunk,:,:,:] = modelmd0[np.arange(
                    modelmd0.shape[0]), iiflat].reshape((n_inchunk, ny, nz, numtrials))

            R2 = np.max(FitHRFR2, axis=-1)  # R2 is XYZ
            HRFindex = np.argmax(FitHRFR2, axis=-1)  # HRFindex is XYZ

            # also, use R2 from each run to select best HRF
            HRFindexrun = np.argmax(FitHRFR2run, axis=-1)

            iiflat = np.tile(HRFindex.flatten()[:, np.newaxis], (1, FitHRFR2run.shape[3])).flatten()
            flatFitHRFR2run = squish(FitHRFR2run,4)

            # using each voxel's best HRF, what are the corresponding R2run values?
            R2run = flatFitHRFR2run[np.arange(flatFitHRFR2run.shape[0]), iiflat].reshape((nx, ny, nz, numruns))

            # FIT TYPE-B MODEL (LSS) INTERLUDE BEGIN

            # if params.wantlss, we have to use the determined HRFindex and
            # re-fit the entire dataset using LSS estimation. this will
            # simply replace 'modelmd' with a new version.
            # so that's what we have to do here.

            if params['wantlss']:

                # initalize
                modelmd = np.zeros((nx*ny*nz, numtrials), dtype=np.float32)
                # X*Y*Z x trialbetas  [the final beta estimates]

                # loop over chunks
                print(
                    '*** FITTING TYPE-B MODEL'
                    '(FITHRF but with LSS estimation) ***\n')

                for zin in tqdm(np.arange(len(chunks)), desc='chunks'):

                    this_chunk = chunks[zin]
                    n_inchunk = len(this_chunk)

                    # loop over possible HRFs
                    for hh in np.arange(nh):

                        # figure out which voxels to process.
                        # this will be a vector of indices into the small
                        # chunk that we are processing.
                        # our goal is to fully process this set of voxels!
                        goodix = np.flatnonzero(
                            HRFindex[this_chunk, :, :] == hh)

                        if goodix.size!=0:

                            data0 = [squish(
                                        x[this_chunk, ...],
                                        3
                                    )[goodix, :]
                                    for x in self.data]  # voxels x time

                            # calculate the corresponding indices relative to the
                            # full volume
                            temp = np.zeros_like(HRFindex)
                            temp[this_chunk, :, :] = 1
                            relix = np.flatnonzero(temp)[goodix]

                            # define options
                            optA = {'extra_regressors': params['extra_regressors'],
                                    'maxpolydeg': params['maxpolydeg'],
                                    'wantpercentbold': params['wantpercentbold'],
                                    'suppressoutput': 1
                                    }

                            # do the GLM
                            cnt = 0
                            for rrr in np.arange(len(designSINGLE)):  # each run
                                for ccc in np.arange(numtrialrun[rrr]):
                                    # each trial
                                    designtemp = designSINGLE[rrr]
                                    designtemp = np.c_[
                                        designtemp[:, cnt+ccc],
                                        np.sum(
                                            designtemp[:, np.setdiff1d(
                                                np.arange(
                                                    designtemp.shape[1]
                                                    ),
                                                cnt+ccc)],
                                            axis=1
                                        )
                                    ]
                                    results0, cache = glm_estimatemodel(
                                        designtemp,
                                        data0[rrr],
                                        stimdur,
                                        tr,
                                        'assume',
                                        params['hrflibrary'][:, hh],
                                        0,
                                        optA
                                    )
                                    modelmd[relix, cnt+ccc] = \
                                        results0['betasmd'][:, 0]

                                cnt = cnt + numtrialrun[rrr]

                # deal with dimensions
                modelmd = np.reshape(modelmd,(nx, ny, nz, numtrials))

            # FIT TYPE-B MODEL (LSS) INTERLUDE END
            # prepare disk/memory output fields

            results_out = {
                'FitHRFR2': FitHRFR2,
                'FitHRFR2run': FitHRFR2run,
                'HRFindex': HRFindex,
                'HRFindexrun': HRFindexrun,
                'R2': R2,
                'R2run': R2run,
                'betasmd': modelmd,
                'meanvol': meanvol
            }

            # save to disk if desired
            if params['wantfileoutputs'][whmodel] == 1:

                if params['wanthdf5'] == 1:
                    file0 = os.path.join(outputdir, 'TYPEB_FITHRF.hdf5')
                else:
                    file0 = os.path.join(outputdir, 'TYPEB_FITHRF.npy')

                print(f'\n*** Saving results to {file0}. ***\n')
                
                if params['wanthdf5'] == 1:
                    hf = h5py.File(file0, 'w')
                    for k, v in results_out.items():
                        if isinstance(v, list):
                            v = np.array(v)
                        if v is not None:
                            hf.create_dataset(k, data=v)
                        else:
                            hf.create_dataset(k, data=h5py.Empty("f"))
                    hf.close()
                else:
                    np.save(file0, results_out)

            # figures?
            if wantfig:
                if xyz:
                    cmap = mpl.colormaps['jet'].resampled(256)
                    filer = os.path.join(figuredir,'HRFindex.png')
                    plt.imsave(
                        filer,
                        np.uint8(255*make_image_stack(HRFindex,[1, nh])),
                        cmap=cmap, vmin=0, vmax=255)

                # more figs
                # Scatter plot
                plt.figure(figsize=(11,12))
                plt.scatter(
                    onoffR2.flatten(),
                    HRFindex.flatten() + 0.2 * np.random.randn(*HRFindex.flatten().shape),
                    c='r',
                    s=9
                )

                # Set YTicks
                plt.yticks(np.arange(1, nh+1))

                # Set YLim
                plt.ylim([-1, nh+1])

                # Set XLabel and YLabel
                plt.xlabel('ON-OFF model R^2')
                plt.ylabel('HRF index (with small amount of jitter)')
                plt.savefig(os.path.join(figuredir, 'onoffR2_vs_HRFindex.png'))
                plt.clf()

                # beta visualisation
                temp = squish(modelmd, 3)[onoffvizix, :]
                if betavizmx is None:
                    betavizmx = np.nanpercentile(np.abs(temp.flatten()),99)
                filer = os.path.join(figuredir, 'betaviz_typeB.png')
                cmap = cmapsign4(256)
                colormap_to_plot = cmaplookup(temp,-betavizmx,betavizmx,0,cmap)
                plt.imsave(filer, colormap_to_plot)

                # dmetric visualization
                if xyz:
                    temp = calcdmetric(modelmd, np.asarray(stimorder))
                    if drng is None:
                        drng = [np.nanmin(temp.flatten()), np.nanmax(temp.flatten())]
                    cmap = mpl.colormaps['hot'].resampled(256)
                    filer = os.path.join(figuredir,'dmetric_typeB.png')
                    plt.imsave(
                        filer,
                        np.uint8(255*make_image_stack(temp, drng)),
                        cmap=cmap, vmin=0, vmax=255
                        )

            # preserve in memory if desired, and then clean up
            if params['wantmemoryoutputs'][whmodel] == 1:
                results['typeb'] = results_out

        # COMPUTE GLMDENOISE REGRESSORS

        # if the user does not want to perform GLMdenoise,
        # we can just skip all of this
        if params['wantglmdenoise'] == 0:

            # just create placeholders
            pcregressors = []
            noisepool = None

        else:

            # figure out the noise pool

            # threshold for non-brain voxels
            thresh = np.percentile(
                meanvol.flatten(),
                params['brainthresh'][0]
            )*params['brainthresh'][1]

            # logical indicating voxels that are bright (brain voxels)
            bright = meanvol > thresh

            # logical indicating voxels with poor R2
            badR2 = onoffR2 < params['brainR2']

            # logical indicating voxels that satisfy all criteria
            if not params['brainexclude']:
                noisepool = (bright * badR2)
            else:
                noisepool = (bright * badR2 * params['brainexclude'])

            # determine noise regressors
            pcregressors = []
            print('*** DETERMINING GLMDENOISE REGRESSORS ***\n')
            polymatrix = []
            for run_i, drun in enumerate(self.data):

                # extract the time-series data for the noise pool
                noise_pool = squish(drun, dimdata)[np.flatnonzero(noisepool), :].T # time x voxels

                # project out polynomials from the data
                # this projects out polynomials
                pmatrix = make_polynomial_matrix(
                    numtimepoints[run_i],
                    params['maxpolydeg'][run_i])

                polymatrix.append(
                    make_projection_matrix(pmatrix))

                noise_pool = polymatrix[run_i].astype(np.float32) @ noise_pool.squeeze() # in case 2D

                noise_pool = normalize(noise_pool, axis=0)

                noise_pool = noise_pool @ noise_pool.T
                unitary = np.linalg.svd(noise_pool)[0]
                unitary = unitary[:, :params['n_pcs']+1]
                unitary = unitary / np.std(unitary, 0, ddof=1)
                pcregressors.append(unitary.astype(np.float32))


        # CROSS-VALIDATE TO FIGURE OUT NUMBER OF GLMDENOISE REGRESSORS
        # if the user does not want GLMdenoise, just set some dummy values
        if params['wantglmdenoise'] == 0:
            pcnum = 0
            xvaltrend = None
            glmbadness = None
            pcvoxels = None

        # in this case, the user decides (and we can skip the cross-validation)
        elif params['pcstop'] <= 0:
            pcnum = -params['pcstop']
            xvaltrend = None
            glmbadness = None
            pcvoxels = None

        # otherwise, we have to do a lot of work
        else:

            # initialize
            # XYZ x 1+npc  [squared beta error for different numbers of PCs]
            glmbadness = np.zeros(
                (numvoxels, 1+params['n_pcs']),
                dtype=np.float32
            )

            # loop over chunks
            print('*** CROSS-VALIDATING DIFFERENT NUMBERS OF REGRESSORS ***\n')
            for zin in tqdm(np.arange(len(chunks)), desc='chunks'):

                this_chunk = chunks[zin]
                n_inchunk = len(this_chunk)

                # loop over possible HRFs
                for hh in np.arange(nh):
                    # figure out which voxels to process.
                    # this will be a vector of indices into the small
                    # chunk that we are processing.
                    # our goal is to fully process this set of voxels!
                    goodix = np.flatnonzero(
                        HRFindex[this_chunk, :, :] == hh)

                    if goodix.size != 0:
                        # skip chunks and hrfs with no hrf found fitting this chunk
                        data0 = \
                            [squish(x[this_chunk, :, :, :], 3)[goodix, :] for x in self.data]

                        # calculate the corresponding indices relative to the
                        # full volume
                        temp = np.zeros_like(HRFindex)
                        temp[this_chunk, :, :] = 1
                        relix = np.flatnonzero(temp)[goodix]

                        # perform GLMdenoise
                        results0 = []
                        for n_pc in range(params['n_pcs']+1):

                            # define options
                            optA = {
                                'maxpolydeg': params['maxpolydeg'],
                                'wantpercentbold': 0,
                                'suppressoutput': 1,
                                'extra_regressors': [None for r in range(numruns)]
                            }
                            if n_pc > 0:
                                for rr in range(numruns):
                                    if params['extra_regressors'][0] is False or \
                                        not np.any(params['extra_regressors'][rr]):
                                        optA['extra_regressors'][rr] = \
                                            pcregressors[rr][:, :n_pc]
                                    else:
                                        optA['extra_regressors'][rr] = \
                                            np.c_[params['extra_regressors'][rr],
                                                pcregressors[rr][:, :n_pc]]
                            # do the GLM
                            temp, cache = glm_estimatemodel(
                                        designSINGLE,
                                        data0,
                                        stimdur,
                                        tr,
                                        'assume',
                                        params['hrflibrary'][:, hh],
                                        0,
                                        optA
                                    )

                            results0.append(temp['betasmd'])
                        glmbadness[relix, :] = calcbadness(
                            params['xvalscheme'],
                            validcolumns,
                            stimix,
                            results0,
                            params['sessionindicator']
                            )  # voxels x regularization levels

            # compute xvaltrend
            ix = np.flatnonzero(
                (onoffR2.flatten() > params['pcR2cutoff']) * (np.asarray(
                    params['pcR2cutoffmask']).flatten()))  # vector of indices

            if ix.size == 0:
                print(
                    'Warning: no voxels passed the pcR2cutoff'
                    'and pcR2cutoffmask criteria. Using the'
                    'best 100 voxels.\n')
                if params['pcR2cutoffmask'] == 1:
                    ix2 = np.flatnonzero(np.ones(onoffR2.shape))
                else:
                    ix2 = np.flatnonzero(params['pcR2cutoffmask'] == 1)

                np.testing.assert_equal(
                    len(ix2) > 0,
                    True,
                    err_msg='no voxels are in pcR2cutoffmask'
                )

                ix3 = np.argsort(onoffR2[ix2])[::-1]
                num = np.min([100, len(ix2)])
                ix = ix2[ix3[range(num)]]

            # NOTE: sign flip so that high is good
            xvaltrend = -np.median(glmbadness[ix, :], axis=0)
            np.testing.assert_equal(np.all(np.isfinite(xvaltrend)), True)

            # create for safe-keeping
            pcvoxels = np.zeros((nx, ny, nz), dtype=bool)
            pcvoxels[np.unravel_index(ix, pcvoxels.shape)] = 1

            # choose number of PCs
            # this is the performance curve that starts
            # at 0 (corresponding to 0 PCs)
            pcnum = select_noise_regressors(xvaltrend, params['pcstop'])

            # deal with dimensions
            glmbadness = np.reshape(glmbadness, [nx, ny, nz, -1])

        # FIT TYPE-C + TYPE-D MODELS [FITHRF_GLMDENOISE, FITHRF_GLMDENOISE_RR]
        # setup
        todo = []
        if params['wantglmdenoise'] and (
             params['wantfileoutputs'][2] or params['wantmemoryoutputs'][2]):
            todo.append(2)  # the user wants the type-C model returned

        if params['wantfracridge'] and (
             params['wantfileoutputs'][3] or params['wantmemoryoutputs'][3]):
            todo.append(3)  # the user wants the type-D model returned

        for whmodel in todo:
            # we need to do some tricky setup
            # if this is just a GLMdenoise case, we need to fake it
            if whmodel == 2:
                # here, we need to fake this in order to get the outputs
                fracstouse = np.array([1])
                fractoselectix = 0
                autoscaletouse = 0  # not necessary, so turn off

            # if this is a fracridge case
            if whmodel == 3:
                # if the user specified only one fraction
                if len(params['fracs']) == 1:

                    # if the first one is 1, this is easy
                    if params['fracs'][0] == 1:
                        fracstouse = np.array([1])
                        fractoselectix = 0
                        autoscaletouse = 0  # not necessary, so turn off

                    # if the first one is not 1, we might need 1
                    else:
                        fracstouse = np.r_[1, params['fracs']]
                        fractoselectix = 1
                        autoscaletouse = params['wantautoscale']

                # otherwise, we have to do costly cross-validation
                else:

                    # set these
                    fractoselectix = None
                    autoscaletouse = params['wantautoscale']

                    # if the first one is 1, this is easy
                    if params['fracs'][0] == 1:
                        fracstouse = params['fracs']

                    # if the first one is not 1, we might need 1
                    else:
                        fracstouse = np.r_[1, params['fracs']]

            # ok, proceed

            # initialize
            # XYZ x trialbetas  [the final beta estimates]
            modelmd = np.zeros((numvoxels, numtrials), dtype=np.float32)
            # XYZ [the R2 for the specific optimal frac]
            R2 = np.zeros((nx, ny, nz), dtype=np.float32)

            # XYZ x runs [the R2 separated by runs for the optimal frac]
            R2run = np.zeros((numvoxels, numruns), dtype=np.float32)

            # XYZ [best fraction]
            FRACvalue = np.zeros((nx, ny, nz), dtype=np.float32)

            if fractoselectix is None:
                # XYZ [rr cross-validation performance]
                rrbadness = np.zeros(
                    (numvoxels, len(params['fracs'])),
                    dtype=np.float32)
            else:
                rrbadness = []

            # XYZ x 2 [scale and offset]
            scaleoffset = np.zeros((numvoxels, 2), dtype=np.float32)

            # loop over chunks
            if whmodel == 2:
                print('\n*** FITTING TYPE-C MODEL (GLMDENOISE) ***\n')
            else:
                print('*** FITTING TYPE-D MODEL (GLMDENOISE_RR) ***\n')

            for z in tqdm(np.arange(len(chunks)), desc='chunks'):
                this_chunk = chunks[z]
                n_inchunk = len(this_chunk)

                # loop over possible HRFs
                for hh in np.arange(nh):
                    # figure out which voxels to process.
                    # this will be a vector of indices into the small
                    # chunk that we are processing.
                    # our goal is to fully process this set of voxels!
                    goodix = np.flatnonzero(
                        HRFindex[this_chunk, :, :] == hh)

                    if goodix.size!=0:

                        data0 = \
                            [squish(x[this_chunk, :, :],3)[goodix, :] for x in self.data]

                        # calculate the corresponding indices relative to the
                        # full volume
                        temp = np.zeros_like(HRFindex)
                        temp[this_chunk, :, :] = 1
                        relix = np.flatnonzero(temp)[goodix]

                        # process each frac
                        results0 = []
                        r20 = []
                        r2run0 = []
                        for fracl in range(len(fracstouse)):

                            # define options
                            optA = {'wantfracridge': 1,
                                    'maxpolydeg': params['maxpolydeg'],
                                    'wantpercentbold': 0,
                                    'suppressoutput': 1,
                                    'frac': fracstouse[fracl],
                                    'extra_regressors': [
                                        None for r in range(numruns)]
                                    }

                            if pcnum > 0:
                                for run_i in range(numruns):
                                    if params['extra_regressors'][0] is False or \
                                        not np.any(params['extra_regressors'][run_i]):

                                        optA['extra_regressors'][run_i] = \
                                            pcregressors[run_i][:, :pcnum]
                                    else:
                                        optA['extra_regressors'][run_i] = \
                                            np.c_[
                                                params['extra_regressors'][run_i],
                                                pcregressors[run_i][:, :n_pc]]

                            # fit the entire dataset using the specific frac
                            temp, cache = glm_estimatemodel(
                                        designSINGLE,
                                        data0,
                                        stimdur,
                                        tr,
                                        'assume',
                                        params['hrflibrary'][:, hh],
                                        0,
                                        optA
                                    )

                            # save some memory
                            results0.append(temp['betasmd'])
                            r20.append(temp['R2'])
                            r2run0.append(temp['R2run'])

                        # perform cross-validation if necessary
                        if fractoselectix is None:

                            # compute the cross-validation performance values
                            rrbadness0 = calcbadness(
                                params['xvalscheme'],
                                validcolumns,
                                stimix,
                                results0,
                                params['sessionindicator'])

                            # this is the weird special case where we have
                            # to ignore the artificially added 1
                            if params['fracs'][0] != 1:
                                FRACindex0 = np.argmin(rrbadness0[:, 1:], axis=1)
                                FRACindex0 = FRACindex0 + 1
                                rrbadness[relix, :] = rrbadness0[:, 1:]
                            else:
                                # pick best frac (FRACindex0 is V x 1 with the
                                # index of the best frac)
                                FRACindex0 = np.argmin(rrbadness0, axis=1)
                                rrbadness[relix, :] = rrbadness0

                        # if we already know fractoselectix, skip the
                        # cross-validation
                        else:
                            FRACindex0 = fractoselectix*np.ones(
                                len(relix),
                                dtype=int)

                        # prepare output
                        # Author: Kendrick Kay
                        #FRACValue[]
                        FRACvalue[np.unravel_index(relix, FRACvalue.shape)] = fracstouse[
                            np.unravel_index(FRACindex0, fracstouse.shape)[0]]
                        for fracl in range(len(fracstouse)):
                            # print(f'model: {whmodel}, frac: {fracl}')
                            # indices of voxels that chose the fraclth frac
                            ii = np.flatnonzero(FRACindex0 == fracl)

                            # scale and offset to match the unregularized result
                            if autoscaletouse:
                                for vv in ii:
                                    X = np.c_[
                                        np.nan_to_num(results0[fracl][vv, :]),
                                        np.ones(numtrials)].astype(np.float32)
                                    # Notice the 0
                                    h = olsmatrix(X) @ results0[0][vv, :].T
                                    if h[0] < 0:
                                        h = np.asarray([1, 0])

                                    scaleoffset[relix[vv], :] = h
                                    modelmd[relix[vv], :] = X @ h

                            else:
                                scaleoffset = np.array([])
                                modelmd[relix[ii], :] = results0[fracl][ii, :]
                            indexings = np.unravel_index(relix[ii], R2.shape)
                            R2[indexings] = r20[fracl][ii]
                            R2run[relix[ii], :] = np.stack(r2run0[fracl])[:, ii].T


            # deal with dimensions
            modelmd = np.reshape(modelmd, [nx, ny, nz, numtrials])
            # deal with the broadcasting of meanvol into the n_conditions
            # of modelmd
            numnewdims = modelmd.ndim - meanvol.ndim
            slicing = [slice(None)] * meanvol.ndim + [np.newaxis] * numnewdims
            modelmd = (modelmd / np.abs(meanvol[tuple(slicing)])) * 100

            R2run = np.reshape(R2run,[nx, ny, nz, numruns])
            if scaleoffset.size > 0:
                scaleoffset = np.reshape(scaleoffset, [nx, ny, nz, 2])

            if fractoselectix is None:
                rrbadness = np.reshape(rrbadness, [nx, ny, nz, -1])

            # save to disk if desired
            if whmodel == 2:
                if params['wanthdf5'] == 1:
                    file0 = os.path.join(
                        outputdir,
                        'TYPEC_FITHRF_GLMDENOISE.hdf5'
                    )
                else:
                    file0 = os.path.join(
                        outputdir,
                        'TYPEC_FITHRF_GLMDENOISE.npy'
                    )

                outdict = {
                    'HRFindex': HRFindex,
                    'HRFindexrun': HRFindexrun,
                    'glmbadness': glmbadness,
                    'pcvoxels': pcvoxels,
                    'pcnum': pcnum,
                    'xvaltrend': xvaltrend,
                    'noisepool': noisepool,
                    'pcregressors': pcregressors,
                    'betasmd': modelmd,
                    'R2': R2,
                    'R2run': R2run,
                    'meanvol':  meanvol
                    }
            elif whmodel == 3:
                if params['wanthdf5'] == 1:
                    file0 = os.path.join(
                        outputdir,
                        'TYPED_FITHRF_GLMDENOISE_RR.hdf5'
                    )
                else:
                    file0 = os.path.join(
                        outputdir,
                        'TYPED_FITHRF_GLMDENOISE_RR.npy'
                    )

                outdict = {
                    'HRFindex': HRFindex,
                    'HRFindexrun': HRFindexrun,
                    'glmbadness': glmbadness,
                    'pcvoxels': pcvoxels,
                    'pcnum': pcnum,
                    'xvaltrend': xvaltrend,
                    'noisepool': noisepool,
                    'pcregressors': pcregressors,
                    'betasmd': modelmd,
                    'R2': R2,
                    'R2run': R2run,
                    'rrbadness': rrbadness,
                    'FRACvalue': FRACvalue,
                    'scaleoffset': scaleoffset,
                    'meanvol':  meanvol
                    }

            if params['wantfileoutputs'][whmodel] == 1:

                print(f'\n*** Saving results to {file0}. ***\n')

                if params['wanthdf5'] == 1:
                    hf = h5py.File(file0, 'w')
                    for k, v in outdict.items():
                        if isinstance(v, list):
                            v = np.array(v)
                        if v is not None:
                            hf.create_dataset(k, data=v)
                        else:
                            hf.create_dataset(k, data=h5py.Empty("f"))
                    hf.close()
                else:
                    np.save(file0, outdict)

            # figures?
            if wantfig:
                if whmodel == 2:
                    if xyz:
                        if noisepool is not None:
                            cmap = mpl.colormaps['gray'].resampled(256)
                            plt.imsave(
                                os.path.join(figuredir,'noisepool.png'),
                                np.uint8(255*make_image_stack(noisepool,[0, 1])),
                                cmap=cmap, vmin=0, vmax=255)

                        if pcvoxels is not None:
                            plt.imsave(
                                os.path.join(figuredir,'pcvoxels.png'),
                                np.uint8(255*make_image_stack(pcvoxels,[0, 1])),
                                cmap=cmap, vmin=0, vmax=255)

                    if xvaltrend is not None:
                        fig = plt.figure()
                        ax = fig.add_subplot(1, 1, 1)
                        ax.plot(range(params['n_pcs']+1), xvaltrend)
                        ax.scatter(pcnum, xvaltrend[pcnum])
                        ax.set(
                            xlabel='# GLMdenoise regressors',
                            ylabel='Cross-val performance (higher is better)')
                        plt.savefig(os.path.join(figuredir, 'xvaltrend.png'))
                        plt.close('all')

                if whmodel == 3:
                    if xyz:
                        cmap = mpl.colormaps['hot'].resampled(256)
                        plt.imsave(
                            os.path.join(figuredir,'typeD_R2.png'),
                            np.uint8(255*make_image_stack(R2,[0, 100])**0.5),
                            cmap=cmap, vmin=0, vmax=255)

                        for rr in range(R2run.shape[-1]):
                            plt.imsave(
                                os.path.join(figuredir,f'typeD_R2_run{rr+1:02}.png'),
                                np.uint8(255*make_image_stack(R2run[:, :, :, rr],[0, 100])**0.5),
                                cmap=cmap, vmin=0, vmax=255)

                        cmap = mpl.colormaps['copper'].resampled(256)
                        plt.imsave(
                            os.path.join(figuredir,'FRACvalue.png'),
                            np.uint8(255*make_image_stack(FRACvalue,[0, 1])),
                            cmap=cmap, vmin=0, vmax=255)

                # beta visualisation
                temp = squish(modelmd, 3)[onoffvizix, :]
                if betavizmx is None:
                    betavizmx = np.nanpercentile(np.abs(temp.flatten()),99)
                typemod='C' if whmodel==2 else 'D'
                filer = os.path.join(figuredir, f'betaviz_type{typemod}.png')
                cmap = cmapsign4(256)
                colormap_to_plot = cmaplookup(temp,-betavizmx,betavizmx,0,cmap)
                plt.imsave(filer, colormap_to_plot)

                if xyz:
                    # detric visualisation
                    temp = calcdmetric(modelmd, np.asarray(stimorder))
                    if drng is None:
                        drng = [np.nanmin(temp.flatten()), np.nanmax(temp.flatten())]
                    cmap = mpl.colormaps['hot'].resampled(256)
                    filer = os.path.join(figuredir,f'dmetric_type{typemod}.png')
                    plt.imsave(
                        filer,
                        np.uint8(255*make_image_stack(temp, drng)),
                        cmap=cmap, vmin=0, vmax=255)


            # preserve in memory if desired
            if params['wantmemoryoutputs'][whmodel] == 1:
                if whmodel == 2:
                    results['typec'] = {
                        'HRFindex': HRFindex,
                        'HRFindexrun': HRFindexrun,
                        'glmbadness': glmbadness,
                        'pcvoxels': pcvoxels,
                        'pcnum': pcnum,
                        'xvaltrend': xvaltrend,
                        'noisepool': noisepool,
                        'pcregressors': pcregressors,
                        'betasmd': modelmd,
                        'R2': R2,
                        'R2run': R2run,
                        'meanvol':  meanvol
                     }
                elif whmodel == 3:
                    results['typed'] = {
                        'HRFindex': HRFindex,
                        'HRFindexrun': HRFindexrun,
                        'glmbadness': glmbadness,
                        'pcvoxels': pcvoxels,
                        'pcnum': pcnum,
                        'xvaltrend': xvaltrend,
                        'noisepool': noisepool,
                        'pcregressors': pcregressors,
                        'betasmd': modelmd,
                        'R2': R2,
                        'R2run': R2run,
                        'rrbadness': rrbadness,
                        'FRACvalue': FRACvalue,
                        'scaleoffset': scaleoffset,
                        'meanvol':  meanvol
                     }

        print('*** All model types done ***\n')
        if not results:
            print('*** nothing selected to return ***\n')
        else:
            print('*** return model types in results ***\n')

        return results
