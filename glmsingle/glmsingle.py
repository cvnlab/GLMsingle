from __future__ import absolute_import, division, print_function
import os
import numpy as np
from tqdm import tqdm
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


__all__ = ["GLM_single"]
dir0 = os.path.dirname(os.path.realpath(__file__))


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
        conditions andan associated cross-validation scheme <params.xvalscheme>
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

       <chunknum> (optional) is the number of voxels that we will process at
         the same time. This number should be large in order to speed
         computation, but should not be so large that you run out of RAM.
         Default: 50000.

        <xvalscheme> (optional) is a cell vector of vectors of run indices,
         indicating the cross-validation scheme. For example, if we have 8
         runs, we could use [[1, 2], [3, 4], [5, 6], [7, 8]] which indicates
         to do 4 folds of cross-validation, first holding out the 1st and 2nd
         runs, then the 3rd and 4th runs, etc.
         Default: {[1] [2] [3] ... [n]} where n is the number of runs.

        <sessionindicator> (optional) is 1 x n (where n is the number of runs)
        with positive integers indicating the run groupings that are
        interpreted as "sessions". The purpose of this input is to allow for
        session-wise z-scoring of single-trial beta weights for the purposes
        of hyperparameter evaluation. Note that the z-scoring has effect only
        INTERNALLY: it is used merely to calculate the cross-validation
        performance and the associated hyperparameter selection; the outputs
        of this function do not reflect z-scoring, and the user may wish to
        apply z-scoring. Default: 1*ones(1,n) which means to interpret allruns
        as coming from the same session.

       *** I/O FLAGS ***

        <wantfileoutputs> (optional) is a logical vector [A B C D] indicating
         which of the four model types to save to disk (assuming that they
         are computed).
         A = 0/1 for saving the results of the ONOFF model
         B = 0/1 for saving the results of the FITHRF model
         C = 0/1 for saving the results of the FITHRF_GLMDENOISE model
         D = 0/1 for saving the results of the FITHRF_GLMDENOISE_RR model
         Default: [1 1 1 1] which means save all computed results to disk.

        <wantmemoryoutputs> (optional) is a logical vector [A B C D] indicating
         which of the four model types to return in the output <results>. The
         user must be careful with this, as large datasets can require a lot of
         RAM. If you do not request the various model types, they will be
         cleared from memory (but still potentially saved to disk).
         Default: [0 0 0 1] which means return only the final type-D model.

        *** GLM FLAGS ***

        <extraregressors> (optional) is time x regressors or a cell vector
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
         is to generate a canonical HRF (see getcanonicalhrf.m).
         Note that the HRF supplied in <hrftoassume> is used in only two
         instances:
         (1) it is used for the simple ONOFF type-A model, and (2) if the
             user sets <wantlibrary> to 0, it is also used for the type-B,
             type-C, and type-D models.

        <hrflibrary> (optional) is time x H with H different HRFs to choose
         from for the library-of-HRFs approach. We automatically normalize
         each HRF to peak at 1.
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

        <brainthresh> (optional) is [A B] where A is a percentile for voxel
         intensity values and B is a fraction to apply to the percentile. These
         parameters are used in the selection of the noise pool.
         Default: [99 0.1].

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

        <fracs> (optional) is a vector of fractions that are greater than 0
         and less than or equal to 1. We automatically sort in descending
         order and ensure the fractions are unique. These fractions indicate
         the regularization levels to evaluate using fractional ridge
         regression (fracridge) and cross-validation.
         Default: fliplr(.05:.05:1).
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
        for key in params.keys():
            if key not in default_params.keys():
                raise ValueError(f"""
                Input parameter not recognized: '{key}'
                Possible input parameters are:\n{list(default_params.keys())}
                """)

        self.params = params

    def fit(self, design, data, stimdur, tr, outputdir=None):
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

        <outputdir> (optional) is a directory to which files will be written.
         (If the directory does not exist, we create it; if the directory
         already exists, we delete its contents so we can start fresh.) If you
         set <outputdir> to None, we will not create a directory and no files
         will be written.
         Default is 'GLMestimatesingletrialoutputs' (created in the current
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

        History:
        [MATLAB]
        - 2020/08/22 - Implement params.sessionindicator. Also, now the
                        cross-validation units now reflect
                        the "session-wise z-scoring" hyperparameter selection
                        approach; thus, the cross-
                        validation units have been CHANGED relative to prior
                        analyses!
        - 2020/05/14 - Version 1.0 released!
                        (Tweak some documentation; output more results; fix a
                        small bug (params.fracs(1)~=1).)
        - 2020/05/12 - Add pcvoxels output.
        - 2020/05/12 - Initial version. Beta version. Use with caution.
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
        numtimepoints = [data[run_i].shape[1] for run_i in range(numruns)]

        if xyz:
            numvoxels = np.prod(xyz)
        else:
            numvoxels = self.data[0].shape[0]

        # inputs
        if 'xvalscheme' not in params:
            params['xvalscheme'] = np.arange(numruns)

        if 'sessionindicator' not in params:
            params['sessionindicator'] = np.ones((1, numruns)).astype(int)

        if 'maxpolydeg' not in params:
            params['maxpolydeg'] = [
                np.arange(
                    round(
                        ((self.data[r].shape[1]*tr)/60)/2) + 1
                    ) for r in np.arange(numruns)]

        if 'hrftoassume' not in params:
            params['hrftoassume'] = normalisemax(
                getcanonicalhrf(stimdur, tr),
                dim='global'
            )

        if 'hrflibrary' not in params:
            params['hrflibrary'] = getcanonicalhrflibrary(stimdur, tr).T

        # deal with length issues and other miscellaneous things
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

        if xyz and outputdir is not None:
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

        if np.any(params['wantfileoutputs']):
            errm = 'specify an <outputdir> in order to get file outputs'
            np.testing.assert_equal(
                type(outputdir),
                str,
                err_msg=errm)

        # deal with special library stuff
        if params['wantlibrary'] == 0:
            params['hrflibrary'] = params['hrftoassume'].reshape(-1,1)

        # calc
        # if the data was passed as 3d, unpack xyz
        if xyz:
            nx, ny, nz = xyz

        nh = params['hrflibrary'].shape[1]

        # figure out chunking scheme
        chunks = chunking(
            np.arange(numvoxels),
            int(np.ceil(numvoxels/np.ceil(numvoxels/params['chunklen']))))

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

        # save to disk if desired
        if params['wantfileoutputs'][whmodel] == 1:
            file0 = os.path.join(outputdir, 'TYPEA_ONOFF.npy')
            print(f'\n*** Saving results to {file0}. ***\n')
            np.save(file0, onoffR2, meanvol, xyz)

        # figures
        if wantfig:
            plt.imshow(
                make_image_stack(onoffR2.reshape(xyz)),
                vmin=0,
                vmax=100,
                cmap='hot'
            )
            ax = plt.gca()
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            plt.colorbar()
            plt.savefig(os.path.join(outputdir, 'onoffR2.png'))
            plt.close('all')
            plt.imshow(make_image_stack(meanvol.reshape(xyz)), cmap='gray')
            ax = plt.gca()
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            plt.colorbar()
            plt.savefig(os.path.join(outputdir, 'meanvol.png'))
            plt.close('all')

        # preserve in memory if desired, and then clean up
        if params['wantmemoryoutputs'][whmodel] == 1:
            results['typea'] = {
                'onoffR2': onoffR2,
                'meanvol': meanvol
            }

        # DETERMINE THRESHOLDS
        if wantfig:
            thresh = findtailthreshold(
                onoffR2.flatten(),
                os.path.join(outputdir, 'onoffR2hist.png'))[0]
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
            HRFindex = np.ones(numvoxels)  # easy peasy

        else:

            # define
            whmodel = 1

            # initialize
            FitHRFR2 = np.zeros(
                (numvoxels, nh),
                dtype=np.float32)
            # X x Y x Z x HRFs with R2 values (all runs)
            FitHRFR2run = np.zeros(
                (numvoxels, numruns, nh),
                dtype=np.float32)
            # X x Y x Z x runs x HRFs with R2 separated by runs
            modelmd = np.zeros(
                (numvoxels, numtrials),
                dtype=np.float32)
            # X x Y x Z x trialbetas
            optC = {
                    'extra_regressors': params['extra_regressors'],
                    'maxpolydeg': params['maxpolydeg'],
                    'wantpercentbold': params['wantpercentbold'],
                    'suppressoutput': 1
            }

            # loop over chunks
            print('*** FITTING TYPE-B MODEL (FITHRF) ***\n')
            for z in tqdm(np.arange(len(chunks)), desc='chunks'):

                this_chunk = chunks[z]
                n_inchunk = len(this_chunk)

                data_chunk = [datx[this_chunk, :] for datx in self.data]
                # do the fitting and accumulate all the betas
                modelmd0 = np.zeros(
                    (n_inchunk, numtrials, nh),
                    dtype=np.float32)
                # someXYZ x trialbetas x HRFs
                for p in np.arange(nh):
                    results0 = glm_estimatemodel(
                        designSINGLE,
                        data_chunk,
                        stimdur,
                        tr,
                        'assume',
                        params['hrflibrary'][:, p],
                        0,
                        optC
                    )[0]

                    FitHRFR2[this_chunk, p] = results0['R2']
                    FitHRFR2run[this_chunk, :, p] = np.stack(
                        results0['R2run']).T
                    modelmd0[:, :, p] = results0['betasmd']

                # keep only the betas we want
                # ii shape someXYZ
                ii = np.argmax(FitHRFR2[this_chunk, :], axis=1)

                # tile it as someXYZ x numtrials
                iiflat = np.tile(
                    ii[:, np.newaxis], numtrials).flatten()

                # someXYZ x numtrials x nh
                modelmd0 = np.reshape(
                    modelmd0, [n_inchunk*numtrials, -1])

                # XYZ by n_trials
                modelmd[this_chunk, :] = modelmd0[np.arange(
                    n_inchunk*numtrials), iiflat].reshape(n_inchunk, -1)

            R2 = np.max(FitHRFR2, axis=1)  # R2 is XYZ
            HRFindex = np.argmax(FitHRFR2, axis=1)  # HRFindex is XYZ

            # also, use R2 from each run to select best HRF
            HRFindexrun = np.argmax(FitHRFR2run, axis=2).flatten()

            FitHRFR2run = np.reshape(
                FitHRFR2run,
                (numvoxels*numruns, nh))

            # using each voxel's best HRF, what are the corresponding R2run
            # values?
            R2run = FitHRFR2run[np.arange(
                numvoxels*numruns),
                HRFindexrun].reshape([numvoxels, -1])

            # FIT TYPE-B MODEL (LSS) INTERLUDE BEGIN

            # if params.wantlss, we have to use the determined HRFindex and
            # re-fit the entire dataset using LSS estimation. this will
            # simply replace 'modelmd' with a new version.
            # so that's what we have to do here.

            if params['wantlss']:

                # initalize
                modelmd = np.zeros((numvoxels, numtrials), dtype=np.float32)
                # X*Y*Z x trialbetas  [the final beta estimates]

                # loop over chunks
                print(
                    '*** FITTING TYPE-B MODEL'
                    '(FITHRF but with LSS estimation) ***\n')

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
                            HRFindex[this_chunk] == hh)

                        data0 = \
                            [x[this_chunk, :][goodix, :] for x in self.data]

                        # calculate the corresponding indices relative to the
                        # full volume
                        temp = np.zeros(HRFindex.shape)
                        temp[this_chunk] = 1
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

            # FIT TYPE-B MODEL (LSS) INTERLUDE END
            
            # if user provided XYZ, reshape disk/memory output fields into XYZ
            if xyz:
                results_out = {'FitHRFR2': np.reshape(FitHRFR2, [nx, ny, nz, nh]),
                     'FitHRFR2run': np.reshape(FitHRFR2run, [nx, ny, nz, numruns, nh]),
                     'HRFindex': np.reshape(HRFindex, [nx, ny, nz]),
                     'HRFindexrun': np.reshape(HRFindexrun, [nx, ny, nz, numruns]),
                     'R2': np.reshape(R2, [nx, ny, nz]),
                     'R2run': np.reshape(R2run, [nx, ny, nz, numruns]),
                     'betasmd': np.reshape(modelmd, [nx, ny, nz, numtrials]),
                     'meanvol':  meanvol
                     }
            else:
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
                file0 = os.path.join(outputdir, 'TYPEB_FITHRF.npy')
                print(f'\n*** Saving results to {file0}. ***\n')
                np.save(
                    file0,
                    results_out
                )

            # figures?
            if wantfig:
                """ TODO
                port normalizerange.m and add to makeimstack
                """
                plt.imshow(
                    make_image_stack(HRFindex.reshape(xyz)),
                    vmin=0,
                    vmax=nh)
                ax = plt.gca()
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                plt.colorbar()
                plt.savefig(os.path.join(outputdir, 'HRFindex.png'))
                plt.close('all')

            # preserve in memory if desired, and then clean up
            if params['wantmemoryoutputs'][whmodel] == 1:
                results['typeb'] = results_out

        # COMPUTE GLMDENOISE REGRESSORS

        # if the user does not want to perform GLMdenoise,
        # we can just skip all of this
        if params['wantglmdenoise'] == 0:

            # just create placeholders
            pcregressors = []
            noisepool = []

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
                noise_pool = np.transpose(drun)[:, noisepool]

                # project out polynomials from the data
                # this projects out polynomials
                pmatrix = make_polynomial_matrix(
                    numtimepoints[run_i],
                    params['maxpolydeg'][run_i])

                polymatrix.append(
                    make_projection_matrix(pmatrix))

                noise_pool = polymatrix[run_i].astype(np.float32) @ noise_pool

                noise_pool = normalize(noise_pool, axis=0)

                noise_pool = noise_pool @ noise_pool.T
                u = np.linalg.svd(noise_pool)[0]
                u = u[:, :params['n_pcs']+1]
                u = u / np.std(u, 0)
                pcregressors.append(u.astype(np.float32))

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
                        HRFindex[this_chunk] == hh)

                    data0 = \
                        [x[this_chunk, :][goodix, :] for x in self.data]

                    # calculate the corresponding indices relative to the
                    # full volume
                    temp = np.zeros(HRFindex.shape)
                    temp[this_chunk] = 1
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
                                if not params['extra_regressors'] or \
                                     not params['extra_regressors'][rr]:

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
                    len(ix2) > 0, True, err_msg='no voxels are in pcR2cutoffmask')

                ix3 = np.argsort(onoffR2[ix2])[::-1]
                num = np.min([100, len(ix2)])
                ix = ix2[ix3[range(num)]]

            # NOTE: sign flip so that high is good
            xvaltrend = -np.median(glmbadness[ix, :], axis=0)
            np.testing.assert_equal(np.all(np.isfinite(xvaltrend)), True)

            # create for safe-keeping
            pcvoxels = np.zeros((numvoxels), dtype=bool)
            pcvoxels[ix] = 1

            # choose number of PCs
            # this is the performance curve that starts
            # at 0 (corresponding to 0 PCs)
            pcnum = select_noise_regressors(xvaltrend, params['pcstop'])

            # deal with dimensions
            # NOTE skip for now
            # glmbadness = np.reshape(glmbadness, [nx, ny, nz, -1])

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
            R2 = np.zeros(numvoxels, dtype=np.float32)

            # XYZ x runs [the R2 separated by runs for the optimal frac]
            R2run = np.zeros((numvoxels, numruns), dtype=np.float32)

            # XYZ [best fraction]
            FRACvalue = np.zeros(numvoxels, dtype=np.float32)

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
                        HRFindex[this_chunk] == hh)

                    data0 = \
                        [x[this_chunk, :][goodix, :] for x in self.data]

                    # calculate the corresponding indices relative to the
                    # full volume
                    temp = np.zeros(HRFindex.shape)
                    temp[this_chunk] = 1
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
                                if not params['extra_regressors'] or \
                                     not params['extra_regressors'][run_i]:

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

                    FRACvalue[relix] = fracstouse[
                        np.unravel_index(FRACindex0, fracstouse.shape)[0]]
                    for fracl in range(len(fracstouse)):
                        # print(f'model: {whmodel}, frac: {fracl}')
                        # indices of voxels that chose the fraclth frac
                        ii = np.flatnonzero(FRACindex0 == fracl)

                        # scale and offset to match the unregularized result
                        if autoscaletouse:
                            for vv in ii:
                                X = np.c_[
                                    results0[fracl][vv, :],
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
                        R2[relix[ii]] = r20[fracl][ii]
                        R2run[relix[ii], :] = np.stack(r2run0[fracl])[:, ii].T

            # deal with dimensions
            modelmd = (modelmd / np.abs(meanvol)[:, np.newaxis]) * 100
            
            if xyz:
                modelmd = np.reshape(modelmd, [nx, ny, nz, numtrials])
                R2 = np.reshape(R2, [nx, ny, nz])
                R2run = np.reshape(R2run, [nx, ny, nz, numruns])
                if scaleoffset.size > 0:
                    scaleoffset = np.reshape(scaleoffset, [nx, ny, nz, 2])

                if fractoselectix is None:
                    rrbadness = np.reshape(rrbadness, [nx, ny, nz, -1])

            # save to disk if desired
            if whmodel == 2:
                file0 = os.path.join(
                    outputdir, 'TYPEC_FITHRF_GLMDENOISE.npy')
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
                file0 = os.path.join(
                    outputdir, 'TYPED_FITHRF_GLMDENOISE_RR.npy')
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
                np.save(file0, outdict)

            # figures?
            if wantfig:
                if whmodel == 2:
                    plt.imshow(
                        make_image_stack(noisepool.reshape(xyz)),
                        vmin=0,
                        vmax=1,
                        cmap='gray'
                    )
                    ax = plt.gca()
                    ax.axes.xaxis.set_ticklabels([])
                    ax.axes.yaxis.set_ticklabels([])
                    plt.colorbar()
                    plt.savefig(os.path.join(outputdir, 'noisepool.png'))
                    plt.close('all')
                    plt.imshow(
                        make_image_stack(pcvoxels.reshape(xyz)),
                        vmin=0,
                        vmax=1,
                        cmap='gray'
                    )
                    ax = plt.gca()
                    ax.axes.xaxis.set_ticklabels([])
                    ax.axes.yaxis.set_ticklabels([])
                    plt.colorbar()
                    plt.savefig(os.path.join(outputdir, 'pcvoxels.png'))
                    plt.close('all')

                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    ax.plot(range(params['n_pcs']+1), xvaltrend)
                    ax.scatter(pcnum, xvaltrend[pcnum])
                    ax.set(
                        xlabel='# GLMdenoise regressors',
                        ylabel='Cross-val performance (higher is better)')
                    plt.savefig(os.path.join(outputdir, 'xvaltrend.png'))
                    plt.close('all')

                if whmodel == 3:
                    plt.imshow(
                        make_image_stack(R2),
                        vmin=0,
                        vmax=100,
                        cmap='hot'
                    )
                    ax = plt.gca()
                    ax.axes.xaxis.set_ticklabels([])
                    ax.axes.yaxis.set_ticklabels([])
                    plt.colorbar()
                    plt.savefig(os.path.join(outputdir, 'typeD_R2.png'))
                    plt.close('all')
                    plt.imshow(
                        make_image_stack(FRACvalue.reshape(xyz)),
                        vmin=0,
                        vmax=1,
                        cmap='copper'
                    )
                    ax = plt.gca()
                    ax.axes.xaxis.set_ticklabels([])
                    ax.axes.yaxis.set_ticklabels([])
                    plt.colorbar()
                    plt.savefig(os.path.join(outputdir, 'FRACvalue.png'))

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
