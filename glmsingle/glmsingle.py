from __future__ import absolute_import, division, print_function
import os
import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

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

__all__ = ["GLM_single"]
dir0 = os.path.dirname(os.path.realpath(__file__))


class GLM_single():

    def __init__(self, params=None):
        """glm singletrial denoise constructor

        This function computes up to four model outputs (called type-A (ONOFF),
        type-B (FITHRF), type-C (FITHRF_GLMDENOISE), and type-D
        (FITHRF_GLMDENOISE_RR)),and either saves the model outputs to disk,
        or returns them in <results>, or both, depending on what the user
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
        
        *** DIAGNOSTIC MODEL (FIR) FLAGS ***
        
        <firdelay> (optional) is the total time duration in seconds over which to estimate 
         the run-wise FIR model (where we assume an ONOFF design matrix in which all 
         conditions are collapsed together). Default: 30.
        
        <firpct> (optional) is a percentile threshold. We average the FIR model 
         R2 values across runs and then select voxels that pass this threshold.
         These voxels are used for the FIR timecourse summaries. Default: 99.

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

        We return model results in the output variable <results>.
        These results are saved to disk in files called 'TYPEA...',
        'TYPEB...', and so on. There are various outputs for each 
        of the four model types:

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
                    
        Note that not all outputs exist for every model type.


        We also return design-related results in the output variable <resultsdesign>.
        These results are saved to disk to a file called 'DESIGNINFO...'.
        The outputs include:

        <design> is as specified by the user (with possibly some minor regularization)

        <stimdur> is as specified by the user

        <tr> is as specified by the user

        <opt> is as specified by the user (with possibly some minor regularization)

        <designSINGLE> is a single-trial design matrix corresponding to <design>

        <stimorder> is a row vector indicating which condition (1-indexed)
        each trial (in chronological order) belongs to

        <numtrialrun> is a row vector with the number of trials in each run

        <condcounts> is a row vector with the number of trials
        associated with each condition

        <condinruns> is a row vector with the number of runs that
        each condition shows up in

        <endbuffers> is a row vector with the number of seconds after the 
        last trial onset in each run


        We also return diagnostic FIR-related results --- these are saved
        to disk to a file called 'RUNWISEFIR...'. The outputs include:

        <firR2> is the R2 of the FIR model for each run (X x Y x Z x run).

        <firtcs> is the estimated FIR timecourse for each run (X x Y x Z x 1 x time x run).
        Note that the first time point is coincident with trial onset and the
        time points are at the sampling rate corresponding to <tr>.

        <firavg> is the estimated FIR timecourse in each run (time x run).
        These are obtained by calculating the median timecourse
        across the "best" voxels (see opt.firpct).

        <firgrandavg> is the average of <firavg> across runs (time x 1).
        

        *** FIGURES: ***

        If <outputdir> is set appropriately, we will generate a variety of useful
        figures and save them to disk. Note that if you provide your data in 3D
        format (e.g. X x Y x Z x T), we will be able to write out a number of
        additional useful slice inspections that you will not get if you provide
        your data in collapsed format (e.g. XYZ x T).

        betaviz_type[B,C,D].png - an image visualization of betas obtained
        under the type-B, type-C, and type-D models. The matrix dimensions
        are 1,000 voxels x trials. We choose 1,000 voxels equally spaced in 
        descending order from the 100th to 75th percentiles of
        the R^2 values produced by the ONOFF model. The colormap is
        cmapsign4.py (blueish colors to black to reddish colors) from 
        -X to X where X is the 99th percentile of the absolute value 
        of the betas in the first model that is actually
        computed (typically, this will be the type-B model). 

        dmetric_type[B,C,D].png - a "deviation from zero" metric calculated
        based on the betas obtained under the type-B, type-C, and type-D models.
        We use a hot colormap ranging between the min and max of the values
        obtained for the first model that is computed (typically, this will be 
        the type-B model).

        FRACvalue.png - chosen fractional ridge regression value
        (copper colormap between 0 and 1)

        HRFindex.png - 1-index of chosen HRF
        (jet colormap between 1 and the number of HRFs in the library)

        meanvol.png - simply the mean across all data volumes

        noisepool.png - voxels selected for the noise pool (white means selected)

        onoffR2_vs_HRFindex.png - scatter plot of the R^2 of the ONOFF model 
        against the chosen HRF index. All voxels are shown. A small amount of 
        jitter is added to the HRF index in order to aid visibility.

        onoffR2.png - R^2 of the ONOFF model (sqrt hot colormap between 0% and 100%)

        onoffR2hist.png - depicts the finding of an automatic threshold on the ONOFF
        model R^2. This is used in determining the noise pool (but can be 
        overridden by opt.brainR2).

        pcvoxels.png - voxels used to summarize GLMdenoise cross-validation results
        (white means selected)

        runwiseFIR_R2_runXX.png - for each run, the R^2 of the diagnostic FIR model
        (sqrt hot colormap between 0% and 100%)

        runwiseFIR_R2_runavg.png - simply the average of the R^2 across runs

        runwiseFIR.png - Upper left shows run-wise FIR estimates. The estimates reflect
        the mean FIR timecourse averaged across a set of "best" voxels (see opt.firpct). 
        The mean of these mean FIR timecourses across runs is indicated by the thick 
        red line. Upper right shows FIR amplitudes at the peak time observed in the
        grand mean timecourse (indicated by the dotted black line). Bottom left shows 
        the HRFs in the library as colored lines and the "assumed HRF" as a thick 
        black line. Note that these reflect any user-specified customization (as 
        controlled via opt.hrftoassume and opt.hrflibrary).

        typeD_R2_runXX.png - the R^2 of the final type-D model computed using data
        from individual runs (sqrt hot colormap between 0% and 100%)

        typeD_R2.png - the R^2 of the final type-D model (using all data) 

        xvaltrend.png - shows the cross-validation performance for different numbers
        of GLMdenoise regressors. Note that the y-axis units are correct but not 
        easy to interpret.

        """

        # DEAL WITH INPUTS
        params = self.params

        # initialise return
        results = {}

        # xyz can either be a tuple of dimensions x y z
        # or a boolean indicating that data was 2D
        data, design, xyz, numcond = check_inputs(data, design)

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

        # additional check for the file format
        if 'wanthdf5' not in params:
            params['wanthdf5'] = 0

        if 'sessionindicator' not in params:
            params['sessionindicator'] = np.ones((1, numruns)).astype(int)

        if 'maxpolydeg' not in params:
            params['maxpolydeg'] = [
                np.arange(
                    alt_round(
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

        if xyz and figuredir is not None:
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
            
        # calculate number of trials for each condition
        condcounts = []  # 1 x cond with counts
        for p in range(numcond):
            condcounts.append(np.sum(np.asarray(stimorder) == p))

        # calculate for each condition, how many runs it shows up in
        condinruns = []  # 1 x cond with counts
        for p in range(numcond):
            condinruns.append(np.sum([np.sum(np.array(x) == p) > 0 for x in stimix]))

        # calculate buffer at the end of each run
        endbuffers = []  # 1 x runs with number of seconds
        for p in range(len(design)):
            temp = np.nonzero(np.sum(design[p], axis=1))[0] + 1  # 1-indices of when trials happen
            temp = design[p].shape[0] - temp[-1]  # number of volumes AFTER last trial onset 
            endbuffers.append(temp * tr)  # number of seconds AFTER last trial onset for which we have data

        # do some diagnostics
        print("*** DIAGNOSTICS ***:")
        print(f"There are {len(design)} runs.")
        print(f"The number of conditions in this experiment is {numcond}.")
        print(f"The stimulus duration corresponding to each trial is {stimdur:.2f} seconds.")
        print(f"The TR (time between successive data points) is {tr:.2f} seconds.")
        print(f"The number of trials in each run is: {numtrialrun}.")
        print(f"The number of trials for each condition is: {condcounts}.")
        print(f"For each condition, the number of runs in which it appears: {condinruns}.")
        print(f"For each run, how much ending buffer do we have in seconds? {endbuffers}.")
                
        # issue warning if trials get too close to the end
        if any(endbuffer < 8 for endbuffer in endbuffers):
            print('Warning: You have specified trial onsets that occur less than 8 seconds from the end of at least one of the runs. This may cause estimation problems! As a solution, consider simply omitting specification of these ending trials from the original design matrix.')

        # construct a nice output dict for this design-related stuff
        varstoinsert = ['design', 'stimdur', 'tr', 'params', 'designSINGLE', 'stimorder', 'numtrialrun', 'condcounts', 'condinruns', 'endbuffers']
        resultsdesign = {}
        for var in varstoinsert:
            resultsdesign[var] = locals()[var]

        if isinstance(outputdir, str):
            file0 = os.path.join(outputdir, 'DESIGNINFO.npy')
            print(f"*** Saving design-related results to {file0}. ***")
            np.save(file0, resultsdesign, allow_pickle=True)
            
            
        # FIT DIAGNOSTIC RUN-WISE FIR MODEL

        # The approach:
        # (1) Every stimulus is treated as the same.
        # (2) We fit an FIR model up to 30 s.
        # (3) Each run is fit completely separately.

        # collapse all conditions and fit each run separately
        skip = False
        
        if skip is False:
            print("*** FITTING DIAGNOSTIC RUN-WISE FIR MODEL ***")
            design0 = [np.sum(x, axis=1)[:, np.newaxis] for x in self.design]
            
            firR2 = [] # X x Y x Z x runs (R2 of FIR model for each run)
            firtcs = [] # X x Y x Z x 1 x time x runs (FIR timecourse for each run)
            glm_params = {
                    'extra_regressors': params['extra_regressors'],
                    'maxpolydeg': params['maxpolydeg'],
                    'wantpercentbold': params['wantpercentbold'],
                    'suppressoutput': 1
                }

            for p, data_p in enumerate(self.data):

                # fit the model for each run separately
                results0, cache0 = glm_estimatemodel(design0[p], data_p, stimdur, tr, 'fir', int(np.floor(params['firdelay'] / tr)),
                                              0, glm_params)
                firR2.append(results0['R2'])
                firtcs.append(results0['betasmd'])

            del results0  # clear results0

            # stack arrays along the appropriate dimensions
            firR2 = np.stack(firR2, axis=-1)
            firtcs = np.stack(firtcs, axis=-1)

            # calculate the mean R2 and threshold
            firR2mn = np.mean(firR2, axis=-1)
            firthresh = np.nanpercentile(firR2mn.ravel(), params['firpct'])
            firix = np.where(firR2mn > firthresh)

            # calculate timecourse averages
            firavg = []
            for rr, data_rr in enumerate(self.data):
                temp = firtcs[..., rr].reshape(-1, firtcs.shape[-2])[firix]
                firavg.append(np.median(temp, axis=0))

            firavg = np.stack(firavg, axis=-1)
            firgrandavg = np.mean(firavg, axis=-1)
            
        if wantfig:
            # Prepare the figure
            fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))

            # Subplot 1
            cmap0 = plt.cm.turbo(np.linspace(0, 1, len(self.data)))
            h = []
            legendlabs = []

            for rr in range(len(self.data)):
                h_rr, = ax[0, 0].plot(np.arange(0, tr * (firavg.shape[0]), tr), firavg[:, rr], 'o-', color=cmap0[rr, :], linewidth = 1, markerfacecolor='none')
                h.append(h_rr)
                legendlabs.append(f'Run {rr + 1}')

            h_avg, = ax[0, 0].plot(np.arange(0, tr * (len(firgrandavg)), tr), firgrandavg, 'r-', linewidth=2)
            
            h.append(h_avg)
            legendlabs.append('Run Avg')
            mxix = np.argmax(firgrandavg)
            ax[0, 0].axhline(0, color='k', linestyle='-')
            ax[0, 0].axvline(mxix * tr, color='k', linestyle=':')
            # Get the current axis limits
            xmin, xmax = ax[0, 0].get_xlim()
            # Update the lower limit to be -2/3 of the upper limit
            ax[0, 0].set_xlim(xmin=xmin, xmax=1.5*xmax)
            #ax[0, 0].set_xlim([0, len(firgrandavg) + int(0.5 * len(firgrandavg)) - 1])
            ax[0, 0].set_xlabel('Time from trial onset (s)')
            ax[0, 0].set_ylabel('BOLD (%)')
            ax[0, 0].legend(h, legendlabs, loc='upper right', fontsize = 5.5)

            # Subplot 2
            ax[0, 1].bar(np.arange(len(self.data)), firavg[mxix, :])
            ax[0, 1].set_xticks(np.arange(len(self.data)))
            ax[0, 1].set_xlabel('Run number')
            ax[0, 1].set_ylabel('BOLD at peak time (%)')
            # Get the current axis limits
            ymin, ymax = ax[0, 1].get_ylim()

            # Update the lower limit to be -2/3 of the upper limit
            ax[0, 1].set_ylim(ymin=-2/3 * ymax, ymax=ymax)

            # Subplot 3
            cmap0 = plt.cm.turbo(np.linspace(0, 1, nh))
            h = []
            legendlabs = []

            for hh in range(nh):
                h_hh, = ax[1, 0].plot(np.arange(0, tr * (params['hrflibrary'].shape[0]), tr), params['hrflibrary'][:, hh], '-', color=cmap0[hh, :], linewidth = 1)
                h.append(h_hh)
                legendlabs.append(f'HRFindex{hh + 1}')

            h_assume, = ax[1, 0].plot(np.arange(0, tr * (len(params['hrftoassume'])), tr), params['hrftoassume'], 'k-', linewidth=2)
            legendlabs.append('HRFassume')
            h.append(h_assume)
            ax[1, 0].set_xlim(ax[0, 0].get_xlim())
            ax[1, 0].axhline(0, color='k', linestyle='-')
            ax[1, 0].set_xlabel('Time from trial onset (s)')
            ax[1, 0].set_ylabel('BOLD (a.u.)')
            ax[1, 0].legend(h, legendlabs, loc='upper right', fontsize = 5.5)
            
            # Remove top and right spines for subplots
            for a in [ax[0, 0], ax[0, 1], ax[1, 0]]:
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)

            # Remove all axes and boxes for the bottom right subplot
            ax[1, 1].axis('off')
            
            # Save the figure
            plt.savefig(f'{figuredir}/runwiseFIR.png')                
            plt.show()
            
            for rr in range(firR2.shape[-1]):
                plot_firR2 = firR2[...,rr].reshape(xyz)
                img_data = make_image_stack(plot_firR2)
                img_data_normalized = (img_data / 100) ** 0.5
                plt.imshow(img_data_normalized, cmap='hot', norm=Normalize(vmin=0, vmax=1), interpolation='none')
                plt.axis('off')
                plt.savefig(f'{figuredir}/runwiseFIR_R2_run{rr:02d}.png', bbox_inches='tight')
                plt.close()

            img_data_avg = make_image_stack(firR2mn.reshape(xyz))
            img_data_avg_normalized = (img_data_avg / 100) ** 0.5
            plt.imshow(img_data_avg_normalized, cmap='hot', norm=Normalize(vmin=0, vmax=1), interpolation='none')
            plt.axis('off')
            plt.savefig(f'{figuredir}/runwiseFIR_R2_runavg.png', bbox_inches='tight')
            plt.close()
                        
        if isinstance(outputdir, str):
            file0 = os.path.join(outputdir, 'RUNWISEFIR.npy')
            print(f'*** Saving FIR results to {file0}. ***')
            np.save(file0, {'firR2': firR2, 
                            'firtcs': firtcs, 
                            'firavg': firavg, 
                            'firgrandavg': firgrandavg}, allow_pickle=True)
                                
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
        
        # Determine onoffvizix for beta inspection
        onoffvizix = np.argsort(np.nan_to_num(onoffR2.ravel(), nan=-np.inf))  # Ascending order; ensure NaN is treated as -Inf
        step = -0.25/1000
        idx = np.floor(len(onoffvizix) * np.arange(1,0.75,step)).astype(int) - 1
        onoffvizix = onoffvizix[idx] # 1000 voxels from 100th percentile to 75th percentile
        
        # save to disk if desired
        if params['wantfileoutputs'][whmodel] == 1:
            if params['wanthdf5'] == 1:
                file0 = os.path.join(outputdir, 'TYPEA_ONOFF.hdf5')
            else:
                file0 = os.path.join(outputdir, 'TYPEA_ONOFF.npy')

            print(f'\n*** Saving results to {file0}. ***\n')
            # if user provided XYZ, reshape disk/memory output fields into XYZ
            if xyz:
                results_out = {
                    'onoffR2': np.reshape(onoffR2, [nx, ny, nz]),
                    'meanvol': np.reshape(meanvol, [nx, ny, nz]),
                    'betasmd': np.reshape(betasmd, [nx, ny, nz])
                    }
            else:
                results_out = {
                    'onoffR2': onoffR2,
                    'meanvol': meanvol,
                    'betasmd': betasmd
                }
            if params['wanthdf5'] == 1:
                hf = h5py.File(file0, 'w')
                for k, v in results_out.items():
                    hf.create_dataset(k, data=v)
                hf.close()
            else:
                np.save(file0, results_out)

        # figures
        if wantfig:
            if xyz:
                # only plot this if data was provided as a volume.
                plt.imshow(
                    make_image_stack(onoffR2.reshape(xyz)),
                    vmin=0,
                    vmax=100,
                    cmap='hot',
                    interpolation='none'
                )
                ax = plt.gca()
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                plt.colorbar()
                plt.savefig(os.path.join(figuredir, 'onoffR2.png'))
                plt.close('all')
                plt.imshow(make_image_stack(meanvol.reshape(xyz)), cmap='gray',interpolation='none')
                ax = plt.gca()
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                plt.colorbar()
                plt.savefig(os.path.join(figuredir, 'meanvol.png'))
                plt.close('all')
                
        # preserve in memory if desired, and then clean up
        if params['wantmemoryoutputs'][whmodel] == 1:
            if xyz:
                results['typea'] = {
                    'onoffR2': onoffR2.reshape(xyz),
                    'meanvol': meanvol.reshape(xyz),
                    'betasmd': betasmd.reshape(xyz)
                }
            else:
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
                results_out = {
                    'FitHRFR2': np.reshape(FitHRFR2, [nx, ny, nz, nh]),
                    'FitHRFR2run': np.reshape(FitHRFR2run, [nx, ny, nz, numruns, nh]),
                    'HRFindex': np.reshape(HRFindex, [nx, ny, nz]),
                    'HRFindexrun': np.reshape(HRFindexrun, [nx, ny, nz, numruns]),
                    'R2': np.reshape(R2, [nx, ny, nz]),
                    'R2run': np.reshape(R2run, [nx, ny, nz, numruns]),
                    'betasmd': np.reshape(modelmd, [nx, ny, nz, numtrials]),
                    'meanvol':  np.reshape(meanvol, [nx, ny, nz])
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

                if params['wanthdf5'] == 1:
                    file0 = os.path.join(outputdir, 'TYPEB_FITHRF.hdf5')
                else:
                    file0 = os.path.join(outputdir, 'TYPEB_FITHRF.npy')

                print(f'\n*** Saving results to {file0}. ***\n')

                if params['wanthdf5'] == 1:
                    hf = h5py.File(file0, 'w')
                    for k, v in results_out.items():
                        hf.create_dataset(k, data=v)
                    hf.close()
                else:
                    np.save(file0, results_out)

            # figures?
            if wantfig:
                """ TODO
                port normalizerange.m and add to makeimstack
                """
                plt.imshow(
                    make_image_stack(HRFindex.reshape(xyz)),
                    vmin=0,
                    vmax=nh,
                    interpolation='none')
                ax = plt.gca()
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                plt.colorbar()
                plt.savefig(os.path.join(figuredir, 'HRFindex.png'))
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
                if xyz:
                    outdict = {
                        'HRFindex': HRFindex.reshape(xyz),
                        'HRFindexrun': HRFindexrun,
                        'glmbadness': glmbadness,
                        'pcvoxels': pcvoxels,
                        'pcnum': pcnum,
                        'xvaltrend': xvaltrend,
                        'noisepool': noisepool.reshape(xyz),
                        'pcregressors': pcregressors,
                        'betasmd': modelmd,
                        'R2': R2,
                        'R2run': R2run,
                        'meanvol':  meanvol.reshape(xyz)
                        }
                else:
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
                        hf.create_dataset(k, data=v)
                    hf.close()
                else:
                    np.save(file0, outdict)

            # figures?
            if wantfig:
                if whmodel == 2:
                    if noisepool is not None:
                        plt.imshow(
                            make_image_stack(noisepool.reshape(xyz)),
                            vmin=0,
                            vmax=1,
                            cmap='gray',
                            interpolation='none'
                        )
                        ax = plt.gca()
                        ax.axes.xaxis.set_ticklabels([])
                        ax.axes.yaxis.set_ticklabels([])
                        plt.colorbar()
                        plt.savefig(os.path.join(figuredir, 'noisepool.png'))
                        plt.close('all')

                    if pcvoxels is not None:
                        plt.imshow(
                            make_image_stack(pcvoxels.reshape(xyz)),
                            vmin=0,
                            vmax=1,
                            cmap='gray',
                            interpolation='none'
                        )
                        ax = plt.gca()
                        ax.axes.xaxis.set_ticklabels([])
                        ax.axes.yaxis.set_ticklabels([])
                        plt.colorbar()
                        plt.savefig(os.path.join(figuredir, 'pcvoxels.png'))
                        plt.close('all')
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
                    plt.imshow(
                        make_image_stack(R2),
                        vmin=0,
                        vmax=100,
                        cmap='hot',
                        interpolation='none'
                    )
                    ax = plt.gca()
                    ax.axes.xaxis.set_ticklabels([])
                    ax.axes.yaxis.set_ticklabels([])
                    plt.colorbar()
                    plt.savefig(os.path.join(figuredir, 'typeD_R2.png'))
                    plt.close('all')
                    plt.imshow(
                        make_image_stack(FRACvalue.reshape(xyz)),
                        vmin=0,
                        vmax=1,
                        cmap='copper',
                        interpolation='none'
                    )
                    ax = plt.gca()
                    ax.axes.xaxis.set_ticklabels([])
                    ax.axes.yaxis.set_ticklabels([])
                    plt.colorbar()
                    plt.savefig(os.path.join(figuredir, 'FRACvalue.png'))
                    plt.close('all')

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

        return results, resultsdesign
