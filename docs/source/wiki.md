# WIKI

## Basic information

GLMsingle is introduced and described in the following pre-print:

[Prince, J.S., Charest, I., Kurzawski, J.W., Pyles, J.A., Tarr, M.J., Kay, K.N. GLMsingle: a toolbox for improving single-trial fMRI response estimates. bioRxiv (2022).](https://doi.org/10.1101/2022.01.31.478431)

GLMsingle is an analysis technique (an algorithm) for obtaining accurate
estimates of single-trial beta weights in fMRI data.

If you have questions or discussion points, please use the Discussions feature
of this github repository, or alternatively, e-mail Kendrick (kay@umn.edu). If
you find a bug, please let us know by raising a Github issue.

## Example scripts

We provide a number of example scripts that demonstrate usage of GLMsingle.

You can browse these example scripts here:

- [Python - example 1](https://htmlpreview.github.io/?https://github.com/kendrickkay/GLMsingle/blob/main/examples/example1.html)
- [Python - example 2](https://htmlpreview.github.io/?https://github.com/kendrickkay/GLMsingle/blob/main/examples/example2.html)
- [MATLAB - example 1](https://htmlpreview.github.io/?https://github.com/kendrickkay/GLMsingle/blob/main/matlab/examples/example1preview/example1.html)
- [MATLAB - example 2](https://htmlpreview.github.io/?https://github.com/kendrickkay/GLMsingle/blob/main/matlab/examples/example2preview/example2.html)

## Tips on usage

### What's the deal with `sessionindicator`?

If your experiment wants to analyze/combine data from multiple distinct scan
sessions (multiple scanner visits), there is the possibility that after basic
pre-processing, there may be substantial abrupt differences in percent BOLD
signal change across responses observed on different days. The
`sessionindicator` option allows you to tell the code how runs are grouped, and
it will internally use this information to z-score responses within sessions in
order to better estimate cross-validation performance. This normalization is
just done internally (under the hood) and does not propagate to the final
outputs.

### Can I exert control over the noise pool voxels?

The default behavior is to automatically select noise pool voxels based on
passing a simple signal intensity threshold (i.e. a simple mask that ignores
out-of-brain voxels) and on having very low amounts of BOLD variance related to
the experiment (i.e. a "R<sup>2</sup>" threshold). If you want to specifically
control the voxels that are used for the noise pool, you can achieve this by
setting `brainthresh` to [99 0] (which allows all voxels to pass the intensity
threshold), `brainR2` to 100 (which allows potentially all voxels in) and then
setting `brainexclude` to be all voxels that you do NOT want in the noise pool.

## FAQ

### What are the main things that GLMsingle does?

The main components of GLMsingle include:

1. a "library of HRFs" technique where an empirically derived set of HRF
   timecourses (from the subjects in the NSD dataset) are used as potential HRFs
   for each voxel in your dataset,
2. the GLMdenoise technique where data-driven nuisance regressors are obtained
   and added into the GLM (using cross-validation for determining how many
   nuisance regressors to add), and
3. ridge regression as a way to improve robustness of single-trial beta
   estimates. The technique relies on heavy amounts of computation, but is
   implemented in a relatively efficient manner, and could therefore serve as a
   go-to tool for deriving beta estimates from many types of fMRI experimental
   data, and especially for condition-rich designs with few repeats per
   condition.

### How does GLMsingle achieve denoising?

We can consider each of the three components.

1. Mismodeling the timecourse of a voxel can lead to suboptimal results (i.e.
   results that are, in a sense, "noisy"); GLMsingle attempts to remedy this by
   using a well regularized HRF selection strategy where a simple "index" is
   learned for each voxel.
2. fMRI data suffer from very large amounts of spatially correlated noise (e.g.
   due to head motion, physiological noise, etc.) --- by deriving these noise
   sources from the data themselves, GLMsingle is able to provide some amount of
   "modeling" out the noise.
3. Finally, fMRI designs often involve substantial overlap of the response
   across successive trials. From a statistical perspective, this is going to
   hurt estimation efficiency. Ridge regression is a method that induces some
   amount of shrinkage bias to help improve out-of-sample generalization.

Note that these three components are heterogenous with regards to the idea of
"fitting" the data. For the HRF component, the presumption is that we probably
have at least enough data to estimate an HRF index for each voxel; hence the
philosophy is to indeed fit the data. For the GLMdenoise component, the goal is
to try to add regressors to the model to better fit the data, while
acknowledging that at some point, overfitting is going to occur (and the
algorithm attempts to determine that point). For the ridge regression component,
things are a bit different. The point of ridge regression is to allow for the
possibility that due to measurement noise, single-trial beta estimates are
inaccurate and we want to actually limit the extent to which we fit the data.
Thus, somewhat counter-intuitively, ridge regression _increases_ the residuals
of the model fit.

### In GLMsingle, the GLMdenoise and ridge regression (RR) components of the method require experimental conditions to repeat across runs. How should I think about whether this is appropriate for my experiment?

In order to determine the hyperparameter settings, it is indeed necessary for
some amount of repeated trials to exist in the data that are given to GLMsingle.
It is true that, in a sense, any component of the fMRI data that does not repeat
for the repeated trials associated with a condition is thought of a "noise" by
the cross-validation process. Hence, there is some potential risk that
components of interest might be removed. However, for the most part, we believe
that the risk of this is actually minimal. This can be seen by thinking about
the nature of the effects of GLMdenoise and ridge regression. GLMdenoise is
simply trying to capture variance that appears to be spatially/globally
correlated across voxels in the dataset; hence, its flexibility is actually
quite limited. Ridge regression can only dampen the instabilities of beta
estimates that persists across temporally nearby trials and in a uniform manner
for all trials in the experiment; hence, its flexibility is also quite limited.

### What are the metrics of quality that are relevant here?

The philosophy behind GLMsingle lies primarily in cross-validation. Internal to
the algorithm is the use of cross-validation to determine the two main
"hyperparameters" that are used --- one is the number of nuisance regressors
(which are derived via PCA on a noise pool of voxels) to add into the GLM model,
and the other is the amount of ridge regularization to be applied to each voxel.
The idea is that the hyperparameters that best generalize to the unregularized
single-trial betas in some left-out data are the appropriate parameters to use.

More generally, the quality metric being invoked here is "reliability" or
"reproducibility". Typically, in an experiment you have experimental conditions
that are presented multiple times. The assumption is that the signal induced by
a given condition should reproduce across different trials.

The concept of noise ceiling (described at length in the NSD data paper) is
essentially a sophisticated method to quantify reliability. A simpler metric
that can assess reliability is simply taking the responses obtained for a voxel
and checking its reliability across different trials. For example, you could
split the data into two halves and correlate the signal estimated on one half
with the signal estimated on the other half.

### I noticed that GLMsingle involves some internal cross-validation. Is this a problem for decoding-style analyses where we want to divide the data into a training set and a test set?

It is true that GLMsingle makes use of all of the data that it is presented
with: for example, if you give it 10 runs of time-series data, its outputs are
dependent (in complex ways) on all of the data. However, we don't think that
this poses any major "circularity"-type problems. All that the algorithm knows
about are the onsets of your experimental conditions and some specification of
when you think that conditions repeat, insofar that you expect some component of
the response to be reproducible across trials. The repeats are used (in
leave-one-run-out cross-validation) to determine the setting of the
hyperparameters. GLMsingle has no access to your scientific hypotheses, and it
is hard to see how it could bias the single-trial beta estimates in favor of one
hypothesis over another. There is one exception, however. If the primary outcome
you are trying to demonstrate is that responses to the same condition are
reliable across runs, then that is, in a sense, exactly what the algorithm is
trying to achieve --- so in that scenario, you might not want to give all of the
data to GLMsingle. Instead, you could divide your data into two halves (for
example), independently give each half to GLMsingle, and then test your
hypothesis that indeed responses are reliable across the two halves.

### I noticed that GLMsingle needs some conditions to repeat across runs. But my experiment is not fully balanced in this regard (or I have just a few repeats). Is this a problem?

In general, we don't think that perfect balancing is that critical. The reason
lies in thinking about the nature of the technique --- the repeats are simply
used to set the hyperparameters (number of nuisance regressors; fractional
regularization amount for each voxel). In our experience, even just a handful of
repeats appears to be sufficient to robustly guide the estimation of the
hyperparameters. Thus, even if you can code only a subset of the trials in your
experiment as condition repeats, this may be sufficient to guide the
hyperparameter estimation, and the overall results from GLMsingle might be quite
good. (In fact, there are very few repeats in the NSD and BOLD5000 datasets,
which are the main two datasets demonstrated in the GLMsingle pre-print.)

### How does R<sup>2</sup> start becoming meaningful in the full FITHRF_GLMDENOISE_RR version?

For a single-trial design matrix, note that in a sense the predictors are
extremely flexible and are happy to capture essentially all or almost all of the
variance in the time-series data from a voxel, even if that voxel contains no
actual signal. Thus, for `ASSUMEHRF` or `FITHRF` or `FITHRF_GLMDENOISE` type
models, the R<sup>2</sup> values from these models are more or less meaningless
(everything looks "good"). However, the RR technique (as a direct consequence of
the fact that it will shrink betas to 0 in accordance to the cross-validated
generalizability of the single-trial beta estimates) will essentially leave
unperturbed the good voxels that have good SNR and aggressively shrink the bad
voxels with little or no SNR. As a consequence, the variance explained by the
beta estimates that are produced by ridge regression will be directly related to
the "goodness" of the voxel as determined by the cross-validation procedure.
Thus, the ridge regression results will have high R<sup>2</sup> values for the
voxels that seem to have reproducible beta estimates across runs.

### Why does ridge regression improve beta estimates?

Ridge regression imposes a shrinkage prior on regression weights, with the exact
amount of regularization controlled by the hyperparameter. In regression
problems where there are correlations across the predictors, ordinary
least-squares estimates of regression weights are unbiased but can suffer from
high variance (i.e. affected by noise). By imposing some amount of shrinkage,
the ability of the estimated regression weights to be more generalizable to
unseen data can be improved. However, these are general statistical concepts. In
the specific case of fMRI designs involving closely spaced trials, there are
high levels of positive correlations between temporally adjacent trials. In a
sense, you can think of this situation as there being limited amounts of data
(statistical power) to disentangle the responses due to adjacent trials.
Ordinary least squares tries to estimate these responses but will generally lead
to noisy beta estimates with large magnitudes (either positive and/or negative).
Ridge regression allows the estimation to impose a prior that effectively
downweights the noisy and limited amounts of data that inform the distinction
between adjacent trials; we can view it as imposing some amount of "temporal
smoothing" (pushing the beta weights from adjacent trials to be more similar in
magnitude) with the goal of attaining an overall solution that has better
out-of-sample performance. Also, in GLMsingle, note that a different shrinkage
hyperparameter is selected for each voxel: this is important since voxels vary
substantially in their signal-to-noise ratios.

### What are the units of the single-trial beta weights, and are there any interpretation issues?

The default behavior of GLMsingle is to return beta weights in units of percent
signal change (by dividing by the mean signal intensity observed at each voxel
and multiplying by 100). However, there can be tricky issues having to do with
the selection of the HRF. If the HRF used in a GLM is incorrect or different
from the underlying HRF, there can easily be gross "gain" or "scale" differences
in the obtained percent signal change units of the betas. One of the things that
GLMsingle attempts to do for you is to estimate a more proper HRF timecourse (as
opposed to assuming a canonical HRF). It is the case that depending on how you
use GLMsingle, you may encounter some scale issues. For example, if you use
GLMsingle to analyze data from one scan session and then separately analyze data
from a second scan session, there will likely be some differences across the two
sets of results. First, there could be real (physiological) changes in the
percent signal change across different days. Second, each set of results may
have a different HRF identified and used for a given voxel; thus, the percent
signal change units in the betas for a given voxel will have a different
ultimate meaning for the two sets of results. It is an open question how to
"best" normalize or handle the betas from different scan sessions to more
accurately get at the underlying neural activity. However, a quick and dirty
heuristic is to normalize the betas, e.g., by z-scoring, or scaling, depending
on your overall scientific goals.

### If the HRF changes from voxel to voxel (or area to area), doesn't that pose some interpretation difficulties or confounding issues?

This is an important issue. In essence, what's at stake is the interpretation of
the absolute magnitudes of beta weights that are derived from a GLM analysis.
Typically, the amplitude of an evoked hemodynamic response is expressed in terms
of percent signal change (PSC) (e.g. by dividing the amplitude increase by some
baseline signal level and then multiplying by 100). Now, if all
voxels/areas/subjects shared exactly the same hemodynamic timecourse shape (i.e.
HRF) and we used this HRF in the analysis, then things would be easy. However,
there are real differences in timecourse shape across voxels/areas/subjects.
(And timecourses can actually also change across scan sessions for the same
voxel(s), due to physiology changes.) Thus, an approach that just assumes a
single fixed HRF has the advantage of being easy to think about, but already
will produce biases in PSC magnitudes. (For example, if you just change the
single fixed HRF used in an analysis, you will likely see that some voxel's
magnitudes go up and other voxels' magnitudes go down.) Now, consider the
approach of tailoring the HRF used in the analysis for individual voxels.
Certainly, the interpretation has to follow carefully -- the set of betas we
observe from a given voxel needs to be interpreted as the amplitude of responses
that appear to be present in the data _assuming_ the HRF selected for that
voxel. Note that in GLMsingle, the library of HRFs are normalized such that the
peak HRF value is 1 for each of the HRFs, so you can conveniently think of the
betas that result for a given HRF as literally the amplitude. Nonetheless, it
remains an open question how much we can actually interpret differences in BOLD
amplitudes across voxels/areas. For example, if one region seems to have higher
PSC than another region (even within the same subject), this doesn't necessarily
mean that the underlying local neural activity is actually higher in the first
region. Finally, we note that one approach to "get rid" of these types of
amplitude issues is to normalize (e.g. z-score) the responses you observe from a
voxel (or region) over the course of a scan session. Of course, one should think
carefully about the implications of such an approach for downstream analyses...

### What is the interpretation of the scale and offset?

By default, after the application of ridge regression, GLMsingle applies a
post-hoc scale and offset to the single-trial betas obtained for a given voxel
to best match what is obtained in the unregularized case. The reason for this is
that due to shrinkage, the betas for a voxel are generally "shrunken" down close
to 0, and therefore have some bias to be small in size. The theory is that we
can undo some of the bias by applying a simple scale and offset to the beta
weights. For many post-hoc uses of the beta estimates, a scale and offset is
probably not a big deal, but certainly one should think hard about whether or
not this matters to the analyses you are trying to do. One can always omit the
scale and offset (via appropriate setting of input options) and/or avoid ridge
regression altogether.

### In the NSD data paper, in the calculation of the noise ceiling, the beta weights are z-scored. What's going on there?

The NSD experiment involves aggregating responses across many distinct scan
sessions. Z-scoring is proposed as a potential (simple) method to reduce
instabilities that exist across different scan sessions. Obviously, z-scoring
can throw away relevant information, so one should be careful in that regard.

## Things to watch out for

If you use the input option to use a canonical HRF (instead of the library of
HRFs), the filename that is written is still "`FITHRF`" (even though the results
reflect what the user specified).
