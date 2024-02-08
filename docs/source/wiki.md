# WIKI

## Basic information

GLMsingle is introduced and described in the following paper:

[Prince, J.S., Charest, I., Kurzawski, J.W., Pyles, J.A., Tarr, M., Kay, K.N. Improving the accuracy of single-trial fMRI response estimates using GLMsingle. *eLife* (2022).](https://doi.org/10.7554/eLife.77599)

GLMsingle is an analysis technique (an algorithm) for obtaining accurate
estimates of single-trial beta weights in fMRI data.

If you have questions or discussion points, please use the Discussions feature
of this github repository, or if you find a bug, please let us know by raising a Github Issue.

## Example scripts

We provide a number of example scripts that demonstrate usage of GLMsingle.

You can browse these example scripts here:

- [Python - example 1](https://htmlpreview.github.io/?https://github.com/kendrickkay/GLMsingle/blob/main/examples/example1.html)
- [Python - example 2](https://htmlpreview.github.io/?https://github.com/kendrickkay/GLMsingle/blob/main/examples/example2.html)
- [MATLAB - example 1](https://htmlpreview.github.io/?https://github.com/kendrickkay/GLMsingle/blob/main/matlab/examples/example1preview/example1.html)
- [MATLAB - example 2](https://htmlpreview.github.io/?https://github.com/kendrickkay/GLMsingle/blob/main/matlab/examples/example2preview/example2.html)

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

### I'm concerned about the noise pool. What if there are signals in there?

This is an interesting issue to think about. The short answer is that GLMsingle guards against improper use of nuisance regressors through cross-validation. If, for some reason, there are valid experimental signals being learned from the noise pool, GLMsingle will tend to not use these signals since they will likely degrade the cross-validation performance.

Note that even if the noise pool includes "good" voxels, improvement in beta estimates are still possible. This is something demonstrated in Figure 6 of Kay et al. Frontiers 2013, where we deliberately included the entire brain in the noise pool (just to see what would happen). The intuition is that results will depend on the specific mixture of noise and signal being learned in the data-derived nuisance regressors. If the nuisance regressors have a little bit of signal mixed in, they can still be useful as an approximate model of the noise.

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

### I noticed that GLMsingle needs some conditions to repeat across runs. But my experiment is not fully balanced in this regard (or I have just a few repeats). Is this a problem?

In general, we don't think that perfect balancing is critical. The reason
lies in thinking about the nature of the technique --- the repeats are simply
used to set the hyperparameters (number of nuisance regressors; fractional
regularization amount for each voxel). In our experience, even just a handful of
repeats appears to be sufficient to robustly guide the estimation of the
hyperparameters. Thus, even if you can code only a subset of the trials in your
experiment as condition repeats, this may be sufficient to guide the
hyperparameter estimation, and the overall results from GLMsingle might be quite
good. (In fact, there are very few repeats in the NSD and BOLD5000 datasets,
which are the main two datasets demonstrated in the GLMsingle pre-print.)

## Pre-processing choices

GLMsingle should be applied to pre-processed fMRI data. But which types of pre-processing are suitable? Please review the GLMsingle manuscript for detailed information, but here we comment on some types of pre-processing that have come up in conversations with users.

* Skull stripping / brain masking -- This is totally fine. In fact, one should be able to run GLMsingle even on surface-prepared data (where voxels not in gray matter are already excluded).

* High-pass temporal filtering -- This is unnecessary. In terms of high-pass filtering (where low frequency time series components are removed), GLMsingle automatically includes a set of polynomials to model the baseline signal level which may drift over the course of each run. Thus, one does not need to high-pass filter the fMRI data in pre-processing. (However, if you do include it, it actually won't make a big difference, since the polynomials will essentially model the components that were removed. But note that this assumes that the mean of each voxel has at least been retained, since GLMsingle uses this spatial information.)

* Low-pass temporal filtering -- This is not very typical, and also not necessary. The HRF timecourse and ridge regression components of GLMsingle are intended to compensate for high temporal frequency noise in the data.

* Projecting out nuisance components -- Pipelines often remove/project-out nuisance components (e.g. motion regressors, ICA derived noise, low-rank approaches like NORDIC, etc.) from fMRI time series in the course of pre-processing. While you can do this, and GLMsingle will likely still work, this is not quite recommended. This is because such pre-filtering approaches risk bias. And, GLMsingle's approach is to attempt to learn these types of nuisance components from the data themselves, so it is a bit ironic to use both approaches simultaneously. Moreover, it is possible that using a pre-filtering approach will actually cause strange complications with GLMsingle, so beware.

* Spatial smoothing -- This is fine, if you want to do this as a pre-processing step. Note that you might want to consider running GLMsingle on non-smoothed data, and then decide whether to apply spatial smoothing at a later analysis step (e.g. on the single-trial betas delivered by GLMsingle).

* Registration / spatial changes -- It is typical to register fMRI data to an atlas space, or to a subject's anatomy, and/or to correct for head displacements within and across runs from a subject. These types of spatial changes are all fine, and do not fundamentally interact with the issues dealt with by GLMsingle.

* Grand intensity scaling -- Some pipelines may scale a given dataset to help normalize units across subjects and/or sessions. This is likely fine, as long as it is a scaling (and not an additive offset). (Scaling preserves percent signal change, and preserves most of the important properties of a dataset.) However, we would have some worry if the scaling were applied differently for different voxels/brain regions and/or differently for different runs within a session, as such aggressive normalization may corrupt signals in undesirable ways.

## Things to watch out for when applying GLMsingle

### Edge / head-motion effects

During motion correction, parts of your volume that move outside of the imaged field-of-view will no longer have any valid data. Typical approaches are to "zero out" such areas. These areas are typically on the edge slices of your acquisition.  You will want to make sure that this is done carefully and that, for example, a voxel doesn't sometimes have valid data and sometimes does not have valid data.  The safest approach is to completely zero out the data for a voxel that does not have full data for all of the runs that you are analyzing with GLMsingle.

### Data from multiple scan sessions

BOLD response amplitudes can change substantially across scan sessions (days) for a single individual. Thus, if you are applying GLMsingle to data concatenated across days, this may pose some problems (since the amplitude changes will appear to be "noise" to GLMsingle and may confuse it). One approach is to use the `sessionindicator` input (see below). Another approach is to just apply GLMsingle separately to each scan session. Note that this approach is effective, but does push the issue down the line: you will eventually have to decide whether and how you will want to normalize betas from different scan sessions.

### Filenames

If you use the input option `wantlibrary=0` to use a canonical HRF
(instead of the library of HRFs), the filename that is written is still 
"`FITHRF`" (even though the results reflect what the user specified).
The reason is just due to the internal code architecture (fitting with a 
library consisting of one HRF is still treated as "`FITHRF`").

## Tips on usage

### My experiment design is not quite synchronized with my fMRI data.

GLMsingle requires the design matrix and the fMRI data to be synchronized at some level of granularity. For example, if the design matrix is specified at 1-s increments, the fMRI data need to also be prepared at 1-s increments. Sometimes, users have experiments where events do not occur at exactly the time points of the fMRI data. In order to accommodate these scenarios, one can perform some upsampling/resampling of the fMRI data and/or the design matrix in order to get the two objects to match. (You might find https://github.com/Charestlab/pyslicetime/blob/master/slicetime/tseriesinterp.py and/or https://github.com/cvnlab/knkutils/blob/master/timeseries/tseriesinterp.m useful for changing the sampling rate of fMRI time series data.)

One drawback is the increase in memory and disk space if upsampling is performed. However, keep in mind that one could just upsample to a moderate level (e.g. 1 s) and then perform a little bit of "rounding the design matrix to the nearest TR/volume". Rounding might introduce fairly negligible levels of inaccuracy given the sluggishness of the HRF.

### How do I deal with experimental conditions with different durations?

GLMsingle is designed to estimate betas (response amplitudes) for trials that are expected to share similar timecourse shapes. For example, if an experiment has brief events (e.g. 2-s long), GLMsingle convolves an HRF with a 2-s long square wave to generate the expected timecourse shape, and the idea is to estimate betas that modulate the amplitude of this timecourse shape for the different experimental conditions. If an experiment has long events (e.g. 16-s long blocks), GLMsingle will similarly convolve an HRF with a 16-s long square wave to generate the expected timecourse shape. The timecourse shapes in the two situations are very different.

A major challenge is how to interpret betas when they reflect amplitudes of timecourses reflecting different durations. This is a tricky general problem, and GLMsingle is not particularly suited to resolving that situation.

Note that within a range, the expected timecourse shape undergoes fairly modest changes. For example, the timecourse shape resulting with convolution of a fixed HRF with a 1-s square wave is quite similar to the result of convolution of that fixed HRF with a 2-s square wave. Certainly, the amplitude is very different (as expected); but the shape is fairly similar. So, one approach is to code events generically as, say, 1.5-s in duration, and this will allow the different activity induced by the 1-s and 2-s events to show up in the estimated response amplitude (betas).

Another potential idea is to code a long event as a string of successive short events. For example, suppose that you have some trials that are 2-s in duration, and that occasionally you have a trial that is 6-s in duration. If you are willing to believe that the 6-s duration event is homogeneous in its neural activity over time, you could code this event as a series of 3 2-s events, and label each of these events as reflecting the same condition. In this way, you still conform to the idea that the timecourse of the hemodynamic response is fixed and that you expect the timecourse shape to summate over time. Once you get the individual single-trial betas out, you can simply, for example, average over the 3 individual betas, if you like.

### Can I exert control over the noise pool voxels?

The default behavior is to automatically select noise pool voxels based on
passing a simple signal intensity threshold (i.e. a simple mask that ignores
out-of-brain voxels) and on having very low amounts of BOLD variance related to
the experiment (i.e. a "R<sup>2</sup>" threshold). If you want to specifically
control the voxels that are used for the noise pool, you can achieve this by
setting `brainthresh` to [99 0] (which allows all voxels to pass the intensity
threshold), `brainR2` to 100 (which allows potentially all voxels in) and then
setting `brainexclude` to be all voxels that you do NOT want in the noise pool.

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

### Why do the betas look crazy outside the brain?

GLMsingle's approach is to estimate percent signal change (PSC) by dividing estimated response amplitudes by the mean signal intensity at each given voxel. In voxels that have very little MR signal (e.g. voxels that are outside the brain), the PSC values might blow up. This is fine and not a reason for concern, as you should simply ignore the data from voxels outside of the brain. Alternatively, if you want to apply a brain mask to your data prior to GLMsingle, that would be fine too.

Likewise, given GLMsingle's algorithmic procedures, it will attempt to identify the optimal HRF even for voxels outside of the brain. This is of course somewhat nonsensical if there is no real signal to begin with. The user can/should simply ignore HRF estimates for these voxels.

### How do I interpret the fractional ridge regression values?

In cases where there are strong BOLD responses from the experiment, you should notice that some voxels in the brain have large fractions (i.e. fractions near 1). This indicates that very little regularization is recommended for these voxels. However, if signal-to-noise ratio is weak, the optimal fractions might be near 0. By default, the smallest fraction that is evaluated is 0.05. Hence, it might be the case that most (or all) voxels might have the optimal fraction selected to be 0.05. The interpretation of this, if it occurs, is that heavy regularization of single-trial betas is necessary to improve generalization performance. This does indicate that signals appear to be weak; however, it does not necessarily indicate that there is a problem with the data or analysis per se.

Furthermore, if you inspect the typeD_R2 and the FRACvalue outputs from GLMsingle, we have noticed that these metrics tend accentuate high-SNR voxels and therefore make it somewhat difficult to distinguish low-SNR voxels from surrounding noise voxels. Bear in mind that even if a voxel doesn't "show up" when inspecting these metrics, the single-trial betas that are estimated for the voxel can still have meaningful signals contained within them. We suggest that ultimate determination of whether voxels contain signals should be guided by the user's analyses of the single-trial betas that are provided by GLMsingle.

## Designing design matrices

### How should I design my design matrix?

For some experiments, thinking about how to setup the design matrix is a major challenge that deserves careful thought. For example, consider an experiment where there are occasional one-back events, and these events are actually not of interest. Moreover, suppose you do not want to assume that the brain response to these one-back events are somehow similar to other trials in the experiment. Then, what you could do is to code these one-back events as unique conditions that have only one presentation. For example, if there are 30 one-back events, you could just add 30 new columns to your design matrix and indicate the onset of each one-back event in one of the columns. Then, after you obtain betas from GLMsingle, you can either just ignore all of the betas associated with those columns, or use them for some purpose!

Note that it is okay if your final design matrices have one or more blank columns. For example, perhaps the trials associated with a given condition do not occur in a given run, but rather occur in other runs.

### Do I need blanks?

In general, you should definitely include blanks or dead time in your experiment. And you should not explicitly code those periods in the design matrix. The reason is that GLMsingle uses a set of polynomials per run to model the baseline signal, and the baseline signal is estimated based on sections of your experiment that are not explicitly coded as experimental events. Note that the interpretation of the single-trial beta amplitudes is that they are evoked responses in the BOLD signal above and beyond whatever the baseline signal is (and this baseline signal may drift to some extent up and down over the course of the run).

If you try to use a design matrix where "everything" is coded, this is problematic because of the inability to estimate the baseline signal level. In these cases, consider omitting the coding of some of the experimental events (e.g. "fixation periods"); in doing so, the idea is that you are trying to estimate changes in the BOLD response **relative** to what the BOLD signal level is during these omitted periods.

### What is the minimum trial separation?

Since the BOLD response is very sluggish, the response to two successive trials can overlap a lot. And, the more closely spaced your trials, the larger the overlap. It is an empirical question as to how effectively GLMsingle can separate the response amplitudes to closely spaced trials. Clearly, the closer the spacing, the harder it is to accurately estimate responses from nearby trials; however, the counteracting factor here is that with more closely spaced trials, a larger number of trials can be conducted (which counteracts the loss of power).

We do not yet know what the "minimum trial separation" is. Good results were obtained in the Natural Scenes Dataset with a trial separation of 4 s --- in that experiment, an image was shown for 3 s, and then there was a 1 s gap before the next image. Note that the gap here is not very relevant from GLMsingle's point of view. It would be perfectly fine to have, e.g., continuous "mini blocks" that last for a 4-s trial and then immediately move on to the next 4-s trial.

We would guess that going faster than 4 s would be risky. Perhaps 3 s would still deliver reasonable results, but faster than that would seem risky (but is an open question). Keep in mind that in GLMsingle, you code the onsets of trials, so what we are talking about here is the separation between the onsets of successive trials. Also, keep in mind that a separate issue is the `stimdur` input, which controls the duration of the expected hemodynamic response to a given trial. For example, in a situation where successive trials are spaced 6 seconds apart, if the `stimdur` input is 6 seconds, this has different collinearity properties compared to the situation where the `stimdur` input is 1 second.

### What about trial subcomponents?

In some cognitive experiments, there are multiple stages to a trial. For example, one might get a cue, preparatory period, a stimulus presentation, another cue to indicate to make a response, and/or an explicit motor (button press) period.

It is up to the experimenter how to design the modeling approach. From GLMsingle's point of view, it does not distinguish (nor care about) these subcomponents. One approach is to code each subcomponent that you wish to model as a distinct condition. For example, you could code the cue for condition A as "condition A1" and the stimulus for condition A as "condition A2". The theory and interpretation is that you are attempting to estimate a separate response amplitude associated with the cue (and its associated brain activity) and the stimulus (and its associated brain activity). As usual, you will have to make some choices as to whether to treat different presentations of a given type of trial as counting as "repeats" or not.

Alternatively, it is totally valid to treat all subcomponents of a trial as contributing to a single response amplitude. From a statistical point of view, this is an easier and more straightforward approach. Ultimately, it depends on your analysis approach and what types of information you are hoping to extract from the data.

## Additional questions/discussion

### What does GLMsingle think a "signal" is?

GLMsingle treats responses evoked by an experimental condition as a signal
if they tend to replicate across repeated trials of that condition.
If the evoked response is zero, then clearly there is no signal.
If the evoked response has some variability across trials, that is okay.
The general goal is to attempt to analyze a set of data in order
to improve the replicability of the condition responses.

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
voxels that seem to have reproducible beta estimates across runs. (One 
note: The R<sup>2</sup> values reflect the explanatory power of the model with
the shrunken beta weights. However, by default we apply a post-hoc scaling and 
offset to the (potentially shrunken) betas from each voxel to match the overall
mean of the unregularized betas. Thus, there is a minor mismatch, in that the
R<sup>2</sup> does not quite correspond to the post-hoc scaled/offset betas.)

Note that it is for these reasons that we write a figure inspection of
the typeD model R<sup>2</sup> values (typeD_R2.png), which includes RR.
The R<sup>2</sup> values from the other models are not very informative.

### Why does HRF selection improve beta estimates?

The extent to which using a better HRF improves GLM outcomes (like beta estimates) depends on how different the shape of the timecourse of the chosen HRF is from some alternative default HRF. If the shape is only slightly different, beta estimates will only slightly change. If the shape is radically different, the beta estimates will change substantially.

In the brain, hemodynamic timecourses can vary due to variations in vasculature distributions (e.g. close to a vein, far from a vein). If the imaging resolution is high, these variations can be substantial. How much impact timecourse variations have can also depend on the experiment. In general, block designs tend to create experimental predictors that are less affected by HRF variations than event-related designs. Thus, for event-related designs especially, getting the HRF right becomes more critical for achieving accurate beta estimates.

Of course, one general consideration here is that the ability to achieve benefits from HRF selection does depend on signal to noise ratio and the amount of data available. If signal to noise is very weak, it may be practically impossible to accurately estimate HRFs. In such cases, the user can consider fixing the HRF to a canonical HRF, or perhaps to a subject-specific HRF (for example, as estimated through the diagnostic FIR model).

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

### Let's talk about the HRF library. How do I know if the library is appropriate for my data?

The HRF library was based on a large survey of "activated" voxels in the visual-memory experiment conducted as part of the Natural Scenes Dataset (NSD). The voxels tended to be in the "back of the brain" but we did not restrict this to be the case (hence, some frontal voxels contributed too). Results were combined across 8 subjects and we attempted to generate a library that is representative of all 8 subjects.

While we think the library should work well as a generic library, certainly it could be valuable to perform detailed and further assessments of this issue. In fact, one could perform finite impulse response (FIR) modeling and derive a new library of HRFs for a specific set of data using the procedures detailed in the NSD paper, and then hook that library into GLMsingle.

### Why isn't the HRF selection cross-validated?

GLMdenoise has a natural "unregularized" state: do not include any derived nuisance regressors as they may introduce overfitting. Ridge regression also has a natural "unregularized" state: perform ordinary least squares estimates and do not introduce any shrinkage bias. The selection of the HRF, however, is a bit different. First, we always need some HRF (it is not as if we can have a model without some HRF timecourse). Second, we lack a baseline HRF, as the canonical HRF used in GLMsingle for the ON-OFF model is really just a very approximate HRF. The different candidates in the HRF library do not really count as different levels of regularization (for which cross-validation might be applied). Rather, each HRF candidate is itself a fully valid HRF choice. In the current approach, we are fully accepting of the idea of chooosing the candidate that works the best (i.e. maximizes the fit to the data). (However, it is certainly true that the statistical approach could be radically altered in such a way that there is some deliberate bias to choose a modal HRF and to only deviate from the modal HRF if the data allow it. This could be a workable approach, but comes with its own set of challenges, such as computational time and added complexity.)

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
scale and offset (via appropriate setting of input option `wantautoscale`) 
and/or avoid ridge regression altogether (via `wantfracridge`).

### In the NSD data paper, in the calculation of the noise ceiling, the beta weights are z-scored. What's going on there?

The NSD experiment involves aggregating responses across many distinct scan
sessions. Z-scoring is proposed as a potential (simple) method to reduce
instabilities that exist across different scan sessions. (Obviously, z-scoring
can throw away relevant information, so one should be careful in that regard.)
Omitting the z-scoring is perfectly fine as an alternative approach, and one
can analyze a set of data accordingly.

### If some of my conditions happen only once, and other conditions have multiple repeats, does that somehow make the singleton conditions different or strange?

No, there shouldn't be any major worry here. The analysis components, generally speaking, treat each single trial equally. The repeats are used essentially just to determine the hyperparameters (i.e. number of nuisance regressors, amount of ridge regression shrinkage per voxel).

## Basic checklist for calling GLMsingle correctly

### Data

- The data should be your fMRI data (after minimal pre-processing).
- Are the voxel dimensions correct (e.g. XYZ dimensions)?
- Are the number of runs correct?
- Are the number of volumes in each run correct?
- Do you know your voxel size?
- Do you know what the TR associated with your prepared fMRI data is? Is it correct?
- Can you confirm that the fMRI volumes are sane? If you plot a slice from an example volume, does it look like a brain? If you compare a volume from the beginning of a run (or session) to a volume from the end of a run (and/or session), does it look quite similar (and stable)?
- Look at the time-series data for an example voxel (just choose one). Does it look like a reasonable time-series? Are the units reasonable (e.g. arbitrary raw scanner units fluctuating around 1000 or 500 or something like that?)?
- Can you confirm that if you, say, watch a movie of your brain volumes over time, that the brain imaging is stable and that there are no massive transients or artifacts?
- Are all of your runs consistently prepared and have the same format (aside from possible differences in the number of volumes in different runs)?
  
### Design

- The design specifies how you think about the structure of your experiment.
- Do you have a strategy for how you are trying to model your data?
- Have you thought about the <stimdur> input?
- Did you code your trial onsets correctly? Double-check that each column of your design matrix has onsets matched to your fMRI volumes.
- Consider performing a careful visual inspection of the design matrix that is fed to GLMsingle. Does it match your expectation with respect to the timing of your experiment?
- Are you sure the coding of the design matrix makes sense for your experiment?
- Did you make sure to NOT code blank/null trials?
- Did you make sure that the coding of the design matrix is correct in all of your runs?
