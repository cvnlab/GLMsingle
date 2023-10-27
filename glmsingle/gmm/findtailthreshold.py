import numpy as np
from glmsingle.utils.robustrange import robustrange
from glmsingle.utils.picksubset import picksubset
from sklearn.mixture import GaussianMixture as gmdist
import matplotlib.pyplot as plt


def findtailthreshold(v, figpath=None):
    """
     function [f,mns,sds,gmfit] = findtailthreshold(v,wantfig)

     <v> is a vector of values
     <wantfig> (optional) is whether to plot a diagnostic figure. Default: 1.

     Fit a Gaussian Mixture Model (with n=2) to the data and
     find the point at which the posterior probability is
     equal (50/50) across the two Gaussians. This serves
     as a nice "tail threshold".

     To save on computational load, we take a random subset of
     size 1000000 if there are more than that number of values.
     We also use some discretization in computing our solution.

     return:
     <f> as the threshold
     <mns> as [A B] with the two means (A < B)
     <sds> as [C D] with the corresponding std devs
     <gmfit> with the gmdist object (the order might not
       be the same as A < B)

     example:
     from numpy.random import randn
     f, mns, sds, gmfit = findtailthreshold(np.r_[randn(1000), 5+3*randn(500)], figpath='test.png')
    """

    # internal constants
    numreps = 3  # number of restarts for the GMM
    maxsz = 1000000  # maximum number of values to consider
    nprecision = 500
    # linearly spaced values between lower and upper robust range

    # inputs
    if figpath is None:
        wantfig = 0
    else:
        wantfig = 1

    # quick massaging of input
    v2 = v[np.isfinite(v)]
    if len(v2) > maxsz:
        print('warning: too big, so taking a subset')
        v2, _, _ = picksubset(v2, maxsz)

    # fit mixture of two gaussians
    gmfit = gmdist(n_components=2, tol=1e-10, reg_covar=0, n_init=numreps).fit(v2.reshape(-1, 1))

    # figure out a nice range
    rng = robustrange(v2.flatten())[0]

    # include the smaller of the two distribution means if necessary
    rng[0] = np.min([rng[0], np.min(gmfit.means_.flatten())])

    # include the bigger of the two distribution means if necessary
    rng[1] = np.max([rng[1], np.max(gmfit.means_.flatten())])

    # OLD
    # rng = robustrange(v2.flatten())[0]

    # evaluate posterior
    allvals = np.linspace(rng[0], rng[1], num=nprecision)
    checkit = gmfit.predict_proba(allvals.reshape(-1, 1))

    # figure out crossing
    if checkit[-1,0] > 0.5:
        whdist = 0 # if the first distribution is the higher one on the right
    else:
        whdist = 1 # if the second distribution is the higher one on the right

    for ix in range(checkit.shape[0] - 1, -1, -1):
        if checkit[ix, whdist] <= 0.5:
            break

    # warn if necessary
    if checkit[ix, whdist] > 0.5:
        print('warning: no crossing of 0.5 found. results may be inaccurate!')

    # OLD
    # np.testing.assert_equal(
    #    np.any(checkit[:, 0] > .5) and np.any(checkit[:, 0] < .5),
    #    True,
    #    err_msg='no crossing of 0.5 detected')
    # ix = np.argmin(np.abs(checkit[:, 0]-.5))

    # return it
    f = allvals[ix]

    # prepare other outputs
    mns = gmfit.means_.flatten()
    sds = np.sqrt(gmfit.covariances_.flatten())
    if mns[1] < mns[0]:
        mns = mns[[1, 0]]
        sds = sds[[1, 0]]

    # start the figure
    if wantfig:
        # make figure
        plt.plot(allvals, checkit)
        plt.plot([allvals[ix], allvals[ix]], plt.ylim(), 'k-', linewidth=2)
        plt.title('Posterior Probabilities')
        plt.savefig(figpath)
        plt.close('all')

    return f, mns, sds, gmfit
