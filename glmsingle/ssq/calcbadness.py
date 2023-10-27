import numpy as np
import copy
from glmsingle.utils.zerodiv import zerodiv


def calcbadness(xvals, validcolumns, stimix, results, sessionindicator):
    """
    badness = calcbadness(xvals,validcolumns,stimix,results,sessionindicator)

    Arguments:
    __________

    <xvals>:

    is a list vector of vectors of run indices

    <validcolumns>:

    is a list vector, each element is the vector of trial indices
    associated with the run

    <stimix>:

    is a list vector, each element is the vector of actual condition
    numbers occurring with a given run

    <results>:

    is a 1 x n with results. the first one is SPECIAL and is unregularized.

    <sessionindicator>

    is 1 x RUNS with positive integers indicating run groupings for sessions.
    this is used only to perform the session-wise z-scoring for the purposes
    of hyperparameter evaluation.

    Returns
    __________

    <badness>:

    voxels x hyperparameters with the sum of the squared error from
    cross-validation.
    the testing data consists of the beta weights from results[0],
    i.e. unregularized beta weights.

    note that the squared error is expressed in the z-score units
    (given that we z-score the single-trial beta weights prior to evaluation
    of the different hyperparameters).

    note:
    the unregularized betas set the stage for the session-wise normalization:
    for each session, we determine a fixed mu and sigma that are applied to
    the session under all of the various regularization levels.
    """
    # initialize

    badness = np.zeros(
        (results[0].shape[0], len(results))
        )

    # calc
    alltheruns = np.arange(len(validcolumns))

    # z-score transform the single-trial beta weights
    if np.max(sessionindicator) == 1:
        sessions = [1]
    else:
        sessions = range(1, np.max(sessionindicator) + 1)

    for sess in sessions:

        wh = np.flatnonzero(np.array(sessionindicator) == sess)

        whcol = np.concatenate([validcolumns[x] for x in wh])

        # mean of unregularized case
        mn = np.mean(results[0][:, whcol], axis=1)

        # std dev of unregularized case
        sd = np.std(results[0][:, whcol], axis=1, ddof=1)

        resultsdm = copy.deepcopy(results)
        for runis in range(len(resultsdm)):
            rundemean = results[runis][:, whcol]-mn[:, np.newaxis]
            resultsdm[runis][:, whcol] = zerodiv(rundemean, sd, val=0, wantcaution=0)

    # do cross-validation
    for xx in range(len(xvals)):

        # calc
        # which runs are training, e.g. [1 2 5 6 7 8 9 10 11 12]
        testix = xvals[xx]
        trainix = np.setdiff1d(alltheruns, testix)

        # calc
        # we need to check whether we need
        # to concatenate multiple test runs
        if testix.size > 1:
            # vector of trial indices in the testing data
            testcols = np.concatenate([validcolumns[x] for x in testix])

            # vector of condition-ids in the testing data
            testids = np.concatenate([stimix[x] for x in testix])
        else:
            # default leave-one-out case.
            # vector of trial indices in the testing data

            testcols = validcolumns[testix]

            # vector of condition-ids in the testing data
            testids = stimix[testix]

        # vector of trial indices in the training data
        traincols = np.concatenate([validcolumns[x] for x in trainix])

        # vector of condition-ids in the training data
        trainids = np.concatenate([stimix[x] for x in trainix])

        # calculate cross-validation performance
        for pcr in range(len(results)):
            # hashrec = cell(1,max(testids));  # speed-up by caching results
            for trial in range(len(testids)):

                # which training trials match the current condition-id?
                haveix = np.flatnonzero(trainids == testids[trial])

                if haveix.size > 0:

                    # NOTE:
                    # testcols(trial) tells us which trial in the testing runs
                    # to pull betas for (these are 0-based trial numbers)
                    # traincols(haveix) tells us the corresponding trials
                    # (isolated within the training runs) to pull betas for
                    # (these are 1-based trial numbers)

                    # compute squared error of all training betas against the
                    # current testing beta, and accumulate!!
                    betas_1 = resultsdm[pcr][:, traincols[haveix]]
                    betas_2 = resultsdm[0][:, testcols[trial]]

                    badness[:, pcr] = badness[:, pcr] + np.sum(
                        (betas_1-betas_2[:, np.newaxis])**2, axis=1)
                    # NOTICE the use of results(0)

    return badness
