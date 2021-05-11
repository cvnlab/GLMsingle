import numpy as np


def robustrange(m):
    """
     robustrange(m)

     <m> is an array

     figure out a reasonable range for the values in <m>,
     that is, one that tries to include as many of the values
     as possible, but also tries to exclude the effects of potential
     outliers (so that we don't get a really large range).
     see the code for specific details on how we do this.

     return:
      <f> as [<mn> <mx>]
      <mn> as the minimum value of the range
      <mx> as the maximum value of the range

     example:
     import matplotlib.pyplot as plt
     x = np.random.randn(10000)**2
     plt.hist(x,100)
     rng = robustrange(x)[0]
     plt.plot(
         [rng[0], rng[0]],
         plt.ylim(),
         color='r',
         linestyle='-',
         linewidth=2)
     plt.plot(
         [rng[1], rng[1]],
         plt.ylim(),
         color='r',
         linestyle='-',
         linewidth=2)
     plt.title(f'range is {rng[0]:.20f} {rng[1]:.20f}')

    """
    # absolute min and max

    absmn = np.min(m.flatten())
    absmx = np.max(m.flatten())

    # percentiles
    vals = np.percentile(m.flatten(), [.1, 10, 50, 90, 99.9])

    # percentile-based min and max
    pmn = vals[2] - 5*(vals[2]-vals[1])
    pmx = vals[2] + 5*(vals[3]-vals[2])

    # whether to rerun (recursively)
    rerun = 0

    # deal with max
    if vals[4] <= pmx:  # if the 99.9 is reasonably small, use it
        if absmx <= vals[2] + 1.1*(vals[4]-vals[2]):
            # actually, if the absolute max isn't too big, use that
            finalmx = absmx
        else:
            finalmx = vals[4]

    else:
        # hmm, something is funny.  probably there are outliers.
        # let's chop off and re-run.
        rerun = 1
        m = m[np.logical_not(m > pmx)]

    # deal with min
    if vals[0] >= pmn:
        if absmn >= vals[2] - 1.1*(vals[2]-vals[0]):
            finalmn = absmn
        else:
            finalmn = vals[0]

    else:
        rerun = 1
        m = m[np.logical_not(m < pmn)]

    # rerun without outliers and output
    if rerun:
        f, mn, mx = robustrange(m)
        return f, mn, mx
    else:
        f = [finalmn, finalmx]
        mn = finalmn
        mx = finalmx
        return f, mx, mn
