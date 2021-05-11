import numpy as np


def picksubset(m, num, seed=None):
    """
     function [f,idx,fnot] = picksubset(m,num,seed)

     <m> is a numpy array
     <num> is
       X indicating the size of the subset to pick out
     <seed> (optional) is the rand state to use.
       default: 0.

     return:
      <f> as a vector with a random subset of <m>.
      <idx> as a vector of the indices of the elements that we picked.
      <fnot> as a vector with the remaining elements of <m>.

     note that if you try to pick out a subset bigger than <m>,
     we will just return as many elements as there are in <m>.

     example:
     from numpy.random import randn
     picksubset(randn(10,10),10)
    """
    # input
    if seed is None:
        seed = 0

    # do it
    np.random.seed(seed)

    nm = len(m.flatten())

    idx = np.random.permutation(range(nm))[range(np.min([num, nm]))]

    f = m.flatten()[idx]
    notf = np.ones(m.flatten().shape)
    notf[idx] = 0
    fnot = m.flatten()[notf]

    np.random.seed(None)

    return f, idx, fnot
