import numpy as np


def squish(m, num):
    """
    f = squish(m,num)

     <m> is a matrix
     <num> is the positive number of initial dimensions to squish together

     return <m> squished.

     example:
     a = np.asarray([[1,2],[3,4]])
     b = np.asarray([1,2,3,4])
     np.testing.assert_array_equal(squish(a,2), b.T)
    """
    msize = m.shape

    # calculate the new dimensions
    newdim = np.r_[np.prod(msize[:num]), msize[num:]].tolist()

    # do the reshape
    f = np.reshape(m, newdim)
    # tack on a 1 to handle the special case of squishing everything together

    return f
