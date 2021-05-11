import numpy as np


def chunking(vect, num, chunknum=None):
    """ chunking
    Input:
        <vect> is a array
        <num> is desired length of a chunk
        <chunknum> is chunk number desired (here we use a 1-based
              indexing, i.e. you may want the frist chunk, or the second
              chunk, but not the zeroth chunk)
    Returns:
        [numpy array object]:

        return a numpy array object of chunks.  the last vector
        may have fewer than <num> elements.

        also return the beginning and ending indices associated with
        this chunk in <xbegin> and <xend>.

    Examples:

        a = np.empty((2,), dtype=np.object)
        a[0] = [1, 2, 3]
        a[1] = [4, 5]
        assert(np.all(chunking(list(np.arange(5)+1),3)==a))

        assert(chunking([4, 2, 3], 2, 2)==([3], 3, 3))

    """
    if chunknum is None:
        nchunk = int(np.ceil(len(vect)/num))
        f = []
        for point in range(nchunk):
            f.append(vect[point*num:np.min((len(vect), int((point+1)*num)))])

        return np.asarray(f)
    else:
        f = chunking(vect, num)
        # double check that these behave like in matlab (xbegin)
        xbegin = (chunknum-1)*num+1
        # double check that these behave like in matlab (xend)
        xend = np.min((len(vect), chunknum*num))

        return np.asarray(f[num-1]), xbegin, xend
