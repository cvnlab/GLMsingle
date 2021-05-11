import numpy as np


def isrowvector(m):
    """Check if the array has only one dimension

    f = isrowvector(m)

    function f = isrowvector(m)

    <m> is a matrix

    return whether <m> is 1 x n where n >= 0.
    specifically:
    f = isvector(m) & size(m,1)==1;

    example:
    isrowvector([[1,2]])
    isrowvector([[1]])
    isrowvector(np.zeros(1))
    not isrowvector([])
    """

    if not isinstance(m, np.ndarray):
        m = np.asarray(m)
    f = m.shape[0] == 1
    return f
