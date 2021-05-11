import numpy as np
from math import floor, ceil


def make_image_stack(m):
    """make image stack from 3D volume
    <m> is a 3D matrix.  if more than 3D, we reshape to be 3D.
    we try to make the imagestack as square as possible
    (e.g. for 16 images, we would use [4 4]).
    find the minimum possible to fit all the images in.    
    Args:
        m ([3D volume]): this can be a 3D volume of any sort.
    Returns:
        [2D stack]: 2D stack of slices
    """

    m = np.array(m)

    bordersize = 1
    # calc
    nrows, ncols, numim = m.shape
    mx = np.nanmax(m.ravel())

    # calculate csize

    rows = floor(np.sqrt(numim))
    cols = ceil(numim / rows)
    csize = [rows, cols]

    rowstop = rows - 1

    # calc
    chunksize = csize[0] * csize[1]
    numchunks = ceil(numim / chunksize)

    # total cols and rows for adding border to slices
    tnrows = nrows + bordersize
    tncols = ncols + bordersize

    # make a zero array of chunksize
    # add border
    mchunk = np.zeros((tnrows, tncols, chunksize))
    mchunk[:, :, :numim] = mx
    mchunk[:-1, :-1, :numim] = m

    # combine images

    flatmap = np.zeros((tnrows * rows, tncols * cols))
    ci = 0
    ri = 0
    for plane in range(chunksize):
        flatmap[ri: ri + tnrows, ci: ci + tncols] = mchunk[:, :, plane]
        ri += tnrows
        # if we have filled rows rows, change column
        # and reset r
        if plane != 0 and ri == tnrows * rows:
            ci += tncols
            ri = 0

    return flatmap
