import numpy as np
import matplotlib.pyplot as plt
from glmsingle.utils.splitmatrix import splitmatrix
from glmsingle.utils.normalizerange import normalizerange

def make_image_stack(m, wantnorm=0, addborder=1, csize=None, bordersize=1):
    """
    <m> is a 3D matrix. If more than 3D, we reshape to be 3D.
    We automatically convert to float format for the purposes of this function.
    
    <wantnorm> (optional) is
      0 means no normalization
      [A, B] means normalize and threshold values such that A and B map to 0 and 1.
      X means normalize and threshold values such that X percentile
        from lower and upper end map to 0 and 1. If the X percentile
        from the two ends are the same, then map everything to 0.
      -1 means normalize to 0 and 1 using -max(abs(m(:))) and max(abs(m(:)))
      -2 means normalize to 0 and 1 using 0 and max(m(:))
      -3 means normalize to 0 and 1 using min(m(:)) and max(m(:))
      default: 0.

    <addborder> (optional) is
      0 means do not add border
      1 means add border at the right and bottom of each image.
        the border is assigned the maximum value.
      2 means like 1 but remove the final borders at the right and bottom.
      -1 means like 1 but assign the border the middle value instead of the max.
      -2 means like 2 but assign the border the middle value instead of the max.
      j means like 1 but assign the border a value of 0.
      2*j means like 2 but assign the border a value of 0.
      NaN means plot images into figure windows instead of returning a matrix.
        each image is separated by one matrix element from surrounding images.
        in this case, <wantnorm> should not be 0.
      default: 1.
      
    <csize> (optional) is [X Y], a 2D matrix size according
      to which we concatenate the images (row then column).
      default is [], which means try to make as square as possible
      (e.g. for 16 images, we would use [4 4]).
      special case is -1 which means use [1 size(m,3)].
      another special case is [A 0] or [0 A] in which case we
      set 0 to be the minimum possible to fit all the images in.

    <bordersize> (optional) is number of pixels in the border in the case that
      <addborder> is not NaN.  default: 1.

    If <addborder> is not NaN, then return a 3D matrix.  The first two dimensions 
    contain images concatenated together, with any extra slots getting filled 
    with the minimum value. The third dimension contains additional sets of images
    (if necessary).

    If <addborder> is NaN, then make a separate figure window for each set of images.
    In each figure window, we plot individual images using imagesc scaled to the range [0,1].
    We return <f> as [].

    Note that in the case of <wantnorm>, if the determined range has min and max equal 
    to each other, we just give up and return an image that is all zeros.
    """
    # calc
    nrows, ncols = m.shape[:2]

    # make double if necessary
    m = np.float64(m)
    if type(wantnorm) is not list:
        wantnorm=[wantnorm]
    wantnorm = np.array(wantnorm).astype(np.float64)

    # make <m> 3D if necessary
    m = np.reshape(m,[nrows,ncols,-1])

    # find range, normalize
    if len(wantnorm)==2:
        if wantnorm[0] == wantnorm[1]:
            m = np.zeros_like(m) # avoid error from normalizerange.m
        else:
            m = normalizerange(m,0,1,wantnorm[0],wantnorm[1])
        
        mn = 0
        mx = 1
    else:

        if wantnorm==0:
            mn, mx = np.nanmin(m.flatten()), np.nanmax(m.flatten())
        elif wantnorm==-1:        
            rng = np.r_[-np.max(np.abs(m.latten())), np.max(np.abs(m.flatten()))]
        elif wantnorm==-2:
            rng = np.r_[0, np.max(m.flatten())]
        elif wantnorm==-3:
            rng = np.r_[np.min(m.flatten()), np.max(m.flatten())]
        else:
            rng = np.percentile(m.flatten(),[wantnorm, 100-wantnorm])        
    
        if wantnorm != 0:
            if rng[0] == rng[1]:
                m = np.zeros_like(m) # avoid error from normalizerange.m
            else:
                m = normalizerange(m,0,1,rng[0],rng[1])
            
            mn = 0
            mx = 1
    
    md = (mn+mx)/2

    # number of images
    numim = m.shape[2]

    # calculate csize if necessary
    if csize is None:
        rows = int(np.floor(np.sqrt(numim)))  # MAKE INTO FUNCTION?
        cols = int(np.ceil(numim/rows))
        csize = (rows, cols)
    elif csize==-1:
        csize = (1, numim)
    elif csize[0]==0:
        csize[0] = int(np.ceil(numim/csize[1]))
    elif csize[1]==0:
        csize[1] = int(np.ceil(numim/csize[0]))


    # calc
    chunksize = np.prod(csize)
    numchunks = int(np.ceil(numim/chunksize))

    # convert to cell vector, add some extra matrices if necessary
    m = splitmatrix(m,2)
    m.extend([np.full_like(m[0], mn) for _ in range(numchunks*chunksize - numim)])

    # add border?
    if np.imag(addborder) or addborder:
        for p in range(len(m)):
            # figure out new size including border size
            sz = list(np.array(m[p].shape[:2]) + bordersize)
            bordercolor = 0 if np.imag(addborder) else (mx if addborder > 0 else md)
            new_im = np.ones(sz) * bordercolor
            new_im[:-bordersize, :-bordersize] = m[p].squeeze()
            m[p] = new_im

    # combine images
    f = []
    for p in range(numchunks):
        temp = [np.reshape(x, (x.shape[0], x.shape[1])) for x in m[p*chunksize:(p+1)*chunksize]]
        temp = [np.vstack(temp[i*csize[0]:(i+1)*csize[0]]) for i in range(csize[1])]
        f.append(np.hstack(temp))
    f = np.dstack(f)

    return f.squeeze()

