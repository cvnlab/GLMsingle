import numpy as np
from scipy.interpolate import interp1d
from glmsingle.utils.resamplingindices import resamplingindices
from glmsingle.utils.colors import colors
import matplotlib.pyplot as plt

def cmapturbo(n=256):

    # cmapturbo(n)
    #
    # <n> (optional) is the desired number of entries (no more than 256).
    #   Default: 256.
    #
    # Return a jet-like colormap that is more perceptually uniform.
    #
    # example:
    # plt.imshow(np.rand((100,100)))
    # plt.colormap(cmapturbo)
    # plt.colorbar()
    # plt.show
    ################%%%%%%%%%%%
    
    # See also https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html
    
    # Look-up table for Turbo colormap.
    # Includes example usage and comparison with other colormaps.
    # Adapted for Matlab from https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
    
    # Nikolas Karalis, 21 August 2019
  
    ###########
    cols = colors()
    ncols, nchan = cols.shape

    # calc
    xx = resamplingindices(1,ncols,n)
    
    # do it
    f = []
    for p in range(nchan):
      f.append(np.interp(xx, range(ncols), cols[:,p].T)) 

    return np.asarray(f).T
    