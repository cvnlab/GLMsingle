import numpy as np

def alt_round(x):
    """
     alt_round(x)
     
     <x> is an array
     
     simulate the rounding behavior of matlab where 0.5 rounds 
     to 1 and -.5 rounds to -1. (python rounds ties to the
     nearest even integer.)
     
     return:
      an array of rounded values (as integers)
    
     example:
     import numpy as np
     x = np.array([-1, -0.5, 0, 0.5, 0.7, 1.0, 1.5, 2.1, 2.5, 2.6, 3.5])
     y = alt_round(x)

    """
    return (np.sign(x) * np.ceil( np.floor( np.abs(x) * 2 ) / 2 )).astype(int)
