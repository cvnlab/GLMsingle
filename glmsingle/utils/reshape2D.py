import numpy as np


def reshape2D(m, dim):
    """
    Reshape matrix 'm' to 2D, moving the specified dimension 'dim' to the beginning.
    
    m: input matrix
    dim: target dimension (0-based indexing)
    
    Returns a reshaped 2D matrix
    """
    dimorder = np.r_[dim, np.setdiff1d(np.arange(np.max([m.ndim,dim])), dim)]

    # Permute the dimensions of m based on the new order and then reshape it into a 2D array
    return np.reshape(custom_permute(m, dimorder), (custom_size(m, dim), -1), order='F')


def reshape2D_undo(f, dim, msize):
    """
    Undo the reshaping performed by reshape2D.
    
    f: reshaped matrix
    dim: target dimension (0-based indexing)
    msize: original shape of the matrix
    
    Returns the matrix reshaped back to its original shape
    """
    # figure out the permutation order that was used in reshape2D
    dimorder = np.r_[dim, np.setdiff1d(range(np.max([len(msize),dim])),dim)]

    if dim >= len(msize):  # If the specified dimension is greater than the number of dimensions of the original matrix
        reshapesize = [1] + list(msize)
    else:
        reshapesize = np.array(msize)[dimorder]

    # Reshape the matrix back to its original shape and transpose to the original dimension order
    return np.transpose(f.reshape(reshapesize, order='F'), np.argsort(dimorder)).squeeze()


def custom_permute(arr, order):
    """
    Permute the dimensions of the array according to the order provided.
    Can expand the number of dimensions.
    """
    # Add new axes for all dimensions not in the original shape
    for dim in order:
        if dim >= len(arr.shape):
            arr = np.expand_dims(arr, dim)
    # Transpose to the desired order
    return np.transpose(arr, order)


def custom_size(arr, dim):
    """
    Return the size of the specified dimension.
    If the dimension doesn't exist, return 1.
    """
    if dim < len(arr.shape):
        return arr.shape[dim]  
    else:
        return 1