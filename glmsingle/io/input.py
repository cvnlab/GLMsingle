import numpy
import nibabel


def load_nifti(fpath):
    """Read Nifti file and return data in time x voxels shape.

    Uses nibabel
    
    Args:
        fpath (str): Full path to a nifti bold file
    
    Returns:
        ndarray: two-dimensional array with data in time x voxels
    """

    img = nibabel.load(fpath)
    data_xyzt = img.get_data()
    data_txyz = numpy.moveaxis(data_xyzt, -1, 0)
    return data_txyz.reshape([data_txyz.shape[0], -1])
