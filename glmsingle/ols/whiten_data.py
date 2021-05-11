
import numpy as np
from glmsingle.ols.make_poly_matrix import (make_polynomial_matrix,
                                            make_projection_matrix)
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


def construct_projection_matrix(ntimepoints,
                                extra_regressors=None,
                                poly_degs=None):
    """[summary]

    Arguments:
        data {[type]} -- [description]
        design {[type]} -- [description]

    Keyword Arguments:
        extra_regressors {bool} -- [description] (default: {False})
        poly_degs {[type]} -- [description] (default: {np.arange(5)})

    Returns:
        [type] -- [description]
    """
    if poly_degs is None:
        poly_degs = np.arange(5)

    polynomials = make_polynomial_matrix(ntimepoints, poly_degs)

    if extra_regressors is not None and extra_regressors.size > 0:
        polynomials = np.c_[polynomials, extra_regressors]

    return make_projection_matrix(polynomials)


def whiten_data(data, design, extra_regressors=False, poly_degs=None):
    """[summary]

    Arguments:
        data {[type]} -- [description]
        design {[type]} -- [description]

    Keyword Arguments:
        extra_regressors {bool} -- [description] (default: {False})
        poly_degs {[type]} -- [description] (default: {np.arange(5)})

    Returns:
        [type] -- [description]
    """
    if poly_degs is None:
        poly_degs = np.arange(5)

    # whiten data
    whitened_data = []
    whitened_design = []

    for i, (y, X) in enumerate(zip(data, design)):
        polynomials = make_polynomial_matrix(X.shape[0], poly_degs)
        if extra_regressors:
            if extra_regressors[i].any():
                polynomials = np.c_[polynomials, extra_regressors[i]]

        project_matrix = make_projection_matrix(polynomials)
        whitened_design.append(project_matrix @ X)
        whitened_data.append(project_matrix @ y)

    return whitened_data, whitened_design
