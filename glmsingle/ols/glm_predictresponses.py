import numpy as np
from glmsingle.design.construct_stim_matrices import construct_stim_matrices
from glmsingle.design.convolve_design import convolve_design
from glmsingle.utils.squish import squish


def glm_predictresponses(model, design, tr, numtimepoints, dimdata=0):
    """
     responses = glm_predictresponses(model,design,tr,numtimepoints,dimdata)

    Arguments:

     <model> is one of the following:
       A where A is X x Y x Z x conditions x time with the timecourse of the
         response of each voxel to each condition.  XYZ can be collapsed.
       {B C} where B is time x 1 with the HRF that is common to all voxels and
         all conditions and C is X x Y x Z x conditions with the amplitude of
         the response of each voxel to each condition
       Note that in both of these cases, the first time point is assumed to be
       coincident with condition onset.
     <design> is the experimental design.  There are three possible cases:
       1. A where A is a matrix with dimensions time x conditions.
          Each column should be zeros except for ones indicating condition
          onsets.
          (Fractional values in the design matrix are also allowed.)
       2. {A1 A2 A3 ...} where each of the A's are like the previous case.
          The different A's correspond to different runs, and different runs
          can have different numbers of time points.
       3. {{C1_1 C2_1 C3_1 ...} {C1_2 C2_2 C3_2 ...} ...} where Ca_b
          is a vector of onset times for condition a in run b.  Time starts
          at 0 and is coincident with the acquisition of the first volume.
          This case is compatible only with the common-HRF <model>.
     <tr> is the sampling rate in seconds
     <numtimepoints> is a vector with the number of time points in each run

     Given various inputs, compute the predicted time-series response.

     Returns:
     <responses> as XYZ x time or a list vector of elements that are
       each XYZ x time.  The format of <responses> will be a matrix in
       the case that <design> is a matrix (case 1) and will be a cell vector
       in the other cases (cases 2 and 3).

     History:
     - 2013/05/12: allow <design> to specify onset times; add <tr>,
                   <numtimepoints> as inputs
     - 2013/05/12: update to indicate fractional values in design matrix are
                   allowed.
     - 2012/12/03: *** Tag: Version 1.02 ***
     - 2012/11/2 - Initial version.
    """

    # calc
    ismatrixcase = type(design) != list

    if not ismatrixcase:
        if design[0].ndim == 1:
            # handle special case of onoff design
            design = [p[:, np.newaxis] for p in design]
    
    if type(model) == list:
        modelshape = model[0]['betasmd'].shape        
    else:
        modelshape = model['betasmd'].shape

    if len(modelshape) == 2:
        # case x*x*z conditions
        dimdata = 0
        xyzsize = [modelshape[0]]
    else:
        dimdata = 2
        xyzsize = list(modelshape[:dimdata+1])
    

    # make cell
    if type(design) is not list:
        design = [design]

    # loop over runs
    responses = []
    desopt = {'n_times': numtimepoints, 'tr': tr}
    for p in range(len(design)):

        # case of shared HRF
        if 'hrfknobs' in model:

            # convolve with HRF
            dm = convolve_design(
              design[p],
              model['hrfknobs'],
              desopt).astype(np.float32)

            tempdata = squish(model['betasmd'], dimdata+1).T 

            tempmodel = dm @ tempdata
            # weight by the amplitudes # X x Y x Z x time
            responses.append(
                np.reshape(
                  tempmodel.T,
                  xyzsize + [numtimepoints[p]]
                )
            )

        # case of individual timecourses
        else:

            # length of each timecourse (L)
            timecourselen = model['betasmd'].shape[-1]

            # expand design matrix using delta functions
            temp = construct_stim_matrices(
                design[p].T,
                prenumlag=0,
                postnumlag=timecourselen-1
            ).astype(np.float32)  # time*L x conditions

            # weight design matrix by the timecourses
            tempdata = squish(np.transpose(squish(model['betasmd'], dimdata+1), [2, 1, 0]),2)
            
            tempmodel = temp @ tempdata

            responses.append(            
                np.reshape(
                    tempmodel.T,
                   xyzsize + [numtimepoints[p]])
            )

    # undo cell if necessary
    if ismatrixcase:
        responses = np.stack(responses)

    return responses
