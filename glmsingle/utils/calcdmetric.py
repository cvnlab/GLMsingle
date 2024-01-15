import numpy as np

def calcdmetric(modelmd, stimorder):

    max_stim = np.max(stimorder)
    tempmn_list = []
    tempsd_list = []

    for pin in range(max_stim):
        ix = np.where(stimorder == pin)[0]
        if len(ix) > 0:
            tempmn_list.append(np.mean(modelmd[:, :, :, ix], axis=3))
        if len(ix) > 1:
            tempsd_list.append(np.std(modelmd[:, :, :, ix], axis=3))

    tempmn = np.stack(tempmn_list, axis=3)

    # output array of NaNs if no repeats
    if len(tempsd_list) == 0:
        f = np.nan * np.ones_like(tempmn)
    else:
        tempsd = np.stack(tempsd_list, axis=3)
        f = np.sqrt(np.mean(tempmn**2, axis=3)) / np.sqrt(np.mean(tempsd**2, axis=3))

    return f