import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd
from glmsingle.design.make_design_matrix import make_design
from glmsingle.glmsingle import GLM_single
import time

sub = 2
ses = 1
stimdur = 0.5
tr = 2

proj_path = os.path.join(
    '/home',
    'adf',
    'charesti',
    'data',
    'arsa-fmri',
    'BIDS')

data_path = os.path.join(
    proj_path,
    'derivatives',
    'fmriprep',
    'sub-{}',
    'ses-{}',
    'func')

design_path = os.path.join(
    proj_path,
    'sub-{}',
    'ses-{}',
    'func')

runs = glob.glob(
    os.path.join(data_path.format(sub, ses), '*preproc*nii.gz'))
runs.sort()
runs = runs[:-1]

eventfs = glob.glob(
    os.path.join(design_path.format(sub, ses), '*events.tsv'))
eventfs.sort()

runs = runs[:3]
eventfs = eventfs[:3]

data = []
design = []

for i, (run, eventf) in enumerate(zip(runs, eventfs)):
    print(f'run {i}')
    y = nib.load(run).get_fdata().astype(np.float32)
    dims = y.shape
    # y = np.moveaxis(y, -1, 0)
    # y = y.reshape([y.shape[0], -1])

    n_volumes = y.shape[-1]

    # Load onsets and item presented
    onsets = pd.read_csv(eventf, sep='\t')["onset"].values
    items = pd.read_csv(eventf, sep='\t')["stimnumber"].values
    n_events = len(onsets)

    # Create design matrix
    events = pd.DataFrame()
    events["duration"] = [stimdur] * n_events
    events["onset"] = np.round(onsets)
    events["trial_type"] = items

    # pass in the events data frame. the convolving of the HRF now
    # happens internally
    design.append(
        make_design(events, tr, n_volumes)
        )
    data.append(y)

opt = {'wantlss': 0}
outputdir = 'GLMestimatesingletrialoutputs'

start_time = time.time()
gst = GLM_single(opt)

results = gst.fit(
    design,
    data,
    stimdur,
    tr,
    outputdir=outputdir)


elapsed_time = time.time() - start_time
print(
    'elapsedtime: ',
    f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
)
