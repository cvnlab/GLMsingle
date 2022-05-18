#!/group/tuominen/anaconda3/bin/python3
# Import libraries

import os
import numpy as np
import nibabel as nib
import pandas as pd
from bids import BIDSLayout
from glmsingle.glmsingle import GLM_single

def grab_data(root):
    ''' Use pybids to easily iter over dataset generated using fMRIprep. '''
    # Load data
    raw = BIDSLayout(os.path.join(root,'rawdata'))
    processed = BIDSLayout(os.path.join(root,'derivatives','iter4'))
    # Print summary data
    print(raw)
    print(processed)
    # Get data per run
    bold_info = raw.get(extension='.nii.gz', suffix='bold', return_type='filename')[0]
    metadata = raw.get_metadata(bold_info)
    slicedata = nib.load(bold_info).get_fdata().astype(np.float32)
    ''' To make an event file GLMsingle compatible, calculate total
    acquisition time rounded to nearest second. Total time will represent
    number of rows in new event file. '''
    # Print run data
    tr = metadata['RepetitionTime']
    num_slices = slicedata.shape[3]
    total_dur = round(tr * num_slices)
    print('TR:', tr, '\n Number of slices acquired:', num_slices,
          '\n Total run duration:', total_dur)
    return raw, processed, tr, num_slices, total_dur

def make_new_df(root, raw, subjects, total_dur):
    ''' Make an empty dataframe, # of rows is equal to total run duration (seconds),
    because we will upsample the TR to 1s to reduce jitter. # of columns equal to
    total unique event types in .tsv file. In this case 3 unique event types exist,
    as we ignore the event type representing intertrial intervals (ITIs). '''
    # Grab event files
    eventfiles = raw.get(subject=subjects, extension='.tsv', suffix='events', return_type='filename')
    # Load into pandas
    for eventfile in eventfiles:
        design = pd.read_csv(eventfile, sep='\t')
        # Parse each event type, as indicated by value, where each integer represents a different event type
        CSplus_shock = design.loc[design['value'] == 3]
        CSplus = design.loc[design['value'] == 1]
        CSminus = design.loc[design['value'] == 2]
        # Make new df
        new_design = pd.DataFrame(columns = ['TR', 'CSplusshock','CSplus','CSminus'])
        new_design['TR'] = [i for i in range(1, (total_dur+1))]
        # For each stimulus type, mark the onset of the stimulus with 1. All other values should be 0
        for cs in [CSplus_shock, CSplus, CSminus]:
            for i in range(0, len(cs.index)):
                tr_start = int(round(cs.iloc[i,0]))
                if cs is CSplus_shock:
                    new_design.iloc[tr_start, :] =  [tr_start, 1, 0, 0]
                elif cs is CSplus:
                    new_design.iloc[tr_start, :] =  [tr_start, 0, 1, 0]
                elif cs is CSminus:
                    new_design.iloc[tr_start, :] =  [tr_start, 0, 0, 1]
                else:
                    new_design.iloc[tr_start, :] =  [tr_start, 0, 0, 0]
        # Fill NaNs with 0s
        new_design.fillna(0, inplace=True)
        # Give the event file a new name
        new_name = raw.parse_file_entities(eventfile)
        new_name['suffix'] = 'events_glmsingle'
        pattern = 'sub-{}_ses-{}_task-{}_run-{}_{}.csv'
        # Save the event file in derivatives, exclude TR column
        new_design.to_csv(os.path.join(root, 'derivatives', 'iter4',
        'sub-'+new_name['subject'], 'ses-'+new_name['session'], 'func',
        pattern.format(new_name['subject'],
        new_name['session'], new_name['task'], new_name['run'],
        new_name['suffix'])), columns=['CSplusshock','CSplus','CSminus'],
        index=False, header=False)
        print('glmsingle event file saved:', new_name['subject'], 'run:', new_name['run'])
def upsample_bold(processed, tr, tr_new, num_slices, subjects):
    ''' Upsample task-based fMRI data to 1s using pyslicetime scripts:
    https://github.com/Charestlab/pyslicetime/blob/master/example_nipype.py.
    Output will have suffix _stc appended.'''
    from slicetime.nipype_interface import SliceTime
    # Grab all preprocessed runs
    bold_runs = processed.get(subject=subjects, extension='.nii.gz', suffix='bold', return_type='filename')
    # Filter for resolution - here we exclude res-2
    bold_runsf = [b for b in bold_runs if 'res-2' not in b]
    # Import parameters
    tr_old = tr
    tr_new = tr_new
    num_slices = num_slices
    # sliceorder needs to be 1-based (see fakeout below)
    slicetimes = np.flip(np.arange(0, tr_old, tr_old/num_slices, dtype=object)).tolist()
    # Iter over runs
    for run in bold_runsf:
        print('Upsampling:', run)
        st = SliceTime()
        st.inputs.in_file = run
        st.inputs.tr_old = tr
        st.inputs.tr_new = tr_new
        st.inputs.slicetimes = slicetimes
        res = st.run()

def smooth_bold(processed, smooth_num, subjects):
    ''' Smooth data prior to first-level analysis using FSL in nipype.
    Output will have suffix _smooth appended.'''
    from nipype.interfaces.fsl import Smooth
    # Grab all upsampled runs
    bold_runs = processed.get(subject=subjects, extension='.nii.gz', suffix='stc', return_type='filename')
    # Smooth data before inputting to GLMsingle
    for run in bold_runs:
        print('Smoothing:', os.path.basename(run))
        path = os.path.dirname(run)
        os.chdir(path)
        sm = Smooth()
        sm.inputs.output_type = 'NIFTI_GZ'
        sm.inputs.in_file = run
        sm.inputs.fwhm = smooth_num
        res = sm.run()

def run_1st_level(processed, stimdur, tr_new, subjects):
    ''' Input several runs per subject to perform cross-validation
    in GLMdenoise (step 3 of GLMsingle). Imaging data and event data
    should be appended into separate lists. '''
    # Get all subjects
    data = []
    design = []
    # Loop over subjects
    for sub in subjects:
        # Grab all upsampled and smoothed runs
        bold_runs = processed.get(subject=subjects, extension='.nii.gz', suffix='smooth', return_type='filename')
        # Grab all GLMsingle-compatible event files
        eventfiles = processed.get(subject=subjects, extension='.csv', suffix='glmsingle', return_type='filename')
        # Append all runs for a single subject session
        # fMRI data
        for run in bold_runs:
            r = nib.load(run).get_fdata().astype(np.float32)
            data.append(r)
        # design matrices
        for event in eventfiles:
            ev = pd.read_csv(event, header=None).to_numpy(dtype='float32')
            design.append(ev)
        # Initialize model
        gs = GLM_single()
        # Name output folders
        metadata = processed.parse_file_entities(bold_runs[0])
        outdir = os.path.join(os.path.dirname(bold_runs[0]), metadata['task'] + '_concat_glmsingle')
        # Run GLMsingle
        results = gs.fit(
        design,
        data,
        stimdur,
        tr_new,
        outputdir=outdir)
        # Clear data
        data.clear()
        design.clear()

def main():
    # Hardcoded file path to existing BIDS dataset
    root = '/group/tuominen/EmoSal_BIDS'
    # Stimulus duration
    stimdur = 4
    # Desired repetition time
    tr_new = 1
    # Desired smoothing
    smooth_num = 6
    # Desired subjects
    subjects = ['avl003', 'avl004', 'avl005', 'avl006', 'avl007', 'avl009', 'avl010',
    'avl011', 'avl012', 'avl-013r', 'avl014', 'avl016', 'avl017', 'avl018', 'avl019',
    'avl021', 'avl022', 'avl024', 'avl025', 'avl027', 'avl028', 'avl200', 'avl201']
    # Execute functions
    ''' Can comment out intermediate steps if already complete. '''
    raw, processed, tr, num_slices, total_dur = grab_data(root)
    make_new_df(root, raw, subjects, total_dur)
    upsample_bold(processed, tr, tr_new, num_slices, subjects)
    smooth_bold(processed, smooth_num, subjects)
    run_1st_level(processed, stimdur, tr_new, subjects)
if __name__ == "__main__":
# execute only if run as a script
    main()
