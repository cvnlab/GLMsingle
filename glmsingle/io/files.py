from glmsingle.glmsingle import GLM_single
from glmsingle.io.output import Output
from glmsingle.io.input import load_nifti
from pprint import pprint
import nibabel
import pandas


def run_files(bold_files, event_files, tr, stimdur=None, out=None):
    """Run glmdenoise on the provided image and event files

    Args:
        bold_files (list): List of filepaths to .nii bold files
        event_files (list): List of filepaths to .tsv event files
        tr (float): Repetition time used across scans
        stimdur (float): stimulus duration
    """

    msg = 'need same number of image and event files'
    assert len(bold_files) == len(event_files), msg
    if out is None:
        out = Output()
    out.configure_from(sample_file=bold_files[0])
    data = [load_nifti(f) for f in bold_files]
    design = [pandas.read_csv(f, delimiter='\t') for f in event_files]

    opt = {'wantlss': 0}
    outputdir = 'GLMsingletrialoutputs'

    gst = GLM_single(opt)

    gst.fit(
        design,
        data,
        stimdur,
        tr,
        outputdir=outputdir)
