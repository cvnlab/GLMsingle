from glmdenoise.io.directory import run_bids_directory
import os


def run_public(dataset_url, sub=None, task=None):
    """Download dataset by datalad URL and denoise it.

    Example: `glmdenoise ///workshops/nih-2017/ds000114`
    
    Args:
        dataset_url (str): URL of dataset, e.g. '///workshops/nih-2017/ds000114'
        subject (str, optional): BIDS identifier of one subject to run.
            Defaults to None, meaning all subjects.
        task (str, optional): Name of specific task to run.
            Defaults to None, meaning all tasks.
    """

    missing_datalad_msg = """
    You're trying to run GLMdenoise on a publicly available dataset.
    
    This requires:
    - datalad   >= 0.11.4       (pip install datalad)
    - git-annex >= 6.20180913   (on Ubuntu 19.04: apt install git-annex)
    """
    try:
        import datalad.api
    except ImportError:
        print(missing_datalad_msg)
        # Unix error code for 'Package not installed': 65
        # See /usr/include/asm-generic/errno.h
        exit(65)

    root_dir = os.path.expanduser('~/datalad')
    dataset_dir = os.path.join(root_dir, dataset_url.lstrip('/'))
    try:
        datalad.api.install(
            source=dataset_url, 
            path=dataset_dir, 
            recursive=True, 
            get_data=True
        )
    except datalad.support.exceptions.IncompleteResultsError:
        print('Could not download all files, data potentially incomplete')
    
    run_bids_directory(dataset_dir, sub=sub, task=task)
