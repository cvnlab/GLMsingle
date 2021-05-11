from glmsingle.io.files import run_files
from glmsingle.io.bids import BidsDirectory
from glmsingle.io.output import Output


def run_bids_directory(directory='.', sub_num=None, sub=None, task=None, stimdur=None):
    """Run glmdenoise on a whole or part of a dataset in a BIDS directory

    Args:
        directory (str, optional): Root data directory containing BIDS.
            Defaults to '.'
        sub_num (int, optional): Number of one subject to run.
            Defaults to None.
        sub (string, optional): BIDS identifier of one subject to run.
            Defaults to None.
        task (string, optional): Name of specific task to run.
            Defaults to None.
    """

    bids = BidsDirectory(directory)
    bids.index()
    return run_bids(bids, sub_num=sub_num, sub=sub, task=task, stimdur=stimdur)


def run_bids(bids, sub_num=None, sub=None, task=None, ses=None, stimdur=None):
    """Recursively run GLMsingle on subjects, tasks, sessions

    This uses a bids object that is able to find data in the BIDS
    directory it represents.
    See `run_bids_directory` to call this with a directory path.

    Args:
        bids (glmsingle.io.bids.BidsDirectory): Wrapper for pybids
        sub_num (int, optional): Number of one subject to run.
            Defaults to None.
        sub (str, optional): BIDS identifier of one subject to run.
            Defaults to None.
        task (str, optional): Name of specific task to run.
            Defaults to None.
        ses (str, optional): Session identifier to run.
            Defaults to None, meaning run all sessions one by one

    """
    if sub and task and ses:
        return run_bids_subset(bids, sub, task, ses, stimdur)
    elif sub and task:
        sessions = bids.get_sessions_for_task_and_subject(task, sub)
        for ses in sessions:
            run_bids(bids, sub=sub, task=task, ses=ses, stimdur=stimdur)
        if sessions == []:
            run_bids_subset(bids, sub, task, stimdur=stimdur)
    elif sub:
        tasks = bids.get_tasks_for_subject(sub)
        for task in tasks:
            run_bids(bids, sub=sub, task=task, stimdur=stimdur)
    elif sub_num:
        sub = bids.subject_id_from_number(sub_num)
        assert sub, 'Could not match subject index to a subject ID'
        run_bids(bids, sub=sub, stimdur=stimdur)
    else:
        subs = bids.get_preprocessed_subjects_ids()
        for sub in subs:
            run_bids(bids, sub=sub, stimdur=stimdur)


def run_bids_subset(bids, sub, task, ses=None, stimdur=None):
    """Run GLMdenoise on a subset of files in a BIDS dir defined by the args

    This uses a bids object that is able to find data in the BIDS
    directory it represents.
    See `run_bids_directory` to call this with a directory path.

    Args:
        bids (glmdenoise.io.bids.BidsDirectory): Wrapper for pybids
        sub (str): BIDS identifier of one subject to run.
        task (str): Name of specific task to run.
        ses (str, optional): Session identifier to run.
            Defaults to None, meaning there are no sessions.

    """
    bold_files = bids.get_filepaths_bold_runs(sub, task, ses)
    if not bold_files:
        msg = 'No preprocessed runs found for subject {} task {} session {}'
        print(msg.format(sub, task, ses))
        return
    event_files = bids.get_filepaths_event_runs(sub, task, ses)
    bold_files, event_files = bids.match_run_files(bold_files, event_files)
    metas = bids.get_metas_boldfiles(bold_files)
    key = 'RepetitionTime'
    trs = [meta[key] for meta in metas if key in meta]
    assert trs, 'RepetitionTime not specified in metadata'
    assert len(set(trs)) == 1, 'RepetitionTime varies across runs'
    out = Output()
    out.fit_bids_context(bids, sub=sub, task=task, ses=ses, stimdur=stimdur)
    return run_files(bold_files, event_files, tr=trs[0], stimdur=stimdur, out=out)
