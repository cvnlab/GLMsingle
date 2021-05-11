import argparse
import os
from glmsingle.io.directory import run_bids_directory
from glmsingle.io.public import run_public


def main():
    """This function gets called when the user executes `glmsingle`.

    It defines and interprets the console arguments, then calls
    the relevant python code.
    """

    parser = argparse.ArgumentParser(prog='glmsingle')
    parser.add_argument(
        'dataset',
        nargs='?',
        default='.',
        help='Data directory containing BIDS, or name of public dataset.')
    parser.add_argument(
        '--subject',
        default=None,
        help='Subject number. If not specified, will run for each subject.')
    parser.add_argument(
        '--task',
        default=None,
        help='Task name. If not specified, will run on all tasks.')
    parser.add_argument(
        '--stimdur',
        default=None,
        help='Stimulus duration. How long was the stimulus presented for?')
    args = parser.parse_args()
    if args.dataset[:3] == '///':
        run_public(args.dataset, args.subject, args.task, args.stimdur)
    else:
        run_bids_directory(args.dataset, args.subject, args.task, args.stimdur)
