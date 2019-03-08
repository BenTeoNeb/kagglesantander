"""
Misc utils
"""
import importlib
import sys
import pandas as pd

from lib.constants import TMP_FOLDER, DATA_FOLDER, SUBMISSION_FOLDER

def make_submission_from_hdf(filename, index=None):
    """
    Format a submission file from a hdf file
    """

    df_preds = pd.read_hdf(TMP_FOLDER + filename + '.hdf', key='df')

    if index is not None:
        submission = (
            pd.DataFrame(df_preds).merge(index.reset_index(), on=df_preds.index)[['ID_code', 0]]
            .rename(columns={0:'target'}))
    else:
        submission = pd.read_csv(DATA_FOLDER + 'sample_submission.csv')
        submission['target'] = df_preds
    submission.to_csv(SUBMISSION_FOLDER + filename + '.csv', index=False)

    return submission

def force_import(custom_module):
    """
    Force module import even if the module is already loaded
    """

    # Check if custom modules are already loaded
    force_module_reload = False
    if custom_module in sys.modules:
        force_module_reload = True

    # Do the import anyway to get the module object
    module = importlib.import_module(custom_module)

    # Force reload if necessary
    if force_module_reload:
        module = importlib.reload(module)
        print("-> Forced custom module reload", module)

    return module
