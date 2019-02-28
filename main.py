"""
Main program
"""
# Standard import below
import pandas as pd

from lib.utils import force_import

#Custom modules to import
dataload = force_import('lib.dataload')
model = force_import('lib.model')

if __name__ == '__main__':
    df_train, df_target, df_test = dataload.load_data(read=True)

    df_train.head()
