# Imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

DATA_FOLDER = '/Users/benjaminfarcy/workdir/kaggle/elo/data/'
PRED_FOLDER = './preds/'

from dataload import load_elo_data
from featureng import feature_engineering
from model import train_lgbm_fold

## Load data

df_train, df_target, df_test, df_merchants, df_transactions = load_elo_data(DATA_FOLDER, load_everything=True)

#%%
train, test = feature_engineering(df_train, df_test, df_transactions)
print("wtd")

remove_cols = ['card_id']
selected_features = list(set(list(train.columns)) - set(remove_cols))
models, importances = train_lgbm_fold(
    train,
    selected_features,
    df_target['target'],
    n_splits=5,
    repeat_cv=1,
    n_max_estimators=10000
    )
