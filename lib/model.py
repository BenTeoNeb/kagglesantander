#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import warnings

from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, roc_auc_score, auc
from sklearn.tree import DecisionTreeRegressor

import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error

import seaborn as sns
import json
from lib.constants import TMP_FOLDER

import matplotlib.pyplot as plt

def train_lgbm_fold_classif(df, df_test, features, df_target,
                            repeat_cv=1, n_splits=5, n_max_estimators=10000,
                            verbose_round=100,
                            write=True
                            ):

    print('== INIT ==')

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'bagging_fraction': 0.5919630103966947,
        'bagging_freq': 4,
        'feature_fraction': 0.7482831185653955,
        'lambda_l1': 0.14085228532750096,
        'lambda_l2': 0.5484258671438155,
        'max_bin': 285,
        'max_depth': 2,
        'min_data_in_leaf': 49,
        'num_leaves': 2,
        'learning_rate': 0.1
        }
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'bagging_fraction': 0.8559810628748616,
        'bagging_freq': 40,
        'feature_fraction': 0.5007475282515805,
        'lambda_l1': 0.8838329545574001,
        'lambda_l2': 0.011652675357265607,
        'max_bin': 241,
        'max_depth': 10,
        'min_data_in_leaf': 55,
        'num_leaves': 2,
        'learning_rate': 0.05
        }
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'bagging_fraction': 0.5152176302473519,
        'bagging_freq': 72,
        'feature_fraction': 0.5187053325849631,
        'lambda_l1': 0.6835775824393563,
        'lambda_l2': 0.3102783166293553,
        'max_bin': 204,
        'max_depth': 2,
        'min_data_in_leaf': 40,
        'num_leaves': 2,
        'learning_rate': 0.05
        }

    X = df[features].values
    y = df_target.values.ravel()

    importances = pd.DataFrame()

    print('== START MODEL TRAIN')
    df_oof_preds = pd.DataFrame(np.zeros((len(df), repeat_cv)))
    df_preds = pd.DataFrame(np.zeros((len(df_test), repeat_cv)))
    for i in range(repeat_cv):
        print("== REPEAT CV", i)
        oof_preds = y.copy().astype('float')
        train_preds = y.copy().astype('float')
        #kfold = KFold(n_splits=n_splits, shuffle=True, random_state=i)
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=i)
        for fold, (train_index, val_index) in enumerate(kfold.split(X, y)):
            print("==== CV", fold)
            trn_data = lgb.Dataset(X[train_index], label=y[train_index])
            val_data = lgb.Dataset(X[val_index], label=y[val_index])
            model = lgb.train(
                params,
                trn_data,
                n_max_estimators,
                valid_sets=[trn_data, val_data],
                verbose_eval=verbose_round,
                early_stopping_rounds=100)

            oof_preds[val_index] = model.predict(X[val_index],
                                                 num_iteration=model.best_iteration)

            train_preds[train_index] = model.predict(X[train_index],
                                                     num_iteration=model.best_iteration)

            df_preds[i] += model.predict(df_test[features],
                                         num_iteration=model.best_iteration) / n_splits

            imp_df = pd.DataFrame()
            imp_df['feature'] = features
            imp_df['gain'] = model.feature_importance()
            imp_df['model'] = i
            imp_df['fold'] = fold
            importances = pd.concat([importances, imp_df], axis=0, sort=False)

        cv_score = roc_auc_score(y, oof_preds)
        tr_score = roc_auc_score(y, train_preds)

        print("REPEAT CV:", i, "CV SCORE:", cv_score, "TR SCORE", tr_score)
        df_oof_preds[i] = oof_preds

        filename = None
        if write:
            cv_score_str = "CV_{:<7.5f}".format(cv_score)
            tr_score_str = "TR_{:<7.5f}".format(tr_score)
            root_filename = 'lgbm_classif_' + cv_score_str + '_' + tr_score_str
            pred_root_filename = 'preds_' + root_filename
            filename = root_filename + '.hdf'
            df_preds[i].to_hdf(TMP_FOLDER + 'preds_' + filename, 'df')
            df_oof_preds[i].to_hdf(TMP_FOLDER + 'oof_' + filename, 'df')
            with open(TMP_FOLDER + root_filename + '_params.json', 'w') as outfile:
                json.dump(params, outfile)

    return importances, df_oof_preds, df_preds, pred_root_filename

def plot_importances(importances_, num_features=2000):
    mean_gain = importances_[['gain', 'feature']].groupby('feature').mean()
    importances_['mean_gain'] = importances_['feature'].map(mean_gain['gain'])
    plt.figure(figsize=(8, 18))
    data_imp = importances_.sort_values('mean_gain', ascending=False)
    per_feature = data_imp['model'].nunique() * data_imp['fold'].nunique()
    print(per_feature)
    sns.barplot(x='gain', y='feature', data=data_imp[:per_feature*num_features])
    plt.tight_layout()
    plt.savefig('importances.png')
    plt.show()

    return data_imp
