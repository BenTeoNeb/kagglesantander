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
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

def train_lgbm_fold(df, df_test, features, df_target,
                    repeat_cv=1, n_splits=5, n_max_estimators=10000,
                   verbose_round=100):

    print('== INIT')

    params = {
        'class_weights': None,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'max_depth': 5,
        'learning_rate': 0.01,
        'verbose': 0,
        'colsample_bytree': 0.9,
        'min_child_samples': 20,
        'min_split_gain': 0.01,
        'reg_alpha': 0.01,
        'reg_lambda': 0.01,
        'subsample': 0.9,
        'subsample_freq': 1,
        'subsample_for_bin': 200000,
        'early_stopping_round': 100,
        'n_estimators': n_max_estimators
    }

    X = df[features].values
    y = df_target.values

    importances = pd.DataFrame()
    models = []

    print('== START MODEL TRAIN')
    df_oof_preds = pd.DataFrame(np.zeros((len(df), repeat_cv)))
    df_preds = pd.DataFrame(np.zeros((len(df_test), repeat_cv)))
    for i in range(repeat_cv):
        print("== REPEAT CV", i)
        preds = y.copy()
        kfold = KFold(n_splits = n_splits, shuffle = True, random_state = i)
        rmse_list = []
        cv_index=0
        for fold, (train_index, val_index) in enumerate(kfold.split(X,y)):
            print("==== CV", fold)
            trn_x, trn_y = X[train_index], y[train_index]
            val_x, val_y = X[val_index], y[val_index]
            model = lgb.LGBMRegressor(**params)
            model.fit(
                trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                verbose=verbose_round
            )
            preds[val_index] = model.predict(X[val_index],
                                             num_iteration=model.best_iteration_)
            rmse = np.sqrt(mean_squared_error(y[val_index], preds[val_index]))
            rmse_list.append(rmse)
            cv_index=cv_index+1

            df_preds[i] += model.predict(df_test[features]) / n_splits

            imp_df = pd.DataFrame()
            imp_df['feature'] = features
            imp_df['gain'] = model.feature_importances_
            imp_df['model'] = i
            imp_df['fold'] = fold
            importances = pd.concat([importances, imp_df], axis=0, sort=False)

            models.append(model)

        rmse_cv = np.mean(rmse_list)

        print("RMSE:", rmse_cv)
        df_oof_preds[i] = preds

    return models, importances, df_oof_preds, df_preds

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

def make_predictions(df, features, models):
    list_preds = []
    df['pred_mean'] = 0
    for index, model in enumerate(models):
        pred_name = 'model_' + str(index)
        df[pred_name] = model.predict(df[features], num_iteration=model.best_iteration_)
        list_preds.append(pred_name)
        df['pred_mean'] = df['pred_mean'] + df[pred_name]
    df['pred_mean'] = df['pred_mean'] / len(models)

    return df, list_preds
