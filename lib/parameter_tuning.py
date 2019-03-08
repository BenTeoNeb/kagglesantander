"""
Parameter tuning with hyperopt
"""

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from hyperopt import tpe, hp, fmin, Trials, space_eval
from sklearn.datasets import make_classification
import json

from lib.constants import TMP_FOLDER
from lib import dataload


def cross_val_score_lgb(x, y, params, n_splits=3,
                        n_max_estimators=20000,
                        early_stopping_rounds=100):
    """
    Cross val score wrapper for lgb binary classifier with
    auc score
    """

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = y.copy().astype('float')
    train_preds = y.copy().astype('float')
    for fold, (train_index, val_index) in enumerate(kfold.split(x, y)):
        trn_data = lgb.Dataset(x[train_index], label=y[train_index])
        val_data = lgb.Dataset(x[val_index], label=y[val_index])
        model = lgb.train(
            params,
            trn_data,
            n_max_estimators,
            valid_sets=[trn_data, val_data],
            verbose_eval=-1,
            early_stopping_rounds=early_stopping_rounds)
        oof_preds[val_index] = model.predict(x[val_index], num_iteration=model.best_iteration)
        train_preds[train_index] = model.predict(x[train_index], num_iteration=model.best_iteration)
    cv_score = roc_auc_score(y, oof_preds)
    tr_score = roc_auc_score(y, train_preds)

    return cv_score, tr_score

def params_helper(args):
    # Learning rate can be put high for hyperparameter search
    # and then set to a lower value for real model training

    fixed_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'verbose': -1,
        'learning_rate': 0.2
    }

    int_params = [
        'max_bin',
        'max_depth',
        'min_data_in_leaf',
        'num_leaves',
        'bagging_freq'
    ]

    opt_params = {}
    for key, value in args.items():
        if key in int_params:
            opt_params[key] = int(value)
        else:
            opt_params[key] = value

    params = {**fixed_params, **opt_params}

    return params

def objective_func(args):

    params = params_helper(args)

    df_train, df_target, _ = dataload.load_data(read=True, reduce_mem=False)

    remove_cols = ['target', 'ID_code']
    features = list(set(df_train.columns) - set(remove_cols))

    x = df_train[features].values
    y = df_target['target'].values

    cv_score, tr_score = cross_val_score_lgb(x, y.ravel(), params)

    # Base loss on cross val score and
    # penalize loss with spread between cv and tr, to avoid overfitting
    loss = (1-cv_score) + 2.0*(abs(cv_score-tr_score))

    print('======================')
    print('params:', params)
    print('loss:' + str(loss))
    print('auc:' + str(cv_score) + ' ' + str(tr_score))
    print('======================')

    return loss

def find_optimal_params(max_evals=1000, write=True):
    param_space = {
        'num_leaves': hp.quniform('num_leaves', 2, 50, 1),
        'max_depth': hp.quniform('max_depth', 2, 10, 1),
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', 20, 100, 1),
        'max_bin': hp.quniform('max_bin', 200, 300, 1),
        'lambda_l1': hp.uniform('lambda_l1', 0, 1),
        'lambda_l2': hp.uniform('lambda_l2', 0, 1),
        'feature_fraction': hp.uniform('feature_fraction', 0.5, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1.0),
        'bagging_freq': hp.quniform('bagging_freq', 1, 100, 2)
    }
    trials = Trials()
    best_classifier = fmin(objective_func,
                           param_space,
                           algo=tpe.suggest,
                           trials=trials,
                           max_evals=max_evals)
    best_params = space_eval(param_space, best_classifier)
    print(best_params)

    if write:
        with open(TMP_FOLDER + 'best_params.json', 'w') as outfile:
            json.dump(best_params, outfile)

    return best_params
