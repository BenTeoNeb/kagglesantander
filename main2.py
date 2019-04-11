"""
Main program
"""
# Standard import below
from lib.utils import force_import

#Custom modules to import
dataload = force_import('lib.dataload')
model = force_import('lib.model')
utils = force_import('lib.utils')
parameter_tuning = force_import('lib.parameter_tuning')

if __name__ == '__main__':
    best_params = parameter_tuning.find_optimal_params(max_evals=100)
    params = parameter_tuning.params_helper(best_params)

    print("==== ============================== ====")
    print("==== FINISHED HYPERPARAMETER SEARCH ====")
    print("==== ============================== ====")
    print("")
    print(" .. Retraining best w best params .. ")

    #df_train, df_target, _ = dataload.load_data(read=True, reduce_mem=False)

    df_train = pd.read_hdf('./data_tmp/new_train.hdf', key='df')

    remove_cols = ['target', 'ID_code']
    features = list(set(df_train.columns) - set(remove_cols))

    x = df_train[features].values
    y = df_train['target'].values

    cv_score, tr_score = parameter_tuning.cross_val_score_lgb(x, y.ravel(), params)
    print(cv_score, tr_score)
