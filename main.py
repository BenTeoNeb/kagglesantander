"""
Main program
"""
# Standard import below
from lib.utils import force_import

#Custom modules to import
dataload = force_import('lib.dataload')
model = force_import('lib.model')
utils = force_import('lib.utils')

if __name__ == '__main__':
    df_train, df_target, df_test = dataload.load_data(read=True, reduce_mem=False)

    remove_cols = ['target', 'ID_code']
    features = list(set(df_train.columns) - set(remove_cols))

    model.train_lgbm_fold_classif(df_train, df_test, features, df_target,
                                  repeat_cv=1, n_splits=4,
                                  n_max_estimators=1000
                                  )

    utils.make_submission_from_hdf('preds_lgbm_classif_CV_0.88763_TR_0.98783')
