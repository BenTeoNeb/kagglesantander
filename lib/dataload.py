"""
Data loading
Missing data handling
"""

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from lib.constants import DATA_FOLDER, TMP_FOLDER


def reduce_mem_usage(df, verbose=True):
    """
    Downcast types to reduce memory usage
    """

    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def impute_from_key(data, col_to_impute, key, verbose=False):
    """
    In data, where col_to_impute is null, impute it if key is unique identifier.
    """

    before_na = data[col_to_impute].isna().sum()
    key_uniques = data[key + [col_to_impute]].drop_duplicates()
    key_uniques = key_uniques.loc[(key_uniques[col_to_impute].notnull())]

    # take only unique merchants for the `key` combination
    key_unique = (
        key_uniques.groupby(key)[col_to_impute].count().reset_index(name="n_imputes")
    )
    key_unique = key_unique.loc[key_unique["n_imputes"] == 1]
    key_unique = key_unique.merge(key_uniques, on=key)
    key_unique.drop("n_imputes", axis=1, inplace=True)

    # rename the merchant_id so we can join it more easily later on
    key_unique.columns = key + ["imputed_col"]

    # merge back to coalesce
    data = data.merge(key_unique, on=key, how="left")
    data[col_to_impute] = data[col_to_impute].combine_first(data["imputed_col"])

    data.drop("imputed_col", axis=1, inplace=True)

    after_na = data[col_to_impute].isna().sum()

    if verbose:
        print(
            "Imputed",
            before_na - after_na,
            "values. From",
            before_na,
            "to ",
            after_na,
            "null values",
        )
    return data


def load_data(write=True, read=False, reduce_mem=True, features=False):
    """
    Load Data
    """
    if read:
        print("... Reading ...")
        if features:
            df_train = pd.read_hdf(TMP_FOLDER + "df_train_fe.hdf", key="df")
            df_target = pd.read_hdf(TMP_FOLDER + "df_target_fe.hdf", key="df")
            df_test = pd.read_hdf(TMP_FOLDER + "df_test_fe.hdf", key="df")
        else:
            df_train = pd.read_hdf(TMP_FOLDER + "df_train.hdf", key="df")
            df_target = pd.read_hdf(TMP_FOLDER + "df_target.hdf", key="df")
            df_test = pd.read_hdf(TMP_FOLDER + "df_test.hdf", key="df")
        if reduce_mem:
            df_train = reduce_mem_usage(df_train)
            df_target = reduce_mem_usage(df_target)
            df_test = reduce_mem_usage(df_test)

    else:
        print("-- Train data")
        df_train = pd.read_csv(DATA_FOLDER + "train.csv")
        df_target = pd.DataFrame(df_train["target"])

        print("-- Test data")
        df_test = pd.read_csv(DATA_FOLDER + "test.csv")

        if reduce_mem:
            df_train = reduce_mem_usage(df_train)
            df_test = reduce_mem_usage(df_test)
            df_target = reduce_mem_usage(df_target)

        if write:
            print("... Writing ...")
            df_train.to_hdf(TMP_FOLDER + "df_train.hdf", "df")
            df_test.to_hdf(TMP_FOLDER + "df_test.hdf", "df")
            df_target.to_hdf(TMP_FOLDER + "df_target.hdf", "df")
    print("-- Done")
    return df_train, df_target, df_test
