"""
Data loading
Missing data handling
"""

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

def reduce_mem_usage(df, verbose=True):
    """
    Downcast types to reduce memory usage
    """

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            'Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'
            .format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def impute_from_key(data, col_to_impute, key, verbose=False):
    """
    In data, where col_to_impute is null, impute it if key is unique identifier.
    """

    before_na = data[col_to_impute].isna().sum()
    key_uniques = data[key + [col_to_impute]].drop_duplicates()
    key_uniques = key_uniques.loc[(key_uniques[col_to_impute].notnull())]

    # take only unique merchants for the `key` combination
    key_unique = key_uniques.groupby(key)[col_to_impute].count().reset_index(name = 'n_imputes')
    key_unique = key_unique.loc[key_unique['n_imputes']==1]
    key_unique = key_unique.merge(key_uniques, on = key)
    key_unique.drop('n_imputes', axis=1, inplace=True)

    # rename the merchant_id so we can join it more easily later on
    key_unique.columns = key + ['imputed_col']

    # merge back to coalesce
    data = data.merge(key_unique, on=key, how='left')
    data[col_to_impute] = data[col_to_impute].combine_first(data['imputed_col'])

    data.drop('imputed_col', axis=1, inplace=True)

    after_na = data[col_to_impute].isna().sum()

    if verbose:
        print('Imputed', before_na - after_na,
              'values. From', before_na, 'to ', after_na, 'null values')
    return data

def load_elo_data(data_folder, load_everything=False, write=True, read=False):
    """
    Load ELO data
    """
    if read:
        print('... Reading ...')
        df_train = reduce_mem_usage(pd.read_hdf(data_folder + 'df_train.hdf', key='df'))
        df_test = reduce_mem_usage(pd.read_hdf(data_folder + 'df_test.hdf', key='df'))
        df_target = pd.read_hdf(data_folder + 'df_target.hdf', key='df')
        df_merchants = pd.read_hdf(data_folder + 'df_merchants.hdf', key='df')
        df_transactions = reduce_mem_usage(pd.read_hdf(data_folder + 'df_transactions.hdf', key='df'))
    else:
        df_header = pd.read_excel(data_folder + 'Data_Dictionary.xlsx')
        display(df_header.head(len(df_header)))

        print('-- Train data')
        df_train = reduce_mem_usage(pd.read_csv(data_folder + 'train.csv'))
        df_target = pd.DataFrame(df_train['target'])

        print('-- Test data')
        df_test = reduce_mem_usage(pd.read_csv(data_folder + 'test.csv'))
        df_test['first_active_month'] = (
            df_test['first_active_month']
            .fillna(value=df_test['first_active_month'].value_counts().index[0]))

        print('-- Merchants')
        df_merchants = reduce_mem_usage(pd.read_csv(data_folder + 'merchants.csv'))
        #df_merchants['category_2'].fillna(6.0, inplace=True)
        #df_merchants.fillna(0, inplace=True)

        print('-- Transactions')
        df_transactions = reduce_mem_usage(pd.read_csv(data_folder + 'new_merchant_transactions.csv'))
        df_transactions['source'] = 0
        if load_everything:
            df_all_transactions = reduce_mem_usage(pd.read_csv(data_folder + 'historical_transactions.csv'))
            df_all_transactions['source'] = 1
            df_transactions = pd.concat([df_transactions, df_all_transactions])
        # Missing data
        # Impute merchants
        key = ['card_id','city_id','category_1','installments','category_3',
                  'merchant_category_id','category_2','state_id','subsector_id']
        df_transactions = impute_from_key(df_transactions, 'merchant_id', key, verbose=True)
        # Fill what is possible from merchants for category_2
        buf = df_transactions.merge(df_merchants[['merchant_id', 'category_2']], on=['merchant_id'])
        df_transactions['category_2'] = buf['category_2_x'].combine_first(buf['category_2_y'])
        df_transactions['category_2'].fillna(6.0, inplace=True)
        # New category for category_3
        df_transactions['category_3'].fillna('D', inplace=True)
        # Re-impute merchants
        key = ['card_id','city_id','category_1','installments','category_3',
                  'merchant_category_id','category_2','state_id','subsector_id']
        df_transactions = impute_from_key(df_transactions, 'merchant_id', key, verbose=True)
        # Deal with other merchants
        df_transactions['merchant_id'].fillna('M_ID_UNKNOWN', inplace=True)

        if write:
            print('... Writing ...')
            df_train.to_hdf(data_folder + 'df_train.hdf', 'df')
            df_test.to_hdf(data_folder + 'df_test.hdf', 'df')
            df_target.to_hdf(data_folder + 'df_target.hdf', 'df')
            df_merchants.to_hdf(data_folder + 'df_merchants.hdf', 'df')
            df_transactions.to_hdf(data_folder + 'df_transactions.hdf', 'df')

    return df_train, df_target, df_test, df_merchants, df_transactions
