#%%
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.stats import mode
import datetime
import time

def perform_agg_dict(data, agg_dict_ref, groupcol):
    agg_dict = {}
    for col in agg_dict_ref.keys():
        agg_dict[col] = {}
        for aggfunc in agg_dict_ref[col]:
            if isinstance(aggfunc, str):
                func_name = aggfunc
            else:
                func_name = aggfunc.__name__
            agg_dict[col][col + '_' + "-".join(groupcol) + '_' + func_name] = aggfunc

    tmp = data.groupby(groupcol).agg(agg_dict)
    tmp.columns = tmp.columns.droplevel()
    tmp = tmp.reset_index()
    return tmp

def get_mode(series):
    """
    Get the modality of a serie.
    """
    return mode(series)[0][0]

def timekeeper(start):
    elapsed = time.time() - start
    print("Took", np.round(elapsed), '[s]',
          np.round(elapsed/60), '[m]')

def feature_engineering_prepare_transactions(df_transactions, df_merchants, write=True, read=False):

    if read:
        start = time.time()
        print("--- Reading prepared transactions and merchants")
        df_transactions = pd.read_hdf('./df_transactions_prepared.hdf', key='df')
        df_merchants = pd.read_hdf('./df_merchants_prepared.hdf', key='df')
        timekeeper(start)
    else:
        start = time.time()
        # Merchants
        print("- Merchants")
        df_merchants['numerical_1'] = np.round(df_merchants['numerical_1'] / 0.009914905 + 5.79639, 0)
        df_merchants['numerical_2'] = np.round(df_merchants['numerical_2'] / 0.009914905 + 5.79639, 0)
        drop_merchants = ['merchant_category_id', 'merchant_group_id', 'subsector_id',
                      'category_1', 'city_id', 'state_id', 'category_2']
        df_merchants = df_merchants.drop(drop_merchants, axis=1).drop_duplicates()
        df_merchants = df_merchants[~df_merchants.duplicated(subset=['merchant_id'], keep='first')]
        df_merchants = df_merchants.replace([np.inf, -np.inf], np.nan)
        for col in ['avg_sales_lag3', 'avg_sales_lag6', 'avg_sales_lag12',
                    'avg_purchases_lag3', 'avg_purchases_lag6', 'avg_purchases_lag12',
                   ]:
            df_merchants[col] = (df_merchants[col].fillna(df_merchants[col].median()))

        # Use describe to get a most frequent dummy merchant
        describe = df_merchants.describe(include='all').reset_index()
        describe = describe[(describe['index'] == 'top') | (describe['index'] == '50%')].transpose()
        describe = pd.DataFrame(describe[2].combine_first(describe[8]))
        dummy_merchant = describe.transpose().drop('index', axis=1)
        dummy_merchant['merchant_id'] = 'M_ID_UNKNOWN'
        df_merchants = df_merchants.append(dummy_merchant)

        # Transactions
        print("- Transactions")
        df_transactions['purchase_amount'] = np.round(df_transactions['purchase_amount'] / 0.00150265118 + 497.06, 2)
        df_transactions['purchase_date'] = pd.to_datetime(df_transactions['purchase_date'])
        df_transactions['year'] = df_transactions['purchase_date'].dt.year
        df_transactions['weekofyear'] = df_transactions['purchase_date'].dt.weekofyear
        df_transactions['month'] = df_transactions['purchase_date'].dt.month
        df_transactions['dayofweek'] = df_transactions['purchase_date'].dt.dayofweek
        df_transactions['weekend'] = (df_transactions['purchase_date'].dt.weekday >=5).astype(int)
        df_transactions['hour'] = df_transactions['purchase_date'].dt.hour
        df_transactions['month_diff'] = ((datetime.datetime.today() - df_transactions['purchase_date']).dt.days)//30
        df_transactions['month_diff'] += df_transactions['month_lag']
        ## enrich with merchants
        df_transactions = df_transactions.merge(df_merchants, on='merchant_id', how='left')
        # Fix types
        for col in ['active_months_lag3', 'active_months_lag6', 'active_months_lag12',
                    'numerical_1', 'numerical_2']:
            df_transactions[col] = df_transactions[col].astype(np.int8)
        for col in ['avg_sales_lag3', 'avg_sales_lag6', 'avg_sales_lag12',
                    'avg_purchases_lag3', 'avg_purchases_lag6', 'avg_purchases_lag12',
                   ]:
            df_transactions[col] = df_transactions[col].astype(np.float64)
        # Label encoding of df_transactions

        print("-- Label encoding")
        col_tolabelencode = ['authorized_flag',
                             'category_4',
                             'category_1',
                             'category_3',
                             'most_recent_sales_range',
                             'most_recent_purchases_range'
                             ]
        le = LabelEncoder()
        for col in col_tolabelencode:
            le.fit(df_transactions[col].unique())
            df_transactions[col] = le.transform(df_transactions[col])

        timekeeper(start)
        if write:
            start = time.time()
            print("--- Writing ...")
            df_transactions.to_hdf('./df_transactions_prepared.hdf', 'df')
            df_merchants.to_hdf('./df_merchants_prepared.hdf', 'df')
            timekeeper(start)

    return df_transactions, df_merchants

def feature_engineering_transactions_aggregations_0(df_transactions, id="default", write=True, read=False):

    if id != "default":
        filename = './tmp_' + str(id) + '.hdf'
    else:
        filename = './tmp_0.hdf'

    if read:
        start = time.time()
        print("--- Reading Transaction aggregations at card id level - 0")
        tmp = pd.read_hdf(filename, key='df')
        timekeeper(start)
    else:
        start = time.time()
        # Get card_id level infos
        print("--- Transaction aggregations at card id level")
        groupcol = ['card_id']
        agg_dict_ref_col = {
            'purchase_date': ['min', 'max', 'nunique'],
            'year': ['min', 'max', 'median', 'nunique'],
            'weekofyear': ['min', 'max', 'mean', 'nunique'],
            'month': ['min', 'max', 'mean', 'nunique'],
            'dayofweek': ['min', 'max', 'mean', 'nunique', get_mode],
            'weekend': ['mean', 'nunique'],
            'hour': ['min', 'mean', 'median', 'max', 'nunique'],
            'purchase_amount': ['sum', 'max', 'min', 'mean', 'median', 'var', 'std'],
            'numerical_1': ['sum', 'max', 'min', 'mean', 'var', 'std'],
            'numerical_2': ['sum', 'max', 'min', 'mean', 'var', 'std'],
            'avg_sales_lag3': ['sum', 'max', 'min', 'mean', 'var', 'std'],
            'avg_sales_lag6': ['sum', 'max', 'min', 'mean', 'var', 'std'],
            'avg_sales_lag12': ['sum', 'max', 'min', 'mean', 'var', 'std'],
            'avg_purchases_lag3': ['sum', 'max', 'min', 'mean', 'var', 'std'],
            'avg_purchases_lag6': ['sum', 'max', 'min', 'mean', 'var', 'std'],
            'avg_purchases_lag12': ['sum', 'max', 'min', 'mean', 'var', 'std'],
            'month_lag': ['mean', 'max', 'min', 'std'],
            'card_id': ['size', 'count'],
            'city_id': ['nunique', get_mode],
            'state_id': ['nunique', get_mode],
            'subsector_id': ['nunique', get_mode],
            'merchant_id': ['nunique', get_mode],
            'merchant_category_id': ['nunique', get_mode],
            'installments': ['nunique', get_mode],
            'authorized_flag': ['nunique', 'mean', get_mode],
            'category_4': ['nunique', 'mean', get_mode],
            'category_1': ['nunique', 'mean', get_mode],
            'category_3': ['nunique', get_mode],
            'most_recent_sales_range': ['nunique', get_mode],
            'most_recent_purchases_range': ['nunique', get_mode],
            'month_diff': ['mean']
        }
        tmp = perform_agg_dict(df_transactions, agg_dict_ref_col, groupcol)
        print('---- Aggregations done' )
        tmp['purchase_date_diff'] = (tmp['purchase_date_card_id_max'] -
                                     tmp['purchase_date_card_id_min']).dt.days
        tmp['purchase_date_average'] = tmp['purchase_date_diff']/tmp['card_id_card_id_size']
        tmp['purchase_date_max_uptonow'] = (datetime.datetime.today() - tmp['purchase_date_card_id_max']).dt.days
        tmp['purchase_date_min_uptonow'] = (datetime.datetime.today() - tmp['purchase_date_card_id_min']).dt.days
        timekeeper(start)

        if write:
            start = time.time()
            print("--- Writing ...")
            tmp.to_hdf(filename, 'df')
            timekeeper(start)

    return tmp

def feature_engineering_transactions_aggregations_1(df_transactions, write=True, read=False):

    if read:
        start = time.time()
        print("--- Reading Transaction aggregations at card id level - 1")
        tmp = pd.read_hdf('./tmp_1.hdf', key='df')
        timekeeper(start)
    else:
        start = time.time()
        # Get card_id level infos
        print("--- Transaction aggregations at card id, month_lag level")
        groupcol = ['card_id', 'month_lag']
        agg_dict_ref_col = {
            'purchase_amount': ['sum', 'max', 'min', 'mean', 'median', 'var', 'std'],
            'installments': ['nunique', 'sum', 'mean'],
        }
        tmp = perform_agg_dict(df_transactions, agg_dict_ref_col, groupcol)

        # regroup at card_id level
        tmp = tmp.groupby('card_id').agg(['mean', 'std'])
        tmp.columns = ['_'.join(col).strip() for col in tmp.columns.values]
        tmp.reset_index(inplace=True)

        print('---- Aggregations done' )
        timekeeper(start)

        if write:
            start = time.time()
            print("--- Writing ...")
            tmp.to_hdf('./tmp_1.hdf', 'df')
            timekeeper(start)

    return tmp

def feature_engineering(df_train, df_test, df_transactions, df_merchants):

    # Flag target outliers
    df_train['outlier'] = 0
    df_train.loc[df_train['target'] < -30, 'outlier'] = 1
    df_test['outlier'] = 0
    # Merge train/test
    train_test = pd.concat([df_train.drop('target', axis=1), df_test])

    df_transactions, df_merchants = feature_engineering_prepare_transactions(df_transactions, df_merchants, read=True)
    tmp = feature_engineering_transactions_aggregations_0(df_transactions, read=True, id="full")
    tmp1 = feature_engineering_transactions_aggregations_1(df_transactions, read=True)
    tmp2 = feature_engineering_transactions_aggregations_0(df_transactions[df_transactions['source'] == 0],
                                                           id="source0",
                                                           read=True)
    tmp2.columns = [col + '_source0' for col in tmp2.columns]
    tmp2.rename(columns={'card_id_source0': 'card_id'}, inplace=True)
    tmp3 = feature_engineering_transactions_aggregations_0(df_transactions[df_transactions['source'] == 1],
                                                           id="source1",
                                                           read=True)
    tmp3.columns = [col + '_source1' for col in tmp3.columns]
    tmp3.rename(columns={'card_id_source1': 'card_id'}, inplace=True)
    print('--- Merging back' )
    start = time.time()
    train_test = train_test.merge(tmp, on='card_id', how='left')
    print('    -' )
    train_test = train_test.merge(tmp1, on='card_id', how='left')
    print('    -' )
    train_test = train_test.merge(tmp2, on='card_id', how='left')
    print('    -' )
    train_test = train_test.merge(tmp3, on='card_id', how='left')
    print('---- Merging done' )
    timekeeper(start)

    # Based on prefered merchant of customer: merchant_id_card_id_get_mode, merge back merchant info
    train_test.rename(columns={"merchant_id_card_id_get_mode": "merchant_id"}, inplace=True)
    subset_merchant = ['merchant_id',
                       'numerical_1', 'numerical_2',
                       'avg_sales_lag3', 'avg_purchases_lag3', 'active_months_lag3',
                       'avg_sales_lag6', 'avg_purchases_lag6', 'active_months_lag6',
                       'avg_sales_lag12', 'avg_purchases_lag12', 'active_months_lag12',
                      ]
    train_test = train_test.merge(df_merchants[subset_merchant],
                     on=['merchant_id'], how='left')

    """
    # One hot encoding of df_transactions
    print("-- One hot encoding")
    col_toonehot = [
        'category_3',
        'category_2',
        'most_recent_sales_range',
        'most_recent_purchases_range'
        ]
    df_transactions = pd.get_dummies(df_transactions, columns=col_toonehot)

    # Split transactions based on city_id.
    df_transactions_nocity = pd.DataFrame(df_transactions[df_transactions['city_id'] == -1])
    df_transactions_city = pd.DataFrame(df_transactions[df_transactions['city_id'] != -1])
    """

    # Main data
    print("- TrainTest")
    # Extract info from date
    train_test['first_active_month_category'] = train_test['first_active_month']
    train_test['first_active_month_date'] = pd.to_datetime(train_test['first_active_month'])
    train_test['first_active_month'] = train_test['first_active_month_date'].dt.month
    train_test['first_active_year'] = train_test['first_active_month_date'].dt.year
    train_test['elapsed_time'] = (datetime.date(2018, 2, 1) - train_test['first_active_month_date'].dt.date).dt.days
    train_test = train_test.drop('first_active_month_date', axis=1)

    # Fix types
    for col in ['active_months_lag3', 'active_months_lag6', 'active_months_lag12',
            'numerical_1', 'numerical_2']:
        train_test[col] = train_test[col].astype(np.int8)
    for col in ['avg_sales_lag3', 'avg_sales_lag6', 'avg_sales_lag12',
                'avg_purchases_lag3', 'avg_purchases_lag6', 'avg_purchases_lag12',
               ]:
        train_test[col] = train_test[col].astype(np.float64)

    # Label encoding
    print("-- Label encoding")
    col_tolabelencode = ['first_active_year',
                         'first_active_month_category']
    le = LabelEncoder()
    for col in col_tolabelencode:
        le.fit(train_test[col].unique())
        train_test[col] = le.transform(train_test[col])

    # One hot encoding
    print("-- One hot encoding")
    col_toonehot = [
        'feature_1',
        'feature_2',
        ]
    train_test = pd.get_dummies(train_test, columns=col_toonehot)

    # Remove some columns
    remove_cols = ['card_id', 'merchant_id',
                   'merchant_id_card_id_get_mode_source0',
                   'merchant_id_card_id_get_mode_source1',
                   'purchase_date_card_id_min',
                   'purchase_date_card_id_min_source0',
                   'purchase_date_card_id_min_source1',
                   'purchase_date_card_id_max_source0',
                   'purchase_date_card_id_max_source1',
                   'purchase_date_card_id_max',
                   'outlier']
    selected_features = list(set(list(train_test.columns)) - set(remove_cols))
    train_test = train_test[selected_features]

    # Resplit
    print("- Resplit train/test")
    train = train_test.iloc[:len(df_train)]
    test = train_test.iloc[len(df_train):]

    print("- Write")
    start = time.time()
    train.to_hdf('./train_fe.hdf', 'df')
    test.to_hdf('./test_fe.hdf', 'df')
    print("-- Done")
    timekeeper(start)

    return train, test, df_transactions, df_merchants, tmp
