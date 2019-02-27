        tmp = perform_agg_dict(df_transactions, agg_dict_ref_col, groupcol)
        print("---- Aggregations done")
        tmp["purchase_date_diff"] = (
            tmp["purchase_date_card_id_max"] - tmp["purchase_date_card_id_min"]
        ).dt.days
        tmp["purchase_date_average"] = (
            tmp["purchase_date_diff"] / tmp["card_id_card_id_size"]
        )
        tmp["purchase_date_max_uptonow"] = (
            datetime.datetime.today() - tmp["purchase_date_card_id_max"]
        ).dt.days
        tmp["purchase_date_min_uptonow"] = (
            datetime.datetime.today() - tmp["purchase_date_card_id_min"]
        ).dt.days
        timekeeper(start)

        if write:
            start = time.time()
            print("--- Writing ...")
            tmp.to_hdf(filename, "df")
            timekeeper(start)

    return tmp


def feature_engineering_transactions_aggregations_1(
    df_transactions, write=True, read=False
):

    if read:
        start = time.time()
        print("--- Reading Transaction aggregations at card id level - 1")
        tmp = pd.read_hdf("./tmp_1.hdf", key="df")
        timekeeper(start)
    else:
        start = time.time()
        # Get card_id level infos
        print("--- Transaction aggregations at card id, month_lag level")
        groupcol = ["card_id", "month_lag"]
        agg_dict_ref_col = {
            "purchase_amount": ["sum", "max", "min", "mean", "median", "var", "std"],
            "installments": ["nunique", "sum", "mean"],
        }
        tmp = perform_agg_dict(df_transactions, agg_dict_ref_col, groupcol)

        # regroup at card_id level
        tmp = tmp.groupby("card_id").agg(["mean", "std"])
        tmp.columns = ["_".join(col).strip() for col in tmp.columns.values]
        tmp.reset_index(inplace=True)

        print("---- Aggregations done")
        timekeeper(start)

        if write:
            start = time.time()
            print("--- Writing ...")
            tmp.to_hdf("./tmp_1.hdf", "df")
            timekeeper(start)

    return tmp


def feature_engineering(df_train, df_test, df_transactions, df_merchants):

    # Flag target outliers
    df_train["outlier"] = 0
    df_train.loc[df_train["target"] < -30, "outlier"] = 1
    df_test["outlier"] = 0
    # Merge train/test
    train_test = pd.concat([df_train.drop("target", axis=1), df_test])

    df_transactions, df_merchants = feature_engineering_prepare_transactions(
        df_transactions, df_merchants, read=True
    )
    tmp = feature_engineering_transactions_aggregations_0(
        df_transactions, read=True, id="full"
    )
    tmp1 = feature_engineering_transactions_aggregations_1(df_transactions, read=True)
    tmp2 = feature_engineering_transactions_aggregations_0(
        df_transactions[df_transactions["source"] == 0], id="source0", read=True
    )
    tmp2.columns = [col + "_source0" for col in tmp2.columns]
    tmp2.rename(columns={"card_id_source0": "card_id"}, inplace=True)
    tmp3 = feature_engineering_transactions_aggregations_0(
        df_transactions[df_transactions["source"] == 1], id="source1", read=True
    )
    tmp3.columns = [col + "_source1" for col in tmp3.columns]
    tmp3.rename(columns={"card_id_source1": "card_id"}, inplace=True)
    print("--- Merging back")
    start = time.time()
    train_test = train_test.merge(tmp, on="card_id", how="left")
    print("    -")
    train_test = train_test.merge(tmp1, on="card_id", how="left")
    print("    -")
    train_test = train_test.merge(tmp2, on="card_id", how="left")
    print("    -")
    train_test = train_test.merge(tmp3, on="card_id", how="left")
    print("---- Merging done")
    timekeeper(start)

    # Based on prefered merchant of customer: merchant_id_card_id_get_mode, merge back merchant info
    train_test.rename(
        columns={"merchant_id_card_id_get_mode": "merchant_id"}, inplace=True
    )
    subset_merchant = [
        "merchant_id",
        "numerical_1",
        "numerical_2",
        "avg_sales_lag3",
        "avg_purchases_lag3",
        "active_months_lag3",
        "avg_sales_lag6",
        "avg_purchases_lag6",
        "active_months_lag6",
        "avg_sales_lag12",
        "avg_purchases_lag12",
        "active_months_lag12",
    ]
    train_test = train_test.merge(
        df_merchants[subset_merchant], on=["merchant_id"], how="left"
    )

    # Main data
    print("- TrainTest")
    # Extract info from date
    train_test["first_active_month_category"] = train_test["first_active_month"]
    train_test["first_active_month_date"] = pd.to_datetime(
        train_test["first_active_month"]
    )
    train_test["first_active_month"] = train_test["first_active_month_date"].dt.month
    train_test["first_active_year"] = train_test["first_active_month_date"].dt.year
    train_test["elapsed_time"] = (
        datetime.date(2018, 2, 1) - train_test["first_active_month_date"].dt.date
    ).dt.days
    train_test = train_test.drop("first_active_month_date", axis=1)

    # Fix types
    for col in [
        "active_months_lag3",
        "active_months_lag6",
        "active_months_lag12",
        "numerical_1",
        "numerical_2",
    ]:
        train_test[col] = train_test[col].astype(np.int8)
    for col in [
        "avg_sales_lag3",
        "avg_sales_lag6",
        "avg_sales_lag12",
        "avg_purchases_lag3",
        "avg_purchases_lag6",
        "avg_purchases_lag12",
    ]:
        train_test[col] = train_test[col].astype(np.float64)

    # Label encoding
    print("-- Label encoding")
    col_tolabelencode = ["first_active_year", "first_active_month_category"]
    le = LabelEncoder()
    for col in col_tolabelencode:
        le.fit(train_test[col].unique())
        train_test[col] = le.transform(train_test[col])

    # One hot encoding
    print("-- One hot encoding")
    col_toonehot = ["feature_1", "feature_2"]
    train_test = pd.get_dummies(train_test, columns=col_toonehot)

    # Remove some columns
    remove_cols = [
        "card_id",
        "merchant_id",
        "merchant_id_card_id_get_mode_source0",
        "merchant_id_card_id_get_mode_source1",
        "purchase_date_card_id_min",
        "purchase_date_card_id_min_source0",
        "purchase_date_card_id_min_source1",
        "purchase_date_card_id_max_source0",
        "purchase_date_card_id_max_source1",
        "purchase_date_card_id_max",
        "outlier",
    ]
    selected_features = list(set(list(train_test.columns)) - set(remove_cols))
    train_test = train_test[selected_features]

    # Resplit
    print("- Resplit train/test")
    train = train_test.iloc[: len(df_train)]
    test = train_test.iloc[len(df_train) :]

    print("- Write")
    start = time.time()
    train.to_hdf("./train_fe.hdf", "df")
    test.to_hdf("./test_fe.hdf", "df")
    print("-- Done")
    timekeeper(start)

    return train, test, df_transactions, df_merchants, tmp
