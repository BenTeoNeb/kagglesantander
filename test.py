print("hello")

## Load data
from dataload import load_elo_data
df_train, df_target, df_test, df_merchants, df_transactions = load_elo_data(DATA_FOLDER, load_everything=True)
