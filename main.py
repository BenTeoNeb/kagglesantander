from lib import func_AA

func_AA.hello()

## Load data
from dataload import load_elo_data

df_train, df_target, df_test, df_merchants, df_transactions = load_elo_data(
    DATA_FOLDER, load_everything=True
)

df = df2(df.stuff(1).stuff(2))
