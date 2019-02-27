"""
Code dump on feature selection, need to tidy this up. Ideas are:
* Measure univariate pdf difference between df_test and df_train and discard feature when too high
https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/77537
* Use of boruta package: https://github.com/scikit-learn-contrib/boruta_py
* Measure difference between feature importance on model and feature importance when model is fitted to noise
https://www.kaggle.com/ogrellier/feature-selection-with-null-importances
https://academic.oup.com/bioinformatics/article/26/10/1340/193348
* Adversarial feature selection
"""
# Code dump

# Kolmogorov-Smirnov
from scipy.stats import ks_2samp
list_p_value =[]

for i in tqdm(df_train_columns):
    list_p_value.append(ks_2samp(df_test[i] , df_train[i])[1])

Se = pd.Series(list_p_value, index = df_train_columns).sort_values()
list_discarded = list(Se[Se < .1].index)
