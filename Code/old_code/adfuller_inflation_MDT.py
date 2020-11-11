from functions.utils import adfuller_test, cointegration_test
from functions.MDT_functions import MDT
import matplotlib.pyplot as plt
import tensorly as tl
import pandas as pd
import numpy as np
import time

n_train = 52
r = 8
filepath = 'https://raw.githubusercontent.com/selva86/datasets/master/Raotbl6.csv'
df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
X = df.to_numpy()
X = X.T
df_train = pd.DataFrame(X[:, :n_train].T)

df_differenced = df_train.diff().dropna()
df_differenced = df_differenced.diff().dropna()
X_d = df_differenced.to_numpy()
X_d = X_d.T



X_hat, _ = MDT(X_d, r)
X_hat_vec = tl.unfold(X_hat, -1).T
df_hat = pd.DataFrame(X_hat_vec.T)

plt.figure(figsize = (12,5))
plt.plot(X_hat_vec[0,...].T)
plt.show()

counter = 0
indices = []
index = 0
for name, column in df_hat.iteritems():
    c = adfuller_test(column, name=column.name)
    if c==0:
        indices.append(index)
    index += 1
    counter += c
    #print('\n')
print(counter, "/", X_hat_vec.shape[0], "Series are Stationary After 2nd Differencing and MDT")





def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc









