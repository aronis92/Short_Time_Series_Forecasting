from functions.utils import adfuller_test, cointegration_test
from functions.MDT_functions import MDT
import matplotlib.pyplot as plt
import tensorly as tl
import pandas as pd
import numpy as np
import time

n_train = 52
n_val = 5
n_test = 5
filepath = 'https://raw.githubusercontent.com/selva86/datasets/master/Raotbl6.csv'
df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
#cointegration_test(df)
X = df.to_numpy()
X = X.T

X_train = X[:, :n_train]
df_train = pd.DataFrame(X_train.T)

counter = 0
indices = []
index = 0
for name, column in df_train.iteritems():
    c = adfuller_test(column, name=column.name)
    if c==0:
        indices.append(index)
    index += 1
    counter += c
    #print('\n')
print(counter, "/", X_train.shape[0], "Series are Stationary")


df_differenced = df_train.diff().dropna()
counter = 0
indices = []
index = 0
for name, column in df_differenced.iteritems():
    c = adfuller_test(column, name=column.name)
    if c==0:
        indices.append(index)
    index += 1
    counter += c
    #print('\n')
print(counter, "/", X_train.shape[0], "Series are Stationary After 1st Differencing")


df_differenced = df_differenced.diff().dropna()
counter = 0
indices = []
index = 0
for name, column in df_differenced.iteritems():
    c = adfuller_test(column, name=column.name)
    if c==0:
        indices.append(index)
    index += 1
    counter += c
    #print('\n')
print(counter, "/", X_train.shape[0], "Series are Stationary After 2nd Differencing")












