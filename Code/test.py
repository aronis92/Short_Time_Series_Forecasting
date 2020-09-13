import pandas as pd
import numpy as np
from fbprophet import Prophet

n_data = 100
n_predictions = 1
i = 1
#n_dim = data.shape[1]

data = pd.read_csv('data/nasdaq100/small/nasdaq100_padding.csv',  nrows = n_data)
# data = data.to_numpy()

ds = pd.date_range('2015-02-24', periods = n_data, freq='D')
ds = pd.DataFrame(ds.date)
ds = ds.rename(columns={0:'ds'})
new_data = pd.concat([ds, data], axis = 1)

X = new_data.iloc[:, [0, i]]
X = X.rename(columns={X.columns[1]:'y'})

m = Prophet()
m.fit(X)

future = m.make_future_dataframe(periods = n_predictions)
future.tail()