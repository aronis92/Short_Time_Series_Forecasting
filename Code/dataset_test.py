import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from functions.utils import get_data


data = sm.datasets.macrodata.load_pandas().data
X = data[['realgdp', 'realcons', 'realinv', 'realgovt', 'realdpi', 'cpi', 'm1', 'tbilrate', 'unemp', 'pop', 'infl', 'realint']]
X = X.to_numpy()
X = X.T
plt.figure(figsize = (12,5))
plt.title("US Macroeconomic dataset")
plt.plot(X.T)
plt.show()



X, _, _ = get_data(dataset = "inflation", Ns = [52, 1, 1])
plt.figure(figsize = (12,5))
plt.title("Inflation dataset")
plt.plot(X.T)
plt.show()

X, _, _ = get_data(dataset = "nasdaq", Ns = [52, 1, 1])
plt.figure(figsize = (12,5))
plt.title("NASDAQ dataset")
plt.plot(X.T)
plt.show()

X, _, _ = get_data(dataset = "traffic", Ns = [52, 1, 1])
plt.figure(figsize = (12,5))
plt.title("Traffic dataset")
plt.plot(X.T)
plt.show()

X, _, _ = get_data(dataset = "book", Ns = [52, 1, 1])
plt.figure(figsize = (12,5))
plt.title("Book dataset")
plt.plot(X.T)
plt.show()

X, _, _ = get_data(dataset = "yahoo", Ns = [52, 1, 1])
plt.figure(figsize = (12,5))
plt.title("Yahoo dataset")
plt.plot(X.T)
plt.show()