import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from functions.utils import get_data
from stationarity_tests import stationarity_tests

#                             Index  Var x Time
datasets = ['macro', #__________0     12 x 203
            'elnino', #_________1     12 x 61
            'copper', #_________2      5 x 25
            'fertility', #______3    192 x 52
            'stackloss', #______4      4 x 21
            'nightvisitors', #__5      8 x 56
            'mortality', #______6      2 x 72
            'ozone', #__________7      8 x 203
            'inflation', #______8      8 x 123
            'nasdaq', #_________9     82 x 40560
            'traffic', #________10   228 x 40
            'yahoo', #__________11     5 x 2469
            'book'] #___________12     3 x sum(Ns)

Ns = [50, 1, 1]
data_name = datasets[9]

X, _, _ = get_data(data_name, Ns)

plt.figure(figsize = (12,5))
plt.plot(X.T)
plt.show()



ds = [i for i in range(1, 8)]
rs = [i for i in range(2, 11)]
print("Dataset: " + data_name + " \n")
stationarity_tests(data_name, ds, rs, Ns)






