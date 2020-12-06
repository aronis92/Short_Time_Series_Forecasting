import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import make_interp_spline, BSpline

def get_results(name):
    '''Arrays 4 (volumes 1_5_10_15) x 5 (Algorithms)'''
    if name == "yahoo":
        macro = np.array([[0.0510, 0.0314, 0.0432, 0.0551, 0.0276],
                          [0.1223, 0.1244, 0.0986, 0.1286, 0.0878],
                          [0.1383, 0.1470, 0.1254, 0.1320, 0.1530],
                          [0.1626, 0.1818, 0.1345, 0.1294, 0.1415]])
        return macro, "Yahoo Stocks"
    elif name == "nasdaq":
        elnino = np.array([[0.2046, 0.2191, 0.1998, 0.2141, 0.1938],
                           [0.2205, 0.2373, 0.2014, 0.2143, 0.2046],
                           [2.3643, 1.6568, 0.2112, 0.2321, 0.2022],
                           [3.8674, 2.1464, 0.2353, 0.2237, 0.2031]])
        return elnino, "NASDAQ"



data, data_name = get_results("nasdaq")


plt.figure(figsize = (12, 5))
plt.title(data_name + " Dataset",#" Comparison on a 20-point Train set",
          fontname = 'Arial',
          fontweight = 'bold', 
          fontstyle = 'oblique',
          fontsize = 16, 
          y = 1.05)
plt.xlabel('Forecasting Horizon')
plt.ylabel("NRMSE")
plt.ylim(0.19, 0.255)
plt.xticks(ticks=[i for i in range(4)],
           labels=['1', '5', '10', '15'])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
plt.plot(data[:,0], label="AR_SC", marker='o')
plt.plot(data[:,1], label="AR_MC", marker='o')
plt.plot(data[:,2], label="Prophet", marker='o')
plt.plot(data[:,3], label="BHT_AR_SC", marker='o')
plt.plot(data[:,4], label="BHT_AR_MC", marker='o')
plt.legend(fancybox = True, framealpha = 1, shadow = True, borderpad = 1, ncol = 3, fontsize = 12)
plt.show()

