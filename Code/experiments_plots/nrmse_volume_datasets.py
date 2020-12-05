import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import make_interp_spline, BSpline

def get_results(name):
    '''Arrays 4 (percentages 100_75_50_25) x 5 (Algorithms)'''
    if name == "macro":
        macro = np.array([[0.0150, 0.0066, 0.0179, 0.0078, 0.0057],
                          [0.0056, 0.0045, 0.0095, 0.0046, 0.0025],
                          [0.0087, 0.0085, 0.0086, 0.0052, 0.0032],
                          [0.0053, 0.0079, 0.0070, 0.0080, 0.0037]])
        return macro, "US Macroeconomic"
    elif name == "elnino":
        elnino = np.array([[0.0259, 0.0334, 0.0218, 0.0160, 0.0164],
                           [0.0236, 0.0326, 0.1515, 0.0245, 0.0103],
                           [0.0196, 0.0332, 0.0454, 0.0104, 0.0093],
                           [0.0198, 0.0392, 0.0131, 0.0114, 0.0107]])
        return elnino, "El-Nino"
    elif name == "ozone":
        ozone = np.array([[0.6900, 0.2908, 0.3077, 0.3052, 0.1707],
                          [0.0734, 0.1062, 0.4138, 0.0587, 0.0416],
                          [0.2089, 0.2354, 0.2734, 0.1038, 0.1127],
                          [0.4789, 0.4007, 0.3045, 0.2941, 0.1947]])
        return ozone, "Ozone"
    elif name == "night":    
        night = np.array([[0.0680, 0.1310, 0.1865, 0.0453, 0.0826],
                          [0.0880, 0.0813, 0.2802, 0.0950, 0.0705],
                          [0.0830, 0.0720, 0.2028, 0.1074, 0.0546],
                          [0.2657, 0.4195, 0.2226, 0.1192, 0.0718]])
        return night, "Night Visitors"
                 
                 

data, data_name = get_results("ozone")


plt.figure(figsize = (12, 5))
plt.title(data_name + " Dataset",#" Comparison on a 20-point Train set",
          fontname = 'Arial',
          fontweight = 'bold', 
          fontstyle = 'oblique',
          fontsize = 16, 
          y = 1.05)
plt.xlabel('Training Set Volume')
plt.ylabel("NRMSE")
#plt.ylim(0.008, 0.07)
plt.xticks(ticks=[i for i in range(4)],
           labels=['100%', '75%', '50%', '25%'])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
plt.plot(data[:,0], label="AR_SC", marker='o')
plt.plot(data[:,1], label="AR_MC", marker='o')
plt.plot(data[:,2], label="Prophet", marker='o')
plt.plot(data[:,3], label="BHT_AR_SC", marker='o')
plt.plot(data[:,4], label="BHT_AR_MC", marker='o')
plt.legend(fancybox = True, framealpha = 1, shadow = True, borderpad = 1, ncol = 2, fontsize = 12)
plt.show()

