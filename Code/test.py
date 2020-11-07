import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def plot_results(data, title, ytitle):
    plt.figure(figsize = (12,5))
    plt.title(title)
    plt.xlabel('Dataset Volume')
    plt.ylabel(ytitle)
    plt.xticks(ticks=[0, 1, 2, 3, 4],
               labels=['100 points', '40 points', '20 points', '10 points', '5 points'])
    
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    plt.plot(data[:,0], label="AR", marker='o')
    plt.plot(data[:,1], label="VAR", marker='o')
    plt.plot(data[:,2], label="Prophet", marker='o')
    plt.plot(data[:,3], label="BHT_AR", marker='o')
    plt.plot(data[:,4], label="BHT_VAR", marker='o')
    
    plt.legend(fancybox = True,
               framealpha = 1,
               shadow = True,
               borderpad = 1,
               ncol = 5,
               fontsize = 12)
    
    
#              AR________VAR_______Prophet___BHT-AR____BHT-VAR_
X = np.array([[0.071759, 0.025453, 0.091920, 0.068464, 0.087860],  # 100
              [0.070842, 0.125955, 0.186396, 0.068168, 0.066181],  #  40
              [0.080897, 0.129356, 0.149310, 0.081648, 0.076533],  #  20
              [0.130220, 0.152955, 0.104766, 0.069972, 0.059880],  #  10
              [0.226482, 0.221791, 0.161345, 0.173531, 0.146240]]) #   5

plot_results(X, "Synthetic NRMSE over Volume", "NRMSE")


#              AR________VAR_______Prophet___BHT-AR____BHT-VAR_
X = np.array([[0.000587, 0.003934, 0.005801, 0.000446, 0.000588],  # 100
              [0.000770, 0.001100, 0.001152, 0.000708, 0.000643],  #  40
              [0.001935, 0.002832, 0.001437, 0.000783, 0.000710],  #  20
              [0.001935, 0.002832, 0.001484, 0.001183, 0.001174],  #  10
              [0.002495, 0.001836, 0.004079, 0.001667, 0.001534]]) #   5

plot_results(X, "NASDAQ-100 NRMSE over Volume", "NRMSE")




d = {"A":1}
d["A"] = 2