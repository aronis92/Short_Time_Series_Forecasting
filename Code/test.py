import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0.071759, 0.025453, 0.091920, 0.068464, 0.087860],#100
              [0.070842, 0.125955, 0.186396, 0.068168, 0.066181], #40
              [0.080897, 0.129356, 0.149310, 0.081648, 0.076533], #20
              [0.130220, 0.152955, 0.104766, 0.069972, 0.059880]])#10

def plot_results(data, title, ytitle):
    # plt.plot(epoch[1:], data[1:])
    epoch = [100, 40, 20, 10]
    plt.figure(figsize = (12,5))
    plt.title(title)
    plt.xlabel('Volume')
    plt.ylabel(ytitle)
    plt.xticks(ticks=[0, 1, 2, 3], labels=['100 points', '40 points', '20 points', '10 points'])
    plt.plot(data[:,0], label="AR")
    plt.plot(data[:,1], label="VAR")
    plt.plot(data[:,2], label="Prophet")
    plt.plot(data[:,3], label="BHT_AR")
    plt.plot(data[:,4], label="BHT_VAR")
    plt.legend()
plot_results(X, "Synthetic NRMSE over volume", "NRMSE")




import tensorly as tl
from tensorly.decomposition import tucker


X = np.array([])
