import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0.070842, 0.125955, 0.186396, 0.068168, 0.066181],
              [0.080897, 0.129356, 0.149310, 0.081648, 0.076533],
              [0.130220, 0.152955, 0.104766, 0.069972, 0.059880]])

def plot_results(data, title, ytitle):
    # plt.plot(epoch[1:], data[1:])
    epoch = [40, 20, 10]
    plt.figure(figsize = (12,5))
    plt.title(title)
    plt.xlabel('Volume')
    plt.ylabel(ytitle)

    plt.plot(epoch, data)
    
plot_results(X, "Synthetic NRMSE over volume", "NRMSE")




import tensorly as tl
from tensorly.decomposition import tucker


X = np.array([])
