import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import make_interp_spline, BSpline

names = ["AR_SC", "AR_MC", "Prophet", "BHT_AR_SC", "BHT_AR_MC"]

#              AR_______VAR______Prophet__BHT-AR__BHT-VAR_
data = np.array([[1.0304,  0.0154,  0.0182,  0.0195,  0.0085],  #  1
                 [1.0342,  0.0348,  0.0350,  0.0346,  0.0268],  #  2
                 [1.0343,  0.0502,  0.0209,  0.0472,  0.0298],  #  3
                 [1.0389,  0.0469,  0.0226,  0.0263,  0.0193],  #  4
                 [1.0337,  0.0534,  0.0221,  0.0451,  0.0276],  #  5
                 [1.0380,  0.0690,  0.0252,  0.0245,  0.0223],  #  6
                 [1.0382,  0.0286,  0.0283,  0.0477,  0.0203],  #  7
                 [1.0381,  0.0239,  0.0211,  0.0219,  0.0198],  #  8
                 [1.0378,  0.0745,  0.0307,  0.0208,  0.0204],  #  9
                 [1.0398,  0.0887,  0.0464,  0.0230,  0.0202],])# 10



plt.figure(figsize = (10,5))
plt.title("Synthetic Data NRMSE over Volume",
          fontname = 'Arial',
          fontweight = 'bold', 
          fontstyle = 'oblique',
          fontsize = 16, 
          y = 1)
plt.xlabel('Dataset Volume')
plt.ylabel("NRMSE")
plt.xticks(ticks=[i for i in range(10)],
           labels=[str(i+1) for i in range(10)])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
plt.plot(data[:,1], label="VAR", marker='o')
plt.plot(data[:,2], label="Prophet", marker='o')
plt.plot(data[:,3], label="BHT_AR", marker='o')
plt.plot(data[:,4], label="BHT_VAR", marker='o')
plt.legend(fancybox = True, framealpha = 1, shadow = True, borderpad = 1, ncol = 2, fontsize = 12)
plt.show()



#                 AR_______VAR______Prophet__BHT-AR__BHT-VAR_
data = np.array([[1.0390,  0.0124,  0.0142,  0.0159,  0.0092],  #  1
                 [1.0348,  0.0128,  0.0123,  0.0171,  0.0105],  #  5
                 [1.0346,  0.0210,  0.0226,  0.0217,  0.0182],  # 10
                 [1.0370,  0.0186,  0.0205,  0.0199,  0.0205],])# 15


plt.figure(figsize = (10,5))
plt.title("Synthetic Data NRMSE over Volume",
          fontname = 'Arial',
          fontweight = 'bold', 
          fontstyle = 'oblique',
          fontsize = 16, 
          y = 1)
plt.xlabel('Dataset Volume')
plt.ylabel("NRMSE")
plt.xticks(ticks=[i for i in range(4)],
           labels=['1', '5', '10', '15'])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
plt.plot(data[:,1], label="VAR", marker='o')
plt.plot(data[:,2], label="Prophet", marker='o')
plt.plot(data[:,3], label="BHT_AR", marker='o')
plt.plot(data[:,4], label="BHT_VAR", marker='o')
plt.legend(fancybox = True, framealpha = 1, shadow = True, borderpad = 1, ncol = 2, fontsize = 12)
plt.show()








# plt.figure(figsize = (10,5))
# plt.title("Synthetic Data NRMSE over Volume",
#           fontname = 'Arial',
#           fontweight = 'bold', 
#           fontstyle = 'oblique',
#           fontsize = 16, 
#           y = 0.915)
# plt.xlabel('Dataset Volume')
# plt.ylabel("NRMSE")
# plt.xticks(ticks=[0, 32, 65, 99],
#             labels=['1', '5', '10', '15'])
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)


# x = np.array([0, 1, 2, 3])
# xnew = np.linspace(x.min(), x.max(), 50) 

# for i in range(1, 5):
#     y = data[:, i]
#     spl = make_interp_spline(x, y, k=3)
#     y_smooth = spl(xnew)
#     plt.plot(y_smooth, label = names[i])

# plt.legend(fancybox = True, framealpha = 1, shadow = True, borderpad = 1, ncol = 5, fontsize = 12)
# plt.show()




# plt.figure(figsize = (10,5))
# plt.title("Synthetic Data NRMSE over Volume",
#           fontname = 'Arial',
#           fontweight = 'bold', 
#           fontstyle = 'oblique',
#           fontsize = 16, 
#           y = 0.915)
# plt.xlabel('Dataset Volume')
# plt.ylabel("NRMSE")
# plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#             labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)


# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# xnew = np.linspace(x.min(), x.max(), 1000) 

# for i in range(1, 5):
#     y = data[:, i]
#     spl = make_interp_spline(x, y, k=3)
#     y_smooth = spl(xnew)
#     plt.plot(y_smooth, label = names[i])

# plt.legend(fancybox = True, framealpha = 1, shadow = True, borderpad = 1, ncol = 5, fontsize = 12)
# plt.show()




