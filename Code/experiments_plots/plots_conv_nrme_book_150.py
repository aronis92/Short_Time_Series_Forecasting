import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import make_interp_spline, BSpline


def get_results():
    
    #                        20_1_1        20_5_5    20_10_10    20_15_15
    ar_nrmse = np.array([[0.00995764,	0.0133547,	0.0163530,	0.0149475],
                         [0.01033100,	0.0134014,	0.0171235,	0.0149447],
                         [0.01079490,	0.0134583,	0.0184946,	0.0147677],
                         [0.01142900,	0.0135787,	0.0198199,	0.0147478],
                         [0.01205060,	0.0138050,	0.0206960,	0.0148289],
                         [0.01245180,	0.0138345,	0.0210862,	0.0149548],
                         [0.01245910,	0.0135686,	0.0210956,	0.0151185],
                         [0.01159940,	0.0134228,	0.0208315,	0.0153325],
                         [0.00855693,	0.0133627,	0.0203684,	0.0156029],
                         [0.00607841,	0.0133357,	0.0197533,	0.0157833],
                         [0.00783122,	0.0133227,	0.0190203,	0.0155094],
                         [0.00857614,	0.0133165,	0.0182082,	0.0150113],
                         [0.00787706,	0.0133143,	0.0173840,	0.0147575],
                         [0.00534366,	0.0133150,	0.0166699,	0.0147531],
                         [0.00475496,	0.0133183,	0.0162575,	0.0148411]])

    #                        20_1_1       20_3_3      20_5_5      20_7_7
    var_nrmse = np.array([[0.01612490,	0.0155441,	0.0217135,	0.0147411],
                          [0.01498190,	0.0154604,	0.0217135,	0.0147411],
                          [0.01318390,	0.0154170,	0.0217135,	0.0147411],
                          [0.00955547,	0.0153925,	0.0217135,	0.0147411],
                          [0.00554311,	0.0153768,	0.0217135,	0.0147411],
                          [0.00352717,	0.0153649,	0.0217135,	0.0147411],
                          [0.00299362,	0.0153541,	0.0217135,	0.0147411],
                          [0.00298988,	0.0153427,	0.0217135,	0.0147411],
                          [0.00302146,	0.0153295,	0.0217135,	0.0147411],
                          [0.00303334,	0.0153133,	0.0217135,	0.0147411],
                          [0.00303547,	0.0152930,	0.0217135,	0.0147411],
                          [0.00303462,	0.0152670,	0.0217135,	0.0147411],
                          [0.00303335,	0.0152331,	0.0217135,	0.0147411],
                          [0.00303236,	0.0151893,	0.0217135,	0.0147411],
                          [0.00303171,	0.0151363,	0.0217135,	0.0147411]])

    #                      20_1_1     20_3_3      20_5_5      20_7_7   
    ar_conv = np.array([[1.121730,	1.0800800,	0.975637,	1.0099600],
                        [0.228175,	0.1377990,	0.193908,	0.2737560],
                        [0.210323,	0.0702726,	0.196028,	0.1516500],
                        [0.206397,	0.0867223,	0.198636,	0.0802841],
                        [0.202159,	0.1477210,	0.199170,	0.0570116],
                        [0.196332,	0.2199140,	0.197714,	0.0495962],
                        [0.189920,	0.1904620,	0.194912,	0.0504708],
                        [0.186208,	0.1104190,	0.191273,	0.0597077],
                        [0.196349,	0.0630425,	0.187107,	0.0837050],
                        [0.226392,	0.0409324,	0.182746,	0.1399460],
                        [0.227541,	0.0300316,	0.178295,	0.2250110],
                        [0.213271,	0.0240912,	0.173866,	0.2067390],
                        [0.209510,	0.0208778,	0.170015,	0.1200290],
                        [0.221592,	0.0194057,	0.167706,	0.0739076],
                        [0.243017,	0.0192785,	0.168264,	0.0553496]])

    #                       20_1_1        20_3_3       20_5_5      20_7_7  
    var_conv = np.array([[0.9091480,	1.03800000,	0.75758000,	0.7761190],
                         [0.0378718,	0.02407790,	0.00107698,	0.0011047],
                         [0.0678832,	0.01375210,	0.00107698,	0.0011047],
                         [0.1006110,	0.00822183,	0.00107698,	0.0011047],
                         [0.0927599,	0.00568691,	0.00107698,	0.0011047],
                         [0.0557935,	0.00510962,	0.00107698,	0.0011047],
                         [0.0278581,	0.00570049,	0.00107698,	0.0011047],
                         [0.0170459,	0.00691713,	0.00107698,	0.0011047],
                         [0.0148882,	0.00860837,	0.00107698,	0.0011047],
                         [0.0141027,	0.01081170,	0.00107698,	0.0011047],
                         [0.0138528,	0.01361550,	0.00107698,	0.0011047],
                         [0.0137803,	0.01709030,	0.00107698,	0.0011047],
                         [0.0137629,	0.02121580,	0.00107698,	0.0011047],
                         [0.0137617,	0.02577220,	0.00107698,	0.0011047],
                         [0.0137640,	0.03022530,	0.00107698,	0.0011047]])



    return ar_nrmse, var_nrmse, ar_conv, var_conv


ar_nrmse, var_nrmse, ar_conv, var_conv = get_results()





names = ["AR_SC", "AR_MC", "Prophet", "BHT_AR_SC", "BHT_AR_MC"]
colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple', 
          'tab:pink', 'tab:brown', 'tab:olive', 'tab:gray', 'tab:cyan']
titles = ["Convergence Value Comparison", "NRMSE Value Comparison"]
subtitles = ["Convergence Value", "NRMSE Value"]
n_rows = 2
n = 3
c_1 = 6 #0,1 #2,3 #4,5 #6,7
c_2 = 7
fig = plt.figure(figsize = (12,9))
gs = fig.add_gridspec(n_rows, hspace = 0.3)
axs = gs.subplots(sharex=False, sharey=False)

for k in range(2):
    plt.sca(axs[k])
    plt.xticks(ticks=[i for i in range(15)],
                labels=[str(i+1) for i in range(15)])
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)


fig.suptitle("Train: 150,    Validation: "+ str((n>0)*n*5 + (n==0)*1), 
             fontname = 'Arial', 
             fontweight = 'bold', 
             fontstyle = 'oblique',
             fontsize = 18, 
             y = 0.96)#0.915)

for i in range(n_rows):
    axs[i].set_title(titles[i], fontname = 'Arial', fontweight = 'bold', fontstyle = 'oblique', fontsize = 14)
    axs[i].set_xlabel('Iteration', fontname = 'Arial', fontstyle = 'oblique', fontsize = 12)
    axs[i].set_ylabel(subtitles[i], fontname = 'Arial', fontstyle = 'oblique', fontsize = 12)

axs[0].plot(ar_conv[:, n], label = "BHT_AR_SC Convergence Value", c = colors[c_1], marker='o')
axs[0].plot(var_conv[:, n], label = "BHT_AR_MC Convergence Value", c = colors[c_2], marker='o')   
axs[1].plot(ar_nrmse[:, n], label = "BHT_AR_SC NRMSE Value", c = colors[c_1], marker='o')
axs[1].plot(var_nrmse[:, n], label = "BHT_AR_MC NRMSE Value", c = colors[c_2], marker='o')
axs[0].legend(fancybox = True, framealpha = 1, shadow = True, borderpad = 1, ncol = 1, fontsize = 12)
axs[1].legend(fancybox = True, framealpha = 1, shadow = True, borderpad = 1, ncol = 1, fontsize = 12)
plt.show() #loc=9, 











# fig = plt.figure(figsize = (10,12))
# gs = fig.add_gridspec(5, hspace=0)
# axs = gs.subplots(sharex=True, sharey=False)
# fig.suptitle("BHT_AR_SC Convergences", 
#              fontname = 'Arial', 
#              fontweight = 'bold', 
#              fontstyle = 'oblique',
#              fontsize = 16, 
#              y = 0.915)
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
# for i in range(len(colors)):
#     axs[i].plot(ar_conv[:, i], c = colors[i])
# plt.show()












