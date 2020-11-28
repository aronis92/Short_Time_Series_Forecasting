import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import make_interp_spline, BSpline


def get_results():
    
    #                       20_1_1        20_3_3      20_5_5      20_7_7      20_9_9
    ar_nrmse = np.array([[0.0187236,    0.0416975,  0.0226790,	0.0247324,	0.0243951],
                         [0.0142376,	0.0350507,	0.0216637,	0.0241558,	0.0243979],
                         [0.0132172,	0.0348684,	0.0216516,	0.0242366,	0.0241345],
                         [0.0131082,	0.0407604,	0.0216714,	0.0239625,	0.0241432],
                         [0.0131058,	0.0483836,	0.0216789,	0.0238860,	0.0241427],
                         [0.0131169,	0.0489642,	0.0216810,	0.0238672,	0.0241386],
                         [0.0131278,	0.0417118,	0.0216814,	0.0238622,	0.0241362],
                         [0.0131363,	0.0345469,	0.0216815,	0.0238608,	0.0241352],
                         [0.0131424,	0.0330957,	0.0216815,	0.0238604,	0.0241347],
                         [0.0131466,	0.0381760,	0.0216815,	0.0238603,	0.0241346],
                         [0.0131495,	0.0465376,	0.0216815,	0.0238602,	0.0241345],
                         [0.0131514,	0.0499563,	0.0216815,	0.0238602,	0.0241345],
                         [0.0131527,	0.0443346,	0.0216815,	0.0238602,	0.0241344],
                         [0.0131535,	0.0362698,	0.0216815,	0.0238602,	0.0241344],
                         [0.0131541,	0.0328165,	0.0216815,	0.0238602,	0.0241344]])

    #                        20_1_1       20_3_3      20_5_5      20_7_7      20_9_9
    var_nrmse = np.array([[0.03194850,	0.0492938,	0.0227231,	0.0320201,	0.0377349],
                          [0.07680560,	0.0508151,	0.0228856,	0.0322303,	0.0360845],
                          [0.05252750,	0.0342932,	0.0203912,	0.0323076,	0.0353008],
                          [0.01921540,	0.0262570,	0.0420555,	0.0321941,	0.0349266],
                          [0.01115370,	0.0259613,	0.0268869,	0.0319177,	0.0347383],
                          [0.01074580,	0.0260507,	0.0217184,	0.0315683,	0.0346382],
                          [0.01073110,	0.0261089,	0.0208197,	0.0312454,	0.0345828],
                          [0.01033440,	0.0261401,	0.0205100,	0.0310055,	0.0345511],
                          [0.00950735,	0.0261559,	0.0203639,	0.0308541,	0.0345327],
                          [0.00825799,	0.0261638,	0.0202809,	0.0307691,	0.0345218],
                          [0.00663767,	0.0261676,	0.0202281,	0.0307252,	0.0345153],
                          [0.00476502,	0.0261695,	0.0201931,	0.0307038,	0.0345114],
                          [0.00290562,	0.0261704,	0.0201709,	0.0306936,	0.0345091],
                          [0.00200207,	0.0261708,	0.0201589,	0.0306889,	0.0345077],
                          [0.00317274,	0.0261710,	0.0201563,	0.0306867,	0.0345069]])

    #                      20_1_1     20_3_3      20_5_5      20_7_7      20_9_9
    ar_conv = np.array([[1.102320,	1.224620,	1.317520,	0.953046,	0.946225],
                        [0.272964,	0.381029,	0.258900,	0.259539,	0.300315],
                        [0.219494,	0.291688,	0.225804,	0.264767,	0.210755],
                        [0.210971,	0.265648,	0.223888,	0.212223,	0.164938],
                        [0.209511,	0.273319,	0.223866,	0.185145,	0.158480],
                        [0.208969,	0.277766,	0.223886,	0.181751,	0.158252],
                        [0.208684,	0.271075,	0.223891,	0.181444,	0.158211],
                        [0.208515,	0.262204,	0.223892,	0.181414,	0.158150],
                        [0.208410,	0.256890,	0.223892,	0.181411,	0.158116],
                        [0.208343,	0.258733,	0.223892,	0.181410,	0.158100],
                        [0.208300,	0.268590,	0.223892,	0.181410,	0.158092],
                        [0.208272,	0.276905,	0.223892,	0.181410,	0.158089],
                        [0.208254,	0.273486,	0.223892,	0.181410,	0.158087],
                        [0.208243,	0.264604,	0.223892,	0.181410,	0.158087],
                        [0.208235,	0.257927,	0.223892,	0.181410,	0.158086]])

    #                       20_1_1        20_3_3          20_5_5      20_7_7      20_9_9
    var_conv = np.array([[1.2884000,	0.987948000,	1.04926000,	0.8148250,	1.1439800],
                         [0.1906410,	0.264072000,	0.08999740,	0.0802411,	0.0702909],
                         [0.1289520,	0.287624000,	0.11811400,	0.0549347,	0.0510134],
                         [0.0846980,	0.123203000,	0.09505980,	0.0492809,	0.0384358],
                         [0.0392902,	0.060659800,	0.05297780,	0.0523645,	0.0307742],
                         [0.0200092,	0.030053200,	0.02576020,	0.0613584,	0.0262040],
                         [0.0157486,	0.014559900,	0.01442680,	0.0763676,	0.0234864],
                         [0.0155674,	0.007157200,	0.01008730,	0.0958414,	0.0218676],
                         [0.0167782,	0.003746530,	0.00844730,	0.1111940,	0.0209021],
                         [0.0185592,	0.002174830,	0.00771211,	0.1080180,	0.0203285],
                         [0.0205689,	0.001436170,	0.00728818,	0.0854657,	0.0199926],
                         [0.0225364,	0.001085280,	0.00700259,	0.0608705,	0.0198030],
                         [0.0241303,	0.000918763,	0.00679953,	0.0457930,	0.0197030],
                         [0.0249575,	0.000841617,	0.00665428,	0.0388454,	0.0196545],
                         [0.0246749,	0.000808955,	0.00655142,	0.0350470,	0.0196322]])
    
    return ar_nrmse, var_nrmse, ar_conv, var_conv


ar_nrmse, var_nrmse, ar_conv, var_conv = get_results()





names = ["AR_SC", "AR_MC", "Prophet", "BHT_AR_SC", "BHT_AR_MC"]
colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple', 
          'tab:pink', 'tab:brown', 'tab:olive', 'tab:gray', 'tab:cyan']
titles = ["Convergence Value Comparison", "NRMSE Value Comparison"]
n_rows = 2
n_val = 5
c_1 = 8 #0,1 #2,3 #4,5 #6,7
c_2 = 9
fig = plt.figure(figsize = (12,9))
gs = fig.add_gridspec(n_rows, hspace = 0.3)
axs = gs.subplots(sharex=False, sharey=False)

for k in range(2):
    plt.sca(axs[k])
    plt.xticks(ticks=[i for i in range(15)],
                labels=[str(i+1) for i in range(15)])
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)


fig.suptitle("Train: 20,    Validation: " + str(2*n_val - 1), 
             fontname = 'Arial', 
             fontweight = 'bold', 
             fontstyle = 'oblique',
             fontsize = 18, 
             y = 0.96)#0.915)

for i in range(n_rows):
    axs[i].set_title(titles[i], fontname = 'Arial', fontweight = 'bold', fontstyle = 'oblique', fontsize = 14)
    axs[i].set_xlabel('Iteration', fontname = 'Arial', fontstyle = 'oblique', fontsize = 12)
    axs[i].set_ylabel("Convergence Value", fontname = 'Arial', fontstyle = 'oblique', fontsize = 12)

axs[0].plot(ar_conv[:, n_val-1], label = "BHT_AR_SC Convergence Value", c = colors[c_1], marker='o')
axs[0].plot(var_conv[:, n_val-1], label = "BHT_AR_MC Convergence Value", c = colors[c_2], marker='o')   
axs[1].plot(ar_nrmse[:, n_val-1], label = "BHT_AR_SC NRMSE Value", c = colors[c_1], marker='o')
axs[1].plot(var_nrmse[:, n_val-1], label = "BHT_AR_MC NRMSE Value", c = colors[c_2], marker='o')
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












