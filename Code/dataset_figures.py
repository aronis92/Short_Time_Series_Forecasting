from functions.utils import get_data
import matplotlib.pyplot as plt
import random

# plt.figure(figsize = (12,5))
# plt.plot(X.T)
# plt.show()

#                             Index  Var x Time
datasets = ['macro', #__________0     12 x 203    # DONE
            'elnino', #_________1     12 x 61     # DONE
            'ozone', #__________2      8 x 203    # DONE
            'nightvisitors', #__3      8 x 56     # DONE
            'nasdaq', #_________4     82 x 40560  # DONE
            'yahoo', #__________5     5 x 2469    # DONE
            'stackloss', #______6     4 x 21      # DONE
            'book1'] #__________7     3 x sum(Ns) # DONE 


'''Book Dataset'''
X, _, _ = get_data(datasets[7], [500, 1, 1])
fig = plt.figure(figsize = (12,9))
gs = fig.add_gridspec(4, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle("Synthetic Time Series", fontsize=16, y=0.915)
colors = ['tab:blue', 'tab:orange', 'tab:green']
for i in range(len(colors)):
    axs[0].plot(X[i, :].T, c = colors[i])
    axs[i+1].plot(X[i, :].T, c = colors[i])
plt.savefig('./figures/synthetic.png')
plt.close()



'''Macro Dataset'''
X, _, _ = get_data(datasets[0], [500, 1, 1])
colors = ['tab:blue', 'tab:orange', 'tab:green', 
          'tab:red', 'tab:purple', 'tab:brown', 
          'tab:pink', 'tab:gray', 'tab:olive', 
          'tab:cyan', 'saddlebrown', 'k']
plt.figure(figsize = (12,5))
plt.title("US Macroeconomic Time Series", fontsize=16)
for i in range(len(colors)):
    plt.plot(X[i, :].T, c=colors[i])
plt.savefig('./figures/macro1.png')
plt.close()


fig = plt.figure(figsize = (14,14))
gs = fig.add_gridspec(4, 3, hspace=0.04)
fig.suptitle("US Macroeconomic Time Series", fontsize=16, y=0.91)
axs = gs.subplots(sharex=True, sharey=False)
for i in range(0, X.shape[0], 3):
    axs[int(i/2), 0].plot(X[i, :].T, c=colors[i])
    axs[int(i/2), 1].plot(X[i+1, :].T, c=colors[i+1])
    if i != X.shape[0]-2:
        axs[int(i/2), 0].axes.xaxis.set_visible(False)
        axs[int(i/2), 1].axes.xaxis.set_visible(False)
plt.savefig('./figures/macro2.png')
plt.close()


fig = plt.figure(figsize = (14,14))
gs = fig.add_gridspec(4, 3, hspace=0, wspace=0)
fig.suptitle("US Macroeconomic Time Series", fontsize=16, y=0.91)
axs = gs.subplots(sharex=False, sharey=False)
for i in range(4):
    for j in range(3):
        axs[i, j].plot(X[4*i+j, :].T, c=colors[4*i+j])
        axs[i, j].axes.xaxis.set_visible(False)
        axs[i, j].axes.yaxis.set_visible(False)
plt.show()



'''El Nino'''
X, _, _ = get_data(datasets[1], [500, 1, 1])
colors = ['tab:blue', 'tab:orange', 'tab:green', 
          'tab:red', 'tab:purple', 'tab:brown', 
          'tab:pink', 'tab:gray', 'tab:olive', 
          'tab:cyan', 'saddlebrown', 'k']
plt.figure(figsize = (12,5))
plt.title("El Nino Time Series", fontsize=16)
for i in range(len(colors)):
    plt.plot(X[i, :].T, c=colors[i])
plt.savefig('./figures/elnino1.png')
plt.close()


fig = plt.figure(figsize = (14,14))
gs = fig.add_gridspec(6, 2, hspace=0.04)
fig.suptitle("El Nino Time Series", fontsize=16, y=0.91)
axs = gs.subplots(sharex=True, sharey=False)
for i in range(0, X.shape[0], 2):
    axs[int(i/2), 0].plot(X[i, :].T, c=colors[i])
    axs[int(i/2), 1].plot(X[i+1, :].T, c=colors[i+1])
    if i != X.shape[0]-2:
        axs[int(i/2), 0].axes.xaxis.set_visible(False)
        axs[int(i/2), 1].axes.xaxis.set_visible(False)
plt.savefig('./figures/elnino2.png')
plt.close()


'''Ozone'''
X, _, _ = get_data(datasets[2], [500, 1, 1])
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 
          'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
fig = plt.figure(figsize = (14,14))
gs = fig.add_gridspec(5, 2, hspace=0.15)
fig.suptitle("Ozone Time Series", fontsize=16, y=0.91)
axs = gs.subplots(sharex=False, sharey=False)
for ax in axs[0,:]:
    ax.remove()
axbig = fig.add_subplot(gs[0, :])
for i in range(X.shape[0]):
    axbig.plot(X[i, :].T, c=colors[i])
    if i%2 == 0:
        axs[int(i/2)+1, 0].plot(X[i, :].T, c=colors[i])
    else:
        axs[int(i/2)+1, 1].plot(X[i, :].T, c=colors[i])
    # if i < X.shape[0]-2:
    #     axs[int(i/2)+1, 0].axes.xaxis.set_visible(False)
    #     axs[int(i/2)+1, 1].axes.xaxis.set_visible(False)
plt.savefig('./figures/ozone.png')
plt.close()


'''Night Visitors'''
X, _, _ = get_data(datasets[3], [500, 1, 1])
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 
          'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
fig = plt.figure(figsize = (14,14))
gs = fig.add_gridspec(5, 2, hspace=0.15)
fig.suptitle("Night Visitors Time Series", fontsize=16, y=0.91)
axs = gs.subplots(sharex=False, sharey=False)
for ax in axs[0,:]:
    ax.remove()
axbig = fig.add_subplot(gs[0, :])
for i in range(X.shape[0]):
    axbig.plot(X[i, :].T, c=colors[i])
    if i%2 == 0:
        axs[int(i/2)+1, 0].plot(X[i, :].T, c=colors[i])
    else:
        axs[int(i/2)+1, 1].plot(X[i, :].T, c=colors[i])
    # if i < X.shape[0]-2:
    #     axs[int(i/2)+1, 0].axes.xaxis.set_visible(False)
    #     axs[int(i/2)+1, 1].axes.xaxis.set_visible(False)
plt.savefig('./figures/night_visitors.png')
plt.close()

    
'''Stack Loss'''
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
X, _, _ = get_data(datasets[6], [500, 1, 1])
fig = plt.figure(figsize = (9,9))
gs = fig.add_gridspec(5, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle("Stack Loss Time Series", fontsize=16, y=0.915)
for i in range(len(colors)):
    axs[0].plot(X[i, :].T, c = colors[i])
axs[4].plot(X[0, :].T, c = colors[0])
axs[2].plot(X[1, :].T, c = colors[1])
axs[3].plot(X[2, :].T, c = colors[2])
axs[1].plot(X[3, :].T, c = colors[3])
plt.savefig('./figures/stack_loss.png')
plt.close()


    
'''Yahoo'''
X, _, _ = get_data(datasets[5], [100, 1, 1])
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
plt.figure(figsize = (12,5))
plt.title("Yahoo Stock Time Series", fontsize=16)
for i in range(len(colors)):
    plt.plot(X[i, :].T, c=colors[i])
plt.savefig('./figures/yahoo1.png')
plt.close()


fig = plt.figure(figsize = (12,9))
gs = fig.add_gridspec(4, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle("Yahoo Stock Time Series", fontsize=16, y=0.915)
axs[0].plot(X[0, :].T, c = colors[0])
axs[1].plot(X[1, :].T, c = colors[1])
axs[2].plot(X[2, :].T, c = colors[2])
axs[3].plot(X[3, :].T, c = colors[3])
plt.savefig('./figures/yahoo2.png')
plt.close()



'''nasdaq'''
X, _, _ = get_data(datasets[4], [100, 1, 1])
fig = plt.figure(figsize = (12,9))
gs = fig.add_gridspec(3, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle("NASDAQ Time Series", fontsize=16, y=0.915)
colors = []
for i in range(X.shape[0]):
    r = random.random()
    g = random.random()
    b = random.random()
    colors.append(tuple([r,g,b]))
    axs[0].plot(X[i, :].T, c = colors[i])
axs[1].plot(X[-2, :].T, c = colors[-2])
axs[2].plot(X[-1, :].T, c = colors[-1])
# plt.show()
plt.savefig('./figures/nasdaq.png')
plt.close()


fig = plt.figure(figsize = (12,9))
rows = 7
cols = 4
gs = fig.add_gridspec(rows, cols, hspace=0, wspace=0)
axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle("NASDAQ Time Series", fontsize=16, y=0.915)
for i in range(rows):
    for j in range(cols):
        axs[i, j].plot(X[rows*i+j, :].T, c=colors[rows*i+j])
        axs[i, j].axes.xaxis.set_visible(False)
        axs[i, j].axes.yaxis.set_visible(False)
#plt.show()
plt.savefig('./figures/nasdaq1.png')
plt.close()


fig = plt.figure(figsize = (12,9))
gs = fig.add_gridspec(rows, cols, hspace=0, wspace=0)
axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle("NASDAQ Time Series", fontsize=16, y=0.915)
for i in range(rows):
    for j in range(cols):
        axs[i, j].plot(X[rows*cols + rows*i+j, :].T, c=colors[rows*cols + rows*i+j])
        axs[i, j].axes.xaxis.set_visible(False)
        axs[i, j].axes.yaxis.set_visible(False)
#plt.show()
plt.savefig('./figures/nasdaq2.png')
plt.close()


fig = plt.figure(figsize = (12,9))
rows = 6
cols = 4
gs = fig.add_gridspec(rows, cols, hspace=0, wspace=0)
axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle("NASDAQ Time Series", fontsize=16, y=0.915)
for i in range(rows):
    for j in range(cols):
        axs[i, j].plot(X[2*rows*cols + rows*i+j, :].T, c=colors[2*rows*cols + rows*i+j])
        axs[i, j].axes.xaxis.set_visible(False)
        axs[i, j].axes.yaxis.set_visible(False)
#plt.show()
plt.savefig('./figures/nasdaq3.png')
plt.close()




# '''Macro Dataset'''
# X, _, _ = get_data(datasets[0], [500, 1, 1])
# fig = plt.figure(figsize = (14,14))
# gs = fig.add_gridspec(7, 2, hspace=0.1)
# axs = gs.subplots(sharex=True, sharey=False)
# for ax in axs[0,:]:
#     ax.remove()
# axbig = fig.add_subplot(gs[0, :])
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

# # fig.suptitle('Sharing both axes')
# for i in range(len(colors)):
#     axbig.plot(X[i, :].T, c=colors[i])
    
    

# fig = plt.figure(figsize = (14,14))
# gs = fig.add_gridspec(6, 2, hspace=0.1)
# axs = gs.subplots(sharex=True, sharey=False)
# axbig = fig.add_subplot(gs[0, :])
# axs[1, 0].plot(X[0, :].T, c=colors[0])
# axs[1, 1].plot(X[1, :].T, c=colors[1])
# axs[2, 0].plot(X[2, :].T, c=colors[2])
# axs[2, 1].plot(X[3, :].T, c=colors[3])
# axs[3, 0].plot(X[4, :].T, c=colors[4])
# axs[3, 1].plot(X[5, :].T, c=colors[5])
# axs[4, 0].plot(X[6, :].T, c=colors[6])
# axs[4, 1].plot(X[7, :].T, c=colors[7])
# axs[5, 0].plot(X[8, :].T, c=colors[8])
# axs[5, 1].plot(X[9, :].T, 'tab:blue')
# plt.savefig('./figures/macro2.png')

