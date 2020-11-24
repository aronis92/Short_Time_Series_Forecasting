from functions.utils import get_data
import matplotlib.pyplot as plt

# plt.figure(figsize = (12,5))
# plt.plot(X.T)
# plt.show()

#                             Index  Var x Time
datasets = ['macro', #__________0     12 x 203    # 
            'elnino', #_________1     12 x 61     # 
            'ozone', #__________2      8 x 203    # 
            'nightvisitors', #__3      8 x 56     #
            'nasdaq', #_________4     82 x 40560  # 
            'yahoo', #__________5     5 x 2469    #
            'stackloss', #______6     4 x 21      #
            'book1'] #__________7     3 x sum(Ns) # DONE 

'''Book Dataset'''
X, _, _ = get_data(datasets[7], [500, 1, 1])
fig = plt.figure(figsize = (12,9))
gs = fig.add_gridspec(4, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)
# fig.suptitle('Sharing both axes')
colors = ['tab:blue', 'tab:orange', 'tab:green']
for i in range(len(colors)):
    axs[0].plot(X[i, :].T, c = colors[i])
    axs[i+1].plot(X[i, :].T, c = colors[i])
# axs[1].plot(X[0, :].T, 'tab:blue')
# axs[2].plot(X[1, :].T, 'tab:orange')
# axs[3].plot(X[2, :].T, 'tab:green')
plt.show()



fig = plt.figure(figsize = (14,14))
gs = fig.add_gridspec(7, 2, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)
for ax in axs[0,:]:
    ax.remove()
axbig = fig.add_subplot(gs[0, :])

# fig.suptitle('Sharing both axes')
axbig.plot(X.T)
axs[1, 0].plot(X[0, :].T, 'tab:blue')
axs[1, 1].plot(X[1, :].T, 'tab:orange')
axs[2, 0].plot(X[2, :].T, 'tab:green')
axs[2, 1].plot(X[3, :].T, 'tab:blue')
axs[3, 0].plot(X[4, :].T, 'tab:orange')
axs[3, 1].plot(X[5, :].T, 'tab:green')
axs[4, 0].plot(X[6, :].T, 'tab:blue')
axs[4, 1].plot(X[7, :].T, 'tab:orange')
axs[5, 0].plot(X[8, :].T, 'tab:green')
axs[5, 1].plot(X[9, :].T, 'tab:blue')
axs[6, 0].plot(X[10, :].T, 'tab:orange')
axs[6, 1].plot(X[11, :].T, 'tab:green')
plt.show()

