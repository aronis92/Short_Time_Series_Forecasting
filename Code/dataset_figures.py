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
fig = plt.figure(figsize = (10,8))
gs = fig.add_gridspec(4, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle("Synthetic Time Series", 
             fontname = 'Arial', 
             fontweight = 'bold', 
             fontstyle = 'oblique',
             fontsize = 16, 
             y = 0.915)
colors = ['tab:blue', 'tab:orange', 'tab:green']
for i in range(len(colors)):
    axs[0].plot(X[i, :].T, c = colors[i])
    axs[i+1].plot(X[i, :].T, c = colors[i])
axs[0].set_ylabel('All Variables')
axs[1].set_ylabel('Variable 1')
axs[2].set_ylabel('Variable 2')
axs[3].set_ylabel('Variable 3')
axs[3].set_yticks([42,44,46,48,50])
plt.xlabel('Time')
plt.show()
# plt.savefig('./figures/raw/synthetic.png')
# plt.close()



'''Macro Dataset'''
X, _, _ = get_data(datasets[0], [500, 1, 1])
colors = ['tab:blue', 'tab:orange', 'tab:green', 
          'tab:red', 'tab:purple', 'tab:brown', 
          'tab:pink', 'tab:gray', 'tab:olive', 
          'tab:cyan', 'saddlebrown', 'k']
plt.figure(figsize = (10,4))
plt.title("US Macroeconomic Time Series",
          fontname = 'Arial', 
          fontweight = 'bold', 
          fontstyle = 'oblique',
          fontsize = 16)
for i in range(len(colors)):
    plt.plot(X[i, :].T, c=colors[i])
plt.xlabel('Time')
plt.ylabel('All Variables')
plt.show()
# plt.savefig('./figures/macro1.png')
# plt.close()


fig = plt.figure(figsize = (12, 7))
gs = fig.add_gridspec(4, 3, hspace=0, wspace=0)
fig.suptitle("US Macroeconomic Time Series", 
             fontname = 'Arial', 
             fontweight = 'bold', 
             fontstyle = 'oblique',
             fontsize = 16, 
             y=0.93)
axs = gs.subplots(sharex=True, sharey=False)
for i in range(4):
    for j in range(3):
        axs[i, j].plot(X[i*3 + j, :].T, c=colors[i*3 + j])
        axs[i, j].axes.xaxis.set_visible(False)
        axs[i, j].axes.yaxis.set_visible(False)
fig.text(0.5, 0.1, 'Time', ha='center')
fig.text(0.11, 0.79, 'Variables 1-3', va='center', rotation='vertical')
fig.text(0.11, 0.6, 'Variables 4-6', va='center', rotation='vertical')
fig.text(0.11, 0.41, 'Variables 7-9', va='center', rotation='vertical')
fig.text(0.11, 0.22, 'Variables 10-12', va='center', rotation='vertical')
plt.show()
# plt.savefig('./figures/macro2.png')
# plt.close()


'''El Nino'''
X, _, _ = get_data(datasets[1], [500, 1, 1])
colors = ['tab:blue', 'tab:orange', 'tab:green', 
          'tab:red', 'tab:purple', 'tab:brown', 
          'tab:pink', 'tab:gray', 'tab:olive', 
          'tab:cyan', 'saddlebrown', 'k']
plt.figure(figsize = (10,4))
plt.title("El Nino Time Series",
          fontname = 'Arial', 
          fontweight = 'bold', 
          fontstyle = 'oblique',
          fontsize = 16)
for i in range(len(colors)):
    plt.plot(X[i, :].T, c=colors[i])
plt.xlabel('Time')
plt.ylabel('All Variables')
plt.show()
# plt.savefig('./figures/elnino1.png')
# plt.close()

fig = plt.figure(figsize = (12, 7))
gs = fig.add_gridspec(4, 3, hspace=0, wspace=0)
fig.suptitle("El Nino Time Series", 
             fontname = 'Arial', 
             fontweight = 'bold', 
             fontstyle = 'oblique',
             fontsize = 16, 
             y=0.93)
axs = gs.subplots(sharex=True, sharey=False)
for i in range(4):
    for j in range(3):
        axs[i, j].plot(X[i*3 + j, :].T, c=colors[i*3 + j])
        axs[i, j].axes.xaxis.set_visible(False)
        axs[i, j].axes.yaxis.set_visible(False)
fig.text(0.5, 0.1, 'Time', ha='center')
fig.text(0.11, 0.79, 'Variables 1-3', va='center', rotation='vertical')
fig.text(0.11, 0.6, 'Variables 4-6', va='center', rotation='vertical')
fig.text(0.11, 0.41, 'Variables 7-9', va='center', rotation='vertical')
fig.text(0.11, 0.22, 'Variables 10-12', va='center', rotation='vertical')
plt.show()

# fig = plt.figure(figsize = (14,14))
# gs = fig.add_gridspec(6, 2, hspace=0.04)
# fig.suptitle("El Nino Time Series", fontname='Arial', fontsize=16, y=0.91)
# axs = gs.subplots(sharex=True, sharey=False)
# for i in range(0, X.shape[0], 2):
#     axs[int(i/2), 0].plot(X[i, :].T, c=colors[i])
#     axs[int(i/2), 1].plot(X[i+1, :].T, c=colors[i+1])
#     if i != X.shape[0]-2:
#         axs[int(i/2), 0].axes.xaxis.set_visible(False)
#         axs[int(i/2), 1].axes.xaxis.set_visible(False)
# plt.show()
# plt.savefig('./figures/elnino2.png')
# plt.close()


'''Ozone'''
X, _, _ = get_data(datasets[2], [500, 1, 1])
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 
          'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
plt.figure(figsize = (10,4))
plt.title("Ozone Time Series",
          fontname = 'Arial', 
          fontweight = 'bold', 
          fontstyle = 'oblique',
          fontsize = 16)
for i in range(len(colors)):
    plt.plot(X[i, :].T, c=colors[i])
plt.xlabel('Time')
plt.ylabel('All Variables')
plt.show()

fig = plt.figure(figsize = (12, 7))
gs = fig.add_gridspec(4, 2, hspace=0, wspace=0)
fig.suptitle("Ozone Time Series", 
             fontname = 'Arial', 
             fontweight = 'bold', 
             fontstyle = 'oblique',
             fontsize = 16, 
             y=0.93)
axs = gs.subplots(sharex=True, sharey=False)
for i in range(4):
    for j in range(2):
        axs[i, j].plot(X[i*2 + j, :].T, c=colors[i*2 + j])
        axs[i, j].axes.xaxis.set_visible(False)
        axs[i, j].axes.yaxis.set_visible(False)
fig.text(0.5, 0.1, 'Time', ha='center')
fig.text(0.11, 0.79, 'Variables 1, 2', va='center', rotation='vertical')
fig.text(0.11, 0.6, 'Variables 3, 4', va='center', rotation='vertical')
fig.text(0.11, 0.41, 'Variables 5, 6', va='center', rotation='vertical')
fig.text(0.11, 0.22, 'Variables 7, 8', va='center', rotation='vertical')
plt.show()

# fig = plt.figure(figsize = (14,14))
# gs = fig.add_gridspec(6, 2, hspace=0.15)
# fig.suptitle("Ozone Time Series", 
#              fontname = 'Arial', 
#              fontweight = 'bold', 
#              fontstyle = 'oblique',
#              fontsize = 16, 
#              y=0.93)
# axs = gs.subplots(sharex=False, sharey=False)
# for ax in axs[0,:]:
#     ax.remove()
# axbig = fig.add_subplot(gs[0:2, :])
# for i in range(X.shape[0]):
#     axbig.plot(X[i, :].T, c=colors[i])
#     if i%2 == 0:
#         axs[int(i/2)+2, 0].plot(X[i, :].T, c=colors[i])
#     else:
#         axs[int(i/2)+2, 1].plot(X[i, :].T, c=colors[i])
#     # if i < X.shape[0]-2:
#     #     axs[int(i/2)+1, 0].axes.xaxis.set_visible(False)
#     #     axs[int(i/2)+1, 1].axes.xaxis.set_visible(False)
# plt.show()
# plt.savefig('./figures/ozone.png')
# plt.close()





'''Night Visitors'''
# X, _, _ = get_data(datasets[3], [500, 1, 1])
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 
#           'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
# fig = plt.figure(figsize = (14,14))
# gs = fig.add_gridspec(5, 2, hspace=0.15)
# fig.suptitle("Night Visitors Time Series", fontname='Arial', fontsize=16, y=0.91)
# axs = gs.subplots(sharex=False, sharey=False)
# for ax in axs[0,:]:
#     ax.remove()
# axbig = fig.add_subplot(gs[0, :])
# for i in range(X.shape[0]):
#     axbig.plot(X[i, :].T, c=colors[i])
#     if i%2 == 0:
#         axs[int(i/2)+1, 0].plot(X[i, :].T, c=colors[i])
#     else:
#         axs[int(i/2)+1, 1].plot(X[i, :].T, c=colors[i])
#     # if i < X.shape[0]-2:
#     #     axs[int(i/2)+1, 0].axes.xaxis.set_visible(False)
#     #     axs[int(i/2)+1, 1].axes.xaxis.set_visible(False)
# plt.show()
# plt.savefig('./figures/night_visitors.png')
# plt.close()

X, _, _ = get_data(datasets[3], [500, 1, 1])
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 
          'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
plt.figure(figsize = (10,4))
plt.title("Night Visitors Time Series",
          fontname = 'Arial', 
          fontweight = 'bold', 
          fontstyle = 'oblique',
          fontsize = 16)
for i in range(len(colors)):
    plt.plot(X[i, :].T, c=colors[i])
plt.xlabel('Time')
plt.ylabel('All Variables')
plt.show()

fig = plt.figure(figsize = (12, 7))
gs = fig.add_gridspec(4, 2, hspace=0, wspace=0)
fig.suptitle("Night Visitors Time Series", 
             fontname = 'Arial', 
             fontweight = 'bold', 
             fontstyle = 'oblique',
             fontsize = 16, 
             y=0.93)
axs = gs.subplots(sharex=True, sharey=False)
for i in range(4):
    for j in range(2):
        axs[i, j].plot(X[i*2 + j, :].T, c=colors[i*2 + j])
        axs[i, j].axes.xaxis.set_visible(False)
        axs[i, j].axes.yaxis.set_visible(False)
fig.text(0.5, 0.1, 'Time', ha='center')
fig.text(0.11, 0.79, 'Variables 1, 2', va='center', rotation='vertical')
fig.text(0.11, 0.6, 'Variables 3, 4', va='center', rotation='vertical')
fig.text(0.11, 0.41, 'Variables 5, 6', va='center', rotation='vertical')
fig.text(0.11, 0.22, 'Variables 7, 8', va='center', rotation='vertical')
plt.show()



    
'''Stack Loss'''
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
# X, _, _ = get_data(datasets[6], [500, 1, 1])
# fig = plt.figure(figsize = (9,9))
# gs = fig.add_gridspec(5, hspace=0)
# axs = gs.subplots(sharex=True, sharey=False)
# fig.suptitle("Stack Loss Time Series", 
#              fontname = 'Arial', 
#              fontweight = 'bold', 
#              fontstyle = 'oblique',
#              fontsize = 16, 
#              y=0.93)
# for i in range(len(colors)):
#     axs[0].plot(X[i, :].T, c = colors[i])
# axs[4].plot(X[0, :].T, c = colors[0])
# axs[2].plot(X[1, :].T, c = colors[1])
# axs[3].plot(X[2, :].T, c = colors[2])
# axs[1].plot(X[3, :].T, c = colors[3])
# plt.show()
# plt.savefig('./figures/stack_loss.png')
# plt.close()

# fig = plt.figure(figsize = (12,7))
# gs = fig.add_gridspec(3, 2, hspace=0.15)
# fig.suptitle(" Time Series", 
#              fontname = 'Arial', 
#              fontweight = 'bold', 
#              fontstyle = 'oblique',
#              fontsize = 16, 
#              y=0.93)
# axs = gs.subplots(sharex=False, sharey=False)
# for ax in axs[0,:]:
#     ax.remove()
# axbig = fig.add_subplot(gs[0, :])
# for i in range(X.shape[0]):
#     axbig.plot(X[i, :].T, c=colors[i])
# axs[2,1].plot(X[0, :].T, c = colors[0])
# axs[1,1].plot(X[1, :].T, c = colors[1])
# axs[2,0].plot(X[2, :].T, c = colors[2])
# axs[1,0].plot(X[3, :].T, c = colors[3])

# axs[2,1].axes.xaxis.set_visible(False)
# axs[1,1].axes.xaxis.set_visible(False)
# axs[2,0].axes.xaxis.set_visible(False)
# axs[1,0].axes.xaxis.set_visible(False)

# axs[2,1].axes.yaxis.set_visible(False)
# axs[1,1].axes.yaxis.set_visible(False)
# axs[2,0].axes.yaxis.set_visible(False)
# axs[1,0].axes.yaxis.set_visible(False)
# plt.show()


colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
X, _, _ = get_data(datasets[6], [500, 1, 1])
plt.figure(figsize = (10,4))
plt.title("Stack Loss Time Series",
          fontname = 'Arial', 
          fontweight = 'bold', 
          fontstyle = 'oblique',
          fontsize = 16)
for i in range(len(colors)):
    plt.plot(X[i, :].T, c=colors[i])
plt.xlabel('Time')
plt.ylabel('All Variables')
plt.show()

fig = plt.figure(figsize = (12, 4))
gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
fig.suptitle("Stack Loss Time Series", 
             fontname = 'Arial', 
             fontweight = 'bold', 
             fontstyle = 'oblique',
             fontsize = 16, 
             y=0.94)
axs = gs.subplots(sharex=True, sharey=False)
for i in range(2):
    for j in range(2):
        axs[i, j].plot(X[i*2 + j, :].T, c=colors[i*2 + j])
        axs[i, j].axes.xaxis.set_visible(False)
        axs[i, j].axes.yaxis.set_visible(False)
fig.text(0.51, 0.08, 'Time', ha='center')
fig.text(0.11, 0.7, 'Variables 1, 2', va='center', rotation='vertical')
fig.text(0.11, 0.32, 'Variables 3, 4', va='center', rotation='vertical')
plt.show()





    
'''Yahoo'''
X, _, _ = get_data(datasets[5], [100, 1, 1])
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
plt.figure(figsize = (10,4))
plt.title("Yahoo Stock Time Series", 
             fontname = 'Arial', 
             fontweight = 'bold', 
             fontstyle = 'oblique',
             fontsize = 16)
for i in range(len(colors)):
    plt.plot(X[i, :].T, c=colors[i])
plt.xlabel('Time')
plt.ylabel('All Variables')
plt.show()


fig = plt.figure(figsize = (12,7))
gs = fig.add_gridspec(4, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle("Yahoo Stock Time Series", 
             fontname = 'Arial', 
             fontweight = 'bold', 
             fontstyle = 'oblique',
             fontsize = 16, 
             y=0.92)
axs[0].plot(X[0, :].T, c = colors[0])
axs[1].plot(X[1, :].T, c = colors[1])
axs[2].plot(X[2, :].T, c = colors[2])
axs[3].plot(X[3, :].T, c = colors[3])
fig.text(0.51, 0.08, 'Time', ha='center')
fig.text(0.082, 0.23, 'Variable 1', va='center', rotation='vertical')
fig.text(0.082, 0.41, 'Variable 2', va='center', rotation='vertical')
fig.text(0.082, 0.6, 'Variable 3', va='center', rotation='vertical')
fig.text(0.082, 0.78, 'Variable 4', va='center', rotation='vertical')
plt.show()



'''NASDAQ'''
X, _, _ = get_data(datasets[4], [100, 1, 1])
fig = plt.figure(figsize = (12,6))
gs = fig.add_gridspec(3, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle("NASDAQ Time Series", 
             fontname = 'Arial', 
             fontweight = 'bold', 
             fontstyle = 'oblique',
             fontsize = 16, 
             y=0.93)
colors = []
for i in range(X.shape[0]):
    r = random.random()
    g = random.random()
    b = random.random()
    colors.append(tuple([r,g,b]))
    axs[0].plot(X[i, :].T, c = colors[i])
axs[1].plot(X[-2, :].T, c = colors[-2])
axs[2].plot(X[-1, :].T, c = colors[-1])
plt.xlabel('Time')
axs[0].set_ylabel('All Variables')
axs[1].set_ylabel('Variable 1')
axs[2].set_ylabel('Variable 2')
plt.show()
# plt.savefig('./figures/nasdaq.png')
# plt.close()


fig = plt.figure(figsize = (12,9))
rows = 7
cols = 4
gs = fig.add_gridspec(rows, cols, hspace=0, wspace=0)
axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle("NASDAQ Time Series", 
             fontname = 'Arial', 
             fontweight = 'bold', 
             fontstyle = 'oblique',
             fontsize = 16, 
             y=0.91)
for i in range(rows):
    for j in range(cols):
        axs[i, j].plot(X[rows*i+j, :].T, c=colors[rows*i+j])
        axs[i, j].axes.xaxis.set_visible(False)
        axs[i, j].axes.yaxis.set_visible(False)
fig.text(0.51, 0.1, 'Time', ha='center')
fig.text(0.1, 0.18, 'Variables\n   27-30', va='center', rotation='vertical')
fig.text(0.1, 0.29, 'Variables\n   23-26', va='center', rotation='vertical')
fig.text(0.1, 0.40, 'Variables\n   19-22', va='center', rotation='vertical')
fig.text(0.1, 0.50, 'Variables\n   15-18', va='center', rotation='vertical')
fig.text(0.1, 0.61, 'Variables\n   11-14', va='center', rotation='vertical')
fig.text(0.1, 0.72, 'Variables\n   7-10', va='center', rotation='vertical')
fig.text(0.1, 0.82, 'Variables\n    3-6', va='center', rotation='vertical')
plt.show()
# plt.savefig('./figures/nasdaq1.png')
# plt.close()


fig = plt.figure(figsize = (12,9))
gs = fig.add_gridspec(rows, cols, hspace=0, wspace=0)
axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle("NASDAQ Time Series",
             fontname = 'Arial', 
             fontweight = 'bold', 
             fontstyle = 'oblique',
             fontsize = 16, 
             y=0.91)
for i in range(rows):
    for j in range(cols):
        axs[i, j].plot(X[rows*cols + rows*i+j, :].T, c=colors[rows*cols + rows*i+j])
        axs[i, j].axes.xaxis.set_visible(False)
        axs[i, j].axes.yaxis.set_visible(False)
fig.text(0.51, 0.1, 'Time', ha='center')
fig.text(0.1, 0.18, 'Variables\n   56-58', va='center', rotation='vertical')
fig.text(0.1, 0.29, 'Variables\n   51-54', va='center', rotation='vertical')
fig.text(0.1, 0.40, 'Variables\n   47-50', va='center', rotation='vertical')
fig.text(0.1, 0.50, 'Variables\n   43-46', va='center', rotation='vertical')
fig.text(0.1, 0.61, 'Variables\n   39-42', va='center', rotation='vertical')
fig.text(0.1, 0.72, 'Variables\n   35-38', va='center', rotation='vertical')
fig.text(0.1, 0.82, 'Variables\n   31-34', va='center', rotation='vertical')
plt.show()
# plt.savefig('./figures/nasdaq2.png')
# plt.close()


fig = plt.figure(figsize = (12,9))
rows = 6
cols = 4
gs = fig.add_gridspec(rows, cols, hspace=0, wspace=0)
axs = gs.subplots(sharex=True, sharey=False)
fig.suptitle("NASDAQ Time Series",
             fontname = 'Arial', 
             fontweight = 'bold', 
             fontstyle = 'oblique',
             fontsize = 16, 
             y=0.91)
for i in range(rows):
    for j in range(cols):
        axs[i, j].plot(X[2*rows*cols + rows*i+j, :].T, c=colors[2*rows*cols + rows*i+j])
        axs[i, j].axes.xaxis.set_visible(False)
        axs[i, j].axes.yaxis.set_visible(False)
fig.text(0.51, 0.1, 'Time', ha='center')
fig.text(0.1, 0.19, 'Variables\n   79-82', va='center', rotation='vertical')
fig.text(0.1, 0.31, 'Variables\n   75-78', va='center', rotation='vertical')
fig.text(0.1, 0.435, 'Variables\n   71-74', va='center', rotation='vertical')
fig.text(0.1, 0.565, 'Variables\n   67-70', va='center', rotation='vertical')
fig.text(0.1, 0.69, 'Variables\n   63-66', va='center', rotation='vertical')
fig.text(0.1, 0.815, 'Variables\n   59-62', va='center', rotation='vertical')
plt.show()
# plt.savefig('./figures/nasdaq3.png')
# plt.close()




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

