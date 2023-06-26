from validation import Validation
import torch, time
import numpy as np
import matplotlib.pyplot as plt

#find worst cases from grid search
device = torch.device('cpu', 0)
params = {'device': device, 'dtype': torch.float32}
q_min = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -1, -1, -0.2]).cpu().numpy()
q_max = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1, 1, 1.3]).cpu().numpy()
q_span = q_max - q_min
plt.rcParams.update({'font.size': 12})
fig = plt.figure()
for i in range(9):
    print('Link %d' % (i+1))
    data = np.loadtxt('outliers_link/outliers_link%d.csv' % (i+1), delimiter=',')
    N = data.shape[0]
    input = data[:N, :10]
    nn_dist = data[:N, 10:19]
    mesh_dist = data[:N, 19:]
    print("%d samples loaded" % input.shape[0])

    # normalize input using qmin and qmax
    input_norm = (input - q_min) / q_span
    # calculate distances to qmin and qmax
    smallest_coord = input_norm.min(axis = 1)
    largest_coord = input_norm.max(axis = 1)
    mask_minbound = 1-smallest_coord > largest_coord
    dst_to_bnd = np.zeros(N)
    dst_to_bnd[mask_minbound] = smallest_coord[mask_minbound]
    dst_to_bnd[~mask_minbound] = largest_coord[~mask_minbound]

    idx_smallest = np.argmin(input_norm, axis = 1)
    idx_largest = np.argmax(input_norm, axis = 1)
    idx_close = np.zeros(N)
    idx_close[mask_minbound] = idx_smallest[mask_minbound]
    idx_close[~mask_minbound] = idx_largest[~mask_minbound]
    bin_w = 0.01
    bins = np.arange(0, 1 + bin_w, bin_w)
    ax = fig.add_subplot(3, 3, i+1)
    #ax.hist(dst_to_bnd, bins=bins, edgecolor='none', color=[0, 0.4470, 0.7410])
    ax.hist(idx_close)
    if i in [0, 3, 6]:
        ax.set_ylabel('Count', fontsize=14)
    if i >=6:
        ax.set_xlabel('Closeness to Boundary', fontsize=14)
    ax.set_title(f'Link {i + 1}')

fig.set_size_inches(15, 8)
plt.subplots_adjust(hspace=0.4)

plt.savefig('outliers_hist_idx.png', dpi=300)
plt.show(block=True)
