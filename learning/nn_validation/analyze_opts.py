import numpy as np
# load data.csv
data = np.loadtxt('data.csv', delimiter=',')
input_all = data[:, :10]
nn_dist_all = data[:, 10:19]
mesh_dist_all = data[:, 19:]

print("%d samples loaded" % input_all.shape[0])
# remove negative distances
thr = 3
mindist_nn = nn_dist_all.min(axis=1)
idx_mask = mindist_nn > thr
input = input_all[idx_mask, :]
nn_dist = nn_dist_all[idx_mask, :]
mesh_dist = mesh_dist_all[idx_mask, :]
print("%d samples remain after filtering" % input.shape[0])

n_links = 9

#plot distributions of distances
import matplotlib.pyplot as plt
fig = plt.figure()
for i in range(n_links):
    ax = fig.add_subplot(3, 3, i+1)
    ax.hist(nn_dist[:, i], bins=100, label='NN')
    ax.hist(mesh_dist[:, i], bins=100, label='Mesh')
    ax.legend()
    L1 = np.abs(nn_dist[:, i] - mesh_dist[:, i])
    print('Link %d: L1 error: %4.2f, Max err: %4.2f' % (i, L1.mean(), L1.max()))
plt.show(block=True)

# save data with errors above threshold
all_err = abs(nn_dist_all - mesh_dist_all)
idx_large_err = np.any(all_err > 10, 1)
print(data[idx_large_err * idx_mask, :].shape)
print('Large errors: %d' % idx_large_err.sum())
np.savetxt('data_large_err.csv', data[idx_large_err * idx_mask, :], delimiter=',', fmt='%4.3f')