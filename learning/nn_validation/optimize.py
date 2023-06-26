# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn import metrics
#
# data = np.loadtxt('data_large_err.csv', delimiter=',')
# input = data[:, :10]
# nn_dist = data[:, 10:19]
# mesh_dist = data[:, 19:]

from validation import Validation
import torch, time
import numpy as np
import pyswarms as ps

#find worst cases from grid search
data = np.loadtxt('outliers.csv', delimiter=',')
N = data.shape[0]
input_all = data[:N, :10]
nn_dist_all = data[:N, 10:19]
mesh_dist_all = data[:N, 19:]

print("%d samples loaded" % input_all.shape[0])
# remove negative distances
thr = 1
mindist_nn = nn_dist_all.min(axis=1)
idx_mask = mindist_nn > thr
input = input_all[idx_mask, :]
nn_dist = nn_dist_all[idx_mask, -1]
mesh_dist = mesh_dist_all[idx_mask, -1]
print("%d samples remain after filtering" % input.shape[0])
error = np.abs(nn_dist - mesh_dist)
idx = np.argsort(error)[::-1]
input = input[idx, :]


device = torch.device('cpu', 0)
params = {'device': device, 'dtype': torch.float32}
q_min = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175,	-2.8973, -1, -1, -0.2]).cpu().numpy()
q_max = torch.tensor([2.8973,	1.7628,	2.8973,	-0.0698,  2.8973,	3.7525,	2.8973, 1, 1, 1.3]).cpu().numpy()
q_span = q_max - q_min
bounds = (q_min, q_max)

closeness = []
for i in range(len(input)):
    dst_to_min = np.abs(input[i] - q_min)
    dst_to_max = np.abs(input[i] - q_max)
    closest = np.min(np.append(dst_to_min, dst_to_max))
    closeness.append(closest)
    print(i, '%4.2f' % closest)

close_np = np.array(closeness)
print('mean:', close_np.mean())
print('std:', close_np.std())
print('max:', close_np.max())
print('min:', close_np.min())
v = Validation()
nind = 1366
print("err config:", input[nind])
rand_add = (2*torch.rand(100, 10, **params)-1) * 0.3*q_span
rand_add[:, 0:7] = 0
rand_input = torch.tensor(input[nind], **params) + rand_add
#nn_pred = v.calc_nn_pred(rand_input)
#mesh_pred = v.calc_mesh_mindist(rand_input)
#err = nn_pred - mesh_pred
err = v.calc_err(rand_input)
#print('NN prediction:', nn_pred)
#print('Mesh prediction:', mesh_pred)
print('Max error:', err.max())
print('Min error:', err.min())
print('Mean error:', err.mean())
print('init error:', v.calc_err(torch.tensor(input[0:1], **params)))

# print('\n\n\n')
# time.sleep(0.1)
# # Set-up hyperparameters
# options = {'c1': 0.5, 'c2': 0.3, 'w':0.1}
# # options = {'c1': 0.8, 'c2': 0.03, 'w':0.5, 'bounds': bounds}
# #options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2, 'bounds': bounds}
# LINK = 8
# n_particles = 10
# for i in range(1):
#     print(LINK, i)
#     # Call instance of PSO
#     optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=len(bounds[0]), options=options, bounds=bounds,
#                                         init_pos=input[0:n_particles, :])
#
#     # Perform optimization
#     best_cost, best_pos = optimizer.optimize(v.fitness, iters=500, verbose=True, link=LINK)
#
#     print(f"\nThe best position found is: {best_pos}")
#     print(f"The best cost found is: {best_cost}")
#     print('NN prediction:', v.calc_nn_pred(torch.tensor(best_pos.reshape(1, -1)).to(**params)))
#     print('Mesh prediction:', v.calc_mesh_mindist(torch.tensor(best_pos.reshape(1, -1)).to(**params)))
#     # print(f"The best solution is: {v.fitness(best_pos.reshape(1,-1))}")
#     # with open("bad_data_links/%d_thr.csv" % LINK, "ab") as f:
#     #     np.savetxt(f, best_pos.reshape(1, -1), delimiter=',', fmt='%4.3f')
#
