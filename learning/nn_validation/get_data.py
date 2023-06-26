import sys
sys.path.append('../nn-learning/')
from sdf.robot_sdf import RobotSdfCollisionNet
from scipy.io import loadmat
import torch
import numpy as np
from fk_num import *
import time
from validation import Validation

# torch parameters
device = torch.device('cpu', 0)
params = {'device': device, 'dtype': torch.float32}

v = Validation()
q_min = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175,	-2.8973, -1, -1, -0.2]).to(**params)
q_max = torch.tensor([2.8973,	1.7628,	2.8973,	-0.0698,  2.8973,	3.7525,	2.8973, 1, 1, 1.3]).to(**params)
q_span = q_max - q_min

# generate data
for i in range(10):
    print(i)
    input = q_min + q_span * torch.rand(1000, 10, **params)
    t0 = time.time()
    a = v.calc_nn_pred(input)
    t_nn = time.time()
    b = v.calc_mesh_mindist(input)
    t_mesh = time.time()
    print('NN time: %4.2f s\nMesh time: %4.2f s' % (t_nn-t0, t_mesh-t_nn))
    #print('Mean L1 error: %4.2f' % ((a-b).abs().mean()))
    #print('Max L1 error: %4.2f' % ((a-b).abs().max()))
    data = torch.cat((input, a, b), 1)
    with open("data_.csv", "ab") as f:
        np.savetxt(f, data.numpy(), delimiter=',', fmt='%4.3f')

# f = v.get_mesh_fk(input[0, :7])
# print(f)
# # scatter plot 3d points of f
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for link in f:
#     ax.scatter(link[:, 0], link[:, 1], link[:, 2])
# ax.set_aspect('equal')
# plt.show()
# plt.show(block=True)