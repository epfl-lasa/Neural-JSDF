import torch
import numpy as np
import time
from validation import Validation

device = torch.device('cpu', 0)
params = {'device': device, 'dtype': torch.float32}

inputs = np.loadtxt('data_bad_opt.csv', delimiter=',')
inputs = torch.tensor(inputs, **params)
print("%d samples loaded" % inputs.shape[0])

v = Validation()
q_min = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175,	-2.8973, -1, -1, -0.2]).to(**params)
q_max = torch.tensor([2.8973,	1.7628,	2.8973,	-0.0698,  2.8973,	3.7525,	2.8973, 1, 1, 1.3]).to(**params)
q_span = q_max - q_min

a = v.calc_nn_pred(inputs)
b = v.calc_mesh_mindist(inputs)
print('done')