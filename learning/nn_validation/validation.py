import sys
sys.path.append('../nn-learning/')
from sdf.robot_sdf import RobotSdfCollisionNet
from scipy.io import loadmat
import torch
import numpy as np
from fk_num import *
import time


# torch parameters
device = torch.device('cpu', 0)
params = {'device': device, 'dtype': torch.float32}

class Validation:
    def __init__(self):

        # Load nn model
        fname = 'franka_collision_model.pt'
        self.nn_model = RobotSdfCollisionNet(in_channels=10, out_channels=9, layers=[256] * 4, skips=[])
        self.nn_model.load_weights('../nn-learning/' + fname, params)
        self.nn_model.model.to(**params)
        self.nn_model.model_jit = self.nn_model.model
        self.nn_model.model_jit = torch.jit.script(self.nn_model.model_jit)
        self.nn_model.model_jit = torch.jit.optimize_for_inference(self.nn_model.model_jit)

        # load robot matlab meshes
        data_mat = loadmat('../data-sampling/meshes/mesh_light_pts.mat')['mesh'][0]
        self.meshes = []
        self.faces = []
        self.vertices = []
        self.link_names = []
        self.v_int_pts = []
        for link in data_mat:
            faces = torch.tensor(link[0][0][0].astype(int)).to(**params)
            vertices = torch.tensor(link[0][0][1]).to(**params)
            link_name = link[0][0][2][0]
            int_pts = torch.tensor(link[0][0][3]).to(**params)
            v_int_pts = torch.cat((vertices, int_pts), 0)
            self.meshes.append({'vertices': vertices, 'faces': faces, 'link_name': link_name})
            self.faces.append(faces)
            self.vertices.append(vertices)
            self.link_names.append(link_name)
            self.v_int_pts.append(v_int_pts)

        # specify robot parameters
        dh_a = torch.tensor([0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0])        # "r" in matlab
        dh_d = torch.tensor([0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107])       # "d" in matlab
        dh_alpha = torch.tensor([0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0])  # "alpha" in matlab
        self.dh = torch.vstack((dh_d, dh_a*0, dh_a, dh_alpha)).T.to(**params)          # (d, theta, a (or r), alpha)

    def calc_nn_pred(self, input):
        y_pred = self.nn_model.model_jit(input)
        # mindist, _ = y_pred.min(dim=-1, keepdim=True)
        # return mindist
        return y_pred

    def get_mesh_fk(self, q):
        fk = dh_fk(torch.cat((q, torch.tensor([0]).to(q.device)), 0), self.dh)
        v_vec = []
        for i, P in enumerate(fk):
            v_vec.append(self.v_int_pts[i] @ P[:3,:3].T + P[:3,3:4].T)
        return v_vec

    def get_mindists(self, v_vec, y):
        mindists = []
        for v_link in v_vec:
            dist = torch.cdist(v_link, y.unsqueeze(0), p=2)
            mindist, _ = dist.min(dim=0, keepdim=False)
            mindists.append(mindist)
        return torch.cat(mindists, 0)

    def calc_mesh_mindist(self, input):
        q = input[:, :7]
        y = input[:, 7:]
        n_pts = y.shape[0]
        all_dists = torch.zeros(q.shape[0], 9, **params)
        for i in range(n_pts):
            v = self.get_mesh_fk(q[i])
            mindists = self.get_mindists(v, y[i])
            all_dists[i, :] = mindists
        #res, _ = all_dists.min(dim=1, keepdim=True)
        res = all_dists
        return(100*res)

    def calc_err(self, input):
        q = input[:, :7]
        y = input[:, 7:].to(**params)
        res_meshes = self.calc_mesh_mindist(input)
        res_nn = self.calc_nn_pred(input)
        err = torch.abs(res_meshes - res_nn)
        max_err, _ = err.max(dim=-1)
        return max_err

    def calc_err_thr(self, input, link=None):
        q = input[:, :7]
        y = input[:, 7:].to(**params)
        res_meshes = self.calc_mesh_mindist(input)
        res_nn = self.calc_nn_pred(input)
        err = torch.abs(res_meshes - res_nn)
        THR = 2
        err[(res_meshes < THR) & (res_nn < THR)] = 0
        if link is not None:
            err = err[:, link].reshape(-1, 1)
        max_err, _ = err.max(dim=-1)
        return max_err

    def fitness(self, input, link=None):
        res = self.calc_err_thr(torch.tensor(input).to(**params), link).cpu().numpy()
        return -1*res
