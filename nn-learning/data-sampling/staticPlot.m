clc
clear all
%%
fig_handle = figure('Name','Franka Kinematic Model','Position',[100 100 600 600]);
ax_body = axes(fig_handle,'View',[115 12],'Position',[0.1300 0.1100 0.7750 0.8150]);
axis equal
hold on
light
ax_body.XLim = [-0.9 0.9];
ax_body.YLim = [-0.9 0.9];
ax_body.ZLim = [-0.3 1.3];
patch_handle = [];
load('meshes/mesh_light.mat');
load('meshes/mesh_full.mat');

base = eye(4);
q_min = [-2.8973 -1.7628 -2.8973 -3.0718 -2.8973 -0.0175	-2.8973 0];
q_max = [2.8973	1.7628	2.8973	-0.0698	2.8973	3.7525	2.8973 0.04];
joint_state = [0.6144, -0.0748, -1.4672, -1.6090, -0.1344,  1.5553, -0.7080, 0];
patch_handle = plot_franka_fcn(ax_body,patch_handle,mesh, base, joint_state,[0 0.5 1],[]);







