clc
clear all
close all
addpath("functions/")
%%
% This function is used to presample points around meshes, not to repeat
% this procedure for each q, but use FK to reposition the points
% (because sampling close-to-collision points uniformly takes a long time
% for each robot state q
%
% Input: mesh_light.mat; Output: mesh_light_pts.mat
%%
patch_handle = [];
load('meshes/mesh_light.mat');

%% 
q_min = [-2.8973 -1.7628 -2.8973 -3.0718 -2.8973 -0.0175	-2.8973 0];
q_max = [2.8973	1.7628	2.8973	-0.0698	2.8973	3.7525	2.8973 0.04];
q_span = q_max-q_min;
q_min = q_min-q_span*0.05;
q_max = q_max+q_span*0.05;
N_MESHES = 11;
base = eye(4);

for j = 1:1:N_MESHES
    %generate points inside (and very close) to the link
    bbox_inside = getBBox(mesh{j}.v);
    bbox_inside = scaleBBox(bbox_inside,0.2);
    pts_all = sampleBbox(bbox_inside, 10000);
    [dst, pt_closest] = point2trimesh('Faces',mesh{j}.f,...
                                  'Vertices',mesh{j}.v,...
                                  'QueryPoints',pts_all,...
                                  'Algorithm', 'vectorized');
    %dist_arr(:,j) = dst;
    IN = inpolyhedron(mesh{j}.f, mesh{j}.v, pts_all);
    dst2 = abs(dst);
    dst2(IN) = -1 * dst2(IN);
    %figure(j)

    lbl_inside = dst2<0;
    %plot3(pts_all(lbl_inside,1),pts_all(lbl_inside,2),pts_all(lbl_inside,3),'r.')
    axis equal
    hold on
    mesh{j}.int_pts = pts_all(lbl_inside,:);

    lbl_close = dst2>0 & dst2<0.03;
    %plot3(pts_all(lbl_close,1),pts_all(lbl_close,2),pts_all(lbl_close,3),'b.')
    mesh{j}.close_pts = pts_all(lbl_close,:);

end
%save('meshes/mesh_light_pts.mat','mesh')

%% some plotting (if needed)
% clc
% clear all
% close all
% load('meshes/mesh_light_pts.mat')
% q_min = [-2.8973 -1.7628 -2.8973 -3.0718 -2.8973 -0.0175	-2.8973 0];
% q_max = [2.8973	1.7628	2.8973	-0.0698	2.8973	3.7525	2.8973 0.04];
% q_span = q_max-q_min;
% base = eye(4);
% 
% pts_trans = fk_transform(mesh, base, q_min+q_span/2);
% for i = 1:1:length(pts_trans)
%     plot3(pts_trans{i}(:,1),pts_trans{i}(:,2),pts_trans{i}(:,3),'r.')
%     hold on
%     axis equal
% end