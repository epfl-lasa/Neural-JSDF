clc
clear all
close all
addpath("functions/")
%main function used to create a dataset of robot
%poses(q)-points(y)-distances(d) for regression learning d = f(q,y)
%%
base = eye(4);
patch_handle = [];
load('meshes/mesh_light_pts.mat');

%% 
q_min = [-2.8973 -1.7628 -2.8973 -3.0718 -2.8973 -0.0175	-2.8973 0];
q_max = [2.8973	1.7628	2.8973	-0.0698	2.8973	3.7525	2.8973 0.04];
q_span = q_max-q_min;
q_min = q_min-q_span*0.05;
q_max = q_max+q_span*0.05;

%% Preliminary dataset generation (rough data, no distance calculations yet)
N_MESHES = length(mesh)-2;
N_JPOS = 10; %1 joint position ~ 1000 workspace points. 
%N_JPOS = 5000; %value used in the paper  

N_INSIDE = repmat(25,[N_MESHES,1]);
N_OUTSIDE = repmat(35,[N_MESHES,1]);
N_CLOSE = repmat(20,[N_MESHES,1]);
N_FAR = repmat(20,[N_MESHES,1]);
N_ZERO = repmat(10,[N_MESHES,1]);

box_delta = 0.1;
bbox_far.xmin = -1; bbox_far.xmax = 1;
bbox_far.ymin = -1; bbox_far.ymax = 1;
bbox_far.zmin = -0.2; bbox_far.zmax = 1.3;
q_rand = q_min+rand(N_JPOS,8).*(q_max-q_min);
tic
dataset = [];
parfor i = 1:N_JPOS
    jpos = q_rand(i,:);
    mesh_fk = meshes_fk(mesh, base, jpos);
    pts_all = [];
    [int_pts, out_pts] = fk_transform(mesh, base, jpos);
    for j = 1:1:N_MESHES
        %generate points inside (and very close) to the link
        bbox_inside = getBBox(mesh_fk{j}.V);
        pts_inside = int_pts{j}(randi(length(int_pts{j}),[N_INSIDE(j),1]),:);
        pts_outside = out_pts{j}(randi(length(out_pts{j}),[N_OUTSIDE(j),1]),:);

        %generate points close to the link
        bbox_close = scaleBBox(bbox_inside, box_delta);
        pts_close = sampleBbox(bbox_close, N_CLOSE(j));

        %generate points far from the link
        pts_far = sampleBbox(bbox_far, N_FAR(j));
        pts_mesh = mesh_fk{j}.V(randi(length(mesh_fk{j}.V),[N_ZERO(j),1]),:);
        pts_all = [pts_all; pts_inside; pts_outside; pts_close; pts_far; pts_mesh];
    end
    n_pts = size(pts_all,1);
    dist_arr = zeros(n_pts, N_MESHES);
    for j = 1:1:N_MESHES
        [dst, pt_closest] = point2trimesh('Faces',mesh_fk{j}.F,...
                                      'Vertices',mesh_fk{j}.V,...
                                      'QueryPoints',pts_all,...
                                      'Algorithm', 'vectorized');
        %dist_arr(:,j) = dst;
        IN = inpolyhedron(mesh_fk{j}.F, mesh_fk{j}.V, pts_all);
        dst2 = abs(dst);
        dst2(IN) = -1 * dst2(IN);
        dist_arr(:,j) = dst2;
        %disp(sum(abs(dst2-dst)))
    end

    subdataset = [repmat(jpos(1:end-1),[n_pts, 1]), pts_all, dist_arr];
    dataset = [dataset; subdataset];
end
toc
save('datasets/data_mesh_test.mat','dataset')
%%
% hold on
% axis equal
% camlight
% view(160,20)
% N_DRAW = n_pts;
% for i = 1:1:N_MESHES
%     F = mesh_fk{i}.F;
%     V = mesh_fk{i}.V;
%     tmp = patch('Faces',F,'Vertices',V(:,1:3),'FaceAlpha',0.1,'EdgeAlpha',0);
%     drawnow
%     %pause(1)
% end
% plot3(pts_all(1:N_DRAW,1),pts_all(1:N_DRAW,2),pts_all(1:N_DRAW,3),'r.')
%%
