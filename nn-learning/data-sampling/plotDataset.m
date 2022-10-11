clc
clear all
close all
%%
%% visualization of sampled data. Specify any index below:
idx_plot = 1;
%%
load('datasets/data_mesh_test.mat');
jpos = dataset(:,1:10);
dists = dataset(:,11:end);
q_min = [-2.8973 -1.7628 -2.8973 -3.0718 -2.8973 -0.0175	-2.8973 0];
q_max = [2.8973	1.7628	2.8973	-0.0698	2.8973	3.7525	2.8973 0.04];
uniq_jpos =  unique(jpos(:,1:7),'rows');
N_uniq = size(uniq_jpos,1);
pts_per_jpos = length(dataset)/N_uniq;
n_links = size(dists,2);
for j = 1:1:n_links
    %figure(j)
    %hist(dists(:,j), 100);
    negpts = nnz(dists(:,j)<=0)/length(dists(:,j));
    closepts = nnz(dists(:,j)<=0.03 & dists(:,j)>0)/length(dists(:,j));
    sprintf('Negative points in link %d: %4.2f',[j, negpts])
    sprintf('Close points in link %d: %4.2f',[j, closepts])
    sprintf('Positive points in link %d: %4.2f',[j, 1-negpts])
end
idx_plot = max(idx_plot, N_uniq)-1;
i1 = idx_plot*pts_per_jpos+1;
i2 = i1+pts_per_jpos-1;
dst_arr = dists(i1:i2,:);
% idx_pos = find(all(dst_arr>0,2));
% idx_neg = find(any(dst_arr<=0,2));
ax_body = axes('View',[115 12],'Position',[0.1300 0.1100 0.7750 0.8150]);
hold on 
axis equal
load('meshes/mesh_light.mat');

mesh_fk = meshes_fk(mesh, eye(4), [jpos(i1,1:7) 0]);
N_MESHES = length(mesh_fk)-2;
tmp_handle = plot_franka_fcn(ax_body,[], mesh, eye(4), [jpos(i1,1:7) 0],[0 0.5 1],[]);
for i = i1:1:i2
    hold on
    if all(dst_arr(i-i1+1,:)>0)
        plot3(jpos(i,8),jpos(i,9),jpos(i,10),'b.')
        a = 10;
    else
        plot3(jpos(i,8),jpos(i,9),jpos(i,10),'r.')
    end
    %drawnow
    dist_arr = zeros(1,N_MESHES);
%     for j = 1:1:N_MESHES
%         [dst, pt_closest] = point2trimesh('Faces',mesh_fk{j}.F,...
%                                       'Vertices',mesh_fk{j}.V,...
%                                       'QueryPoints',jpos(i,[8,9,10]),...
%                                       'Algorithm', 'linear');
%         dist_arr(j) = dst;
%         plot3(mesh_fk{j}.V(:,1),mesh_fk{j}.V(:,2),mesh_fk{j}.V(:,3),'g.')
%     end
end
camlight