function patch_handle = plot_franka_fcn(ax_handle,patch_handle,mesh, base, joint_state,clr,hl_idx)
%%
r = [0 0 0 0.0825 -0.0825 0 0.088 0];
d = [0.333 0 0.316 0 0.384 0 0 0.107];
alpha = [0 -pi/2 pi/2 pi/2 -pi/2 pi/2 pi/2 0];

%r = [0 0 0 0.0825 -0.0825 0 0.088];
%d = [0.333 0 0.316 0 0.384 0 0];
%alpha = [0 -pi/2 pi/2 pi/2 -pi/2 pi/2 pi/2];

P = franka_dh_fk(joint_state, r, d, alpha,base);
init = 0;
if isempty(patch_handle)
    init = 1;
end
%hl = {};
%hl = {'hand';'link3'};
%% plotting
%uiaxes(ax_handle);
facecolor = clr;
facealpha = 1;
edgealpha = 0;
idx_order ={'link0';'link1';'link2';'link3';'link4';'link5';'link6';'link7';'hand';'finger';'finger'};
idx_order ={'link0';'link1';'link2';'link3';'link4';'link5';'link6';'link7';'hand'};

%plot base to end-effector (without fingers!)
for i = 1:1:length(idx_order)
    idx = find_idx(idx_order{i},mesh);
    R = P{i}(1:3,1:3);
    T = P{i}(1:3,4);
    V = mesh{idx}.v*R'+T';
    %V = (P{i}*[mesh{idx}.v ones(size(mesh{idx}.v,1),1)]')';
    F = mesh{idx}.f;
%     hl_idx
    if ismember(idx_order{i},idx_order(hl_idx))
        facecolor = [1 0 0];
    else
        facecolor = clr;
    end
    if init
        patch_handle{i} = patch(ax_handle,'Faces',F,'Vertices',V(:,1:3),'facecolor',facecolor,'Facealpha',facealpha,'edgealpha',edgealpha);
    else
        patch_handle{i}.Vertices = V(:,1:3);
        patch_handle{i}.FaceColor = facecolor;
    end
end

% %plot hand - inspired by https://github.com/marcocognetti/FrankaEmikaPandaDynModel/blob/master/matlab/utils/LoadFrankaSTLModel.m
% idx = find_idx('hand',mesh);
% V = (P{9}*[mesh{idx}.v ones(size(mesh{idx}.v,1),1)]')';
% F = mesh{idx}.f;
% if init
%     patch_handle{9} = patch(ax_handle,'Faces',F,'Vertices',V(:,1:3),'facecolor',facecolor,'Facealpha',facealpha,'edgealpha',edgealpha);
% else
%     patch_handle{9}.Vertices = V(:,1:3);
% end

% %plot finger 1
% idx = find_idx('finger',mesh);
% T = eye(4);
% T(1:3,4) = [0 joint_state(8) 0.065]';
% P_f1 = P{9}*T;
% %Tf1 = [0 joint_state(8) 0.065];
% %V = (P{9}*[mesh{idx}.v+Tf1 ones(size(mesh{idx}.v,1),1)]')';
% V = (P_f1*[mesh{idx}.v ones(size(mesh{idx}.v,1),1)]')';
% F = mesh{idx}.f;
% if init
%     patch_handle{10} = patch(ax_handle,'Faces',F,'Vertices',V(:,1:3),'facecolor',facecolor,'Facealpha',facealpha,'edgealpha',edgealpha);
% else
%     patch_handle{10}.Vertices = V(:,1:3);
% end
% %plot finger 2
% T = eye(4);
% T(1:3,1:3) = eul2rotm(deg2rad([180 0 0]),'ZYX');
% T(1:3,4) = [0 -joint_state(8) 0.065]';
% P_f2 = P{9}*T;
% V = (P_f2*[mesh{idx}.v ones(size(mesh{idx}.v,1),1)]')';
% % Rf2 = eul2rotm(deg2rad([180 0 0]),'ZYX');
% % Tf2 = [0 -joint_state(8) 0.065];
% % V = (P{9}*[mesh{idx}.v*Rf2+Tf2 ones(size(mesh{idx}.v,1),1)]')';
% F = mesh{idx}.f;
% if init
%     patch_handle{11} = patch(ax_handle,'Faces',F,'Vertices',V(:,1:3),'facecolor',facecolor,'Facealpha',facealpha,'edgealpha',edgealpha);
% else
%     patch_handle{11}.Vertices = V(:,1:3);
% end
end
