function  [int_pts, close_pts] = fk_transform(mesh, base, joint_state)
%%
r = [0 0 0 0.0825 -0.0825 0 0.088 0];
d = [0.333 0 0.316 0 0.384 0 0 0.107];
alpha = [0 -pi/2 pi/2 pi/2 -pi/2 pi/2 pi/2 0];

P = franka_dh_fk(joint_state, r, d, alpha,base);
idx_order ={'link0';'link1';'link2';'link3';'link4';'link5';'link6';'link7';'hand';'finger';'finger'};
%plot base to end-effector (without fingers!)
int_pts = cell(length(idx_order),1);
close_pts = cell(length(idx_order),1);
for i = 1:1:length(idx_order)
    idx = find_idx(idx_order{i},mesh);
    R = P{i}(1:3,1:3);
    T = P{i}(1:3,4);
    int_pts{i} = mesh{idx}.int_pts*R'+T';
    close_pts{i} = mesh{idx}.close_pts*R'+T';
end
