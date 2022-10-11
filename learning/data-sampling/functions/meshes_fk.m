function mesh_fk = meshes_fk(mesh, base, joint_state)
%%
r = [0 0 0 0.0825 -0.0825 0 0.088 0];
d = [0.333 0 0.316 0 0.384 0 0 0.107];
alpha = [0 -pi/2 pi/2 pi/2 -pi/2 pi/2 pi/2 0];

P = franka_dh_fk(joint_state, r, d, alpha,base);

mesh_fk = cell(length(mesh),1);
for i = 1:1:length(mesh)
    R = P{i}(1:3,1:3);
    T = P{i}(1:3,4);
    mesh_fk{i}.V = mesh{i}.v*R'+T';
    mesh_fk{i}.F = mesh{i}.f;
end
