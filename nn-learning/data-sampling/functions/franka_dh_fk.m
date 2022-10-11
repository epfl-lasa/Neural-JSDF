function P = franka_dh_fk(j_state,r,d,alpha,base)
    A = @(r,d,alpha,q) [cos(q) -sin(q)*cos(alpha)   sin(q)*sin(alpha)	r*cos(q);
                        sin(q)	cos(q)*cos(alpha)	-cos(q)*sin(alpha)  r*sin(q);
                        0       sin(alpha)          cos(alpha)          d;
                        0       0                   0                   1];
    Am =@(r,d,alpha,q) [cos(q) -sin(q) 0 r;
                       sin(q)*cos(alpha) cos(q)*cos(alpha) -sin(alpha) -d*sin(alpha);
                       sin(q)*sin(alpha) cos(q)*sin(alpha) cos(alpha) d*cos(alpha);
                       0    0   0   1];
    %base matrix holds index 1 instead of 0 (because matlab)
    P{1} = base;
    %kinematic chain for 7 joints
    for i = 2:1:8
        P{i} = P{i-1}*Am(r(i-1),d(i-1),alpha(i-1),j_state(i-1));
    end
    
    %transformation for hand end-effector, no joint movement here
    %-pi/4 is a mesh rotation, found manually, not specified anywhere
    P{9} = P{end}*Am(r(end),d(end),alpha(end),-pi/4);
    
    %final two for fingers - inspired by:
    % https://github.com/marcocognetti/FrankaEmikaPandaDynModel/blob/master/matlab/utils/LoadFrankaSTLModel.m

    %finger1
    T = eye(4);
    T(1:3,4) = [0 j_state(8) 0.065]';
    P{10} = P{9}*T;
    
    %finger2
    T = eye(4);
    T(1:3,1:3) = eul2rotm(deg2rad([180 0 0]),'ZYX');
    T(1:3,4) = [0 -j_state(8) 0.065]';
    P{11} = P{9}*T;
end
