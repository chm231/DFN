function normals = sample_fisher_normals(mean_n, kappa, N)
% Sample unit normals from 3D Fisher distribution.

    mean_n = mean_n(:) / norm(mean_n);
    U = rand(N,1);
    phi = 2*pi*rand(N,1);

    if kappa < 1e-10
        cosTheta = 2*U - 1;
    else
        cosTheta = (1./kappa) .* log(exp(-kappa) + U .* (exp(kappa) - exp(-kappa)));
        cosTheta = max(-1, min(1, cosTheta));
    end

    sinTheta = sqrt(max(0, 1 - cosTheta.^2));
    v0 = [sinTheta.*cos(phi), sinTheta.*sin(phi), cosTheta];

    ez = [0;0;1];
    if norm(mean_n - ez) < 1e-12
        R = eye(3);
    elseif norm(mean_n + ez) < 1e-12
        R = [1 0 0; 0 -1 0; 0 0 -1];
    else
        axis_rot = cross(ez, mean_n);
        axis_rot = axis_rot / norm(axis_rot);
        angle = acos(dot(ez, mean_n));
        R = axis_angle_to_rotmat(axis_rot, angle);
    end

    normals = (R * v0.').';
    normals = normals ./ vecnorm(normals, 2, 2);
end

function R = axis_angle_to_rotmat(axis, angle)
    x = axis(1); y = axis(2); z = axis(3);
    c = cos(angle);
    s = sin(angle);
    C = 1 - c;

    R = [x*x*C + c,   x*y*C - z*s, x*z*C + y*s;
         y*x*C + z*s, y*y*C + c,   y*z*C - x*s;
         z*x*C - y*s, z*y*C + x*s, z*z*C + c];
end