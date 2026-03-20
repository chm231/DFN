function centers = sample_centers_from_surface_points_vectorized(box, radius, strike_u, dip_u, centerMode)
% Sample disc centers from random points on fracture surfaces.
%
% report:
%   rho = r * U
% area_uniform:
%   rho = r * sqrt(U)

    cls = class(radius);
    N = numel(radius);

    if isfield(box, 'x0'), x0 = box.x0; else, x0 = 0; end
    if isfield(box, 'y0'), y0 = box.y0; else, y0 = 0; end
    if isfield(box, 'z0'), z0 = box.z0; else, z0 = 0; end

    % 1. Preallocate 'centers' (combining previous 'Pr' logic) to save memory
    centers = zeros(N,3, cls);
    centers(:,1) = x0 + box.dx .* rand(N,1, cls);
    centers(:,2) = y0 + box.dy .* rand(N,1, cls);
    centers(:,3) = z0 + box.dz .* rand(N,1, cls);

    omega = 2*pi * rand(N,1, cls);

    switch lower(centerMode)
        case 'report'
            rho = radius .* rand(N,1, cls);
        case 'area_uniform'
            rho = radius .* sqrt(rand(N,1, cls));
        otherwise
            error('Unknown centerMode: %s', centerMode);
    end

    % 2. In-place operations directly into the 'centers' array without creating temp matrices (tdir, cosw, sinw)
    centers(:,1) = centers(:,1) + rho .* (strike_u(:,1) .* cos(omega) + dip_u(:,1) .* sin(omega));
    centers(:,2) = centers(:,2) + rho .* (strike_u(:,2) .* cos(omega) + dip_u(:,2) .* sin(omega));
    centers(:,3) = centers(:,3) + rho .* (strike_u(:,3) .* cos(omega) + dip_u(:,3) .* sin(omega));
end