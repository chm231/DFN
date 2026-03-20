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

    Pr = zeros(N,3, cls);
    Pr(:,1) = x0 + box.dx .* rand(N,1, cls);
    Pr(:,2) = y0 + box.dy .* rand(N,1, cls);
    Pr(:,3) = z0 + box.dz .* rand(N,1, cls);

    omega = 2*pi .* rand(N,1, cls);
    U = rand(N,1, cls);

    switch lower(centerMode)
        case 'report'
            rho = radius .* U;
        case 'area_uniform'
            rho = radius .* sqrt(U);
        otherwise
            error('Unknown centerMode: %s', centerMode);
    end

    cosw = cos(omega);
    sinw = sin(omega);

    tdir = strike_u .* cosw + dip_u .* sinw;
    centers = Pr + rho .* tdir;
end