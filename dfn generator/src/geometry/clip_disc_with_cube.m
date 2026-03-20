function poly = clip_disc_with_cube(center, normal, radius, modelBox, nSeg)
% Clip a disc with axis-aligned cube
% Return polygon vertices inside cube

    if nargin < 5
        nSeg = 64;
    end

    % basis
    [s, d] = normal_to_strike_dip_basis_vectorized(normal);

    % circle points
    th = linspace(0, 2*pi, nSeg)';
    pts = center + radius*cos(th).*s + radius*sin(th).*d;

    % clipping against cube
    poly = pts;

    % keep region satisfying n·x + d >= 0
    poly = clip_polygon_plane(poly, [ 1  0  0], 0);                    % x >= 0
    poly = clip_polygon_plane(poly, [-1  0  0], double(modelBox.dx));  % x <= dx

    poly = clip_polygon_plane(poly, [ 0  1  0], 0);                    % y >= 0
    poly = clip_polygon_plane(poly, [ 0 -1  0], double(modelBox.dy));  % y <= dy

    poly = clip_polygon_plane(poly, [ 0  0  1], 0);                    % z >= 0
    poly = clip_polygon_plane(poly, [ 0  0 -1], double(modelBox.dz));  % z <= dz

    if isempty(poly)
        poly = [];
    end
end