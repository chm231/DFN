function poly = clip_disc_with_cropbox(center, normal, radius, cropBox, nSeg)

    if nargin < 5
        nSeg = 64;
    end

    [s, d] = normal_to_strike_dip_basis_vectorized(normal);

    th = linspace(0, 2*pi, nSeg)';
    pts = center + radius*cos(th).*s + radius*sin(th).*d;

    poly = pts;

    poly = clip_polygon_plane(poly, [ 1  0  0], -cropBox.xmin); % x >= xmin
    poly = clip_polygon_plane(poly, [-1  0  0],  cropBox.xmax); % x <= xmax

    poly = clip_polygon_plane(poly, [ 0  1  0], -cropBox.ymin); % y >= ymin
    poly = clip_polygon_plane(poly, [ 0 -1  0],  cropBox.ymax); % y <= ymax

    poly = clip_polygon_plane(poly, [ 0  0  1], -cropBox.zmin); % z >= zmin
    poly = clip_polygon_plane(poly, [ 0  0 -1],  cropBox.zmax); % z <= zmax

    if isempty(poly)
        poly = [];
    end
end