function insideMask = disc_fully_inside_box(centers, strike_u, dip_u, radius, box, nSeg)
% Check whether each disc lies fully inside the box by polygon sampling.

    if nargin < 6
        nSeg = 48;
    end

    N = size(centers,1);
    insideMask = false(N,1);

    th = linspace(0, 2*pi, nSeg);
    ct = cos(th(:));
    st = sin(th(:));

    if isfield(box, 'x0'), x0 = box.x0; else, x0 = 0; end
    if isfield(box, 'y0'), y0 = box.y0; else, y0 = 0; end
    if isfield(box, 'z0'), z0 = box.z0; else, z0 = 0; end

    for i = 1:N
        c = centers(i,:);
        r = radius(i);
        s = strike_u(i,:);
        d = dip_u(i,:);

        pts = c + r*ct.*s + r*st.*d;

        insideMask(i) = all(pts(:,1) >= x0 & pts(:,1) <= x0 + double(box.dx) & ...
                            pts(:,2) >= y0 & pts(:,2) <= y0 + double(box.dy) & ...
                            pts(:,3) >= z0 & pts(:,3) <= z0 + double(box.dz));
    end
end