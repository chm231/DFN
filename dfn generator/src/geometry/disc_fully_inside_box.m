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

    for i = 1:N
        c = centers(i,:);
        r = radius(i);
        s = strike_u(i,:);
        d = dip_u(i,:);

        pts = c + r*ct.*s + r*st.*d;

        insideMask(i) = all(pts(:,1) >= 0 & pts(:,1) <= double(box.dx) & ...
                            pts(:,2) >= 0 & pts(:,2) <= double(box.dy) & ...
                            pts(:,3) >= 0 & pts(:,3) <= double(box.dz));
    end
end