function plot_dfn_preview_from_center_normal(center, normal, radius)
% Quick preview only.
% Reconstruct temporary in-plane basis from normal.

    [strike_u, dip_u] = normal_to_strike_dip_basis_vectorized(normal);

    nFrac = size(center,1);
    nCircle = 36;
    th = linspace(0, 2*pi, nCircle);

    for i = 1:nFrac
        c = center(i,:);
        r = radius(i);
        s = strike_u(i,:);
        d = dip_u(i,:);

        pts = c + r*cos(th(:)).*s + r*sin(th(:)).*d;
        fill3(pts(:,1), pts(:,2), pts(:,3), rand(1,3), ...
            'FaceAlpha', 0.18, 'EdgeColor', 'none');
    end
end