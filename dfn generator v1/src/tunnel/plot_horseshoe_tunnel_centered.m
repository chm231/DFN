function plot_horseshoe_tunnel_centered(tunnel)
% Horseshoe tunnel extruded in x-direction
% Coordinates are centered at (0,0,0)

    y0 = tunnel.centerY;
    zf = tunnel.floorZ;
    R  = tunnel.radius;
    Hw = tunnel.wallH;
    x1 = tunnel.xmin;
    x2 = tunnel.xmax;

    zs = zf + Hw;   % springline elevation

    yL = y0 - R;
    yR = y0 + R;

    % roof semicircle
    th = linspace(pi, 0, 100);
    y_arc = y0 + R*cos(th);
    z_arc = zs + R*sin(th);

    % horseshoe boundary in yz-plane
    y2d = [yL, yR, yR, y_arc, yL];
    z2d = [zf, zf, zs, z_arc, zs];

    % front face
    patch(x1*ones(size(y2d)), y2d, z2d, [0.85 0.85 0.85], ...
        'FaceAlpha', 0.08, ...
        'EdgeColor', [0.20 0.20 0.20], ...
        'LineWidth', 0.8);

    % back face
    patch(x2*ones(size(y2d)), y2d, z2d, [0.85 0.85 0.85], ...
        'FaceAlpha', 0.08, ...
        'EdgeColor', [0.20 0.20 0.20], ...
        'LineWidth', 0.8);

    % side surfaces
    for i = 1:(numel(y2d)-1)
        X = [x1 x2 x2 x1];
        Y = [y2d(i) y2d(i) y2d(i+1) y2d(i+1)];
        Z = [z2d(i) z2d(i) z2d(i+1) z2d(i+1)];

        patch(X, Y, Z, [0.82 0.82 0.82], ...
            'FaceAlpha', 0.18, ...
            'EdgeColor', [0.30 0.30 0.30], ...
            'LineWidth', 0.5);
    end
end