function draw_horseshoe_outline_yz(tunnel)

    y0 = tunnel.centerY;
    zf = tunnel.floorZ;
    R  = tunnel.radius;
    Hw = tunnel.wallH;

    zs = zf + Hw;

    yL = y0 - R;
    yR = y0 + R;

    th = linspace(pi, 0, 200);
    y_arc = y0 + R*cos(th);
    z_arc = zs + R*sin(th);

    y2d = [yL, yR, yR, y_arc, yL];
    z2d = [zf, zf, zs, z_arc, zs];

    plot(y2d, z2d, 'k-', 'LineWidth', 1.5);
end