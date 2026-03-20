function inside = in_horseshoe_yz(y, z, tunnel)

    y0 = tunnel.centerY;
    zf = tunnel.floorZ;
    R  = tunnel.radius;
    Hw = tunnel.wallH;

    zs = zf + Hw;   % springline

    % 직벽 구간
    inWall = (z >= zf) & (z <= zs) & (abs(y - y0) <= R);

    % 천장 반원 구간
    inRoof = ((y - y0).^2 + (z - zs).^2 <= R^2) & (z >= zs);

    inside = inWall | inRoof;
end