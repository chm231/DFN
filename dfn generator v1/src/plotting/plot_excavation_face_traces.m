function plot_excavation_face_traces(masterFile, cropBox, xFace, tunnel)

    M = load(masterFile);
    master = M.master;

    % 현재 사용 중인 중심 이동
    centerShift = [12.5 12.5 12.5];

    cropBox_centered.xmin = cropBox.xmin - centerShift(1);
    cropBox_centered.xmax = cropBox.xmax - centerShift(1);
    cropBox_centered.ymin = cropBox.ymin - centerShift(2);
    cropBox_centered.ymax = cropBox.ymax - centerShift(2);
    cropBox_centered.zmin = cropBox.zmin - centerShift(3);
    cropBox_centered.zmax = cropBox.zmax - centerShift(3);

    figure('Color','w','Position',[100 100 900 800]);
    hold on; axis equal; grid on;
    xlabel('y [m]');
    ylabel('z [m]');
    title(sprintf('Excavation face traces at x = %.2f m', xFace));

    % 터널 단면 외곽
    draw_horseshoe_outline_yz(tunnel);

    nTrace = 0;

    for k = 1:numel(master.set_files)

        S = load(master.set_files{k});

        centers = double(S.centers) - centerShift;
        normals = double(S.normals);
        radius  = double(S.radius);

        for i = 1:size(centers,1)

            % 1) crop cube 내부 polygon으로 먼저 clipping
            poly = clip_disc_with_cropbox( ...
                centers(i,:), normals(i,:), radius(i), cropBox_centered);

            if isempty(poly)
                continue;
            end

            % 2) 굴진면 x = xFace 와의 교선 계산
            seg = intersect_polygon_with_xplane(poly, xFace);

            if isempty(seg)
                continue;
            end

            % 3) 마제형 단면 내부에 들어오는 부분만 표시
            plot_segment_inside_horseshoe(seg, tunnel);

            nTrace = nTrace + 1;
        end
    end

    fprintf('Visible fracture traces at x = %.2f : %d\n', xFace, nTrace);

    xlim([tunnel.centerY - tunnel.radius - 1, tunnel.centerY + tunnel.radius + 1]);
    ylim([tunnel.floorZ - 1, tunnel.floorZ + tunnel.wallH + tunnel.radius + 1]);
end