function plot_clipped_dfn_crop(masterFile, cropBox)

    M = load(masterFile);
    master = M.master;

    centerShift = [0 0 0]; % Domain is already centered at origin

    cropBox_centered.xmin = cropBox.xmin - centerShift(1);
    cropBox_centered.xmax = cropBox.xmax - centerShift(1);
    cropBox_centered.ymin = cropBox.ymin - centerShift(2);
    cropBox_centered.ymax = cropBox.ymax - centerShift(2);
    cropBox_centered.zmin = cropBox.zmin - centerShift(3);
    cropBox_centered.zmax = cropBox.zmax - centerShift(3);

    figure('Color','w','Position',[100 100 1200 900]);
    hold on; axis equal; grid on;

    xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
    title('Clipped DFN inside crop box');

    draw_crop_box(cropBox_centered);

    color = [0.7 0.7 0.7];
    countVisible = 0;

    for k = 1:numel(master.set_files)

        S = load(master.set_files{k});

        centers = double(S.centers) - centerShift;
        normals = double(S.normals);
        radius  = double(S.radius);

        for i = 1:size(centers,1)

            poly = clip_disc_with_cropbox( ...
                centers(i,:), normals(i,:), radius(i), cropBox_centered);

            if ~isempty(poly)

                patch(poly(:,1), poly(:,2), poly(:,3), color, ...
                    'FaceAlpha', 0.7, ...
                    'EdgeColor', [0.3 0.3 0.3], ...
                    'LineWidth', 0.5);

                countVisible = countVisible + 1;
            end
        end
    end

    fprintf('Visible clipped fractures in crop box = %d\n', countVisible);

    % (Tunnel plotting functions have been removed)

    view([-35 20]);
    camlight headlight;
    lighting gouraud;
    material dull;
end