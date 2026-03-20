function plot_clipped_dfn(masterFile)

    M = load(masterFile);
    master = M.master;
    modelBox = master.box;

    figure('Color','w','Position',[100 100 1100 900]);
    hold on; axis equal; grid on;

    xlabel('x'); ylabel('y'); zlabel('z');
    title('Clipped DFN inside cube');

    draw_box(modelBox);

    color = [0.7 0.7 0.7];

    for k = 1:numel(master.set_files)

        S = load(master.set_files{k});

        centers = double(S.centers);
        normals = double(S.normals);
        radius  = double(S.radius);

        for i = 1:size(centers,1)

            poly = clip_disc_with_cube( ...
                centers(i,:), normals(i,:), radius(i), modelBox);

            if ~isempty(poly)
                patch(poly(:,1), poly(:,2), poly(:,3), color, ...
                    'FaceAlpha', 0.7, ...
                    'EdgeColor', [0.3 0.3 0.3],'LineWidth', 0.5);
            end
        end
    end

    view([-35 20]);
    camlight headlight;
    lighting gouraud;
end