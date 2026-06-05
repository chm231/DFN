function plot_all_sets_full_cube(masterFile, colorBySet)
    %전체 절리 한 번에 그리는 함수
    if nargin < 2
        colorBySet = true;
    end

    M = load(masterFile);
    master = M.master;
    modelBox = master.box;

    figure('Color','w','Position',[100 100 1100 900]);
    hold on; axis equal; grid on;
    xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
    title('Full DFN in 10×10×10 m cube');

    draw_box(modelBox);

    uniformColor = [0.65 0.78 0.95];

    for k = 1:numel(master.set_files)
        S = load(master.set_files{k});

        centers = double(S.centers);
        normals = double(S.normals);
        radius  = double(S.radius);

        [strike_u, dip_u] = normal_to_strike_dip_basis_vectorized(normals);

        
        faceColor = uniformColor;
        

        plot_discs_full(centers, strike_u, dip_u, radius, faceColor);
        clear S
    end

    view([-35 20]);
    camproj perspective;
    box on;
end


function plot_discs_full(centers, strike_u, dip_u, radius, faceColor)

    nCircle = 36;
    th = linspace(0, 2*pi, nCircle);

    for i = 1:size(centers,1)
        c = centers(i,:);
        r = radius(i);
        s = strike_u(i,:);
        d = dip_u(i,:);

        pts = c + r*cos(th(:)).*s + r*sin(th(:)).*d;

        patch(pts(:,1), pts(:,2), pts(:,3), faceColor, ...
            'FaceAlpha', 0.22, ...
            'EdgeColor', [0.15 0.15 0.15], ...
            'LineWidth', 0.2);
    end
end


function draw_box(modelBox)
    x = double(modelBox.dx); y = double(modelBox.dy); z = double(modelBox.dz);

    plot3([0 x x 0 0],[0 0 y y 0],[0 0 0 0 0],'k-','LineWidth',1);
    plot3([0 x x 0 0],[0 0 y y 0],[z z z z z],'k-','LineWidth',1);

    for xx = [0 x]
        for yy = [0 y]
            plot3([xx xx],[yy yy],[0 z],'k-','LineWidth',1);
        end
    end
end