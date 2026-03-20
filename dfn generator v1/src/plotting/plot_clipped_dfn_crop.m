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
    total_area = 0.0;
    area_per_set = zeros(numel(master.set_files), 1);

    for k = 1:numel(master.set_files)

        S = load(master.set_files{k});

        centers = double(S.centers) - centerShift;
        normals = double(S.normals);
        radius  = double(S.radius);
        
        % 1. AABB Vectorized Pre-filtering (Culling)
        valid_mask = ...
            (centers(:,1) - radius <= cropBox_centered.xmax) & (centers(:,1) + radius >= cropBox_centered.xmin) & ...
            (centers(:,2) - radius <= cropBox_centered.ymax) & (centers(:,2) + radius >= cropBox_centered.ymin) & ...
            (centers(:,3) - radius <= cropBox_centered.zmax) & (centers(:,3) + radius >= cropBox_centered.zmin);
            
        valid_idx = find(valid_mask);
        nValid = length(valid_idx);
        
        % Pre-allocate cell array for parfor results
        polys_ch = cell(nValid, 1);
        
        % 2. Parallel computing for the heavy geometric clipping
        parfor idx = 1:nValid
            i = valid_idx(idx);
            polys_ch{idx} = clip_disc_with_cropbox( ...
                centers(i,:), normals(i,:), radius(i), cropBox_centered);
        end
        
        % 3. Sequential plotting and area computation
        for idx = 1:nValid
            poly = polys_ch{idx};
            if ~isempty(poly)
                patch(poly(:,1), poly(:,2), poly(:,3), color, ...
                    'FaceAlpha', 0.7, ...
                    'EdgeColor', [0.3 0.3 0.3], ...
                    'LineWidth', 0.5);

                countVisible = countVisible + 1;
                
                % Compute polygon area and accumulate
                poly_c = [poly; poly(1,:)];
                cross_sum = sum(cross(poly_c(1:end-1,:), poly_c(2:end,:)), 1);
                a = 0.5 * norm(cross_sum);
                total_area = total_area + a;
                area_per_set(k) = area_per_set(k) + a;
            end
        end
    end

    fprintf('Visible clipped fractures in crop box = %d\n', countVisible);

    % Compute P32
    cx = cropBox_centered.xmax - cropBox_centered.xmin;
    cy = cropBox_centered.ymax - cropBox_centered.ymin;
    cz = cropBox_centered.zmax - cropBox_centered.zmin;
    cropVol = cx * cy * cz;
    p32_crop = total_area / cropVol;

    fprintf('Cropbox Volume = %.2f m^3\n', cropVol);
    
    fprintf('\n--- P32 Summary in Cropbox ---\n');
    for k = 1:numel(master.set_files)
        p32_k = area_per_set(k) / cropVol;
        fprintf('Set %d: Input P32 = %.4f | Realized P32 in Cropbox = %.4f\n', ...
            k, master.P32_input(k), p32_k);
    end
    fprintf('------------------------------\n');
    fprintf('TOTAL Input P32 = %.4f\n', sum(master.P32_input));
    fprintf('TOTAL Realized P32 in Cropbox = %.4f\n\n', p32_crop);

    % (Tunnel plotting functions have been removed)

    view([-35 20]);
    camlight headlight;
    lighting gouraud;
    material dull;
end