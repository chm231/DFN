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
                    'FaceAlpha', 0.5, ...
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
        % Load metadata to get input P32
        S_meta = load(master.set_files{k}, 'metadata');
        input_P32 = master.P32_input(k);
        realized_p32 = p32_k; % This is the realized P32 for the current set k
        
        sizeDist = S_meta.metadata.sizeDist;
        if isfield(sizeDist, 'type') && strcmp(sizeDist.type, 'powerlaw') && isfield(sizeDist, 'r0')
            kr = sizeDist.kr;
            rmax = sizeDist.rmax;
            r0 = sizeDist.r0;       % Baseline empirical min radius (0.28m etc)
            rmin = sizeDist.rmin;   % Render cutoff radius (1.0m)
            
            pow = 2 - kr; % Integral of r^2 * r^{-(kr+1)} = r^{1-kr}
            if abs(pow) < 1e-12
                int_r0 = log(rmax) - log(r0);
                int_rmin = log(rmax) - log(rmin);
            else
                int_r0 = (rmax^pow - r0^pow) / pow;
                int_rmin = (rmax^pow - rmin^pow) / pow;
            end
            % Theoretical true P32 for the r >= rmin interval
            true_geo_P32 = input_P32 * (int_rmin / int_r0);
        else
            true_geo_P32 = input_P32;
            r0 = sizeDist.rmin;
            rmin = sizeDist.rmin;
        end
        
        err_gen = abs(realized_p32 - input_P32) / input_P32 * 100;
        
        fprintf('--- Set %d ---\n', k);
        fprintf('  [1] Empirical Baseline P32 (r >= %.2fm) = %8.4f\n', r0, input_P32);
        fprintf('  [2] Real Geological P32    (r >= %.2fm) = %8.4f <-- True scaled target for Cropbox\n', rmin, true_geo_P32);
        fprintf('  [3] Forced Generator P32   (r >= %.2fm) = %8.4f <-- Unscaled input forced on algo\n', rmin, input_P32);
        fprintf('  [4] Actual Cropbox P32     (Generated)    = %8.4f\n', realized_p32);
        fprintf('  >> Error vs Forced Target = %5.2f %%\n', err_gen);
        fprintf('  >> Over-Spawn Ratio       = %5.2f x (Generated / True Geological)\n\n', realized_p32 / true_geo_P32);
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