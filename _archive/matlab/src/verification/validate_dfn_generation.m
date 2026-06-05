function validate_dfn_generation(masterFile, sets)
% VALIDATE_DFN_GENERATION Runs V&V checks on the generated DFN
% 1) Log-Log Size Distribution (Power-law slope check)
% 2) Stereonet Pole Projection (Trend/Plunge & Fisher distribution)
% 3) Combined Stereonet with Density
%
% Modified to match the stereonet style used in Munier (2006):
% - fractures filtered to 1 < r < 250 m
% - equal weight per fracture pole
% - no P32-based weighting in combined stereonet

    tmp = load(masterFile, 'master');
    set_files = tmp.master.set_files;
    set_names = tmp.master.set_names;
    num_sets = numel(set_files);

    % Radius interval used for stereonet, matching the paper
    rmin_plot = 1.0;
    rmax_plot = 250.0;

    % Prepare figures
    fig_size = figure('Name', 'V&V: Size Distribution (Log-Log)', ...
        'Color', 'w', 'Position', [100, 100, 1200, 600]);
    fig_stereo = figure('Name', 'V&V: Orientation Stereonet (Per Set)', ...
        'Color', 'w', 'Position', [150, 150, 1200, 600]);

    % Data for combined stereonet
    all_X = [];
    all_Y = [];

    for s = 1:num_sets
        fprintf('Validating %s...\n', set_names{s});

        mFile = matfile(set_files{s});
        r = mFile.radius;
        n_raw = mFile.normals; % Nx3 normal vectors

        if isempty(r)
            continue;
        end

        % ---------------------------------------------------------------
        % Radius filter: use only 1 < r < 250 m
        % ---------------------------------------------------------------
        valid_r = (r > rmin_plot) & (r < rmax_plot);
        r = r(valid_r);
        n_raw = n_raw(valid_r, :);

        if isempty(r)
            continue;
        end

        %% --------------------------------------------------------------
        % 1. Size Distribution (Log-Log Plot)
        % ---------------------------------------------------------------
        figure(fig_size);
        subplot(2, ceil(num_sets/2), s);

        num_bins = 150;
        log_r = log10(r);
        [counts, edges] = histcounts(log_r, num_bins);
        bin_centers = (edges(1:end-1) + edges(2:end)) / 2;

        valid = counts > 0;
        x_data = bin_centers(valid);
        y_data = log10(counts(valid));

        plot(x_data, y_data, 'bo', 'MarkerSize', 3, 'MarkerFaceColor', 'b');
        hold on;

        if strcmp(sets(s).sizeDist.type, 'powerlaw')
            kr = sets(s).sizeDist.kr;
            C = y_data(1) + kr * x_data(1);
            y_theory = -kr * x_data + C;
            plot(x_data, y_theory, 'r-', 'LineWidth', 2);
            title(sprintf('%s (kr=%.2f)', strrep(set_names{s}, '_', ' '), kr));
            legend('Generated Data', 'Theoretical Slope', 'Location', 'sw');
        else
            title(sprintf('%s (Exponential)', strrep(set_names{s}, '_', ' ')));
        end

        xlabel('log_{10}(Radius) [m]');
        ylabel('log_{10}(Count)');
        grid on;

        %% --------------------------------------------------------------
        % 2. Stereonet Projection (Equal-Angle Wulff Net Lower Hemisphere)
        % ---------------------------------------------------------------
        figure(fig_stereo);
        subplot(2, ceil(num_sets/2), s);
        hold on;

        theta = linspace(0, 2*pi, 100);
        plot(cos(theta), sin(theta), 'k-', 'LineWidth', 1.5);

        % Lower hemisphere
        n = n_raw;
        flip_idx = n(:,3) > 0;
        n(flip_idx, :) = -n(flip_idx, :);

        % Equal-angle projection (90 deg CCW rotation, North at Top)
        X_proj = -n(:,2) ./ (1 - n(:,3));
        Y_proj =  n(:,1) ./ (1 - n(:,3));

        % Scatter subset for display only
        max_scatter = 5000;
        if length(X_proj) > max_scatter
            idx = randperm(length(X_proj), max_scatter);
            Xs = X_proj(idx);
            Ys = Y_proj(idx);
        else
            Xs = X_proj;
            Ys = Y_proj;
        end
        scatter(Xs, Ys, 3, 'b', 'filled', 'MarkerFaceAlpha', 0.2);

        % Combined plot pooling: count-based, equal weight per fracture
        all_X = [all_X; X_proj];
        all_Y = [all_Y; Y_proj];

        % Expected mean pole
        mean_tr = sets(s).trend;
        mean_pl = sets(s).plunge;
        mean_nv = mean_pole_vector_from_trend_plunge(mean_tr, mean_pl);
        if mean_nv(3) > 0
            mean_nv = -mean_nv;
        end
        mX = -mean_nv(2) / (1 - mean_nv(3));
        mY =  mean_nv(1) / (1 - mean_nv(3));
        plot(mX, mY, 'rp', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

        text(0, 1.05, 'N', 'HorizontalAlignment','center', 'FontWeight','bold');
        text(1.05, 0, 'E', 'HorizontalAlignment','center', 'FontWeight','bold');
        text(0, -1.05, 'S', 'HorizontalAlignment','center', 'FontWeight','bold');
        text(-1.05, 0, 'W', 'HorizontalAlignment','center', 'FontWeight','bold');

        axis equal;
        axis off;
        title(sprintf('%s\n(Trend: %.1f, Plunge: %.1f)', ...
            strrep(set_names{s}, '_', ' '), mean_tr, mean_pl));
    end

    %% --------------------------------------------------------------
    % 3. Combined Stereonet with Density
    % ---------------------------------------------------------------
    if ~isempty(all_X)
        fig_combined = figure('Name', 'V&V: Combined DFN Stereonet (Density)', ...
            'Color', 'w', 'Position', [200, 200, 800, 800]);
        hold on;
        axis equal;
        axis off;

        % 2D histogram of pole density
        grid_res = 120;
        grid_bins = linspace(-1, 1, grid_res);
        [counts, ~, ~] = histcounts2(all_X, all_Y, grid_bins, grid_bins);

        % Smooth density field
        sigma = 2.0;
        kernel_size = 2 * ceil(2 * sigma) + 1;
        [kx, ky] = meshgrid(-(kernel_size-1)/2:(kernel_size-1)/2);
        kernel = exp(-(kx.^2 + ky.^2) / (2 * sigma^2));
        kernel = kernel / sum(kernel(:));

        density = conv2(counts, kernel, 'same');

        % Mask outside stereonet circle
        [GX, GY] = meshgrid(grid_bins(1:end-1) + diff(grid_bins)/2);
        mask = (GX.^2 + GY.^2) <= 1.0;
        density(~mask) = NaN;

        contourf(GX, GY, density, 12, 'EdgeColor', 'none');
        colormap(jet(256));
        hcb = colorbar;
        ylabel(hcb, 'Pole Density (Count-based)');

        % Optional faint pole overlay
        sub_idx = randperm(length(all_X), min(length(all_X), 2000));
        plot(all_X(sub_idx), all_Y(sub_idx), 'k.', 'MarkerSize', 1);

        % Outer circle
        theta = linspace(0, 2*pi, 100);
        plot(cos(theta), sin(theta), 'k-', 'LineWidth', 2.0);

        % Mean pole markers
        for s = 1:num_sets
            m_tr = sets(s).trend;
            m_pl = sets(s).plunge;
            m_nv = mean_pole_vector_from_trend_plunge(m_tr, m_pl);
            if m_nv(3) > 0
                m_nv = -m_nv;
            end
            mX = -m_nv(2) / (1 - m_nv(3));
            mY =  m_nv(1) / (1 - m_nv(3));
            plot(mX, mY, 'rp', 'MarkerSize', 12, 'MarkerFaceColor', 'r', 'LineWidth', 1.0);
            text(mX + 0.05, mY + 0.05, sprintf('S%d', s), ...
                'Color', 'r', 'FontWeight', 'bold');
        end

        text(0, 1.1, 'N', 'HorizontalAlignment','center', 'FontWeight','bold', 'FontSize', 14);
        text(1.1, 0, 'E', 'HorizontalAlignment','center', 'FontWeight','bold', 'FontSize', 14);
        text(0, -1.1, 'S', 'HorizontalAlignment','center', 'FontWeight','bold', 'FontSize', 14);
        text(-1.1, 0, 'W', 'HorizontalAlignment','center', 'FontWeight','bold', 'FontSize', 14);

        title(sprintf('Combined DFN Stereonet Density Projection (%.1f < r < %.1f m)', ...
            rmin_plot, rmax_plot), 'FontSize', 16, 'FontWeight', 'bold');
    end

    disp('▶ V&V Sub-routine execution complete. Count-based combined stereonet rendered.');
end
