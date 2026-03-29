function validate_dfn_generation(masterFile, sets)
% VALIDATE_DFN_GENERATION Runs V&V checks on the generated DFN
% 1) Log-Log Size Distribution (Power-law slope check)
% 2) Stereonet Pole Projection (Trend/Plunge & Fisher distribution)

    tmp = load(masterFile, 'master');
    set_files = tmp.master.set_files;
    set_names = tmp.master.set_names;
    num_sets = numel(set_files);
    
    % Prepare figures
    fig_size = figure('Name', 'V&V: Size Distribution (Log-Log)', 'Color', 'w', 'Position', [100, 100, 1200, 600]);
    fig_stereo = figure('Name', 'V&V: Orientation Stereonet', 'Color', 'w', 'Position', [150, 150, 1200, 600]);
    
    for s = 1:num_sets
        fprintf('Validating %s...\n', set_names{s});
        
        mFile = matfile(set_files{s});
        r = mFile.radius;
        n = mFile.normals; % Nx3 normal vectors
        
        if isempty(r)
            continue;
        end
        
        %% ----------------------------------------------------------------
        % 1. Size Distribution (Log-Log Plot)
        % -----------------------------------------------------------------
        figure(fig_size);
        subplot(2, ceil(num_sets/2), s);
        
        % Histogram of log10(r)
        num_bins = 150;
        log_r = log10(r);
        [counts, edges] = histcounts(log_r, num_bins);
        bin_centers = (edges(1:end-1) + edges(2:end)) / 2;
        
        % Filter empty bins
        valid = counts > 0;
        x_data = bin_centers(valid);
        y_data = log10(counts(valid));
        
        plot(x_data, y_data, 'bo', 'MarkerSize', 3, 'MarkerFaceColor', 'b');
        hold on;
        
        % If powerlaw, plot theoretical line
        if strcmp(sets(s).sizeDist.type, 'powerlaw')
            kr = sets(s).sizeDist.kr;
            % Theoretical slope on log-log is -kr.
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
        
        %% ----------------------------------------------------------------
        % 2. Stereonet Projection (Equal-Angle Wulff Net Lower Hemisphere)
        % -----------------------------------------------------------------
        figure(fig_stereo);
        subplot(2, ceil(num_sets/2), s);
        hold on;
        
        % Draw stereonet outer circle
        theta = linspace(0, 2*pi, 100);
        plot(cos(theta), sin(theta), 'k-', 'LineWidth', 1.5);
        
        % Project poles
        % n is Nx3 normal vector. x=East, y=North, z=Up.
        % Force all normals to point downward (z < 0) for lower hemisphere.
        flip_idx = n(:,3) > 0;
        n(flip_idx, :) = -n(flip_idx, :);
        
        % Equal-Angle Projection formulas
        X_proj = n(:,1) ./ (1 - n(:,3));
        Y_proj = n(:,2) ./ (1 - n(:,3));
        
        % Instead of millions of points, plot a 2D scatter subset
        max_scatter = 5000;
        if length(X_proj) > max_scatter
            idx = randperm(length(X_proj), max_scatter);
            scatter(X_proj(idx), Y_proj(idx), 3, 'b', 'filled', 'MarkerFaceAlpha', 0.2);
        else
            scatter(X_proj, Y_proj, 3, 'b', 'filled', 'MarkerFaceAlpha', 0.2);
        end
        
        % Plot the expected Mean Pole
        mean_tr = sets(s).trend;
        mean_pl = sets(s).plunge;
        mean_n = mean_pole_vector_from_trend_plunge(mean_tr, mean_pl);
        if mean_n(3) > 0, mean_n = -mean_n; end
        mX = mean_n(1) / (1 - mean_n(3));
        mY = mean_n(2) / (1 - mean_n(3));
        plot(mX, mY, 'rp', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
        
        % North/East/South/West labels
        text(0, 1.05, 'N', 'HorizontalAlignment','center', 'FontWeight','bold');
        text(1.05, 0, 'E', 'HorizontalAlignment','center', 'FontWeight','bold');
        text(0, -1.05, 'S', 'HorizontalAlignment','center', 'FontWeight','bold');
        text(-1.05, 0, 'W', 'HorizontalAlignment','center', 'FontWeight','bold');
        
        axis equal; axis off;
        title(sprintf('%s\n(Trend: %.1f, Plunge: %.1f)', strrep(set_names{s}, '_', ' '), mean_tr, mean_pl));
    end
    
    disp('▶ V&V Sub-routine execution complete. All validation geometry plots rendered.');
end
