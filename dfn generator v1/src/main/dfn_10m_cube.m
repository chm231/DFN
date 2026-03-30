clc; clear; close all;

% 현재 파일 위치를 기준으로 상위 폴더(src) 및 하위 폴더들을 MATLAB 경로에 추가
currentDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(currentDir, '..')));

%% =========================================================
% DFN GENERATED IN 300x300x300 m DOMAIN
% THEN VISUALIZED IN CENTRAL 20x20x20 m CROPPED CUBE
%% =========================================================

outDir = 'dfn_output_cube250m';
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

masterFile = fullfile(outDir, 'dfn_master_index.mat');

%% -----------------------------
%  DFN Domain (generation domain)
%% -----------------------------
box.dx = single(250);
box.dy = single(250);
box.dz = single(250);
box.x0 = -box.dx / 2;
box.y0 = -box.dy / 2;
box.z0 = -box.dz / 2;

rng('shuffle');

%% -----------------------------
%  CHOOSE STRATIGRAPHIC SITE
%% -----------------------------
site_name = 'Laxemar'; % <--- Change this to 'Forsmark' or 'Laxemar'

%% -----------------------------
%  DFN SETS
%% -----------------------------
sets = [];

rmin_explicit = 1.0;
rmax_model = 250.0;

switch lower(site_name)
    case 'forsmark'
        % ========================================================
        % Tabe 2-2. Forsmark DFN, version 1.2.
        % ========================================================
        % ----- Set 1
        sets(1).name = 'Set_1';
        sets(1).P32 = 0.602;
        sets(1).sizeDist.type = 'powerlaw';
        sets(1).sizeDist.kr = 2.88;
        sets(1).sizeDist.r0 = 0.28;
        sets(1).sizeDist.rmin = max(rmin_explicit, 0.28);
        sets(1).sizeDist.rmax = rmax_model;
        sets(1).trend = 87.2;
        sets(1).plunge = 1.7;
        sets(1).kappa = 21.66;

        % ----- Set 2
        sets(2).name = 'Set_2';
        sets(2).P32 = 2.069;
        sets(2).sizeDist.type = 'powerlaw';
        sets(2).sizeDist.kr = 3.02;
        sets(2).sizeDist.r0 = 0.25;
        sets(2).sizeDist.rmin = max(rmin_explicit, 0.25);
        sets(2).sizeDist.rmax = rmax_model;
        sets(2).trend = 135.2;
        sets(2).plunge = 2.7;
        sets(2).kappa = 21.54;

        % ----- Set 3
        sets(3).name = 'Set_3';
        sets(3).P32 = 0.448;
        sets(3).sizeDist.type = 'powerlaw';
        sets(3).sizeDist.kr = 2.81;
        sets(3).sizeDist.r0 = 0.14;
        sets(3).sizeDist.rmin = max(rmin_explicit, 0.14);
        sets(3).sizeDist.rmax = rmax_model;
        sets(3).trend = 40.6;
        sets(3).plunge = 2.2;
        sets(3).kappa = 23.90;

        % ----- Set 4
        sets(4).name = 'Set_4';
        sets(4).P32 = 0.226;
        sets(4).sizeDist.type = 'powerlaw';
        sets(4).sizeDist.kr = 2.95;
        sets(4).sizeDist.r0 = 0.15;
        sets(4).sizeDist.rmin = max(rmin_explicit, 0.15);
        sets(4).sizeDist.rmax = rmax_model;
        sets(4).trend = 190.4;
        sets(4).plunge = 0.7;
        sets(4).kappa = 30.63;

        % ----- Set 5
        sets(5).name = 'Set_5';
        sets(5).P32 = 0.605;
        sets(5).sizeDist.type = 'powerlaw';
        sets(5).sizeDist.kr = 2.92;
        sets(5).sizeDist.r0 = 0.25;
        sets(5).sizeDist.rmin = max(rmin_explicit, 0.25);
        sets(5).sizeDist.rmax = rmax_model;
        sets(5).trend = 342.9;
        sets(5).plunge = 80.3;
        sets(5).kappa = 8.18;

    case 'laxemar'
        % ========================================================
        % Table 2-1. Laxemar DFN, version 1.2.
        % ========================================================
        % ----- Set 1
        sets(1).name = 'Set_1';
        sets(1).P32 = 1.310;
        sets(1).sizeDist.type = 'powerlaw';
        sets(1).sizeDist.kr = 2.85;
        sets(1).sizeDist.r0 = 0.328;
        sets(1).sizeDist.rmin = max(rmin_explicit, 0.328);
        sets(1).sizeDist.rmax = rmax_model;
        sets(1).trend = 338.1;
        sets(1).plunge = 4.5;
        sets(1).kappa = 13.06;

        % ----- Set 2
        sets(2).name = 'Set_2';
        sets(2).P32 = 1.026;
        sets(2).sizeDist.type = 'powerlaw';
        sets(2).sizeDist.kr = 3.04;
        sets(2).sizeDist.r0 = 0.977;
        sets(2).sizeDist.rmin = max(rmin_explicit, 0.977);
        sets(2).sizeDist.rmax = rmax_model;
        sets(2).trend = 100.4;
        sets(2).plunge = 0.2;
        sets(2).kappa = 19.62;

        % ----- Set 3
        sets(3).name = 'Set_3';
        sets(3).P32 = 0.975;
        sets(3).sizeDist.type = 'powerlaw';
        sets(3).sizeDist.kr = 3.01;
        sets(3).sizeDist.r0 = 0.858;
        sets(3).sizeDist.rmin = max(rmin_explicit, 0.858);
        sets(3).sizeDist.rmax = rmax_model;
        sets(3).trend = 212.9;
        sets(3).plunge = 0.9;
        sets(3).kappa = 10.46;

        % ----- Set 4 (Exponential per footnote 1)
        sets(4).name = 'Set_4';
        sets(4).P32 = 2.320;
        sets(4).sizeDist.type = 'exponential';
        % The distribution is exponential with parameter lambda = 1/(mean). 
        % Thus r_0 in exponential maps directly to the mean.
        sets(4).sizeDist.r0 = 4.0; % mean = 4 (lambda = 1/4)
        sets(4).sizeDist.rmin = rmin_explicit;
        sets(4).sizeDist.rmax = rmax_model;
        sets(4).trend = 3.3;
        sets(4).plunge = 62.1;
        sets(4).kappa = 10.13;

        % ----- Set 5
        sets(5).name = 'Set_5';
        sets(5).P32 = 1.400;
        sets(5).sizeDist.type = 'powerlaw';
        sets(5).sizeDist.kr = 3.60;
        sets(5).sizeDist.r0 = 0.400;
        sets(5).sizeDist.rmin = max(rmin_explicit, 0.400);
        sets(5).sizeDist.rmax = rmax_model;
        sets(5).trend = 243.0;
        sets(5).plunge = 24.4;
        sets(5).kappa = 23.52;
end

%% -----------------------------
%  OPTIONS
%% -----------------------------
opts.centerMode = 'area_uniform';
opts.plot_all_dfn = false;              % 전체 도메인 시각화 On/Off
opts.plot_tunnel_intersect_dfn = true;  % 터널 교차부위 표출 On/Off
opts.plot_2d_trace_map = true;          % 2D Trace Map (X/Y/Z 슬라이스) On/Off
opts.export_for_python = true;         % Python 블록 탐지용 HDF5 내보내기 On/Off
opts.overlay_tunnel = true;            % 터널 오버레이 On/Off
opts.verbose = true;
opts.run_validation_suite = false; % <--- [검증 스크립트 ON/OFF 토글]

V = double(box.dx) * double(box.dy) * double(box.dz);

master = struct();
master.set_files = cell(numel(sets), 1);
master.set_names = cell(numel(sets), 1);
master.N_per_set = zeros(numel(sets), 1);
master.P32_input = zeros(numel(sets), 1);
master.P32_nominal = zeros(numel(sets), 1);
master.total_N = 0;
master.box = box;

%% =========================================================
%  GENERATE SET-BY-SET IN 25 m DOMAIN
%% =========================================================
for s = 1:numel(sets)
    seti = sets(s);
    
    % --- Apply P32 Scaling Filter to prevent Over-Spawn ---
    if strcmp(seti.sizeDist.type, 'powerlaw') && isfield(seti.sizeDist, 'r0')
        kr = seti.sizeDist.kr;
        rmax = seti.sizeDist.rmax;
        r0 = seti.sizeDist.r0;
        rmin = seti.sizeDist.rmin; % The cutoff generating limit
        
        pow = 2 - kr; % Integration of r^2 * r^{-(kr+1)} = r^{1-kr}
        if abs(pow) < 1e-12
            int_r0 = log(rmax) - log(r0);
            int_rmin = log(rmax) - log(rmin);
        else
            int_r0 = (rmax^pow - r0^pow) / pow;
            int_rmin = (rmax^pow - rmin^pow) / pow;
        end
        % Scale the target P32 down to the proportion valid for r >= rmin
        target_P32 = seti.P32 * (int_rmin / int_r0);
    elseif strcmp(seti.sizeDist.type, 'exponential')
        % For exponential, r0 represents the mean, and baseline P32 represents all fractures (r>=0).
        lambda = 1 / seti.sizeDist.r0;
        rmax = seti.sizeDist.rmax;
        rmin = seti.sizeDist.rmin; % 1.0m cutoff
        
        % Area integral factor: int r^2 * lambda * exp(-lambda*r)
        % Using the analytical definite integral substitution:
        int_func = @(r) -exp(-lambda*r) .* (r.^2 + 2*r/lambda + 2/lambda^2);
        
        int_r0 = int_func(rmax) - int_func(0);
        int_rmin = int_func(rmax) - int_func(rmin);
        
        target_P32 = seti.P32 * (int_rmin / int_r0);
    else
        target_P32 = seti.P32;
    end
    % ------------------------------------------------------

    Ntarget = compute_num_fractures_from_P32(target_P32, seti.sizeDist, V);

    if opts.verbose
        fprintf('\n[%s] target N = %d\n', seti.name, Ntarget);
    end

    mean_n = mean_pole_vector_from_trend_plunge(seti.trend, seti.plunge);
    
    setFile = fullfile(outDir, sprintf('dfn_set_%02d.mat', s));
    mFile = matfile(setFile, 'Writable', true);
    
    % Initialize empty arrays in the mat-file for continuous appending
    mFile.centers = single.empty(0,3);
    mFile.normals = single.empty(0,3);
    mFile.radius  = single.empty(0,1);
    mFile.set_id  = uint16.empty(0,1);

    chunkSize = 2000000; % 2 Million chunk memory blocks
    numChunks = ceil(Ntarget / chunkSize);
    currentIdx = 1;
    P32_nominal = 0;

    for ch = 1:numChunks
        nChunk = min(chunkSize, Ntarget - currentIdx + 1);
        
        radius_ch = single(sample_radius(seti.sizeDist, nChunk));
        normals_ch = single(sample_fisher_normals(mean_n, seti.kappa, nChunk));

        [strike_u, dip_u] = normal_to_strike_dip_basis_vectorized(normals_ch);
        centers_ch = sample_centers_from_surface_points_vectorized(box, radius_ch, strike_u, dip_u, opts.centerMode);

        set_id_ch = uint16(s * ones(nChunk, 1));
        
        idxRange = currentIdx : (currentIdx + nChunk - 1);
        mFile.centers(idxRange, 1:3) = centers_ch;
        mFile.normals(idxRange, 1:3) = normals_ch;
        mFile.radius(idxRange, 1) = radius_ch;
        mFile.set_id(idxRange, 1) = set_id_ch;
        
        P32_nominal = P32_nominal + sum(pi * double(radius_ch).^2) / V;
        currentIdx = currentIdx + nChunk;
    end

    metadata = struct();
    metadata.name = seti.name;
    metadata.trend = seti.trend;
    metadata.plunge = seti.plunge;
    metadata.kappa = seti.kappa;
    metadata.P32_input = seti.P32;
    metadata.P32_nominal = P32_nominal;
    metadata.N = Ntarget;
    metadata.sizeDist = seti.sizeDist;
    metadata.box = box;
    metadata.note = 'Discs are generated in 300 m box; clipping is applied later in crop box.';

    mFile.metadata = metadata;

    master.set_files{s} = setFile;
    master.set_names{s} = seti.name;
    master.N_per_set(s) = Ntarget;
    master.P32_input(s) = seti.P32;
    master.P32_nominal(s) = P32_nominal;
    master.total_N = master.total_N + Ntarget;

    if opts.verbose
        fprintf('[%s] saved -> %s\n', seti.name, setFile);
        fprintf('[%s] nominal P32 = %.4f\n', seti.name, P32_nominal);
    end
end

save(masterFile, 'master', '-v7.3');

fprintf('\nDone. Total fractures = %d\n', master.total_N);

if opts.run_validation_suite
    fprintf('\nRunning Validation Suite...\n');
    validate_dfn_generation(masterFile, sets);
end

%% -----------------------------
%  Define central crop box centered at origin
%% -----------------------------
cx = 50;
cy = 50;
cz = 50;
cropBox.xmin = -cx / 2;
cropBox.xmax =  cx / 2;
cropBox.ymin = -cy / 2;
cropBox.ymax =  cy / 2;
cropBox.zmin = -cz / 2;
cropBox.zmax =  cz / 2;

%% -----------------------------
%  Load Tunnel Polygon First (for Filtering & Overlay)
%% -----------------------------
tunnel_poly_YZ = [];
tunnel_Y = [];
tunnel_Z = [];

tunnel_file = fullfile(currentDir, '..', '..', '..', '단면_폴리곤.dat');
if exist(tunnel_file, 'file')
    poly = read_tunnel_polygon(tunnel_file);
    poly_shifted = poly;
    poly_shifted(:,2) = poly_shifted(:,2) - 4.0; % Z position sync
    poly_shifted(end+1, :) = poly_shifted(1, :); % Close Polygon
    
    tunnel_Y = poly_shifted(:,1);
    tunnel_Z = poly_shifted(:,2);
    tunnel_poly_YZ = [tunnel_Y, tunnel_Z];
else
    disp('WARNING: [단면_폴리곤.dat] file not found in root directory! Tunnel overlay skipped.');
end

%% -----------------------------
%  Plot clipped DFN (Options Based)
%% -----------------------------
if opts.plot_all_dfn
    fig_dfn_3d_all = plot_clipped_dfn_crop(masterFile, cropBox);
    if opts.overlay_tunnel && ~isempty(tunnel_Y)
        overlay_tunnel_3d(fig_dfn_3d_all, cropBox, tunnel_Y, tunnel_Z);
    end
end

if opts.plot_tunnel_intersect_dfn && ~isempty(tunnel_poly_YZ)
    fig_dfn_3d_tunnel = plot_clipped_dfn_crop(masterFile, cropBox, tunnel_poly_YZ);
    if opts.overlay_tunnel && ~isempty(tunnel_Y)
        overlay_tunnel_3d(fig_dfn_3d_tunnel, cropBox, tunnel_Y, tunnel_Z);
    end
end

%% -----------------------------
%  Export Data for Python GPU Block Detection
%% -----------------------------
if opts.export_for_python
    export_dfn_for_python(masterFile, tunnel_poly_YZ, tunnel_Y, tunnel_Z, cropBox);
end

%% -----------------------------
%  2D Trace Maps (X=0, Y=0, Z=0 슬라이스)
%% -----------------------------
if opts.plot_2d_trace_map
    % --- X=0 단면 (YZ 평면: 터널 단면과 동일 방향) ---
    plot_2d_slice_cropbox(masterFile, cropBox, 'x', 0);
    if opts.overlay_tunnel && ~isempty(tunnel_poly_YZ)
        hold on;
        plot(tunnel_poly_YZ(:,1), tunnel_poly_YZ(:,2), ...
             'r-', 'LineWidth', 2.0, 'DisplayName', 'Tunnel');
        legend('show', 'Location', 'best');
    end

    % --- Y=0 단면 (XZ 평면) ---
    plot_2d_slice_cropbox(masterFile, cropBox, 'y', 0);

    % --- Z=0 단면 (XY 평면) ---
    plot_2d_slice_cropbox(masterFile, cropBox, 'z', 0);
end



%% -----------------------------
%  Helper Inline Functions
%% -----------------------------
function overlay_tunnel_3d(fig_handle, cropBox, tunnel_Y, tunnel_Z)
    figure(fig_handle); hold on;
    N = length(tunnel_Y);
    surf_X = repmat([cropBox.xmin; cropBox.xmax], 1, N);
    surf_Y = repmat(tunnel_Y', 2, 1);
    surf_Z = repmat(tunnel_Z', 2, 1);
    
    surf(surf_X, surf_Y, surf_Z, 'FaceColor', 'r', 'FaceAlpha', 0.15, 'EdgeColor', 'r', 'LineWidth', 0.5);
    patch(repmat(cropBox.xmin, N, 1), tunnel_Y, tunnel_Z, 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'r', 'LineWidth', 2.0);
    patch(repmat(cropBox.xmax, N, 1), tunnel_Y, tunnel_Z, 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'r', 'LineWidth', 2.0);
    %disp('Successfully overlaid Solid 3D boundary mesh of structural tunnel.');
end