clc; clear; close all;

% 현재 파일 위치를 기준으로 상위 폴더(src) 및 하위 폴더들을 MATLAB 경로에 추가
currentDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(currentDir, '..')));

%% =========================================================
%  GENERAL 3D ROCK MASS DFN GENERATOR
%  - not FPI-specific
%  - no tunnel-radius censoring
%  - matrix-based storage
%  - single precision
%  - save each set immediately
%% =========================================================

%% -----------------------------
%  OUTPUT
%% -----------------------------
outDir = 'dfn_output_general';
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

masterFile = fullfile(outDir, 'dfn_master_index.mat');

%% -----------------------------
%  MODEL DOMAIN (rock mass box)
%  User-defined representative rock mass volume
%% -----------------------------
box.x0 = single(0);
box.y0 = single(0);
box.z0 = single(0);
box.dx = single(100);
box.dy = single(100);
box.dz = single(100);

rng(1);   % reproducibility

%% -----------------------------
%  DFN SET DEFINITIONS
%  Forsmark statistics from R-06-54 Table 2-2
%  Here, rmin is reinterpreted as explicit DFN resolution,
%  not tunnel radius.
%% -----------------------------
sets = [];

% Example explicit DFN lower cutoff
rmin_explicit = 0.28;    % [m]
rmax_model    = 250.0;   % [m]

% ----- Set 1
sets(1).name   = 'Set_1';
sets(1).trend  = 87.2;
sets(1).plunge = 1.7;
sets(1).kappa  = 21.66;
sets(1).P32    = 0.602;
sets(1).sizeDist.type = 'powerlaw';
sets(1).sizeDist.kr   = 2.88;
sets(1).sizeDist.r0   = 0.28;    % metadata / reference scale
sets(1).sizeDist.rmin = rmin_explicit;
sets(1).sizeDist.rmax = rmax_model;

% ----- Set 2
sets(2).name   = 'Set_2';
sets(2).trend  = 135.2;
sets(2).plunge = 2.7;
sets(2).kappa  = 21.54;
sets(2).P32    = 2.069;
sets(2).sizeDist.type = 'powerlaw';
sets(2).sizeDist.kr   = 3.02;
sets(2).sizeDist.r0   = 0.25;
sets(2).sizeDist.rmin = max(rmin_explicit, 0.25);
sets(2).sizeDist.rmax = rmax_model;

% ----- Set 3
sets(3).name   = 'Set_3';
sets(3).trend  = 40.6;
sets(3).plunge = 2.2;
sets(3).kappa  = 23.90;
sets(3).P32    = 0.448;
sets(3).sizeDist.type = 'powerlaw';
sets(3).sizeDist.kr   = 2.81;
sets(3).sizeDist.r0   = 0.14;
sets(3).sizeDist.rmin = max(rmin_explicit, 0.14);
sets(3).sizeDist.rmax = rmax_model;

% ----- Set 4
sets(4).name   = 'Set_4';
sets(4).trend  = 190.4;
sets(4).plunge = 0.7;
sets(4).kappa  = 30.63;
sets(4).P32    = 0.226;
sets(4).sizeDist.type = 'powerlaw';
sets(4).sizeDist.kr   = 2.95;
sets(4).sizeDist.r0   = 0.15;
sets(4).sizeDist.rmin = max(rmin_explicit, 0.15);
sets(4).sizeDist.rmax = rmax_model;

% ----- Set 5
sets(5).name   = 'Set_5';
sets(5).trend  = 342.9;
sets(5).plunge = 80.3;
sets(5).kappa  = 8.18;
sets(5).P32    = 0.605;
sets(5).sizeDist.type = 'powerlaw';
sets(5).sizeDist.kr   = 2.92;
sets(5).sizeDist.r0   = 0.25;
sets(5).sizeDist.rmin = max(rmin_explicit, 0.25);
sets(5).sizeDist.rmax = rmax_model;

%% -----------------------------
%  OPTIONS
%% -----------------------------
opts.centerMode = 'area_uniform';   % 'report' or 'area_uniform'
opts.makeQuickPlot = true;
opts.maxPlotTotal = 200;
opts.verbose = true;

%% -----------------------------
%  BASIC INFO
%% -----------------------------
V = double(box.dx) * double(box.dy) * double(box.dz);

if opts.verbose
    fprintf('Output folder: %s\n', outDir);
    fprintf('Model box: %.2f x %.2f x %.2f m\n', double(box.dx), double(box.dy), double(box.dz));
    fprintf('Volume   : %.4e m^3\n', V);
    fprintf('Interpretation: general rock-mass DFN, not tunnel/FPI DFN.\n');
end

%% -----------------------------
%  MASTER INDEX
%% -----------------------------
master = struct();
master.set_files   = cell(numel(sets),1);
master.set_names   = cell(numel(sets),1);
master.N_per_set   = zeros(numel(sets),1);
master.P32_input   = zeros(numel(sets),1);
master.P32_nominal = zeros(numel(sets),1);
master.total_N     = 0;
master.box         = box;
master.centerMode  = opts.centerMode;
master.notes       = ['General DFN generator based on R-06-54 DFN core assumptions ', ...
                      '(Poisson position, independent size-position-orientation, Fisher orientation), ', ...
                      'with FPI/tunnel-specific truncation removed.'];

%% -----------------------------
%  FOR QUICK PLOT
%% -----------------------------
plotPack.center   = zeros(0,3);
plotPack.radius   = zeros(0,1);
plotPack.normals  = zeros(0,3);

%% =========================================================
%  GENERATE SET-BY-SET
%% =========================================================
for s = 1:numel(sets)

    seti = sets(s);

    % fracture count from truncated P32
    N = compute_num_fractures_from_P32(seti.P32, seti.sizeDist, V);

    if opts.verbose
        fprintf('\n[%s] N = %d\n', seti.name, N);
    end

    if N <= 0
        warning('Set %s yielded N <= 0. Skipped.', seti.name);
        continue;
    end

    % sample radius
    mean_n  = mean_pole_vector_from_trend_plunge(seti.trend, seti.plunge);
    
    setFile = fullfile(outDir, sprintf('dfn_set_%02d.mat', s));
    mFile = matfile(setFile, 'Writable', true);

    % Initialize empty arrays in the mat-file for continuous appending
    mFile.centers = single.empty(0,3);
    mFile.normals = single.empty(0,3);
    mFile.radius  = single.empty(0,1);
    mFile.set_id  = uint16.empty(0,1);

    chunkSize = 2000000; % 2 Million chunk memory blocks
    numChunks = ceil(N / chunkSize);
    currentIdx = 1;
    P32_nominal = 0;

    for ch = 1:numChunks
        nChunk = min(chunkSize, N - currentIdx + 1);
        
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
        
        % For quick plotting, append some samples from each chunk
        if opts.makeQuickPlot
            nTake = min(max(2, floor(round(opts.maxPlotTotal/numel(sets))/numChunks)), nChunk);
            idxTest = randperm(nChunk, nTake);
            plotPack.center  = [plotPack.center;  double(centers_ch(idxTest,:))];
            plotPack.radius  = [plotPack.radius;  double(radius_ch(idxTest))];
            plotPack.normals = [plotPack.normals; double(normals_ch(idxTest,:))];
        end
        
        currentIdx = currentIdx + nChunk;
    end

    metadata = struct();
    metadata.name         = seti.name;
    metadata.trend        = seti.trend;
    metadata.plunge       = seti.plunge;
    metadata.kappa        = seti.kappa;
    metadata.P32_input    = seti.P32;
    metadata.P32_nominal  = P32_nominal;
    metadata.N            = N;
    metadata.sizeDist     = seti.sizeDist;
    metadata.centerMode   = opts.centerMode;
    metadata.box          = box;
    metadata.generator    = 'main_generate_dfn_general';
    metadata.storageType  = 'matrix_single_setwise_chunked_appended';

    mFile.metadata = metadata;

    master.set_files{s}   = setFile;
    master.set_names{s}   = seti.name;
    master.N_per_set(s)   = N;
    master.P32_input(s)   = seti.P32;
    master.P32_nominal(s) = P32_nominal;
    master.total_N        = master.total_N + N;

    if opts.verbose
        fprintf('[%s] saved -> %s\n', seti.name, setFile);
        fprintf('[%s] nominal realized P32 = %.4f m^2/m^3\n', seti.name, P32_nominal);
    end
end

%% -----------------------------
%  SAVE MASTER INDEX
%% -----------------------------
save(masterFile, 'master', '-v7.3');

fprintf('\n=========================================\n');
fprintf('General DFN generation finished.\n');
fprintf('Total fractures: %d\n', master.total_N);
fprintf('Master index: %s\n', masterFile);
fprintf('=========================================\n');

disp(table((1:numel(sets))', master.N_per_set, master.P32_input, master.P32_nominal, ...
    'VariableNames', {'SetID','N','P32_input','P32_nominal'}));

%% -----------------------------
%  QUICK PREVIEW
%% -----------------------------
if opts.makeQuickPlot && ~isempty(plotPack.center)
    figure('Color','w');
    hold on; axis equal; grid on;
    xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
    title('General 3D DFN preview');

    % draw box
    plot3([box.x0 box.x0+double(box.dx) box.x0+double(box.dx) box.x0 box.x0], ...
          [box.y0 box.y0 box.y0+double(box.dy) box.y0+double(box.dy) box.y0], ...
          [box.z0 box.z0 box.z0 box.z0 box.z0], 'k-');
    plot3([box.x0 box.x0+double(box.dx) box.x0+double(box.dx) box.x0 box.x0], ...
          [box.y0 box.y0 box.y0+double(box.dy) box.y0+double(box.dy) box.y0], ...
          [box.z0 box.z0 box.z0 box.z0 box.z0] + double(box.dz), 'k-');
    for x = [box.x0 box.x0+double(box.dx)]
        for y = [box.y0 box.y0+double(box.dy)]
            plot3([x x], [y y], [box.z0 box.z0+double(box.dz)], 'k-');
        end
    end

    plot_dfn_preview_from_center_normal(plotPack.center, plotPack.normals, plotPack.radius);
    view(3);
end