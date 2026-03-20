clc; clear; close all;

% 현재 파일 위치를 기준으로 상위 폴더(src) 및 하위 폴더들을 MATLAB 경로에 추가
currentDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(currentDir, '..')));

%% =========================================================
% DFN GENERATED IN 300x300x300 m DOMAIN
% THEN VISUALIZED IN CENTRAL 20x20x20 m CROPPED CUBE
%% =========================================================

outDir = 'dfn_output_cube300m';
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

masterFile = fullfile(outDir, 'dfn_master_index.mat');

%% -----------------------------
%  DFN Domain (generation domain)
%% -----------------------------
box.dx = single(500);
box.dy = single(500);
box.dz = single(500);
box.x0 = -box.dx / 2;
box.y0 = -box.dy / 2;
box.z0 = -box.dz / 2;

rng('shuffle');

%% -----------------------------
%  DFN SETS
%% -----------------------------
sets = [];

rmin_explicit = 0.4;
rmax_model = 250.0;

% ----- Set 1
sets(1).name = 'Set_1';
sets(1).trend = 87.2;
sets(1).plunge = 1.7;
sets(1).kappa = 21.66;
sets(1).P32 = 0.602;
sets(1).sizeDist.type = 'powerlaw';
sets(1).sizeDist.kr = 2.88;
sets(1).sizeDist.r0 = 0.28;
sets(1).sizeDist.rmin = max(rmin_explicit, 0.28);
sets(1).sizeDist.rmax = rmax_model;

% ----- Set 2
sets(2).name = 'Set_2';
sets(2).trend = 135.2;
sets(2).plunge = 2.7;
sets(2).kappa = 21.54;
sets(2).P32 = 2.069;
sets(2).sizeDist.type = 'powerlaw';
sets(2).sizeDist.kr = 3.02;
sets(2).sizeDist.r0 = 0.25;
sets(2).sizeDist.rmin = max(rmin_explicit, 0.25);
sets(2).sizeDist.rmax = rmax_model;

% ----- Set 3
sets(3).name = 'Set_3';
sets(3).trend = 40.6;
sets(3).plunge = 2.2;
sets(3).kappa = 23.90;
sets(3).P32 = 0.448;
sets(3).sizeDist.type = 'powerlaw';
sets(3).sizeDist.kr = 2.81;
sets(3).sizeDist.r0 = 0.14;
sets(3).sizeDist.rmin = rmin_explicit;
sets(3).sizeDist.rmax = rmax_model;

% ----- Set 4
sets(4).name = 'Set_4';
sets(4).trend = 190.4;
sets(4).plunge = 0.7;
sets(4).kappa = 30.63;
sets(4).P32 = 0.226;
sets(4).sizeDist.type = 'powerlaw';
sets(4).sizeDist.kr = 2.95;
sets(4).sizeDist.r0 = 0.15;
sets(4).sizeDist.rmin = rmin_explicit;
sets(4).sizeDist.rmax = rmax_model;

% ----- Set 5
sets(5).name = 'Set_5';
sets(5).trend = 342.9;
sets(5).plunge = 80.3;
sets(5).kappa = 8.18;
sets(5).P32 = 0.605;
sets(5).sizeDist.type = 'powerlaw';
sets(5).sizeDist.kr = 2.92;
sets(5).sizeDist.r0 = 0.25;
sets(5).sizeDist.rmin = max(rmin_explicit, 0.25);
sets(5).sizeDist.rmax = rmax_model;

%% -----------------------------
%  OPTIONS
%% -----------------------------
opts.centerMode = 'area_uniform';
opts.verbose = true;

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
    Ntarget = compute_num_fractures_from_P32(seti.P32, seti.sizeDist, V);

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

%% -----------------------------
%  Define central crop box centered at origin
%% -----------------------------
cx = 30;
cy = 30;
cz = 30;
cropBox.xmin = -cx / 2;
cropBox.xmax =  cx / 2;
cropBox.ymin = -cy / 2;
cropBox.ymax =  cy / 2;
cropBox.zmin = -cz / 2;
cropBox.zmax =  cz / 2;

%% -----------------------------
%  Plot clipped DFN in crop box
%% -----------------------------
plot_clipped_dfn_crop(masterFile, cropBox);

%% -----------------------------
%  Plot 2D Trace Slice (YZ, XZ, XY)
%% -----------------------------
% Example: Take a slice exactly at the center of the Y-axis (y = 0.0)
plot_2d_slice_cropbox(masterFile, cropBox, 'y', 0.0);