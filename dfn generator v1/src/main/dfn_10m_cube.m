clc; clear; close all;

%% =========================================================
%  DFN GENERATED IN 25x25x25 m DOMAIN
%  THEN VISUALIZED IN CENTRAL 10x10x10 m CROPPED CUBE
%% =========================================================

outDir = 'dfn_output_cube25m';
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

masterFile = fullfile(outDir, 'dfn_master_index.mat');

%% -----------------------------
%  DFN Domain (generation domain)
%% -----------------------------
box.dx = single(25);
box.dy = single(25);
box.dz = single(25);

rng(1);

%% -----------------------------
%  DFN SETS
%% -----------------------------
sets = [];

rmin_explicit = 0.40;
rmax_model    = 8.0;

% ----- Set 1
sets(1).name   = 'Set_1';
sets(1).trend  = 87.2;
sets(1).plunge = 1.7;
sets(1).kappa  = 21.66;
sets(1).P32    = 0.602;
sets(1).sizeDist.type = 'powerlaw';
sets(1).sizeDist.kr   = 2.88;
sets(1).sizeDist.r0   = 0.28;
sets(1).sizeDist.rmin = max(rmin_explicit, 0.28);
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
sets(3).sizeDist.rmin = rmin_explicit;
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
sets(4).sizeDist.rmin = rmin_explicit;
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
opts.centerMode = 'area_uniform';
opts.verbose = true;

V = double(box.dx) * double(box.dy) * double(box.dz);

master = struct();
master.set_files   = cell(numel(sets),1);
master.set_names   = cell(numel(sets),1);
master.N_per_set   = zeros(numel(sets),1);
master.P32_input   = zeros(numel(sets),1);
master.P32_nominal = zeros(numel(sets),1);
master.total_N     = 0;
master.box         = box;

%% =========================================================
%  GENERATE SET-BY-SET IN 25 m DOMAIN
%% =========================================================
for s = 1:numel(sets)

    seti = sets(s);

    Ntarget = compute_num_fractures_from_P32(seti.P32, seti.sizeDist, V);

    if opts.verbose
        fprintf('\n[%s] target N = %d\n', seti.name, Ntarget);
    end

    radius  = single(sample_radius(seti.sizeDist, Ntarget));
    mean_n  = mean_pole_vector_from_trend_plunge(seti.trend, seti.plunge);
    normals = single(sample_fisher_normals(mean_n, seti.kappa, Ntarget));

    [strike_u, dip_u] = normal_to_strike_dip_basis_vectorized(normals);
    centers = sample_centers_from_surface_points_vectorized( ...
        box, radius, strike_u, dip_u, opts.centerMode);

    set_id = uint16(s * ones(Ntarget,1));

    area_total = sum(pi * double(radius).^2);
    P32_nominal = area_total / V;

    metadata = struct();
    metadata.name         = seti.name;
    metadata.trend        = seti.trend;
    metadata.plunge       = seti.plunge;
    metadata.kappa        = seti.kappa;
    metadata.P32_input    = seti.P32;
    metadata.P32_nominal  = P32_nominal;
    metadata.N            = Ntarget;
    metadata.sizeDist     = seti.sizeDist;
    metadata.box          = box;
    metadata.note         = 'Discs are generated in 25 m box; clipping is applied later in crop box.';

    setFile = fullfile(outDir, sprintf('dfn_set_%02d.mat', s));
    save(setFile, 'centers', 'normals', 'radius', 'set_id', 'metadata', '-v7.3');

    master.set_files{s}   = setFile;
    master.set_names{s}   = seti.name;
    master.N_per_set(s)   = Ntarget;
    master.P32_input(s)   = seti.P32;
    master.P32_nominal(s) = P32_nominal;
    master.total_N        = master.total_N + Ntarget;

    if opts.verbose
        fprintf('[%s] saved -> %s\n', seti.name, setFile);
        fprintf('[%s] nominal P32 = %.4f\n', seti.name, P32_nominal);
    end

    clear centers normals radius set_id metadata strike_u dip_u
end

save(masterFile, 'master', '-v7.3');

fprintf('\nDone. Total fractures = %d\n', master.total_N);

%% -----------------------------
%  Define central 10x10x10 crop box
%% -----------------------------
cropBox.xmin = 7.5;
cropBox.xmax = 17.5;
cropBox.ymin = 7.5;
cropBox.ymax = 17.5;
cropBox.zmin = 7.5;
cropBox.zmax = 17.5;

tunnel.centerY = 0.0;
tunnel.floorZ  = -3.0;
tunnel.radius  = 2.0;
tunnel.wallH   = 2.0;
tunnel.xmin    = -5.0;
tunnel.xmax    =  5.0;


%% -----------------------------
%  Plot clipped DFN in crop box
%% -----------------------------
plot_clipped_dfn_crop(masterFile, cropBox);
plot_excavation_face_traces(masterFile, cropBox, -4.0, tunnel);