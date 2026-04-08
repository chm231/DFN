function export_dfn_for_python(masterFile, tunnel_poly_YZ, tunnel_Y, tunnel_Z, cropBox)
% EXPORT_DFN_FOR_PYTHON - DFN 및 터널 데이터를 Python GPU 파이프라인용 HDF5로 내보냄
%
% 사용법:
%   export_dfn_for_python(masterFile, tunnel_poly_YZ, tunnel_Y, tunnel_Z)
%   export_dfn_for_python(masterFile, tunnel_poly_YZ, tunnel_Y, tunnel_Z, cropBox)
%
% 출력 파일: <masterFile 폴더>/dfn_export_for_python.h5
%
% HDF5 Dataset 구조:
%   /fractures/centers   [N x 3] float32 - 균열 중심점 (m)
%   /fractures/normals   [N x 3] float32 - 균열 법선벡터 (unit)
%   /fractures/radii     [N x 1] float32 - 균열 반경 (m)
%   /fractures/set_id    [N x 1] uint16  - 균열 세트 번호 (1~5)
%   /tunnel/poly_YZ      [M x 2] float32 - 터널 단면 폴리곤 (Y, Z)
%   /tunnel/profile_Y    [K x 1] float32 - 터널 외곽 Y 좌표
%   /tunnel/profile_Z    [K x 1] float32 - 터널 외곽 Z 좌표
%   /meta/domain_box     [1 x 6] float32 - 전체 생성 도메인 [xmin xmax ymin ymax zmin zmax]
%   /meta/crop_box       [1 x 6] float32 - 분석 crop 박스 [xmin xmax ymin ymax zmin zmax]

    if nargin < 5 || isempty(cropBox)
        cropBox = struct();
        cropBox.xmin = []; cropBox.xmax = [];
        cropBox.ymin = []; cropBox.ymax = [];
        cropBox.zmin = []; cropBox.zmax = [];
    end

    M = load(masterFile);
    master = M.master;

    % 최종 출력 경로 (OneDrive)
    [master_dir, ~, ~] = fileparts(masterFile);
    export_path = fullfile(master_dir, 'dfn_export_for_python.h5');

    % ── OneDrive 동기화 잠금 방지: 로컬 temp에 먼저 쓰기 ──────────────
    tmp_path = fullfile(tempdir(), 'dfn_export_for_python_tmp.h5');
    if exist(tmp_path, 'file'), delete(tmp_path); end

    fprintf('\n🐍 Exporting DFN data for Python GPU pipeline...\n');

    % 균열 데이터 통합
    all_c = []; all_n = []; all_r = []; all_sid = [];
    for k = 1:numel(master.set_files)
        S = load(master.set_files{k});
        all_c   = [all_c;   double(S.centers)];
        all_n   = [all_n;   double(S.normals)];
        all_r   = [all_r;   double(S.radius)];
        all_sid = [all_sid; uint16(S.set_id)];
    end
    N = length(all_r);
    fprintf('  - %d fractures gathered.\n', N);

    % ── HDF5 Write (tmp 경로) ──────────────────────────────────────────
    % 균열
    h5create(tmp_path, '/fractures/centers', [N 3], 'Datatype', 'single');
    h5create(tmp_path, '/fractures/normals', [N 3], 'Datatype', 'single');
    h5create(tmp_path, '/fractures/radii',   [N 1], 'Datatype', 'single');
    h5create(tmp_path, '/fractures/set_id',  [N 1], 'Datatype', 'uint16');

    h5write(tmp_path, '/fractures/centers', single(all_c));
    h5write(tmp_path, '/fractures/normals', single(all_n));
    h5write(tmp_path, '/fractures/radii',   single(all_r));
    h5write(tmp_path, '/fractures/set_id',  all_sid);

    % 터널 데이터
    if ~isempty(tunnel_poly_YZ)
        M_poly = size(tunnel_poly_YZ, 1);
        h5create(tmp_path, '/tunnel/poly_YZ', [M_poly 2], 'Datatype', 'single');
        h5write(tmp_path, '/tunnel/poly_YZ', single(tunnel_poly_YZ));
    end
    if ~isempty(tunnel_Y)
        K = length(tunnel_Y);
        h5create(tmp_path, '/tunnel/profile_Y', [K 1], 'Datatype', 'single');
        h5create(tmp_path, '/tunnel/profile_Z', [K 1], 'Datatype', 'single');
        h5write(tmp_path, '/tunnel/profile_Y', single(tunnel_Y(:)));
        h5write(tmp_path, '/tunnel/profile_Z', single(tunnel_Z(:)));
    end

    % 메타 정보 – 전체 도메인 박스
    xmin_d = min(all_c(:,1) - all_r) - 5;
    xmax_d = max(all_c(:,1) + all_r) + 5;
    domain_box = single([xmin_d, xmax_d, ...
                          min(tunnel_poly_YZ(:,1))-15, max(tunnel_poly_YZ(:,1))+15, ...
                          min(tunnel_poly_YZ(:,2))-15, max(tunnel_poly_YZ(:,2))+15]);
    h5create(tmp_path, '/meta/domain_box', [1 6], 'Datatype', 'single');
    h5write(tmp_path, '/meta/domain_box', domain_box);

    % crop_box – Python 분석 도메인
    if ~isempty(cropBox.xmin)
        crop_box_vec = single([cropBox.xmin, cropBox.xmax, ...
                               cropBox.ymin, cropBox.ymax, ...
                               cropBox.zmin, cropBox.zmax]);
    else
        crop_box_vec = domain_box;
    end
    h5create(tmp_path, '/meta/crop_box', [1 6], 'Datatype', 'single');
    h5write(tmp_path, '/meta/crop_box', crop_box_vec);

    % 속성 쓰기 (tmp 파일에 – OneDrive 잠금 없음)
    h5writeatt(tmp_path, '/', 'created_by', 'dfn_10m_cube.m');
    h5writeatt(tmp_path, '/', 'matlab_version', version);
    h5writeatt(tmp_path, '/', 'num_fractures', N);

    % ── 완성된 파일을 OneDrive 최종 경로로 이동 ──────────────────────
    if exist(export_path, 'file'), delete(export_path); end
    movefile(tmp_path, export_path);

    fprintf('  ✅ Exported to: %s\n', export_path);
    fprintf('  - Run Python pipeline:\n');
    fprintf('      cd "%s"\n', fullfile(fileparts(masterFile), '..', '..', '..', 'python'));
    fprintf('      python detect_blocks_gpu.py --input "%s"\n\n', export_path);
end
