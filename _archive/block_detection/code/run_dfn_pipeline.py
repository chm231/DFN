r"""
run_dfn_pipeline.py  –  GPU 가속 3D 블록 탐지 파이프라인 (메인 스크립트)

사용법:
    python run_dfn_pipeline.py --input <dfn_export_for_python.h5>
                                [--voxel_size 0.5]
                                [--halo 6.0]          # crop_box 크기 제한 (미사용 시 전체)
                                [--tol_factor 0.6]    # 균열 슬랩 두께 = tol_factor × voxel_size
                                [--min_voxels 8]
                                [--connectivity 6 or 26]
                                [--outdir ../storage/output/results]
                                [--no_gpu]

    & "C:\Users\user\miniconda3\python.exe" "c:\Users\user\OneDrive\2026-1\3D DFN modeling\dfn_analysis\run_dfn_pipeline.py" --input "c:\Users\user\OneDrive\2026-1\3D DFN modeling\storage\data\dfn_export_for_python.h5" --voxel_size 0.5 --tol_factor 0.6

알고리즘:
    Step 1 – 3D Voxel 분류  (ROCK / FRACTURE / TUNNEL)  → GPU 가속
    Step 2 – 26-conn CCA     (ROCK 영역만)               → GPU/CPU
    Step 3 – 블록 필터링     (터널 접촉 ∩ 경계 미접촉)  → CPU
    Step 4 – 결과 저장 및 시각화
"""

from __future__ import annotations
import sys, argparse, time, json, os
import numpy as np
import h5py

# ── CUDA 경로 자동 설정 (nvidia-cuda-nvrtc-cu12 pip 패키지) ───────────────
def _setup_cuda_path():
    import site
    for sp in site.getsitepackages():
        bin_dir = os.path.join(sp, 'nvidia', 'cuda_nvrtc', 'bin')
        if os.path.isdir(bin_dir):
            root = os.path.join(sp, 'nvidia', 'cuda_nvrtc')
            if not os.environ.get('CUDA_PATH'):
                os.environ['CUDA_PATH'] = root
            os.environ['PATH'] = bin_dir + os.pathsep + os.environ.get('PATH', '')
            return True
    return False
_setup_cuda_path()

# ── 로컬 모듈 ─────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from tunnel_geometry import build_voxel_masks
from block_detector  import (classify_voxels, run_cca,
                              filter_and_stat_blocks, TUNNEL)
from visualize_blocks import (plot_block_3d_pyvista_interactive, plot_block_overview, 
                              plot_block_3d_scatter, plot_block_with_bounding_fractures,
                              plot_all_blocks_with_fractures)
from export_blocks import export_blocks_csv, export_interfaces_csv


# ════════════════════════════════════════════════════════════════════════════
def load_hdf5(path: str) -> dict:
    """HDF5 로드 – MATLAB 전치(transpose) 자동 보정."""
    print(f"\n[Info] HDF5 로드: {path}")
    data = {}
    with h5py.File(path, 'r') as f:
        raw_c = f['/fractures/centers'][:]
        raw_n = f['/fractures/normals'][:]
        data['centers'] = (raw_c.T if raw_c.shape[0] == 3
                           and raw_c.shape[0] < raw_c.shape[1] else raw_c)
        data['normals'] = (raw_n.T if raw_n.shape[0] == 3
                           and raw_n.shape[0] < raw_n.shape[1] else raw_n)
        data['radii']   = f['/fractures/radii'][:].ravel()
        data['set_id']  = (f['/fractures/set_id'][:].ravel()
                           if '/fractures/set_id' in f
                           else np.ones(len(data['radii']), dtype=np.uint16))

        if '/tunnel/poly_YZ' in f:
            raw_p = f['/tunnel/poly_YZ'][:]
            data['poly_YZ'] = (raw_p.T if raw_p.shape[0] == 2
                               and raw_p.shape[0] < raw_p.shape[1] else raw_p)
        if '/tunnel/profile_Y' in f:
            data['profile_Y'] = f['/tunnel/profile_Y'][:].ravel()
            data['profile_Z'] = f['/tunnel/profile_Z'][:].ravel()

        data['domain_box'] = f['/meta/domain_box'][:].ravel()
        data['crop_box']   = (f['/meta/crop_box'][:].ravel()
                              if '/meta/crop_box' in f
                              else data['domain_box'].copy())
        if '/meta/x_start' in f:
            data['x_start'] = float(f['/meta/x_start'][()])
        if '/meta/x_end' in f:
            data['x_end'] = float(f['/meta/x_end'][()])

    N = len(data['radii'])
    print(f"  균열 수  : {N:,}")
    print(f"  도메인   : {data['domain_box']}")
    if 'poly_YZ' in data:
        print(f"  터널 폴리곤: {len(data['poly_YZ'])} 점")
    return data


# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='GPU 가속 3D 블록 탐지')
    parser.add_argument('--input',       required=True)
    parser.add_argument('--voxel_size',  type=float, default=0.5,   help='복셀 크기 (m)')
    parser.add_argument('--halo',        type=float, default=10.0,   help='터널 주변 해석 반경(Halo) 두께 (m). 미지정시 전체 도메인.')
    parser.add_argument('--tol_factor',  type=float, default=0.5,   help='균열 슬랩 두께 계수')
    parser.add_argument('--min_voxels',  type=int,   default=8,     help='최소 블록 복셀 수')
    parser.add_argument('--connectivity',type=int,   default=6,     help='CCA Connectivity (6 or 26)')
    parser.add_argument('--outdir',      default='storage/output/results')
    parser.add_argument('--no_gpu',      action='store_true')
    
    # ── 추가: 시각화 제어 인자 ────────────────────────────────
    parser.add_argument('--target_block', type=int,   default=None,  help='상세 분석할 특정 블록 ID (기본값: 모든 블록 순회 분석)')
    parser.add_argument('--shell_thickness', type=int, default=2,    help='경계 균열 탐색용 쉘 두께 (voxel)')
    parser.add_argument('--min_contact',   type=int,   default=10,   help='최소 접촉 복셀 수 (이하 제외)')
    parser.add_argument('--max_auto_viz', type=int, default=1000, help='자동 상세 시각화 최대 개수')
    parser.add_argument('--show_fractures', action='store_true', help='상세 시각화 시 경계 균열 패치 표시')
    parser.add_argument('--tunnel_x_start', type=float, default=None, help='3D 시각화 시 터널 시작 위치 (기본값: HDF5 x_start 또는 xs[0])')
    parser.add_argument('--tunnel_end_offset', type=float, default=None, help='막장면 뒤 연장할 추가 터널 튜브 길이 (n 미터)')
    
    args = parser.parse_args()

    if args.no_gpu:
        import block_detector as bd; bd.HAS_GPU = False
        import tunnel_geometry as tg; tg.HAS_GPU = False
        print("[!] GPU 비활성화")

    # 출력 폴더 생성
    os.makedirs(args.outdir, exist_ok=True)
    print(f"[Info] 결과 저장: {os.path.abspath(args.outdir)}")

    t0 = time.time()

    # ── 1. 데이터 로드 ───────────────────────────────────────────────────
    data    = load_hdf5(args.input)
    centers = data['centers'].astype(np.float32)
    normals = data['normals'].astype(np.float32)
    radii   = data['radii'].astype(np.float32)

    poly_YZ = data.get('poly_YZ', None)
    if poly_YZ is None:
        print("[ERROR] /tunnel/poly_YZ 없음"); sys.exit(1)
    poly_Y = poly_YZ[:, 0]
    poly_Z = poly_YZ[:, 1]

    domain_box = data['crop_box']
    print(f"\n분석 도메인: {domain_box}")

    # ── 2. 터널 마스크 생성 ──────────────────────────────────────────────
    print(f"\n[Step 1/4] 터널 마스크 + 복셀 그리드 생성 (voxel={args.voxel_size}m)...")
    _, tunnel_mask_xp, _, grid_info = build_voxel_masks(
        poly_Y, poly_Z,
        domain_box=domain_box,
        voxel_size=args.voxel_size,
        halo_dist=args.halo,
    )
    try:
        import cupy as cp
        tunnel_mask = cp.asnumpy(tunnel_mask_xp)
    except Exception:
        tunnel_mask = np.asarray(tunnel_mask_xp)

    # ── 3. 균열 필터링 (crop_box AABB) ───────────────────────────────────
    xmin, xmax, ymin, ymax, zmin, zmax = domain_box.astype(float)
    in_box = (
        (centers[:, 0] + radii >= xmin) & (centers[:, 0] - radii <= xmax) &
        (centers[:, 1] + radii >= ymin) & (centers[:, 1] - radii <= ymax) &
        (centers[:, 2] + radii >= zmin) & (centers[:, 2] - radii <= zmax)
    )
    c_crop = centers[in_box]
    n_crop = normals[in_box]
    r_crop = radii[in_box]
    print(f"\n[Step 2/4] 균열 복셀 분류 ({in_box.sum():,} / {len(radii):,} 균열)...")

    # ── 4. 복셀 분류 (ROCK / FRACTURE / TUNNEL + Owner Tracking) ──────────
    state, fracture_owner = classify_voxels(
        grid_info, c_crop, n_crop, r_crop,
        tunnel_mask=tunnel_mask,
        tol_factor=args.tol_factor,
    )

    # ── 5. Connected Component Analysis (CCA) ────────────────────────────
    print(f"\n[Step 3/4] CCA (블록 번호 부여) 작동 중 (connectivity={args.connectivity})...")
    labels, n_labels = run_cca(state, connectivity=args.connectivity)

    # ── 6. 블록 필터 + 통계 ───────────────────────────────────────────────
    block_info = filter_and_stat_blocks(
        labels, n_labels, state, grid_info,
        min_voxels=args.min_voxels,
        connectivity=args.connectivity,
    )

    # ── 7. 결과 저장 ─────────────────────────────────────────────────────
    print(f"\n[Step 4/4] 결과 저장 및 시각화...")
    elapsed = time.time() - t0

    summary = dict(
        input_file      = args.input,
        voxel_size      = args.voxel_size,
        tol_factor      = args.tol_factor,
        n_fractures_total    = int(len(radii)),
        n_fractures_in_domain= int(in_box.sum()),
        n_blocks        = len(block_info),
        elapsed_sec     = round(elapsed, 2),
        blocks          = block_info,
    )
    json_path = os.path.join(args.outdir, 'block_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  JSON : {json_path}")

    np.save(os.path.join(args.outdir, 'block_labels.npy'), labels)
    np.save(os.path.join(args.outdir, 'voxel_state.npy'),  state)
    np.save(os.path.join(args.outdir, 'fracture_owner.npy'), fracture_owner)

    out_h5 = os.path.join(args.outdir, 'block_results.h5')
    with h5py.File(out_h5, 'w') as f:
        f.create_dataset('labels',         data=labels, compression='gzip')
        f.create_dataset('voxel_state',    data=state,  compression='gzip')
        f.create_dataset('tunnel_mask',    data=tunnel_mask, compression='gzip')
        f.create_dataset('fracture_owner', data=fracture_owner, compression='gzip')
        f.attrs['n_blocks']    = len(block_info)
        f.attrs['voxel_size']  = args.voxel_size
        f.attrs['tol_factor']  = args.tol_factor
        f.attrs['elapsed_sec'] = elapsed
        grp = f.create_group('grid_info')
        grp.attrs['voxel_size'] = grid_info['voxel_size']
        grp.create_dataset('xs', data=grid_info['xs'])
        grp.create_dataset('ys', data=grid_info['ys'])
        grp.create_dataset('zs', data=grid_info['zs'])
    print(f"  HDF5 : {out_h5}")

    # ── 8. CPU 데이터 전송 (시각화 안정성 확보) ──────────────────────────
    if hasattr(labels, 'get'):
        labels_cpu = labels.get()
        state_cpu  = state.get()
        fracture_owner_cpu = fracture_owner.get()
    else:
        labels_cpu = labels; state_cpu = state
        fracture_owner_cpu = fracture_owner

    # ── 9. CSV Data Export ───────────────────────────────────────────────
    export_blocks_csv(block_info, args.outdir)
    export_interfaces_csv(labels, block_info, grid_info, args.outdir)

    param_suffix = f"vs{args.voxel_size}_tol{args.tol_factor}_minv{args.min_voxels}_minc{args.min_contact}"

    # Determine tunnel 3D range
    t_start = args.tunnel_x_start
    if t_start is None:
        t_start = data.get('x_start', float(grid_info['xs'][0]))
        
    t_end = None
    if data.get('x_end') is not None:
        offset = args.tunnel_end_offset if args.tunnel_end_offset is not None else 0.0
        t_end = data['x_end'] + offset
        
    if t_end is None:
        t_end = float(grid_info['xs'][-1])
        
    tunnel_range = (t_start, t_end)
    print(f"\n[Info] 3D 시각화 터널 범위: X = [{tunnel_range[0]:.2f}m, {tunnel_range[1]:.2f}m]")

    # 시각화 0: overview    # 1) 2D Dashboard
    plot_block_overview(
        labels_cpu, state_cpu, grid_info, block_info, tunnel_poly_YZ=poly_YZ,
        save_path=os.path.join(args.outdir, f"block_overview_{param_suffix}.png")
    )
    
    # 2) 3D Scatter
    plot_block_3d_scatter(
        labels_cpu, state_cpu, grid_info, block_info, tunnel_poly_YZ=poly_YZ,
        save_path=os.path.join(args.outdir, f"block_3d_scatter_{param_suffix}.png"),
        tunnel_range=tunnel_range
    )
    
    # ── 10. 최종 통합 시각화 파이프라인 (2단계) ──────────────────────────────
    frac_data = {'centers': c_crop, 'normals': n_crop, 'radii': r_crop}
    
    # [사진 1] 전체 블럭 분포 뷰
    print(f"\n  [Viz] [1/2] 전체 블록 분포 뷰어 창을 엽니다. (사진 1 저장 포함)")
    pyvista_png_blocks = os.path.join(args.outdir, f"pyvista_3d_blocks_{param_suffix}.png")
    plot_block_3d_pyvista_interactive(
        labels_cpu, state_cpu, grid_info, block_info, tunnel_poly_YZ=poly_YZ,
        save_path=pyvista_png_blocks,
        tunnel_range=tunnel_range
    )

    # [사진 2] 전체 블럭 + 균열 인터페이스 뷰
    print(f"\n  [Viz] [2/2] 전체 블록-균열 인터페이스 뷰어 창을 엽니다. (사진 2 저장 포함)")
    pyvista_png_interfaces = os.path.join(args.outdir, f"pyvista_3d_interfaces_{param_suffix}.png")
    plot_all_blocks_with_fractures(
        labels_cpu, state_cpu, fracture_owner_cpu, grid_info, block_info, frac_data,
        tunnel_poly_YZ=poly_YZ,
        shell_thickness=args.shell_thickness,
        min_contact_voxels=args.min_contact,
        save_path=pyvista_png_interfaces,
        tunnel_range=tunnel_range
    )

    print(f"\n{'='*60}")
    print(f"  [Info] 블록 탐지 완료")
    print(f"  - 탐지된 블록: {len(block_info):,}개")
    if block_info:
        vols = [b['volume_m3'] for b in block_info]
        print(f"  - 최대 볼륨  : {max(vols):.3f} m³")
        print(f"  - 중앙값 볼륨: {float(np.median(vols)):.3f} m³")
    print(f"  - 처리 시간  : {elapsed:.1f}초")
    print(f"  - 결과 폴더  : {args.outdir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
