"""
detect_blocks_gpu.py  –  GPU 가속 3D 블록 탐지 파이프라인 (메인 스크립트)

사용법:
    python detect_blocks_gpu.py --input <dfn_export_for_python.h5>
                                [--voxel_size 0.5]
                                [--halo 6.0]          # crop_box 크기 제한 (미사용 시 전체)
                                [--tol_factor 0.6]    # 균열 슬랩 두께 = tol_factor × voxel_size
                                [--min_voxels 8]
                                [--outdir ./results]
                                [--no_gpu]

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
from visualize_blocks import plot_block_3d_pyvista


# ════════════════════════════════════════════════════════════════════════════
def load_hdf5(path: str) -> dict:
    """HDF5 로드 – MATLAB 전치(transpose) 자동 보정."""
    print(f"\n📂 HDF5 로드: {path}")
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
    parser.add_argument('--tol_factor',  type=float, default=0.6,   help='균열 슬랩 두께 계수')
    parser.add_argument('--min_voxels',  type=int,   default=8,     help='최소 블록 복셀 수')
    parser.add_argument('--outdir',      default=None)
    parser.add_argument('--no_gpu',      action='store_true')
    args = parser.parse_args()

    if args.no_gpu:
        import block_detector as bd; bd.HAS_GPU = False
        import tunnel_geometry as tg; tg.HAS_GPU = False
        print("[!] GPU 비활성화")

    # 출력 폴더
    if args.outdir is None:
        args.outdir = os.path.join(
            os.path.dirname(os.path.abspath(args.input)),
            'block_detection_results')
    os.makedirs(args.outdir, exist_ok=True)
    print(f"📁 결과 저장: {args.outdir}")

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
        halo_dist=0.0,   # halo 불필요 (전체 도메인 분석)
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

    # ── 4. 복셀 분류 (ROCK / FRACTURE / TUNNEL) ──────────────────────────
    state = classify_voxels(
        grid_info, c_crop, n_crop, r_crop,
        tunnel_mask=tunnel_mask,
        tol_factor=args.tol_factor,
    )

    # ── 5. CCA ───────────────────────────────────────────────────────────
    print(f"\n[Step 3/4] CCA (26-connectivity)...")
    labels, n_labels = run_cca(state)

    # ── 6. 블록 필터 + 통계 ───────────────────────────────────────────────
    block_info = filter_and_stat_blocks(
        labels, n_labels, state, grid_info,
        min_voxels=args.min_voxels,
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

    out_h5 = os.path.join(args.outdir, 'block_results.h5')
    with h5py.File(out_h5, 'w') as f:
        f.create_dataset('labels',      data=labels, compression='gzip')
        f.create_dataset('voxel_state', data=state,  compression='gzip')
        f.create_dataset('tunnel_mask', data=tunnel_mask, compression='gzip')
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

    # 시각화 (PyVista 대화형 뷰어 실행)
    plot_block_3d_pyvista(
        labels, block_info, grid_info,
        tunnel_poly_YZ=poly_YZ,
        save_path=os.path.join(args.outdir, 'block_3d_pyvista.png'),
    )

    print(f"\n{'='*60}")
    print(f"  ✅ 블록 탐지 완료")
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
