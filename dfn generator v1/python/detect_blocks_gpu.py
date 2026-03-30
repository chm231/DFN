"""
detect_blocks_gpu.py
메인 실행 스크립트 – GPU 가속 3D 블록 탐지 파이프라인

사용법:
    python detect_blocks_gpu.py --input <path/to/dfn_export_for_python.h5>
                                [--voxel_size 0.5]
                                [--halo 6.0]
                                [--min_voxels 8]
                                [--batch 200000]
                                [--outdir ./results]
                                [--no_gpu]

MATLAB 내보내기 HDF5 구조:
    /fractures/centers   [N×3] float32
    /fractures/normals   [N×3] float32
    /fractures/radii     [N×1] float32
    /fractures/set_id    [N×1] uint16
    /tunnel/poly_YZ      [M×2] float32
    /tunnel/profile_Y    [K×1] float32
    /tunnel/profile_Z    [K×1] float32
    /meta/domain_box     [1×6] float32  [xmin xmax ymin ymax zmin zmax]
    /meta/crop_box       [1×6] float32  (선택)
"""

from __future__ import annotations
import sys
import argparse
import time
import json
import os
import numpy as np
import h5py

# ── 로컬 모듈 ─────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from tunnel_geometry import build_voxel_masks
from block_detector import build_face_adjacency, detect_blocks
from visualize_blocks import plot_block_overview, plot_block_3d_scatter


# ════════════════════════════════════════════════════════════════════════════
def load_hdf5(path: str) -> dict:
    """HDF5 파일에서 DFN + 터널 데이터 로드."""
    print(f"\n📂 HDF5 로드: {path}")
    data = {}
    with h5py.File(path, 'r') as f:
        data['centers']  = f['/fractures/centers'][:]   # (N,3)
        data['normals']  = f['/fractures/normals'][:]   # (N,3)
        data['radii']    = f['/fractures/radii'][:].ravel()  # (N,)
        if '/fractures/set_id' in f:
            data['set_id'] = f['/fractures/set_id'][:].ravel()
        else:
            data['set_id'] = np.ones(len(data['radii']), dtype=np.uint16)

        if '/tunnel/poly_YZ' in f:
            data['poly_YZ'] = f['/tunnel/poly_YZ'][:]   # (M,2)
        if '/tunnel/profile_Y' in f:
            data['profile_Y'] = f['/tunnel/profile_Y'][:].ravel()
            data['profile_Z'] = f['/tunnel/profile_Z'][:].ravel()

        data['domain_box'] = f['/meta/domain_box'][:].ravel()   # (6,)
        if '/meta/crop_box' in f:
            data['crop_box'] = f['/meta/crop_box'][:].ravel()
        else:
            data['crop_box'] = data['domain_box'].copy()

    N = len(data['radii'])
    print(f"  균열 수: {N:,}")
    print(f"  도메인: {data['domain_box']}")
    if 'poly_YZ' in data:
        print(f"  터널 폴리곤: {len(data['poly_YZ'])} 점")
    return data


# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='GPU 가속 3D 블록 탐지 (DFN + 터널)')
    parser.add_argument('--input',      required=True,  help='HDF5 입력 파일 경로')
    parser.add_argument('--voxel_size', type=float, default=0.5,   help='복셀 크기 (m)')
    parser.add_argument('--halo',       type=float, default=6.0,   help='터널 영향권 거리 (m)')
    parser.add_argument('--min_voxels', type=int,   default=8,     help='최소 블록 복셀 수')
    parser.add_argument('--batch',      type=int,   default=200_000, help='균열 배치 크기 (GPU)')
    parser.add_argument('--outdir',     default=None, help='결과 저장 폴더 (기본: 입력 파일 옆)')
    parser.add_argument('--no_gpu',     action='store_true', help='GPU 비활성화 (CPU 폴백)')
    args = parser.parse_args()

    if args.no_gpu:
        import block_detector as bd
        import tunnel_geometry as tg
        bd.HAS_GPU = False
        tg.HAS_GPU = False
        print("[!] GPU 비활성화됨.")

    # ── 출력 폴더 ────────────────────────────────────────────────────────
    if args.outdir is None:
        args.outdir = os.path.join(os.path.dirname(os.path.abspath(args.input)),
                                   'block_detection_results')
    os.makedirs(args.outdir, exist_ok=True)
    print(f"📁 결과 저장: {args.outdir}")

    t0 = time.time()

    # ── 1. 데이터 로드 ───────────────────────────────────────────────────
    data = load_hdf5(args.input)

    centers  = data['centers'].astype(np.float32)
    normals  = data['normals'].astype(np.float32)
    radii    = data['radii'].astype(np.float32)

    poly_YZ = data.get('poly_YZ', None)
    if poly_YZ is None:
        print("[ERROR] /tunnel/poly_YZ 데이터가 없습니다. MATLAB에서 터널 폴리곤을 내보내야 합니다.")
        sys.exit(1)

    poly_Y = poly_YZ[:, 0]
    poly_Z = poly_YZ[:, 1]

    # crop_box를 분석 도메인으로 사용
    domain_box = data['crop_box']
    print(f"\n분석 도메인: {domain_box}")

    # ── 2. 복셀 마스크 생성 ──────────────────────────────────────────────
    print(f"\n[Step 1/4] 복셀 마스크 생성 (voxel={args.voxel_size}m, halo={args.halo}m)...")
    voxel_centers, tunnel_mask_gpu, halo_mask_gpu, grid_info = build_voxel_masks(
        poly_Y, poly_Z,
        domain_box=domain_box,
        voxel_size=args.voxel_size,
        halo_dist=args.halo,
    )

    try:
        import cupy as cp
        tunnel_mask = cp.asnumpy(tunnel_mask_gpu)
        halo_mask   = cp.asnumpy(halo_mask_gpu)
    except Exception:
        tunnel_mask = np.asarray(tunnel_mask_gpu)
        halo_mask   = np.asarray(halo_mask_gpu)

    n_halo = halo_mask.sum()
    print(f"  Halo 복셀 수: {n_halo:,}")

    # ── 3. 균열-면 교차 → 연결 행렬 ─────────────────────────────────────
    print(f"\n[Step 2/4] 균열-면 교차 계산 ({len(radii):,} 균열)...")
    # crop_box 경계 내 균열만 필터링 (AABB)
    xmin, xmax, ymin, ymax, zmin, zmax = domain_box.astype(float)
    margin = radii  # 반경만큼 여유
    in_box = (
        (centers[:, 0] + radii >= xmin) & (centers[:, 0] - radii <= xmax) &
        (centers[:, 1] + radii >= ymin) & (centers[:, 1] - radii <= ymax) &
        (centers[:, 2] + radii >= zmin) & (centers[:, 2] - radii <= zmax)
    )
    c_crop = centers[in_box]
    n_crop = normals[in_box]
    r_crop = radii[in_box]
    print(f"  도메인 내 균열: {in_box.sum():,} / {len(radii):,}")

    conn_x, conn_y, conn_z = build_face_adjacency(
        grid_shape=grid_info['shape'],
        fracture_centers=c_crop,
        fracture_normals=n_crop,
        fracture_radii=r_crop,
        grid_info=grid_info,
        halo_mask_cpu=halo_mask,
        batch_fractures=args.batch,
    )

    # ── 4. 블록 탐지 ─────────────────────────────────────────────────────
    print(f"\n[Step 3/4] 블록 탐지 (min_voxels={args.min_voxels})...")
    labels, block_info = detect_blocks(
        conn_x, conn_y, conn_z,
        halo_mask=halo_mask,
        grid_info=grid_info,
        min_voxels=args.min_voxels,
    )

    # ── 5. 결과 저장 ─────────────────────────────────────────────────────
    print(f"\n[Step 4/4] 결과 저장 및 시각화...")
    elapsed = time.time() - t0

    # JSON 통계
    summary = dict(
        input_file=args.input,
        voxel_size=args.voxel_size,
        halo_dist=args.halo,
        n_fractures_total=int(len(radii)),
        n_fractures_in_domain=int(in_box.sum()),
        n_blocks=len(block_info),
        elapsed_sec=round(elapsed, 2),
        blocks=block_info,
    )
    json_path = os.path.join(args.outdir, 'block_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {json_path}")

    # 레이블 배열 저장 (numpy)
    np.save(os.path.join(args.outdir, 'block_labels.npy'), labels.astype(np.int32))
    np.save(os.path.join(args.outdir, 'halo_mask.npy'), halo_mask)
    print(f"  레이블: {os.path.join(args.outdir, 'block_labels.npy')}")

    # HDF5로 결과 저장
    out_h5 = os.path.join(args.outdir, 'block_results.h5')
    with h5py.File(out_h5, 'w') as f:
        f.create_dataset('labels',     data=labels.astype(np.int32),  compression='gzip')
        f.create_dataset('halo_mask',  data=halo_mask,                compression='gzip')
        f.create_dataset('tunnel_mask',data=tunnel_mask,              compression='gzip')
        f.attrs['n_blocks']    = len(block_info)
        f.attrs['voxel_size']  = args.voxel_size
        f.attrs['halo_dist']   = args.halo
        f.attrs['elapsed_sec'] = elapsed
        grp = f.create_group('grid_info')
        grp.attrs['voxel_size'] = grid_info['voxel_size']
        grp.create_dataset('xs', data=grid_info['xs'])
        grp.create_dataset('ys', data=grid_info['ys'])
        grp.create_dataset('zs', data=grid_info['zs'])
    print(f"  HDF5: {out_h5}")

    # 시각화
    plot_block_overview(
        labels, block_info, grid_info,
        tunnel_poly_YZ=poly_YZ,
        save_path=os.path.join(args.outdir, 'block_overview.png'),
    )
    plot_block_3d_scatter(
        labels, block_info, grid_info,
        save_path=os.path.join(args.outdir, 'block_3d_scatter.png'),
    )

    # ── 요약 출력 ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ✅ 블록 탐지 완료")
    print(f"  - 탐지된 블록: {len(block_info)}개")
    if block_info:
        vols = [b['volume_m3'] for b in block_info]
        print(f"  - 최대 볼륨  : {max(vols):.3f} m³")
        print(f"  - 중앙값 볼륨: {np.median(vols):.3f} m³")
    print(f"  - 처리 시간  : {elapsed:.1f}초")
    print(f"  - 결과 폴더  : {args.outdir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
