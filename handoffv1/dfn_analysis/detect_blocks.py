r"""
detect_blocks.py – 조건부 DFN(export JSON) → 복셀 CCA 블록 탐지 + 제거가능성·안전율

입력: export_domain_dfn_json.py 가 만든 dfn_domain_*.json
      (meta.domain: x 범위·yz 범위·터널 단면 다각형 [y,z], fractures: 원판 중심/법선/반지름)

가정(명시):
  * 좌표 [x, y, z], x = 터널 굴진 방향. 터널 단면 다각형 [y, z] 을 x 로 압출한다.
  * 터널은 기본적으로 도메인 x 범위 전체가 굴착된 상태로 본다(전방 구간 굴착 후 블록 예측).
    --tunnel-x-range 로 바꿀 수 있다.
  * 균열 = 유한 원판, 복셀 slab 반두께 tol = tol_factor × voxel_size.
  * CCA = 6-connectivity 기본 (26 은 비교용; 경사 slab 누설로 블록이 합쳐질 수 있음,
    scripts/verify_block_detection.py Test A/B 참조).
  * 블록 = 터널에 접하고 도메인 경계에 닿지 않는 ROCK 컴포넌트 (min_voxels 이상).
  * 안정성 = 중력만, 지보·수압 없음, 절리 인장강도 0 (block_stability.py).

Usage (handoffv1 폴더에서):
    PYTHONPATH=. python -m dfn_analysis.detect_blocks \
        --json demo_output/seed42/export/dfn_domain_x3-13_halo5.json --voxel-size 0.1
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import time
from collections import Counter

import numpy as np

from dfn_analysis.block_detector import classify_voxels, run_cca, filter_and_stat_blocks
from dfn_analysis.block_stability import analyze_block
from dfn_analysis.tunnel_geometry import build_voxel_masks


def git_commit() -> str | None:
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def detect(d, voxel_size, tol_factor=0.6, min_voxels=8, connectivity=6, tunnel_x_range=None,
           gamma=26.0, phi=30.0, cohesion=0.0):
    """export JSON dict → (블록 행 목록, ctx). ctx 에는 격자·라벨 배열 등 후처리용 자료가 담긴다."""
    dom = d['meta']['domain']
    x0, x1 = dom['x_range_m']
    yz = dom['yz_bounds_m']
    domain_box = np.array([x0, x1, yz['y_min'], yz['y_max'], yz['z_min'], yz['z_max']], float)
    poly = np.asarray(dom['tunnel_polygon_yz_m'], float)            # [y, z]
    tx0, tx1 = tunnel_x_range if tunnel_x_range else (x0, x1)

    fr = d['fractures']
    ids = np.array([f['id'] for f in fr])
    frac_label = np.array([f['label'] for f in fr])
    set_id = np.array([f['set_id'] for f in fr])
    centers = np.array([f['center_xyz_m'] for f in fr], dtype=np.float32)
    normals = np.array([f['normal_xyz'] for f in fr], dtype=np.float32)
    radii = np.array([f['radius_m'] for f in fr], dtype=np.float32)

    print(f"[domain] x=[{x0:g},{x1:g}] y=[{yz['y_min']:.2f},{yz['y_max']:.2f}] "
          f"z=[{yz['z_min']:.2f},{yz['z_max']:.2f}] m  | tunnel x=[{tx0:g},{tx1:g}] m")
    print(f"[fractures] {len(fr):,}  (observed {int((frac_label == 'observed').sum()):,} / "
          f"unobserved {int((frac_label == 'unobserved').sum()):,})")

    # ── 1. 격자 + 터널 마스크 ────────────────────────────────────────────
    tunnel_mask, grid = build_voxel_masks(poly[:, 0], poly[:, 1], domain_box,
                                          voxel_size, tunnel_xmin=tx0, tunnel_xmax=tx1)

    # ── 2. 균열 AABB crop → 복셀 분류 ────────────────────────────────────
    in_box = ((centers[:, 0] + radii >= x0) & (centers[:, 0] - radii <= x1) &
              (centers[:, 1] + radii >= domain_box[2]) & (centers[:, 1] - radii <= domain_box[3]) &
              (centers[:, 2] + radii >= domain_box[4]) & (centers[:, 2] - radii <= domain_box[5]))
    idx = np.nonzero(in_box)[0]
    state, owner = classify_voxels(grid, centers[idx], normals[idx], radii[idx],
                                   tunnel_mask, tol_factor)

    # ── 3. CCA + 필터 ────────────────────────────────────────────────────
    labels, n_labels = run_cca(state, connectivity)
    blocks = filter_and_stat_blocks(labels, n_labels, state, grid,
                                    min_voxels=min_voxels, connectivity=connectivity)

    # ── 4. 제거가능성 · 파괴모드 · FS ────────────────────────────────────
    print(f"[stability] {len(blocks)} blocks ...")
    rows = []
    for b in blocks:
        st = analyze_block(b, labels, state, owner, normals[idx], centers[idx], grid,
                           gamma_kN_m3=gamma, phi_deg=phi,
                           cohesion_kPa=cohesion, tol_factor=tol_factor)
        plane_ids = idx[st['plane_owners']] if st['plane_owners'] else np.array([], int)
        mode_ids = idx[st['mode_planes']] if st['mode_planes'] else np.array([], int)
        rows.append(dict(
            rank=b['rank'], label=b['label'], n_voxels=b['n_voxels'],
            volume_voxel_m3=round(b['volume_m3'], 4),
            volume_corr_m3=round(st['volume_corr_m3'], 4),
            cx=round(b['centroid'][0], 3), cy=round(b['centroid'][1], 3), cz=round(b['centroid'][2], 3),
            tunnel_contact_area_m2=round(b['contact_area_m2'], 3),
            n_planes=st['n_planes'],
            n_planes_observed=int((frac_label[plane_ids] == 'observed').sum()),
            plane_fracture_ids=' '.join(str(ids[i]) for i in plane_ids),
            plane_set_ids=' '.join(str(set_id[i]) for i in plane_ids),
            removable=st['removable'], mode=st['mode'],
            fs=st['fs'] if np.isfinite(st['fs']) else 'inf',
            mode_fracture_ids=' '.join(str(ids[i]) for i in mode_ids),
            mode_dir=' '.join(f'{v:g}' for v in st['mode_dir']) if st['mode_dir'] else '',
            weight_kN=round(st['weight_kN'], 2),
        ))


    ctx = dict(domain_box=domain_box, tunnel_x_range=(tx0, tx1), grid=grid, labels=labels,
               state=state, tunnel_mask=tunnel_mask, n_labels=n_labels, in_box=in_box, poly_yz=poly)
    return rows, ctx


def main():
    ap = argparse.ArgumentParser(description='조건부 DFN → CCA 블록 탐지 + 안전율')
    ap.add_argument('--json', required=True, help='export_domain_dfn_json 출력 JSON')
    ap.add_argument('--voxel-size', type=float, default=0.1, help='복셀 크기 [m]')
    ap.add_argument('--tol-factor', type=float, default=0.6, help='균열 slab 반두께 / voxel_size')
    ap.add_argument('--min-voxels', type=int, default=8, help='최소 블록 복셀 수')
    ap.add_argument('--min-volume', type=float, default=None,
                    help='최소 블록 부피 [m3]. 주면 min_voxels = ceil(V/vs^3) 로 덮어쓴다 '
                         '(복셀 크기가 달라도 같은 물리 기준으로 비교)')
    ap.add_argument('--connectivity', type=int, choices=[6, 26], default=6)
    ap.add_argument('--tunnel-x-range', nargs=2, type=float, default=None,
                    help='터널 굴착 x 범위 [m] (기본: 도메인 x 범위 전체)')
    ap.add_argument('--gamma', type=float, default=26.0, help='암석 단위중량 [kN/m3]')
    ap.add_argument('--phi', type=float, default=30.0, help='절리 마찰각 [deg]')
    ap.add_argument('--cohesion', type=float, default=0.0, help='절리 점착력 [kPa]')
    ap.add_argument('--outdir', default=None, help='기본: <json 폴더>/blocks_vs<voxel>')
    args = ap.parse_args()

    if args.min_volume is not None:
        args.min_voxels = max(1, int(np.ceil(args.min_volume / args.voxel_size ** 3)))

    t0 = time.time()
    with open(args.json, encoding='utf-8') as fh:
        d = json.load(fh)
    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.json)),
                                         f'blocks_vs{args.voxel_size:g}')
    os.makedirs(outdir, exist_ok=True)
    print(f"[input] {args.json}")
    print(f"[settings] voxel={args.voxel_size} m  tol_factor={args.tol_factor}  "
          f"connectivity={args.connectivity}  min_voxels={args.min_voxels} "
          f"(min_volume={args.min_volume} m3)  "
          f"gamma={args.gamma} kN/m3  phi={args.phi} deg  c={args.cohesion} kPa")
    rows, ctx = detect(d, args.voxel_size, args.tol_factor, args.min_voxels, args.connectivity,
                       args.tunnel_x_range, args.gamma, args.phi, args.cohesion)
    domain_box, (tx0, tx1), grid = ctx['domain_box'], ctx['tunnel_x_range'], ctx['grid']
    fr, in_box, n_labels = d['fractures'], ctx['in_box'], ctx['n_labels']

    csv_path = os.path.join(outdir, 'blocks.csv')
    fields = list(rows[0]) if rows else ['rank']
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    modes = Counter(r['mode'] for r in rows)
    fs_vals = [r['fs'] for r in rows if r['fs'] != 'inf']
    summary = dict(
        input_json=os.path.abspath(args.json), git_commit=git_commit(),
        source_seed=d['meta'].get('seed'), site=d['meta'].get('site'),
        domain_box_m=domain_box.tolist(), tunnel_x_range_m=[tx0, tx1],
        voxel_size_m=args.voxel_size, tol_factor=args.tol_factor,
        grid_shape=list(grid['shape']), connectivity=args.connectivity,
        min_voxels=args.min_voxels, min_volume_m3=args.min_volume,
        block_filter='ROCK component, touches TUNNEL, not touching domain boundary, n_voxels >= min_voxels',
        stability=dict(gamma_kN_m3=args.gamma, phi_deg=args.phi, cohesion_kPa=args.cohesion,
                       loads='gravity only, no support, no water, zero joint tensile strength'),
        n_fractures_in_json=len(fr), n_fractures_in_domain=int(in_box.sum()),
        n_components=n_labels, n_blocks=len(rows),
        n_removable=sum(r['removable'] for r in rows),
        mode_counts=dict(modes),
        n_fs_below_1=sum(f < 1.0 for f in fs_vals),
        n_fs_below_1p5=sum(f < 1.5 for f in fs_vals),
        total_volume_corr_m3=round(sum(r['volume_corr_m3'] for r in rows), 3),
        elapsed_sec=round(time.time() - t0, 1),
    )
    with open(os.path.join(outdir, 'block_summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 64}")
    print(f"  blocks={len(rows)}  removable={summary['n_removable']}  modes={dict(modes)}")
    print(f"  FS<1: {summary['n_fs_below_1']}   FS<1.5: {summary['n_fs_below_1p5']}   "
          f"total V_corr={summary['total_volume_corr_m3']} m3")
    if rows:
        v = [r['volume_corr_m3'] for r in rows]
        print(f"  V_corr max={max(v):.3f}  median={np.median(v):.3f} m3")
    print(f"  out: {csv_path}")
    print(f"  elapsed {summary['elapsed_sec']} s")
    print('=' * 64)


if __name__ == '__main__':
    main()
