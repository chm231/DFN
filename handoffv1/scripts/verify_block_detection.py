"""
verify_block_detection.py – CCA 블록 탐지·안정성 모듈의 해석해 검증 (논문 2단계)

Test A  6- vs 26-connectivity 누설 시험
        무한 평면 1개(원판 반지름 1000 m)로 자른 정육면체 격자의 ROCK 컴포넌트 수.
        정답 = 2. 이론(디지털 평면): slab 두께 2·tol 이
          ≥ vs·||n||_inf 이면 6-연결로 분리,  ≥ vs·||n||_1 이면 26-연결로도 분리.
        tol = 0.6·vs 에서는 6-연결은 항상 분리, 26-연결은 경사 평면에서 누설될 수 있다.

Test B  해석해 사면체 쐐기 (터널 면 + 균열면 3개)
        B1 천장 낙하 / B2 측벽 단일면 활동 / B3 측벽 쐐기(2면) 활동.
        기준값: 정확한 평면 기하(꼭짓점)로 부피·면적을 구하고, 볼록 블록의
        Goodman–Shi 규칙(d·n_out ≤ 0 ∀면 이면 자유)으로 모드·FS 를 계산.
        복셀 결과(부피·모드·FS)를 복셀 크기별로 비교해 수렴을 확인한다.

좌표: 전역 [x, y, z], x = 터널 굴진, z = Up. 터널 단면 다각형은 [y, z].

실행 (handoffv1 폴더에서):
    PYTHONPATH=. python scripts/verify_block_detection.py
"""

from __future__ import annotations
import argparse
import contextlib
import csv
import io
import os

import numpy as np

from dfn_analysis.block_detector import classify_voxels, run_cca, filter_and_stat_blocks, ROCK
from dfn_analysis.block_stability import analyze_block, GRAVITY
from dfn_analysis.tunnel_geometry import build_voxel_masks


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


# ─────────────────────────────────────────────────────────────────────────
# Test A
# ─────────────────────────────────────────────────────────────────────────
def test_leak(vs=0.2, tol_factor=0.6):
    normals = {
        'axis (0,0,1)': unit([0, 0, 1]),
        'edge (1,1,0)': unit([1, 1, 0]),
        'diag (1,1,1)': unit([1, 1, 1]),
        'oblique (0.3,0.5,0.81)': unit([0.3, 0.5, 0.81]),
    }
    box = np.array([0, 6, 0, 6, 0, 6], dtype=float)
    far = np.array([100.0, 101.0])              # 터널 없음(도메인 밖 다각형)
    tunnel_mask, grid = quiet(build_voxel_masks, far, far, box, vs)
    rows = []
    for name, n in normals.items():
        centers = np.array([[3.01, 2.97, 3.03]], dtype=np.float32)
        state, _ = quiet(classify_voxels, grid, centers, n[None].astype(np.float32),
                         np.array([1000.0], dtype=np.float32), tunnel_mask, tol_factor)
        n6 = quiet(run_cca, state, 6)[1]
        n26 = quiet(run_cca, state, 26)[1]
        rows.append(dict(normal=name, norm_inf=round(np.abs(n).max(), 3),
                         norm_1=round(np.abs(n).sum(), 3),
                         slab_over_vs=round(2 * tol_factor, 2),
                         components_6=n6, components_26=n26))
    return rows


# ─────────────────────────────────────────────────────────────────────────
# Test B – 해석해 쐐기
# ─────────────────────────────────────────────────────────────────────────
CASES = {
    # 천장(z=4) 위 사면체. 모든 n_out 의 z>0 → 낙하
    'B1_roof_fall': dict(
        poly_yz=[[-5, -5], [5, -5], [5, 4], [-5, 4]],
        free_n=[0, 0, -1], free_pt=[0, 0, 4],
        box=[-5, 5, -5, 5, 2, 9],
        apex=[0.13, -0.07, 6.0],
        normals=[[np.sin(np.radians(55)) * np.cos(np.radians(b)),
                  np.sin(np.radians(55)) * np.sin(np.radians(b)),
                  np.cos(np.radians(55))] for b in (10, 130, 250)],
    ),
    # 우측벽(y=4) 쐐기. 바닥면 1개가 받치고 나머지 2면은 들림 → 단일면 활동
    'B2_wall_slide1': dict(
        poly_yz=[[-5, -5], [4, -5], [4, 5], [-5, 5]],
        free_n=[0, -1, 0], free_pt=[0, 4, 0],
        box=[-5, 5, 2, 10, -5, 5],
        apex=[0.3, 6.5, 0.2],
        normals=[[0.1, 0.6, -1.0], [0.8, 0.5, 0.45], [-0.8, 0.5, 0.45]],
    ),
    # 우측벽(y=4) 쐐기. 아래 두 면이 받치고 교선이 터널 쪽으로 기움 → 쐐기 활동
    'B3_wall_slide2': dict(
        poly_yz=[[-5, -5], [4, -5], [4, 5], [-5, 5]],
        free_n=[0, -1, 0], free_pt=[0, 4, 0],
        box=[-5, 5, 2, 10, -5, 5],
        apex=[0.2, 6.5, 0.4],
        normals=[[0.7, 0.6, -1.0], [-0.7, 0.6, -1.0], [0.1, 0.5, 1.0]],
    ),
}


def analytic_reference(case, gamma, phi_deg, c_kpa):
    """정확한 평면 기하로 부피·면적·모드·FS (볼록 사면체)."""
    A = np.asarray(case['apex'], float)
    ns = [unit(n) for n in case['normals']]          # 바깥 법선(꼭짓점 A 통과)
    f = unit(case['free_n'])
    fp = np.asarray(case['free_pt'], float)
    base = {}
    for i, j in ((0, 1), (1, 2), (0, 2)):
        M = np.stack([ns[i], ns[j], f])
        base[(i, j)] = np.linalg.solve(M, [ns[i] @ A, ns[j] @ A, f @ fp])
    v01, v12, v02 = base[(0, 1)], base[(1, 2)], base[(0, 2)]
    V = abs(np.linalg.det(np.stack([v01 - A, v12 - A, v02 - A]))) / 6.0
    # 면 k 의 삼각형 = A + 면 k 를 포함하는 두 밑변 꼭짓점
    tri = {0: (v01, v02), 1: (v01, v12), 2: (v12, v02)}
    areas = [0.5 * np.linalg.norm(np.cross(p - A, q - A)) for p, q in tri.values()]

    W, tan_phi = gamma * V, np.tan(np.radians(phi_deg))
    free = lambda d: all(d @ n <= 1e-9 for n in ns)
    cands = []
    if free(GRAVITY):
        cands.append(('fall', 0.0))
    for k, n in enumerate(ns):
        cos_a = GRAVITY @ n
        if cos_a <= 1e-9:
            continue
        s = unit(GRAVITY - cos_a * n)
        if free(s):
            sin_a = np.sqrt(1 - cos_a ** 2)
            cands.append(('slide_1plane', (W * cos_a * tan_phi + c_kpa * areas[k]) / (W * sin_a)))
    for i, j in ((0, 1), (1, 2), (0, 2)):
        t = unit(np.cross(ns[i], ns[j]))
        t = -t if t @ GRAVITY < 0 else t
        if t @ GRAVITY <= 1e-9:
            continue
        N = np.linalg.lstsq(np.stack([ns[i], ns[j]], 1), W * (GRAVITY - (GRAVITY @ t) * t), rcond=None)[0]
        if N[0] > 0 and N[1] > 0 and free(t):
            cands.append(('slide_2plane',
                          ((N[0] + N[1]) * tan_phi + c_kpa * (areas[i] + areas[j])) / (W * (t @ GRAVITY))))
    mode, fs = min(cands, key=lambda c: c[1]) if cands else ('stable', float('inf'))
    verts = np.stack([A, v01, v12, v02])
    return dict(volume=V, areas=areas, mode=mode, fs=fs, verts=verts)


def voxel_run(case, vs, connectivity, gamma, phi_deg, c_kpa, tol_factor=0.6, min_voxels=8):
    poly = np.asarray(case['poly_yz'], float)
    box = np.asarray(case['box'], float)
    tunnel_mask, grid = quiet(build_voxel_masks, poly[:, 0], poly[:, 1], box, vs)
    A = np.asarray(case['apex'], np.float32)
    centers = np.repeat(A[None], 3, axis=0)
    normals = np.stack([unit(n) for n in case['normals']]).astype(np.float32)
    radii = np.full(3, 50.0, dtype=np.float32)
    state, owner = quiet(classify_voxels, grid, centers, normals, radii, tunnel_mask, tol_factor)
    labels, n_labels = quiet(run_cca, state, connectivity)
    blocks = quiet(filter_and_stat_blocks, labels, n_labels, state, grid,
                   min_voxels=min_voxels, connectivity=connectivity)
    out = []
    for b in blocks:
        st = analyze_block(b, labels, state, owner, normals, centers, grid,
                           gamma_kN_m3=gamma, phi_deg=phi_deg, cohesion_kPa=c_kpa)
        out.append({**b, **st})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--voxel-sizes', nargs='+', type=float, default=[0.4, 0.2, 0.1, 0.05])
    ap.add_argument('--gamma', type=float, default=26.0, help='단위중량 [kN/m3]')
    ap.add_argument('--phi', type=float, default=30.0, help='절리 마찰각 [deg]')
    ap.add_argument('--cohesion', nargs='+', type=float, default=[0.0, 20.0], help='절리 점착력 [kPa]')
    ap.add_argument('--outdir', default='storage/output/block_verification')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print('=' * 78)
    print('Test A: 6- vs 26-connectivity leak (single infinite plane, tol=0.6*vs, vs=0.2 m)')
    print('        correct component count = 2')
    rows_a = test_leak()
    for r in rows_a:
        print(f"  {r['normal']:<24} ||n||inf={r['norm_inf']:.3f} ||n||1={r['norm_1']:.3f} "
              f"slab/vs={r['slab_over_vs']}  ->  6-conn: {r['components_6']}   26-conn: {r['components_26']}")
    with open(os.path.join(args.outdir, 'test_a_connectivity_leak.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_a[0]))
        w.writeheader(); w.writerows(rows_a)

    print('=' * 78)
    print(f'Test B: analytic tetrahedral wedges  (gamma={args.gamma} kN/m3, phi={args.phi} deg)')
    rows_b = []
    for c_kpa in args.cohesion:
        for name, case in CASES.items():
            ref = analytic_reference(case, args.gamma, args.phi, c_kpa)
            assert np.all((ref['verts'] >= np.asarray(case['box'])[::2]) &
                          (ref['verts'] <= np.asarray(case['box'])[1::2])), f'{name}: wedge outside box'
            print(f"\n  {name}  c={c_kpa:g} kPa  | analytic: V={ref['volume']:.4f} m3  "
                  f"mode={ref['mode']}  FS={ref['fs']:.4f}")
            for conn in (6, 26):
                for vs in args.voxel_sizes:
                    blocks = voxel_run(case, vs, conn, args.gamma, args.phi, c_kpa)
                    if blocks:
                        b = max(blocks, key=lambda d: d['volume_m3'])
                        v_err = 100 * (b['volume_m3'] - ref['volume']) / ref['volume']
                        vc_err = 100 * (b['volume_corr_m3'] - ref['volume']) / ref['volume']
                        fs_err = (100 * (b['fs'] - ref['fs']) / ref['fs']
                                  if np.isfinite(ref['fs']) and ref['fs'] > 0 else float('nan'))
                        row = dict(case=name, cohesion_kPa=c_kpa, connectivity=conn, voxel_m=vs,
                                   n_blocks=len(blocks), V_ref=round(ref['volume'], 4),
                                   V_vox=round(b['volume_m3'], 4), V_err_pct=round(v_err, 2),
                                   V_corr=round(b['volume_corr_m3'], 4), V_corr_err_pct=round(vc_err, 2),
                                   mode_ref=ref['mode'], mode_vox=b['mode'],
                                   FS_ref=round(ref['fs'], 4), FS_vox=round(b['fs'], 4),
                                   FS_err_pct=round(fs_err, 2), removable=b['removable'],
                                   n_planes=b['n_planes'])
                    else:
                        row = dict(case=name, cohesion_kPa=c_kpa, connectivity=conn, voxel_m=vs,
                                   n_blocks=0, V_ref=round(ref['volume'], 4), V_vox=None,
                                   V_err_pct=None, V_corr=None, V_corr_err_pct=None, mode_ref=ref['mode'], mode_vox=None,
                                   FS_ref=round(ref['fs'], 4), FS_vox=None, FS_err_pct=None,
                                   removable=None, n_planes=None)
                    rows_b.append(row)
                    print(f"    {conn:>2}-conn vs={vs:<5} blocks={row['n_blocks']}  "
                          f"V={row['V_vox']}  dV={row['V_err_pct']}%  dVcorr={row['V_corr_err_pct']}%  "
                          f"mode={row['mode_vox']}  "
                          f"FS={row['FS_vox']}  dFS={row['FS_err_pct']}%")
    with open(os.path.join(args.outdir, 'test_b_analytic_wedges.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_b[0]))
        w.writeheader(); w.writerows(rows_b)
    print(f"\nCSV written to {args.outdir}/")


if __name__ == '__main__':
    main()
