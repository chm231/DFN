"""
extract_overbreak_map.py – 터널 스캔 점군 → 여굴 깊이 지도 (x, θ) [로컬 실행용]

기밀 스캔 원자료는 로컬에만 두고, 이 스크립트가 만든 격자 지도(npz/csv)만 외부로 가져간다.
출력에는 점 좌표가 없고, 격자 경계(x: 기준 측점으로부터 거리, θ: 각도)와 격자별 통계만 있다.

좌표 (파이프라인과 동일, 오른손 좌표계):
  x = 굴진 방향 (중심선 p0 → p1), z = 연직 위, y = z × x (굴진 방향을 바라볼 때 왼쪽)
  θ = atan2(z − z_c, y − y_c) [deg], (y_c, z_c) = 설계 단면 다각형 꼭짓점 평균
      θ = 90° 천장, −90° 바닥, 0° = +y 측벽(왼쪽), ±180° = −y 측벽(오른쪽)
  여굴 깊이 = r_점 − r_설계(θ)  (단면 중심에서의 방사 거리 차; + 여굴, − 미굴)
  x 는 p0(기준 측점)에서 잰 거리. 막장 trace 자료의 막장면 x 와 같은 원점을 쓰면 바로 비교된다.

가정: 구간 내 터널은 직선(중심선 = p0→p1 직선), 설계 단면은 구간 전체에서 동일하고 볼록(convex).

입력:
  --points  : .xyz/.txt/.csv (공백/쉼표 구분, 앞 3열 = X Y Z, 헤더 줄은 --skip-rows) 또는 .npy (N×3 이상)
  --design  : 설계 단면 [y, z] CSV (중심선 기준, 헤더 없이 두 열, 한 바퀴 순서)
  --axis-p0 / --axis-p1 : 스캔 좌표계의 중심선 위 두 점. 생략하면 점군이 이미 로컬 [x, y, z] 라고 본다.

예:
  python extract_overbreak_map.py --points scan.xyz --design design_yz.csv ^
      --axis-p0 1000.0 2000.0 50.0 --axis-p1 1100.0 2000.0 50.5 --x-range 0 60 --out overbreak_map

필요 라이브러리: numpy (matplotlib 은 --png 에만 선택 사용)
"""

from __future__ import annotations
import argparse
import itertools
import json
import os

import numpy as np

THRESHOLDS_M = (0.05, 0.10, 0.20, 0.30)


def read_chunks(path, skip_rows, chunk):
    """점군을 (n,3) 배열 조각으로 순차 반환 (큰 파일도 메모리 일정)."""
    if path.lower().endswith('.npy'):
        arr = np.load(path, mmap_mode='r')
        for s in range(0, arr.shape[0], chunk):
            yield np.asarray(arr[s:s + chunk, :3], dtype=np.float64)
        return
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        for _ in range(skip_rows):
            next(fh)
        while True:
            lines = list(itertools.islice(fh, chunk))
            if not lines:
                return
            rows = [ln.replace(',', ' ').split()[:3] for ln in lines if ln.strip()]
            yield np.asarray(rows, dtype=np.float64)


def local_frame(p0, p1):
    """스캔 좌표 → 로컬 [x, y, z] 변환 행렬 (행 = ex, ey, ez)."""
    ex = (p1 - p0) / np.linalg.norm(p1 - p0)
    ey = np.cross([0.0, 0.0, 1.0], ex)
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    return np.stack([ex, ey, ez])


def design_radius(theta_rad, poly_yz, center):
    """중심에서 θ 방향 반직선이 설계 다각형과 만나는 거리 (볼록 다각형 가정)."""
    d = np.stack([np.cos(theta_rad), np.sin(theta_rad)], axis=1)          # (n,2)
    A = poly_yz - center
    B = np.roll(A, -1, axis=0)
    E = B - A
    r = np.full(len(theta_rad), np.inf)
    for a, e in zip(A, E):
        den = d[:, 0] * (-e[1]) - d[:, 1] * (-e[0])                     # det([d, -e])
        ok = np.abs(den) > 1e-12
        t = np.where(ok, (a[0] * (-e[1]) - a[1] * (-e[0])) / np.where(ok, den, 1), np.inf)
        u = np.where(ok, (d[:, 0] * a[1] - d[:, 1] * a[0]) / np.where(ok, den, 1), -1)
        hit = ok & (t > 0) & (u >= 0) & (u <= 1)
        r = np.where(hit, np.minimum(r, t), r)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--points', required=True)
    ap.add_argument('--design', required=True, help='설계 단면 [y, z] CSV (중심선 기준)')
    ap.add_argument('--axis-p0', nargs=3, type=float, default=None, help='중심선 기준 측점 (스캔 좌표)')
    ap.add_argument('--axis-p1', nargs=3, type=float, default=None, help='중심선 진행 방향 점 (스캔 좌표)')
    ap.add_argument('--x-range', nargs=2, type=float, required=True, help='사용할 x 구간 [m] (p0 기준)')
    ap.add_argument('--dx', type=float, default=0.25, help='x 격자 [m]')
    ap.add_argument('--dtheta', type=float, default=2.0, help='θ 격자 [deg]')
    ap.add_argument('--max-abs-dev', type=float, default=2.0,
                    help='|r_점 − r_설계| 가 이보다 큰 점은 버림 [m] (장비·버력 등 잡음)')
    ap.add_argument('--skip-rows', type=int, default=0, help='텍스트 파일 헤더 줄 수')
    ap.add_argument('--chunk', type=int, default=2_000_000)
    ap.add_argument('--out', default='overbreak_map', help='출력 파일 접두어')
    ap.add_argument('--png', action='store_true', help='확인용 그림 저장 (matplotlib 필요)')
    args = ap.parse_args()

    poly = np.loadtxt(args.design, delimiter=',' if args.design.endswith('.csv') else None, ndmin=2)[:, :2]
    center = poly.mean(axis=0)
    R = p0 = None
    if args.axis_p0 is not None:
        p0, p1 = np.array(args.axis_p0), np.array(args.axis_p1)
        R = local_frame(p0, p1)

    acc = None
    n_read = n_used = 0
    for pts in read_chunks(args.points, args.skip_rows, args.chunk):
        n_read += len(pts)
        loc = (pts - p0) @ R.T if R is not None else pts                  # 로컬 [x, y, z]
        if acc is None:
            x_edges = np.arange(args.x_range[0], args.x_range[1] + args.dx / 2, args.dx)
            t_edges = np.arange(-180.0, 180.0 + args.dtheta / 2, args.dtheta)
            shape = (len(x_edges) - 1, len(t_edges) - 1)
            acc = dict(count=np.zeros(shape, np.int64), sum=np.zeros(shape),
                       max=np.full(shape, -np.inf),
                       **{f'n_gt_{t:g}': np.zeros(shape, np.int64) for t in THRESHOLDS_M})
        dy, dz = loc[:, 1] - center[0], loc[:, 2] - center[1]
        theta = np.arctan2(dz, dy)
        dev = np.hypot(dy, dz) - design_radius(theta, poly, center)
        ix = np.floor((loc[:, 0] - x_edges[0]) / args.dx).astype(np.int64)
        it = np.floor((np.degrees(theta) + 180.0) / args.dtheta).astype(np.int64)
        it = np.clip(it, 0, shape[1] - 1)
        ok = (ix >= 0) & (ix < shape[0]) & np.isfinite(dev) & (np.abs(dev) <= args.max_abs_dev)
        ix, it, dev = ix[ok], it[ok], dev[ok]
        n_used += len(dev)
        np.add.at(acc['count'], (ix, it), 1)
        np.add.at(acc['sum'], (ix, it), dev)
        np.maximum.at(acc['max'], (ix, it), dev)
        for t in THRESHOLDS_M:
            np.add.at(acc[f'n_gt_{t:g}'], (ix, it), (dev > t).astype(np.int64))
        print(f"  read {n_read:,} points, used {n_used:,}", flush=True)

    with np.errstate(invalid='ignore', divide='ignore'):
        mean = acc['sum'] / acc['count']
        frac = {f'frac_gt_{t:g}m': acc[f'n_gt_{t:g}'] / acc['count'] for t in THRESHOLDS_M}
    mx = np.where(acc['count'] > 0, acc['max'], np.nan)

    np.savez_compressed(args.out + '.npz', x_edges=x_edges, theta_edges=t_edges,
                        count=acc['count'], mean_overbreak_m=mean, max_overbreak_m=mx, **frac)
    meta = dict(
        description='overbreak depth map on unrolled tunnel wall; no point coordinates included',
        x_definition='distance along centerline from axis-p0 [m]' if R is not None else 'input x [m]',
        theta_definition='atan2(z - z_c, y - y_c) deg; 90 crown, -90 invert, 0 = +y wall (left looking along +x)',
        overbreak_definition='radial distance from design-section centroid minus design radius [m]; + = overbreak',
        dx_m=args.dx, dtheta_deg=args.dtheta, x_range_m=[float(x_edges[0]), float(x_edges[-1])],
        max_abs_dev_m=args.max_abs_dev, thresholds_m=list(THRESHOLDS_M),
        n_points_read=int(n_read), n_points_used=int(n_used),
        n_cells=int(acc['count'].size), n_cells_with_points=int((acc['count'] > 0).sum()),
    )
    with open(args.out + '_meta.json', 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"\n[out] {args.out}.npz  ({os.path.getsize(args.out + '.npz') / 1e6:.2f} MB), {args.out}_meta.json")
    print(f"  cells with points: {meta['n_cells_with_points']:,} / {meta['n_cells']:,}")
    print(f"  mean overbreak (all used points): {np.nansum(acc['sum']) / max(n_used, 1):.3f} m")

    if args.png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        im = ax.pcolormesh(x_edges, t_edges, np.ma.masked_invalid(mx).T, cmap='viridis')
        fig.colorbar(im, ax=ax, label='max overbreak in cell [m]')
        ax.set_xlabel('x along centerline from p0 [m]')
        ax.set_ylabel('θ [deg]  (90 crown, 0 +y wall, -90 invert)')
        fig.savefig(args.out + '.png', dpi=150)
        print(f"  figure: {args.out}.png")


if __name__ == '__main__':
    main()
