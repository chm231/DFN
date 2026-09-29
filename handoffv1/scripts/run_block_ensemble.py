"""
run_block_ensemble.py – 조건부 DFN 앙상블 → 확률적 블록 해석 (논문 3단계)

한 번의 역산 결과(pipeline-dir: 복원 균열 + 역산 파라미터)를 고정하고, 미관측 균열의
확률 생성 시드만 바꿔 N 개의 조건부 DFN 실현을 만든다(export_domain_dfn_json --seed s).
각 실현에 detect_blocks.detect() 를 적용해 블록·파괴모드·FS 를 구하고 다음을 집계한다.

  * 실현별 요약: 블록 수, 제거가능 블록 수, FS<1 블록 수·부피, 최대 블록 부피
  * 블록 풀(pool): 전 실현의 블록 목록 (seed 열 포함)
  * 터널 벽면 확률 지도: 터널에 6-이웃으로 접한 암반 쪽 복셀 층(벽면 층)에서
      P_block(x, θ)    = 해당 위치가 (V ≥ min_volume) 블록에 속할 확률
      P_unstable(x, θ) = 해당 위치가 FS<1 블록에 속할 확률
    θ = atan2(z - z_c, y - y_c) [deg], (y_c, z_c) = 터널 단면 다각형 꼭짓점 평균.
    θ = 90° 천장(crown), -90° 바닥(invert), 0° = +y 측벽(굴진 방향을 볼 때 왼쪽),
    ±180° = -y 측벽(오른쪽). (x 굴진, z 위, 오른손 좌표계)
  * 수렴 확인: 실현 수 N/4, N/2, 3N/4, N 에서의 누적 평균

관측(복원) 균열은 모든 실현에서 동일하고, 미관측 균열만 달라진다.

실행 (handoffv1 폴더에서):
    PYTHONPATH=. python scripts/run_block_ensemble.py \
        --pipeline-dir demo_output/seed42 --n-real 50 --voxel-size 0.05
"""

from __future__ import annotations
import argparse
import contextlib
import csv
import io
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy import ndimage as ndi

from dfn_analysis.block_detector import STRUCT6
from dfn_analysis.detect_blocks import detect, git_commit
from dfn_analysis.tunnel_geometry import build_voxel_masks

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def wall_layer(tunnel_mask):
    """터널에 6-이웃으로 접한 비터널 복셀 (실현과 무관한 고정 기하)."""
    return ndi.binary_dilation(tunnel_mask, structure=STRUCT6) & ~tunnel_mask


def one_realization(job):
    seed, a = job
    json_path = os.path.join(a['outdir'], 'json', f'dfn_domain_seed{seed:04d}.json')
    if not os.path.exists(json_path):
        subprocess.run([sys.executable, '-m', 'dfn_analysis.export_domain_dfn_json',
                        '--pipeline-dir', a['pipeline_dir'], '--seed', str(seed),
                        '--rmax-local', str(a['rmax_local']), '--lmin-det', str(a['lmin_det']),
                        '--halo', str(a['halo']), '--ahead', str(a['ahead']), '--out', json_path]
                       + (['--unconditioned'] if a['unconditioned'] else [])
                       + (['--config', a['config']] if a['config'] else []),
                       check=True, cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT),
                       stdout=subprocess.DEVNULL)
    with open(json_path, encoding='utf-8') as fh:
        d = json.load(fh)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        rows, ctx = detect(d, a['voxel_size'], a['tol_factor'], a['min_voxels'], 6, None,
                           a['gamma'], a['phi'], a['cohesion'])
    labels = ctx['labels']
    layer_idx = np.flatnonzero(wall_layer(ctx['tunnel_mask']))
    lab_on_layer = labels.ravel()[layer_idx]
    block_lbls = np.array([r['label'] for r in rows], dtype=np.int64)
    unstable_lbls = np.array([r['label'] for r in rows if r['fs'] != 'inf' and r['fs'] < 1.0],
                             dtype=np.int64)
    hit_block = layer_idx[np.isin(lab_on_layer, block_lbls)]
    hit_unstable = layer_idx[np.isin(lab_on_layer, unstable_lbls)]
    for r in rows:
        r['seed'] = seed
    return seed, rows, hit_block, hit_unstable, len(d['fractures'])


def per_seed_summary(seed, rows, n_frac):
    fs = [r['fs'] for r in rows]
    unstable = [r for r, f in zip(rows, fs) if f != 'inf' and f < 1.0]
    v = [r['volume_corr_m3'] for r in rows]
    vu = [r['volume_corr_m3'] for r in unstable]
    return dict(seed=seed, n_fractures=n_frac, n_blocks=len(rows),
                n_removable=sum(r['removable'] for r in rows),
                n_fall=sum(r['mode'] == 'fall' for r in rows),
                n_slide=sum(r['mode'].startswith('slide') for r in rows),
                n_fs_lt1=len(unstable),
                V_total_m3=round(sum(v), 4), V_fs_lt1_m3=round(sum(vu), 4),
                V_max_m3=round(max(v), 4) if v else 0.0,
                V_max_fs_lt1_m3=round(max(vu), 4) if vu else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pipeline-dir', required=True, help='역산 파이프라인 출력 폴더 (run_demo 출력 등)')
    ap.add_argument('--n-real', type=int, default=50, help='조건부 DFN 실현 수')
    ap.add_argument('--seed0', type=int, default=1, help='첫 생성 시드 (seed0 .. seed0+n-1)')
    ap.add_argument('--voxel-size', type=float, default=0.05)
    ap.add_argument('--tol-factor', type=float, default=0.6)
    ap.add_argument('--min-volume', type=float, default=0.01, help='블록 최소 부피 [m3] (복셀 부피 기준)')
    ap.add_argument('--gamma', type=float, default=26.0)
    ap.add_argument('--phi', type=float, default=30.0)
    ap.add_argument('--cohesion', type=float, default=0.0)
    ap.add_argument('--rmax-local', type=float, default=25.0, help='export: 확률 생성 반지름 상한 [m]')
    ap.add_argument('--lmin-det', type=float, default=0.5, help='export: 검출 하한 [m]')
    ap.add_argument('--halo', type=float, default=5.0, help='export: 터널 단면 halo [m]')
    ap.add_argument('--ahead', type=float, default=10.0, help='export: 마지막 막장면 전방 길이 [m]')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--config', default=None,
                    help='export 에 넘길 dataset config JSON (set별 trend/plunge/kappa → 확률 생성 방향)')
    ap.add_argument('--unconditioned', action='store_true',
                    help='비교용 비조건부 앙상블 (복원 균열 없음, 관측 막장면 조건화 없음)')
    ap.add_argument('--outdir', default=None, help='기본: <pipeline-dir>/block_ensemble_vs<voxel>')
    args = ap.parse_args()

    a = dict(vars(args))
    a['pipeline_dir'] = os.path.abspath(args.pipeline_dir)
    a['config'] = os.path.abspath(args.config) if args.config else None
    a['outdir'] = os.path.abspath(args.outdir or os.path.join(
        args.pipeline_dir,
        f"block_ensemble{'_uncond' if args.unconditioned else ''}_vs{args.voxel_size:g}"))
    a['min_voxels'] = max(1, int(np.ceil(args.min_volume / args.voxel_size ** 3)))
    os.makedirs(os.path.join(a['outdir'], 'json'), exist_ok=True)

    seeds = list(range(args.seed0, args.seed0 + args.n_real))
    print(f"[ensemble] {len(seeds)} realizations (seeds {seeds[0]}..{seeds[-1]}), "
          f"voxel={args.voxel_size} m, 6-conn, min_volume={args.min_volume} m3 "
          f"(min_voxels={a['min_voxels']}), gamma={args.gamma}, phi={args.phi}, c={args.cohesion}")
    if args.unconditioned:
        print(f"[ensemble] UNCONDITIONED: no reconstructed discs, no face conditioning; "
              f"inverted params from {a['pipeline_dir']}")
    else:
        print(f"[ensemble] fixed: observed/reconstructed discs + inverted params from {a['pipeline_dir']}")
    print(f"[ensemble] varied: unobserved (stochastic) discs only")

    t0 = time.time()
    results = []
    with Pool(args.workers) as pool:
        for k, res in enumerate(pool.imap(one_realization, [(s, a) for s in seeds]), 1):
            results.append(res)
            s = per_seed_summary(res[0], res[1], res[4])
            print(f"  [{k:3d}/{len(seeds)}] seed={s['seed']:4d} frac={s['n_fractures']:5d} "
                  f"blocks={s['n_blocks']:3d} removable={s['n_removable']:3d} "
                  f"FS<1={s['n_fs_lt1']:3d} V_FS<1={s['V_fs_lt1_m3']:.3f} m3  "
                  f"({time.time() - t0:.0f} s)", flush=True)

    # ── 격자·벽면 층 재구성 (실현과 무관한 고정 기하, 균열 없이 1회) ──────────────────────────────────────
    with open(os.path.join(a['outdir'], 'json', f'dfn_domain_seed{seeds[0]:04d}.json'),
              encoding='utf-8') as fh:
        d0 = json.load(fh)
    dom = d0['meta']['domain']
    yz = dom['yz_bounds_m']
    box = np.array([*dom['x_range_m'], yz['y_min'], yz['y_max'], yz['z_min'], yz['z_max']], float)
    poly = np.asarray(dom['tunnel_polygon_yz_m'], float)
    with contextlib.redirect_stdout(io.StringIO()):
        tunnel_mask, grid = build_voxel_masks(poly[:, 0], poly[:, 1], box, args.voxel_size)
    layer = wall_layer(tunnel_mask)
    layer_idx = np.flatnonzero(layer)
    cnt_block = np.zeros(layer.size, np.int32)
    cnt_unst = np.zeros(layer.size, np.int32)
    for _, _, hb, hu, _ in results:
        cnt_block[hb] += 1
        cnt_unst[hu] += 1

    ii, jj, kk = np.unravel_index(layer_idx, layer.shape)
    x = grid['xs'][ii]
    yc, zc = poly.mean(axis=0)
    theta = np.degrees(np.arctan2(grid['zs'][kk] - zc, grid['ys'][jj] - yc))
    x_edges = np.arange(grid['xs'][0] - args.voxel_size / 2, grid['xs'][-1] + args.voxel_size, 0.25)
    t_edges = np.arange(-180, 181, 4.0)
    n_bin = np.histogram2d(x, theta, [x_edges, t_edges])[0]
    N = len(results)
    with np.errstate(invalid='ignore', divide='ignore'):
        P_block = np.histogram2d(x, theta, [x_edges, t_edges], weights=cnt_block[layer_idx])[0] / (n_bin * N)
        P_unst = np.histogram2d(x, theta, [x_edges, t_edges], weights=cnt_unst[layer_idx])[0] / (n_bin * N)
    np.savez_compressed(os.path.join(a['outdir'], 'wall_probability.npz'),
                        x_edges=x_edges, theta_edges=t_edges, P_block=P_block, P_unstable=P_unst,
                        n_voxels_per_bin=n_bin, n_real=N, grid_shape=np.array(layer.shape),
                        layer_idx=layer_idx, cnt_block_layer=cnt_block[layer_idx],
                        cnt_unstable_layer=cnt_unst[layer_idx])

    # ── 표 출력 ─────────────────────────────────────────────────────────
    pooled = [r for _, rows, _, _, _ in results for r in rows]
    summ = [per_seed_summary(s, rows, nf) for s, rows, _, _, nf in results]
    with open(os.path.join(a['outdir'], 'ensemble_blocks.csv'), 'w', newline='', encoding='utf-8') as fh:
        if pooled:
            w = csv.DictWriter(fh, fieldnames=['seed'] + [k for k in pooled[0] if k != 'seed'])
            w.writeheader(); w.writerows(pooled)
    with open(os.path.join(a['outdir'], 'ensemble_per_seed.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0]))
        w.writeheader(); w.writerows(summ)

    def stats(key):
        v = np.array([s[key] for s in summ], float)
        return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, np.percentile(v, 5), np.percentile(v, 95)

    p_any = np.mean([s['n_fs_lt1'] > 0 for s in summ])
    meta = dict(
        pipeline_dir=a['pipeline_dir'], git_commit=git_commit(), n_real=N, seeds=seeds,
        unconditioned=args.unconditioned, config=args.config,
        voxel_size_m=args.voxel_size, tol_factor=args.tol_factor, connectivity=6,
        min_volume_m3=args.min_volume, min_voxels=a['min_voxels'],
        stability=dict(gamma_kN_m3=args.gamma, phi_deg=args.phi, cohesion_kPa=args.cohesion),
        export=dict(rmax_local_m=args.rmax_local, lmin_det_m=args.lmin_det, halo_m=args.halo,
                    ahead_m=args.ahead),
        wall_layer='non-tunnel voxels 6-adjacent to tunnel; theta=atan2(z-zc, y-yc), crown=90 deg',
        P_any_block_fs_lt1=float(p_any),
        mean_std_p5_p95={k: [round(float(v), 4) for v in stats(k)]
                         for k in ('n_blocks', 'n_removable', 'n_fs_lt1', 'V_total_m3',
                                   'V_fs_lt1_m3', 'V_max_m3', 'V_max_fs_lt1_m3')},
        elapsed_sec=round(time.time() - t0, 1),
    )
    with open(os.path.join(a['outdir'], 'ensemble_meta.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    print('\n' + '=' * 72)
    print(f"  N={N}  P(at least one FS<1 block) = {p_any:.2f}")
    for k, (m, sd, p5, p95) in zip(meta['mean_std_p5_p95'], map(stats, meta['mean_std_p5_p95'])):
        print(f"  {k:<16} mean={m:8.3f}  sd={sd:7.3f}  p5={p5:7.3f}  p95={p95:7.3f}")
    print("  convergence (running mean of n_fs_lt1 / V_fs_lt1_m3 / P(any FS<1)):")
    for n in sorted({max(1, N // 4), max(1, N // 2), max(1, 3 * N // 4), N}):
        sub = summ[:n]
        print(f"    N={n:4d}  {np.mean([s['n_fs_lt1'] for s in sub]):6.3f}  "
              f"{np.mean([s['V_fs_lt1_m3'] for s in sub]):7.4f}  "
              f"{np.mean([s['n_fs_lt1'] > 0 for s in sub]):5.2f}")
    print(f"  wall map: max P_block={np.nanmax(P_block):.2f}  max P_unstable={np.nanmax(P_unst):.2f}")
    print(f"  out: {a['outdir']}")
    print('=' * 72)

    plot_wall_map(x_edges, t_edges, P_block, P_unst, N, args,
                  os.path.join(a['outdir'], 'wall_probability.png'))


def plot_wall_map(x_edges, t_edges, P_block, P_unst, N, args, path):
    """터널 벽면 펼침 확률 지도 (x vs θ). 단일 색상(blue) 순차 컬러맵."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ramp = ['#ffffff', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
    cmap = LinearSegmentedColormap.from_list('seq_blue', ramp)
    cmap.set_bad('#e6e6e6')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, constrained_layout=True)
    for ax, P, title in ((axes[0], P_block, f'P(block, V ≥ {args.min_volume:g} m³)'),
                         (axes[1], P_unst, 'P(block with FS < 1)')):
        vmax = max(np.nanmax(P), 1e-3)
        im = ax.pcolormesh(x_edges, t_edges, np.ma.masked_invalid(P).T, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('x — tunnel advance [m] (global)')
        for t, name in ((90, 'crown'), (0, '+y wall (left)'), (-90, 'invert'), (180, '−y wall (right)')):
            ax.axhline(t, color='#9a9a9a', lw=0.6, ls=':')
            ax.text(x_edges[-1], t, f' {name}', va='center', fontsize=7, color='#555555')
        cb = fig.colorbar(im, ax=ax, pad=0.12)
        cb.set_label('probability', fontsize=8)
    axes[0].set_ylabel('θ around tunnel section [deg]  (atan2(z−z_c, y−y_c))')
    axes[0].set_yticks([-180, -90, 0, 90, 180])
    fig.suptitle(f'Unrolled tunnel wall — {N} conditional DFN realizations, voxel {args.voxel_size:g} m, '
                 f'6-connectivity, φ={args.phi:g}°, c={args.cohesion:g} kPa (gray = no wall voxel)',
                 fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  figure: {path}")


if __name__ == '__main__':
    main()
