"""
compare_truth_vs_ensembles.py – 참값 DFN vs 조건부 / 비조건부 앙상블 블록 예측 비교 (논문 4단계)

질문: 막장면 관측(복원 균열 + 조건화)이 전방 블록 위험 예측을 얼마나, 어디까지 개선하는가?

  * 참값: 합성 DFN(h5) 전체에서 같은 도메인(export JSON 의 meta.domain)과 만나는 원판을 골라
          detect_blocks.detect() 를 동일 설정으로 적용.
  * 예측: run_block_ensemble.py 의 wall_probability.npz (조건부 / --unconditioned).
          벽면 층 복셀별 p = (그 복셀이 블록 / FS<1 블록에 속한 실현 수) / N.
  * 채점: 마지막 막장면으로부터의 거리 구간(기본 2 m)별로
          Brier score (낮을수록 좋음), BSS = 1 - BS_cond / BS_uncond (>0 이면 조건화가 개선),
          AUC (참값 사건 복셀을 높은 p 로 순위화하는 능력, 0.5 = 무작위),
          평균 예측 p vs 참값 사건 비율 (전체 보정).
    블록 단위로는 구간별 참값 블록 수가 앙상블 5–95% 범위 안에 드는지 본다.

주의: 참값은 실현 1개이므로 복셀 단위 점수는 표본 변동이 크다. 결론은 여러 참값 시드로 확인해야 한다.
      Laxemar Set 4(지수분포)는 파이프라인에서 확률 생성하지 않으므로(관측 복원만) 참값에는
      있고 앙상블 전방에는 없다 → --truth-exclude-sets 4 로 그 영향을 분리해 볼 수 있다.

실행 (handoffv1 폴더에서):
    PYTHONPATH=. python scripts/compare_truth_vs_ensembles.py --pipeline-dir demo_output/seed42
"""

from __future__ import annotations
import argparse
import contextlib
import csv
import io
import json
import os

import h5py
import numpy as np
from scipy.stats import rankdata

from dfn_analysis.detect_blocks import detect
from dfn_analysis.export_domain_dfn_json import distance_to_domain


def load_truth_json(dfn_h5, dom_meta, exclude_sets):
    """참값 DFN 중 도메인과 만나는 원판만 export JSON 형식 dict 로."""
    x0, x1 = dom_meta['x_range_m']
    poly = np.asarray(dom_meta['tunnel_polygon_yz_m'], float)
    halo = dom_meta['tunnel_halo_m']
    fr = []
    with h5py.File(dfn_h5, 'r') as f:
        n = f['fractures/centers'].shape[0]
        for s in range(0, n, 2_000_000):
            c = f['fractures/centers'][s:s + 2_000_000]
            r = f['fractures/radii'][s:s + 2_000_000].ravel()
            keep = distance_to_domain(c, x0, x1, poly, halo, 'box') <= r
            if not keep.any():
                continue
            nrm = f['fractures/normals'][s:s + 2_000_000][keep]
            sid = f['fractures/set_id'][s:s + 2_000_000].ravel()[keep]
            for ci, ni, ri, si in zip(c[keep], nrm, r[keep], sid):
                if int(si) in exclude_sets:
                    continue
                fr.append(dict(id=len(fr), set_id=int(si), label='truth',
                               center_xyz_m=ci.tolist(), normal_xyz=ni.tolist(), radius_m=float(ri)))
    return dict(meta=dict(domain=dom_meta), fractures=fr)


def auc(p, y):
    """Mann–Whitney AUC (동순위 평균 순위). 한 클래스뿐이면 nan."""
    y = y.astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float('nan')
    ranks = rankdata(p)                      # 동순위는 평균 순위
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pipeline-dir', required=True)
    ap.add_argument('--voxel-size', type=float, default=0.05)
    ap.add_argument('--band', type=float, default=2.0, help='막장면 거리 구간 폭 [m]')
    ap.add_argument('--cond-dir', default=None, help='조건부 앙상블 폴더 (기본: block_ensemble_vs<voxel>)')
    ap.add_argument('--uncond-dir', default=None, help='비조건부 앙상블 폴더 (기본: block_ensemble_uncond_vs<voxel>)')
    ap.add_argument('--truth-exclude-sets', nargs='*', type=int, default=[])
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()

    pdir = args.pipeline_dir
    ens = {'cond': args.cond_dir or os.path.join(pdir, f'block_ensemble_vs{args.voxel_size:g}'),
           'uncond': args.uncond_dir or os.path.join(pdir, f'block_ensemble_uncond_vs{args.voxel_size:g}')}
    meta = {k: json.load(open(os.path.join(v, 'ensemble_meta.json'), encoding='utf-8'))
            for k, v in ens.items()}
    npz = {k: np.load(os.path.join(v, 'wall_probability.npz')) for k, v in ens.items()}
    for k in ('voxel_size_m', 'tol_factor', 'min_voxels', 'stability'):
        assert meta['cond'][k] == meta['uncond'][k], f'ensemble settings differ: {k}'
    assert np.array_equal(npz['cond']['layer_idx'], npz['uncond']['layer_idx'])
    m = meta['cond']
    sfx = ('_noset' + ''.join(map(str, args.truth_exclude_sets))) if args.truth_exclude_sets else ''
    outdir = args.outdir or os.path.join(pdir, f'truth_comparison_vs{args.voxel_size:g}{sfx}')
    os.makedirs(outdir, exist_ok=True)

    # ── 참값 블록 ────────────────────────────────────────────────────────
    json0 = sorted(os.listdir(os.path.join(ens['cond'], 'json')))[0]
    dom = json.load(open(os.path.join(ens['cond'], 'json', json0), encoding='utf-8'))['meta']['domain']
    d_truth = load_truth_json(os.path.join(pdir, 'dfn_export_for_python.h5'), dom,
                              set(args.truth_exclude_sets))
    print(f"[truth] {len(d_truth['fractures']):,} true discs intersect the domain "
          f"(excluded sets: {args.truth_exclude_sets or 'none'})")
    with contextlib.redirect_stdout(io.StringIO()):
        rows, ctx = detect(d_truth, m['voxel_size_m'], m['tol_factor'], m['min_voxels'], 6, None,
                           m['stability']['gamma_kN_m3'], m['stability']['phi_deg'],
                           m['stability']['cohesion_kPa'])
    unst = [r for r in rows if r['fs'] != 'inf' and r['fs'] < 1.0]
    print(f"[truth] blocks={len(rows)}  FS<1={len(unst)}  "
          f"V_FS<1={sum(r['volume_corr_m3'] for r in unst):.3f} m3")

    layer_idx = npz['cond']['layer_idx']
    assert tuple(npz['cond']['grid_shape']) == ctx['labels'].shape
    lab = ctx['labels'].ravel()[layer_idx]
    y = {'block': np.isin(lab, [r['label'] for r in rows]),
         'unstable': np.isin(lab, [r['label'] for r in unst])}
    p = {(k, ev): npz[k][f'cnt_{"block" if ev == "block" else "unstable"}_layer'] / meta[k]['n_real']
         for k in ens for ev in y}

    ii = np.unravel_index(layer_idx, ctx['labels'].shape)[0]
    dist = ctx['grid']['xs'][ii] - dom['last_face_x_m']
    edges = np.arange(0, dist.max() + args.band, args.band)

    # ── 복셀 단위 점수 ───────────────────────────────────────────────────
    out = []
    print(f"\n{'event':<9}{'band [m]':<11}{'truth rate':>11}{'p_cond':>9}{'p_unc':>9}"
          f"{'BS_cond':>10}{'BS_unc':>10}{'BSS':>8}{'AUC_c':>8}{'AUC_u':>8}")
    for ev in ('block', 'unstable'):
        for lo, hi in list(zip(edges[:-1], edges[1:])) + [(0.0, edges[-1])]:
            sel = (dist >= lo) & (dist < hi)
            yy = y[ev][sel].astype(float)
            pc, pu = p[('cond', ev)][sel], p[('uncond', ev)][sel]
            bs_c, bs_u = np.mean((pc - yy) ** 2), np.mean((pu - yy) ** 2)
            row = dict(event=ev, band_m=f'{lo:g}-{hi:g}', n_voxels=int(sel.sum()),
                       truth_rate=round(yy.mean(), 4), mean_p_cond=round(pc.mean(), 4),
                       mean_p_uncond=round(pu.mean(), 4), BS_cond=round(bs_c, 5),
                       BS_uncond=round(bs_u, 5),
                       BSS_cond_vs_uncond=round(1 - bs_c / bs_u, 4) if bs_u > 0 else None,
                       AUC_cond=round(auc(pc, yy), 3), AUC_uncond=round(auc(pu, yy), 3))
            out.append(row)
            print(f"{ev:<9}{row['band_m']:<11}{row['truth_rate']:>11.4f}{row['mean_p_cond']:>9.4f}"
                  f"{row['mean_p_uncond']:>9.4f}{bs_c:>10.5f}{bs_u:>10.5f}"
                  f"{row['BSS_cond_vs_uncond'] if row['BSS_cond_vs_uncond'] is not None else float('nan'):>8.3f}"
                  f"{row['AUC_cond']:>8.3f}{row['AUC_uncond']:>8.3f}")
    with open(os.path.join(outdir, 'voxel_scores_by_distance.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)

    # ── 블록 단위: 구간별 FS<1 블록 수, 참값 vs 앙상블 분포 ─────────────
    print(f"\nFS<1 blocks per band (block centroid): truth vs ensemble mean [p5, p95]")
    blk = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        t = sum(lo <= r['cx'] - dom['last_face_x_m'] < hi for r in unst)
        row = dict(band_m=f'{lo:g}-{hi:g}', truth_n_fs_lt1=t)
        for k in ens:
            cnt = {s: 0 for s in meta[k]['seeds']}
            with open(os.path.join(ens[k], 'ensemble_blocks.csv'), encoding='utf-8') as fh:
                for r in csv.DictReader(fh):
                    if r['fs'] != 'inf' and float(r['fs']) < 1.0 and \
                            lo <= float(r['cx']) - dom['last_face_x_m'] < hi:
                        cnt[int(r['seed'])] += 1
            v = np.array(list(cnt.values()))
            row.update({f'{k}_mean': round(v.mean(), 2), f'{k}_p5': np.percentile(v, 5),
                        f'{k}_p95': np.percentile(v, 95),
                        f'{k}_P_ge1': round((v >= 1).mean(), 2)})
        blk.append(row)
        print(f"  {row['band_m']:<7} truth={t:2d}   cond {row['cond_mean']:5.2f} "
              f"[{row['cond_p5']:g},{row['cond_p95']:g}] P(>=1)={row['cond_P_ge1']:.2f}   "
              f"uncond {row['uncond_mean']:5.2f} [{row['uncond_p5']:g},{row['uncond_p95']:g}] "
              f"P(>=1)={row['uncond_P_ge1']:.2f}")
    with open(os.path.join(outdir, 'block_counts_by_distance.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(blk[0]))
        w.writeheader(); w.writerows(blk)
    with open(os.path.join(outdir, 'truth_blocks.csv'), 'w', newline='', encoding='utf-8') as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    json.dump(dict(pipeline_dir=os.path.abspath(pdir), truth_exclude_sets=args.truth_exclude_sets,
                   n_truth_discs=len(d_truth['fractures']), truth_n_blocks=len(rows),
                   truth_n_fs_lt1=len(unst), ensembles={k: v for k, v in ens.items()},
                   n_real={k: meta[k]['n_real'] for k in ens}, band_m=args.band,
                   voxel_size_m=m['voxel_size_m'], min_voxels=m['min_voxels'],
                   stability=m['stability'], git_commit=m['git_commit']),
              open(os.path.join(outdir, 'comparison_meta.json'), 'w'), indent=2)
    print(f"\nout: {outdir}")


if __name__ == '__main__':
    main()
