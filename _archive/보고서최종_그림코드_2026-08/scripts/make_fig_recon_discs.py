"""보고서 그림 6-12(복원 균열 3D 렌더) 재생성 — v1 복원 결과.

reconstructed_discs.csv 를 읽어 절리군별로 PyVista 어두운 배경 렌더를 저장한다.
색: determined=초록, shrinkage=주황, lower_bound=흰색 (보고서 그림 6-12와 동일 규약).

실행:
    python scripts/make_fig_recon_discs.py \
        [--recon-csv storage/output/pipeline_v1_laxemar/reconstruct/reconstructed_discs.csv] \
        [--sets 4 5] [--outdir docs/figures] [--prefix fig_recon_discs]
출력:
    <outdir>/<prefix>_set<N>_v1.png
"""
import argparse
import csv
import os

import numpy as np
import pyvista as pv

STATUS_COLOR = {"determined": "#2ecc71", "shrinkage": "#e67e22", "lower_bound": "#f5f5f5"}
STATUS_OPACITY = {"determined": 0.85, "shrinkage": 0.55, "lower_bound": 0.35}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon-csv",
                    default="storage/output/pipeline_v1_laxemar/reconstruct/reconstructed_discs.csv")
    ap.add_argument("--sets", nargs="+", type=int, default=[4, 5])
    ap.add_argument("--outdir", default="docs/figures")
    ap.add_argument("--prefix", default="fig_recon_discs")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.recon_csv, encoding="utf-8")))
    os.makedirs(args.outdir, exist_ok=True)

    for set_id in args.sets:
        sub = [r for r in rows if int(r["set_id"]) == set_id]
        if not sub:
            print(f"[skip] set {set_id}: disc 없음")
            continue
        pl = pv.Plotter(off_screen=True, window_size=(1100, 1100))
        pl.set_background("#14181f")
        counts = {}
        for r in sub:
            status = r.get("radius_status", "lower_bound")
            counts[status] = counts.get(status, 0) + 1
            center = np.array([float(r["cx"]), float(r["cy"]), float(r["cz"])])
            normal = np.array([float(r["nx"]), float(r["ny"]), float(r["nz"])])
            radius = float(r["radius"])
            disc = pv.Polygon(center=center, radius=radius, normal=normal, n_sides=48)
            pl.add_mesh(disc, color=STATUS_COLOR.get(status, "white"),
                        opacity=STATUS_OPACITY.get(status, 0.5),
                        show_edges=True, edge_color="#aaaaaa", line_width=0.5)
        pl.camera_position = "iso"
        out = os.path.join(args.outdir, f"{args.prefix}_set{set_id}_v1.png")
        pl.screenshot(out)
        pl.close()
        print(f"[*] written: {out}  (set {set_id}, disc {len(sub)}개, {counts})")


if __name__ == "__main__":
    main()
