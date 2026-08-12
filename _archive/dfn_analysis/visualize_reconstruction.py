"""
원판 복원 검증 시각화 — 막장면 trace vs 복원 원판.

두 뷰:
  (1) 3D 개요: 막장면 윤곽 + 관측 trace(파랑) + 복원 원판 원(반지름 상태별 색).
  (2) 면별 2D 오버레이: 각 면에서 관측 trace(파랑) vs 복원 원판이 그 면에 남기는
      window-clip chord(빨강). 빨강이 파랑을 덮으면 복원이 관측을 재현한 것.

입력: reconstructed_discs.csv + trace h5(관측 trace·폴리곤·면위치).
좌표계: x=East(터널축), y=North, z=Up. 막장면 = x=상수 평면.
"""
import argparse
import csv
import math
import os

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dfn_analysis.generate_conditional_hidden_dfn import visible_trace_on_face, _ccw_polygon
from dfn_analysis.reconstruct_discs_from_traces import _plane_basis

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

RSTATUS_COLOR = {"determined": "#1a9850", "shrinkage": "#f46d43", "lower_bound": "#999999"}


def load_discs(path, sets=None, multi_only=False, max_discs=None):
    discs = []
    for r in csv.DictReader(open(path)):
        if sets and int(r["set_id"]) not in sets:
            continue
        if multi_only and int(r.get("n_faces", 1)) < 2:
            continue
        discs.append(dict(
            set_id=int(r["set_id"]),
            center=np.array([float(r["cx"]), float(r["cy"]), float(r["cz"])]),
            normal=np.array([float(r["nx"]), float(r["ny"]), float(r["nz"])]),
            radius=float(r["radius"]),
            rstatus=r.get("radius_status", "lower_bound"),
            n_faces=int(r.get("n_faces", 1)),
        ))
    if max_discs and len(discs) > max_discs:
        idx = np.linspace(0, len(discs) - 1, max_discs).astype(int)
        discs = [discs[i] for i in idx]
    return discs


def load_observed(trace_h5, sets=None):
    with h5py.File(trace_h5, "r") as f:
        g = f["traces"]
        p0 = g["p0_xyz"][...].astype(float)
        p1 = g["p1_xyz"][...].astype(float)
        sid = g["set_id"][...].astype(int)
        fx = g["face_x_m"][...].astype(float)
        poly = np.array(f["meta"]["tunnel_poly_yz"])
        faces = list(np.array(f["meta"]["face_x_positions_m"]).astype(float))
    keep = [i for i in range(len(sid)) if (not sets or sid[i] in sets)]
    return p0[keep], p1[keep], sid[keep], fx[keep], poly, faces


def _seg_dir(a, b):
    d = b - a
    nn = np.linalg.norm(d)
    return d / nn if nn > 1e-9 else d


def _point_seg_dist(p, a, b):
    ab = b - a
    L2 = float(ab @ ab)
    t = 0.0 if L2 < 1e-12 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def is_reproduced(o0, o1, chords, angle_tol_deg=15.0, dist_tol=0.3):
    """관측 trace(o0,o1)가 복원 chord 목록 중 하나로 재현되는가(2D y-z).
    방향 정렬 + 관측 중점이 복원 chord 선분에 근접 시 재현으로 판정."""
    od = _seg_dir(o0, o1)
    omid = 0.5 * (o0 + o1)
    cos_thr = math.cos(math.radians(angle_tol_deg))
    for r0, r1 in chords:
        rd = _seg_dir(r0, r1)
        if abs(float(od @ rd)) < cos_thr:
            continue
        if _point_seg_dist(omid, r0, r1) <= dist_tol:
            return True
    return False


def circle_points(center, normal, radius, n=48):
    u, v = _plane_basis(normal / np.linalg.norm(normal))
    th = np.linspace(0, 2 * np.pi, n)
    return center[None, :] + radius * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)


def plot_3d(discs, obs, faces, poly_ccw, out):
    p0, p1, sid, fx, poly, _ = obs
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    # 막장면 윤곽(각 x)
    for xf in faces:
        loop = np.column_stack([np.full(len(poly_ccw) + 1, xf),
                                np.r_[poly_ccw[:, 0], poly_ccw[0, 0]],
                                np.r_[poly_ccw[:, 1], poly_ccw[0, 1]]])
        ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color="0.6", lw=0.8)
    # 관측 trace
    for i in range(len(p0)):
        ax.plot([p0[i, 0], p1[i, 0]], [p0[i, 1], p1[i, 1]], [p0[i, 2], p1[i, 2]],
                color="#2166ac", lw=1.4, alpha=0.8)
    # 복원 원판(원)
    seen = set()
    for d in discs:
        c = circle_points(d["center"], d["normal"], d["radius"])
        col = RSTATUS_COLOR.get(d["rstatus"], "#999999")
        lbl = d["rstatus"] if d["rstatus"] not in seen else None
        seen.add(d["rstatus"])
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color=col, lw=0.8, alpha=0.55, label=lbl)
    ax.plot([], [], color="#2166ac", label="관측 trace")
    ax.set_xlabel("x — 터널축 [m]"); ax.set_ylabel("y — North [m]"); ax.set_zlabel("z — Up [m]")
    ax.set_title(f"복원 원판 vs 관측 trace (3D)  |  disc {len(discs)}개")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_box_aspect((max(faces) - min(faces) + 2,
                       np.ptp(poly_ccw[:, 0]), np.ptp(poly_ccw[:, 1])))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[3d]", out)


def plot_faces(discs, obs, faces, poly_ccw, outdir):
    """면 하나당 개별 figure 저장: 관측 trace(파랑) vs 복원 원판 chord(빨강)."""
    p0, p1, sid, fx, poly, _ = obs
    polyc = np.r_[poly_ccw, poly_ccw[:1]]
    for xf in faces:
        fig, ax = plt.subplots(figsize=(9, 7.5))
        ax.plot(polyc[:, 0], polyc[:, 1], color="0.5", lw=1.2, label="관측창")
        # 이 면의 복원 원판 chord (y-z, window-clip)
        chords = []
        for d in discs:
            seg = visible_trace_on_face(d["center"], d["normal"], d["radius"], xf, poly_ccw)
            if seg is not None:
                chords.append((seg[0][1:], seg[1][1:]))
        for r0, r1 in chords:
            ax.plot([r0[0], r1[0]], [r0[1], r1[1]], color="#d73027", lw=1.2, alpha=0.6)
        # 관측 trace: 재현/미재현 판정 후 색 구분
        n_obs = n_rep = 0
        for i in range(len(p0)):
            if abs(fx[i] - xf) >= 0.5:
                continue
            n_obs += 1
            o0, o1 = p0[i, 1:], p1[i, 1:]
            if is_reproduced(o0, o1, chords):
                ax.plot([o0[0], o1[0]], [o0[1], o1[1]], color="#4393c3", lw=1.6, alpha=0.75)
                n_rep += 1
            else:  # 미재현 강조
                ax.plot([o0[0], o1[0]], [o0[1], o1[1]], color="#111111", lw=2.6, alpha=0.95)
        n_unrep = n_obs - n_rep
        ax.plot([], [], color="#4393c3", lw=1.6, label=f"관측(재현됨) {n_rep}")
        ax.plot([], [], color="#111111", lw=2.6, label=f"관측(미재현) {n_unrep}")
        ax.plot([], [], color="#d73027", lw=1.2, label=f"복원 원판 chord {len(chords)}")
        rate = 100 * n_rep / n_obs if n_obs else 0.0
        ax.set_title(f"면 x={xf:g} m  |  관측 {n_obs}  →  재현 {n_rep} / 미재현 {n_unrep}  "
                     f"(재현율 {rate:.0f}%)", fontsize=13, fontweight="bold")
        ax.set_xlabel("y — North [m]"); ax.set_ylabel("z — Up [m]")
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        out = os.path.join(outdir, f"reconstruction_face_x{xf:g}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("[face]", out)


def main():
    ap = argparse.ArgumentParser(description="원판 복원 검증 시각화")
    ap.add_argument("--recon-csv", required=True)
    ap.add_argument("--trace-h5", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sets", nargs="+", type=int, default=None, help="특정 set만")
    ap.add_argument("--multi-face-only", action="store_true", help="다면 관통 disc만")
    ap.add_argument("--max-discs", type=int, default=None, help="3D 과밀 방지용 표본 수")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    obs = load_observed(args.trace_h5, set(args.sets) if args.sets else None)
    poly_ccw = _ccw_polygon(obs[4])
    faces = obs[5]
    discs = load_discs(args.recon_csv, set(args.sets) if args.sets else None,
                       args.multi_face_only, args.max_discs)
    print(f"복원 disc {len(discs)}개, 관측 trace {len(obs[0])}개, 면 {faces}")
    plot_3d(discs, obs, faces, poly_ccw, os.path.join(args.outdir, "reconstruction_3d.png"))
    plot_faces(discs, obs, faces, poly_ccw, args.outdir)


if __name__ == "__main__":
    main()
