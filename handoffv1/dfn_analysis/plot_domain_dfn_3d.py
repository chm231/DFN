"""3D view of the exported domain DFN (observed vs unobserved).

export_domain_dfn_json.py 가 만든 JSON 을 읽어 도메인·터널과 함께 원판을 그린다.
관측 복원 균열과 확률 생성 균열을 색으로 구분하고, 기굴착 구간에서는 복원 균열이
터널 천장·측벽에 남기는 절리선을 함께 표시하는 것이 이 그림의 요점이다.

원판은 도메인 안쪽이면서 터널 공동 바깥인 부분만 그린다. 그래야 균열이 벽면에
드러나는 모습(daylighting)이 보인다.

시각화 전용 — 기하는 계산하지도 바꾸지도 않는다 (CLAUDE.md §12).
좌표 규약: x = 터널 굴진 방향(East), y = North, z = Up.

Usage
-----
    python dfn_analysis/plot_domain_dfn_3d.py --json <경로>          # PNG 저장
    python dfn_analysis/plot_domain_dfn_3d.py --json <경로> --window # 대화형 창
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv

REPO = Path(__file__).resolve().parent.parent

C_OBSERVED = "#1f5fa8"     # 관측 복원 균열
C_UNOBSERVED = "#d98a3a"   # 확률 생성 균열
C_TUNNEL = "#9a9a9a"
C_FACE = "#5a5a5a"
C_DOMAIN = "#222222"
C_WALLTRACE = "#c1272d"    # 벽면 절리선


def drop_repeated_vertices(poly_yz: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """닫는 점이 중복 저장된 다각형에서 길이 0 인 변을 제거한다."""
    keep = np.linalg.norm(np.roll(poly_yz, -1, axis=0) - poly_yz, axis=1) > tol
    return poly_yz[keep]


def signed_poly_dist(pts_yz: np.ndarray, poly_yz: np.ndarray) -> np.ndarray:
    """다각형까지의 부호 거리(내부 음수)."""
    A = poly_yz
    B = np.roll(A, -1, axis=0)
    AB = B - A
    AP = pts_yz[:, None, :] - A[None, :, :]
    t = np.clip(np.einsum("nmj,mj->nm", AP, AB) / np.einsum("mj,mj->m", AB, AB), 0.0, 1.0)
    proj = A[None, :, :] + t[:, :, None] * AB[None, :, :]
    d = np.linalg.norm(pts_yz[:, None, :] - proj, axis=2).min(axis=1)

    y, z = pts_yz[:, 0], pts_yz[:, 1]
    inside = np.zeros(len(pts_yz), dtype=bool)
    for i in range(len(A)):
        y1, z1 = A[i]
        y2, z2 = B[i]
        straddles = (z1 > z) != (z2 > z)
        dz = (z2 - z1) if abs(z2 - z1) > 1e-300 else 1e-300  # 수평 변은 straddles=False
        inside ^= straddles & (y < (y2 - y1) * (z - z1) / dz + y1)
    return np.where(inside, -d, d)


def sdf_domain_minus_tunnel(pts: np.ndarray, box: dict, poly_yz: np.ndarray,
                            tun_lo: float, tun_hi: float) -> np.ndarray:
    """도메인 직육면체 안이면서 터널 공동 밖인 영역의 부호 거리(내부 음수)."""
    d_box = np.maximum.reduce([
        box["x_min"] - pts[:, 0], pts[:, 0] - box["x_max"],
        box["y_min"] - pts[:, 1], pts[:, 1] - box["y_max"],
        box["z_min"] - pts[:, 2], pts[:, 2] - box["z_max"],
    ])
    d_tunnel = np.maximum.reduce([
        tun_lo - pts[:, 0], pts[:, 0] - tun_hi,
        signed_poly_dist(pts[:, 1:3], poly_yz),
    ])
    return np.maximum(d_box, -d_tunnel)


def prism(poly_yz: np.ndarray, x0: float, x1: float) -> pv.PolyData:
    """yz 다각형을 x 방향으로 밀어낸 닫힌 기둥."""
    pts = np.column_stack([np.full(len(poly_yz), x0), poly_yz])
    face = pv.PolyData(pts, faces=np.concatenate([[len(pts)], np.arange(len(pts))]))
    return face.extrude((x1 - x0, 0.0, 0.0), capping=True).triangulate().clean()


def ring(poly_yz: np.ndarray, x: float) -> pv.PolyData:
    """x = const 평면 위의 닫힌 다각형 윤곽선."""
    pts = np.column_stack([np.full(len(poly_yz) + 1, x),
                           np.vstack([poly_yz, poly_yz[:1]])])
    return pv.lines_from_points(pts)


def disc_mesh(centers, normals, radii, seg: int = 28) -> pv.PolyData:
    """원판들을 하나의 PolyData 로 합친다."""
    ref = np.where(np.abs(normals[:, 2:3]) > 0.95,
                   np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    u = np.cross(ref, normals); u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(normals, u)
    th = np.linspace(0.0, 2.0 * np.pi, seg, endpoint=False)
    ring_pts = (np.cos(th)[None, :, None] * u[:, None, :]
                + np.sin(th)[None, :, None] * v[:, None, :])
    pts = (centers[:, None, :] + radii[:, None, None] * ring_pts).reshape(-1, 3)
    faces = np.hstack([np.full((len(centers), 1), seg),
                       np.arange(len(centers) * seg).reshape(len(centers), seg)])
    return pv.PolyData(pts, faces=faces.ravel())


def trace_lines(traces) -> pv.PolyData | None:
    """벽면 절리선 선분들을 하나의 선 메쉬로."""
    if not traces:
        return None
    segs = [pv.Line(tuple(t["p0_xyz_m"]), tuple(t["p1_xyz_m"])) for t in traces]
    return pv.merge(segs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--min-radius", type=float, default=1.5,
                    help="이 반지름 미만 균열은 그리지 않는다(가독성) [m].")
    ap.add_argument("--window", action="store_true", help="대화형 창도 함께 띄운다.")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    doc = json.load(open(args.json, encoding="utf-8"))
    dom = doc["meta"]["domain"]
    if dom.get("shape") != "box":
        raise SystemExit("이 스크립트는 --domain-shape box 로 만든 JSON 만 그린다.")
    poly = drop_repeated_vertices(np.array(dom["tunnel_polygon_yz_m"], dtype=np.float64))
    x0, x1 = dom["x_range_m"]
    tun_lo, tun_hi = dom["excavated_tunnel_x_range_m"]
    b = dom["yz_bounds_m"]
    box = dict(x_min=x0, x_max=x1, y_min=b["y_min"], y_max=b["y_max"],
               z_min=b["z_min"], z_max=b["z_max"])

    F = doc["fractures"]
    R = np.array([f["radius_m"] for f in F])
    keep = R >= args.min_radius
    C = np.array([f["center_xyz_m"] for f in F])[keep]
    N = np.array([f["normal_xyz"] for f in F])[keep]
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    R = R[keep]
    lab = np.array([f["label"] for f in F])[keep]

    pl = pv.Plotter(off_screen=not args.window, window_size=[1800, 1200])
    pl.set_background("white")
    pl.enable_depth_peeling(10)

    # 기굴착 터널(solid) + 마지막 막장면
    pl.add_mesh(prism(poly, tun_lo, tun_hi), color=C_TUNNEL, opacity=0.65)
    face = pv.PolyData(np.column_stack([np.full(len(poly), tun_hi), poly]),
                       faces=np.concatenate([[len(poly)], np.arange(len(poly))]))
    pl.add_mesh(face, color=C_FACE, opacity=0.85)
    pl.add_mesh(face.extract_feature_edges(), color="black", line_width=4)

    # 굴착 예정 터널 윤곽 (미관측 구간)
    pl.add_mesh(ring(poly, x1), color="black", line_width=2)
    for y, z in poly[::3]:
        pl.add_mesh(pv.Line((tun_hi, y, z), (x1, y, z)), color="black", line_width=1)

    # 도메인 직육면체
    pl.add_mesh(pv.Box(bounds=(box["x_min"], box["x_max"], box["y_min"], box["y_max"],
                               box["z_min"], box["z_max"])).extract_all_edges(),
                color=C_DOMAIN, line_width=3)

    # 균열 원판 — 도메인 안 & 터널 공동 밖만
    n_drawn = {}
    for label, color, opac in (("unobserved", C_UNOBSERVED, 0.18),
                               ("observed", C_OBSERVED, 0.85)):
        m = lab == label
        n_drawn[label] = int(m.sum())
        if not m.any():
            continue
        mesh = disc_mesh(C[m], N[m], R[m]).triangulate()
        mesh["d"] = sdf_domain_minus_tunnel(mesh.points, box, poly, tun_lo, tun_hi)
        pl.add_mesh(mesh.clip_scalar(scalars="d", value=0.0, invert=True),
                    color=color, opacity=opac)

    # 터널 벽면(천장·측벽)의 복원 균열 절리선
    walls = trace_lines(doc.get("tunnel_wall_traces", []))
    if walls is not None:
        pl.add_mesh(walls, color=C_WALLTRACE, line_width=5)

    pl.add_legend(
        [[f"observed  {n_drawn.get('observed', 0)}", C_OBSERVED],
         [f"unobserved  {n_drawn.get('unobserved', 0)}", C_UNOBSERVED],
         [f"traces on tunnel wall  {len(doc.get('tunnel_wall_traces', []))}", C_WALLTRACE],
         ["excavated tunnel", C_FACE],
         ["domain box", C_DOMAIN]],
        bcolor="white", size=(0.23, 0.18), loc="upper right", face="rectangle")
    pl.add_text(f"Domain x=[{x0:g}, {x1:g}] m box, tunnel bbox + {dom['tunnel_halo_m']:g} m halo"
                f"   excavated x=[{tun_lo:g}, {tun_hi:g}] m"
                f"   (discs with radius >= {args.min_radius:g} m)",
                position="upper_left", font_size=13, color="black")
    pl.add_axes(xlabel="x (advance)", ylabel="y", zlabel="z", color="black")

    out_dir = args.out_dir or args.json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    focal = np.array([0.5 * (x0 + x1), 0.0, 0.0])

    def shoot(offset, name, up=(0.0, 0.0, 1.0), zoom=1.0):
        pl.camera.position = tuple(focal + np.array(offset, dtype=float))
        pl.camera.focal_point = tuple(focal)
        pl.camera.up = up
        pl.reset_camera()
        pl.camera.zoom(zoom)
        pl.render()
        p = out_dir / name
        pl.screenshot(str(p))
        print(f"[out] {p}")

    shoot((-30.0, -46.0, 26.0), "domain_dfn_3d_iso.png", zoom=1.25)
    shoot((0.0, 0.0, 60.0), "domain_dfn_3d_top.png", up=(1.0, 0.0, 0.0), zoom=1.25)
    shoot((-6.0, -55.0, 6.0), "domain_dfn_3d_side.png", zoom=1.25)

    print(f"[drawn] observed {n_drawn.get('observed', 0)} / "
          f"unobserved {n_drawn.get('unobserved', 0)}  (min_radius={args.min_radius:g} m)")
    if args.window:
        pl.show()
    else:
        pl.close()


if __name__ == "__main__":
    main()
