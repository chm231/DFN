"""Domain-restricted conditional DFN export (JSON) for downstream block detection.

블록 생성·안정성 평가 담당자에게 전달할 균열 목록을 JSON으로 내보낸다.
대상 영역은 "터널 단면 다각형 + halo" 를 마지막 막장면 전방으로 연장한 기둥이다.

    도메인 = { (x,y,z) : x0 <= x <= x1,  dist((y,z), 터널 단면 다각형) <= halo }

균열 구성 (라벨 2종):
  * observed   — 관측 절리선에서 복원한 균열 (reconstruct/reconstructed_discs.csv)
  * unobserved — 역산 파라미터로 확률적으로 생성한 균열. 관측 막장면에 절리선을
                 남기는 것은 제거한다(그 영역은 observed 균열이 이미 설명한다).

좌표 규약: x = East = 터널 굴진 방향, y = North, z = Up. 막장면은 x = const 평면.
생성/조건화 함수는 generate_conditional_hidden_dfn 에서 그대로 재사용한다.

Usage
-----
    python dfn_analysis/export_domain_dfn_json.py \
        --pipeline-dir storage/output/pipeline_v2_laxemar \
        --halo 5.0 --ahead 10.0
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

try:
    from dfn_analysis import generate_conditional_hidden_dfn as G
except ImportError:  # 단독 스크립트 실행(스크립트 폴더가 sys.path 에 있는 경우)
    import generate_conditional_hidden_dfn as G

REPO = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------
# 도메인 기하: (y,z) 평면에서 다각형까지의 거리 (내부는 0)
# ----------------------------------------------------------------------
def dist_to_polygon(pts_yz: np.ndarray, poly_yz: np.ndarray) -> np.ndarray:
    """각 점에서 다각형까지의 유클리드 거리. 다각형 내부면 0."""
    P = np.asarray(pts_yz, dtype=np.float64)
    A = np.asarray(poly_yz, dtype=np.float64)
    B = np.roll(A, -1, axis=0)
    AB = B - A
    AP = P[:, None, :] - A[None, :, :]
    denom = np.maximum(np.einsum("mj,mj->m", AB, AB), 1e-300)
    t = np.clip(np.einsum("nmj,mj->nm", AP, AB) / denom, 0.0, 1.0)
    proj = A[None, :, :] + t[:, :, None] * AB[None, :, :]
    d_edge = np.linalg.norm(P[:, None, :] - proj, axis=2).min(axis=1)

    # ray casting (수평 방향) 내부 판정
    y, z = P[:, 0], P[:, 1]
    inside = np.zeros(len(P), dtype=bool)
    for i in range(len(A)):
        y1, z1 = A[i]
        y2, z2 = B[i]
        straddles = (z1 > z) != (z2 > z)
        dz = z2 - z1 if abs(z2 - z1) > 1e-300 else 1e-300
        y_cross = (y2 - y1) * (z - z1) / dz + y1
        inside ^= straddles & (y < y_cross)
    return np.where(inside, 0.0, d_edge)


def yz_bounds(poly_yz: np.ndarray, halo: float):
    """직사각형 도메인의 yz 범위 = 터널 단면 bounding box + halo."""
    return (poly_yz[:, 0].min() - halo, poly_yz[:, 0].max() + halo,
            poly_yz[:, 1].min() - halo, poly_yz[:, 1].max() + halo)


def distance_to_domain(centers: np.ndarray, x0: float, x1: float,
                       poly_yz: np.ndarray, halo: float, shape: str) -> np.ndarray:
    """중심점에서 도메인까지의 거리. 도메인이 x구간 × yz영역의 곱집합이므로
    두 축 거리의 유클리드 합성이 정확한 거리가 된다."""
    dx = np.maximum(0.0, np.maximum(x0 - centers[:, 0], centers[:, 0] - x1))
    if shape == "box":
        y_lo, y_hi, z_lo, z_hi = yz_bounds(poly_yz, halo)
        dy = np.maximum(0.0, np.maximum(y_lo - centers[:, 1], centers[:, 1] - y_hi))
        dz = np.maximum(0.0, np.maximum(z_lo - centers[:, 2], centers[:, 2] - z_hi))
        dyz = np.hypot(dy, dz)
    else:
        dyz = np.maximum(0.0, dist_to_polygon(centers[:, 1:3], poly_yz) - halo)
    return np.hypot(dx, dyz)


def select_intersecting(discs, x0, x1, poly_yz, halo, shape):
    """원판의 경계구가 도메인과 만나는 것만 남긴다(보수적 포함 기준)."""
    if not discs:
        return []
    centers = np.array([d["center"] for d in discs], dtype=np.float64)
    radii = np.array([d["radius"] for d in discs], dtype=np.float64)
    keep = distance_to_domain(centers, x0, x1, poly_yz, halo, shape) <= radii
    return [d for d, k in zip(discs, keep) if k]


# ----------------------------------------------------------------------
# 터널 벽면(천장·측벽) 위의 복원 균열 절리선
# ----------------------------------------------------------------------
def _clip_param(s_lo, s_hi, value_at_0, slope, lo, hi, tol=1e-12):
    """직선 매개변수 s 구간을 lo <= value_at_0 + slope*s <= hi 로 좁힌다."""
    if abs(slope) < tol:
        return (s_lo, s_hi) if lo - 1e-9 <= value_at_0 <= hi + 1e-9 else None
    a = (lo - value_at_0) / slope
    b = (hi - value_at_0) / slope
    s_lo = max(s_lo, min(a, b))
    s_hi = min(s_hi, max(a, b))
    return (s_lo, s_hi) if s_hi > s_lo else None


def disc_tunnel_surface_traces(center, normal, radius, poly_yz, x_lo, x_hi,
                               min_len=0.05):
    """원판이 터널 벽면(단면 다각형을 x 로 밀어낸 옆면)에 남기는 절리선 선분들.

    벽면은 다각형 변마다 하나씩인 평면 사각형이다. 각 사각형에 대해
    원판 평면과의 교선을 구하고, 원판 내부(현)와 사각형 범위로 자른다.
    """
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    c = np.asarray(center, dtype=np.float64)
    ex = np.array([1.0, 0.0, 0.0])
    out = []
    for i in range(len(poly_yz)):
        a = poly_yz[i]
        b = poly_yz[(i + 1) % len(poly_yz)]
        e = b - a
        edge_len = float(np.linalg.norm(e))
        if edge_len < 1e-12:
            continue
        d = np.array([0.0, e[0] / edge_len, e[1] / edge_len])  # 둘레 방향
        m = np.cross(ex, d)                                     # 벽면 법선
        t = np.cross(n, m)
        nt = float(np.linalg.norm(t))
        if nt < 1e-9:      # 원판이 벽면과 평행
            continue
        t /= nt
        q = np.array([0.0, a[0], a[1]])                         # 벽면 위 한 점
        try:
            p0 = np.linalg.solve(np.array([n, m, t]), np.array([n @ c, m @ q, t @ q]))
        except np.linalg.LinAlgError:
            continue
        # 원판 내부의 현: |p0 + s t - c| <= radius
        w = c - p0
        s_mid = float(w @ t)
        h2 = radius ** 2 - (float(w @ w) - s_mid ** 2)
        if h2 <= 0.0:
            continue
        h = float(np.sqrt(h2))
        span = _clip_param(s_mid - h, s_mid + h, float(p0[0]), float(t[0]), x_lo, x_hi)
        if span is None:
            continue
        span = _clip_param(span[0], span[1], float((p0 - q) @ d), float(t @ d),
                           0.0, edge_len)
        if span is None:
            continue
        s0, s1 = span
        if (s1 - s0) < min_len:
            continue
        out.append((p0 + s0 * t, p0 + s1 * t))
    return out


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pipeline-dir", type=Path,
                    default=REPO / "storage/output/pipeline_v2_laxemar")
    ap.add_argument("--halo", type=float, default=5.0,
                    help="터널 단면 다각형 바깥으로의 확장 폭 [m].")
    ap.add_argument("--ahead", type=float, default=10.0,
                    help="마지막 막장면에서 굴진 방향(+x)으로의 연장 길이 [m].")
    ap.add_argument("--rmax-local", type=float, default=10.0,
                    help="확률 생성 균열의 반지름 상한 [m].")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lmin-det", type=float, default=0.5,
                    help="검출 하한 [m]. 관측 막장면에서 이 길이 이상으로 보였어야 하는 "
                         "확률 생성 균열만 제거한다.")
    ap.add_argument("--keep-adoptions", default="deterministic_disc,orientation_only")
    ap.add_argument("--sets", nargs="+", type=int, default=None)
    ap.add_argument("--exclude-sets", nargs="+", type=int, default=[])
    ap.add_argument("--config", default=None,
                    help="set별 dist_type/r0 override JSON (지수분포 set 등).")
    ap.add_argument("--dfn-h5", type=Path, default=None,
                    help="터널 단면 다각형을 읽을 DFN HDF5. 기본 = pipeline-dir 안의 파일.")
    ap.add_argument("--domain-shape", choices=["box", "polygon"], default="box",
                    help="도메인 단면: 터널 bounding box + halo(직사각형) 또는 단면 다각형 + halo.")
    ap.add_argument("--x-range", nargs=2, type=float, default=None,
                    help="도메인 x 범위 [m]. 주면 --ahead 대신 이 값을 쓴다.")
    ap.add_argument("--tunnel-x-range", nargs=2, type=float, default=None,
                    help="기굴착 터널 벽면 x 범위 [m]. 기본 = 첫 막장면 ~ 마지막 막장면.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pdir = args.pipeline_dir

    # --- 입력 ---
    keep = tuple(a.strip() for a in args.keep_adoptions.split(","))
    visible_all = G.load_visible_discs(pdir / "reconstruct/reconstructed_discs.csv", keep)
    observed, _ = G.load_observed_traces(pdir / "trace_dataset/trace_dataset_3d.csv")
    params = G.load_inverted_params(pdir / "kr/kr_summary_by_set.csv", pdir / "p32/p32_summary.csv")

    if args.config:
        cfg = json.load(open(args.config, encoding="utf-8"))
        for sid_str, s in cfg.get("sets", {}).items():
            sid = int(sid_str)
            params.setdefault(sid, {})
            if "dist_type" in s:
                params[sid]["dist_type"] = str(s["dist_type"])
            if "r0" in s:
                params[sid]["r0"] = float(s["r0"])

    exclude = set(args.exclude_sets)
    target_sets = args.sets if args.sets is not None else sorted(params.keys())
    target_sets = [s for s in target_sets if s not in exclude]

    face_xs = sorted(observed.keys())
    last_face_x = max(face_xs)
    dfn_h5 = args.dfn_h5 or (pdir / "dfn_export_for_python.h5")
    with h5py.File(dfn_h5, "r") as f:
        poly = np.array(f["tunnel/poly_YZ"], dtype=np.float64)
        site = f["meta/site"][()].decode() if "meta/site" in f else ""
    poly_ccw = G._ccw_polygon(poly)

    if args.x_range is not None:
        x0, x1 = args.x_range
    else:
        x0, x1 = last_face_x, last_face_x + args.ahead
    tun_lo, tun_hi = (args.tunnel_x_range if args.tunnel_x_range is not None
                      else (min(face_xs), last_face_x))
    print(f"[domain] x = [{x0:.2f}, {x1:.2f}] m   단면 = "
          f"{'터널 bounding box' if args.domain_shape == 'box' else '터널 단면 다각형'}"
          f" + halo {args.halo:.1f} m")
    print(f"[domain] 관측 막장면: {face_xs}  |  기굴착 터널 벽면 x=[{tun_lo:g},{tun_hi:g}] m")

    # --- 확률 생성용 박스 (도메인 bbox를 rmax_local 만큼 확장) ---
    ymin, ymax, zmin, zmax = yz_bounds(poly_ccw, args.halo)
    m = args.rmax_local
    box = dict(x0=x0 - m, dx=(x1 - x0) + 2 * m,
               y0=ymin - m, dy=(ymax - ymin) + 2 * m,
               z0=zmin - m, dz=(zmax - zmin) + 2 * m)
    print(f"[gen box] x[{box['x0']:.1f},{box['x0']+box['dx']:.1f}] "
          f"y[{box['y0']:.1f},{box['y0']+box['dy']:.1f}] z[{box['z0']:.1f},{box['z0']+box['dz']:.1f}]"
          f"  V={box['dx']*box['dy']*box['dz']:,.0f} m^3")

    # --- 생성 → 관측 막장면 조건화 → 도메인 절단 ---
    hidden_all = G.generate_hidden_discs(params, visible_all, box, args.rmax_local,
                                         args.seed, target_sets)
    hidden_kept, n_removed = G.remove_face_intersecting(hidden_all, face_xs, poly_ccw,
                                                        args.lmin_det)
    print(f"[condition] 확률 생성 {len(hidden_all):,} → 관측 막장면 검출 제거 {n_removed:,} "
          f"(lmin_det={args.lmin_det:g} m) → 잔존 {len(hidden_kept):,}")

    vis_in = select_intersecting(visible_all, x0, x1, poly_ccw, args.halo, args.domain_shape)
    hid_in = select_intersecting(hidden_kept, x0, x1, poly_ccw, args.halo, args.domain_shape)
    print(f"[domain] observed   {len(vis_in):,} / {len(visible_all):,}")
    print(f"[domain] unobserved {len(hid_in):,} / {len(hidden_kept):,}")

    # --- JSON ---
    fractures = []
    for d in vis_in:
        c, n = d["center"], d["normal"]
        fractures.append({
            "id": len(fractures),
            "set_id": int(d["set_id"]),
            "label": "observed",
            "center_xyz_m": [round(float(v), 4) for v in c],
            "normal_xyz": [round(float(v), 6) for v in n],
            "radius_m": round(float(d["radius"]), 4),
            "reconstruction": d["adoption"],
        })
    for d in hid_in:
        c, n = d["center"], d["normal"]
        fractures.append({
            "id": len(fractures),
            "set_id": int(d["set_id"]),
            "label": "unobserved",
            "center_xyz_m": [round(float(v), 4) for v in c],
            "normal_xyz": [round(float(v), 6) for v in n],
            "radius_m": round(float(d["radius"]), 4),
            "reconstruction": "stochastic",
        })

    # --- 기굴착 터널 벽면(천장·측벽)에 나타나는 복원 균열 절리선 ---
    wall_traces = []
    for f_rec, d in zip(fractures[:len(vis_in)], vis_in):
        for p_a, p_b in disc_tunnel_surface_traces(d["center"], d["normal"], d["radius"],
                                                   poly_ccw, tun_lo, tun_hi):
            wall_traces.append({
                "fracture_id": f_rec["id"],
                "set_id": int(d["set_id"]),
                "p0_xyz_m": [round(float(v), 4) for v in p_a],
                "p1_xyz_m": [round(float(v), 4) for v in p_b],
                "length_m": round(float(np.linalg.norm(p_b - p_a)), 4),
            })
    n_frac_on_wall = len({w["fracture_id"] for w in wall_traces})
    print(f"[wall] 터널 벽면 절리선 {len(wall_traces):,}개 "
          f"(복원 균열 {n_frac_on_wall}개에서, 총 연장 "
          f"{sum(w['length_m'] for w in wall_traces):.1f} m)")

    set_params = {}
    for sid in sorted(set(f["set_id"] for f in fractures)):
        p = params.get(sid, {})
        set_params[str(sid)] = {
            "dist_type": p.get("dist_type"),
            "kr": p.get("kr"),
            "P32_per_m": p.get("P32"),
            "rmin_m": p.get("rmin"),
            "stochastic_generated": sid in target_sets and "P32" in p,
        }

    doc = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_pipeline": str(pdir).replace("\\", "/"),
            "site": site,
            "coordinate_system": {
                "x": "터널 굴진 방향 (East)", "y": "North", "z": "Up", "units": "m",
                "face_plane": "막장면은 x = const 평면",
            },
            "domain": {
                "last_face_x_m": last_face_x,
                "x_range_m": [x0, x1],
                "shape": args.domain_shape,
                "tunnel_halo_m": args.halo,
                "yz_bounds_m": (dict(zip(["y_min", "y_max", "z_min", "z_max"],
                                         [round(float(v), 4)
                                          for v in yz_bounds(poly_ccw, args.halo)]))
                                if args.domain_shape == "box" else None),
                "excavated_tunnel_x_range_m": [tun_lo, tun_hi],
                "description": (
                    f"x = [{x0:g}, {x1:g}] m 구간, 터널 "
                    f"{'단면 bounding box' if args.domain_shape == 'box' else '단면 다각형'}"
                    f"에서 바깥으로 {args.halo:g} m 확장한 직육면체 영역. "
                    f"x=[{tun_lo:g},{tun_hi:g}] m 는 기굴착 구간, "
                    f"x=[{tun_hi:g},{x1:g}] m 는 미관측 전방 구간이다."),
                "tunnel_polygon_yz_m": [[round(float(a), 4), round(float(b), 4)] for a, b in poly_ccw],
                "observed_face_x_m": [float(v) for v in face_xs],
            },
            "geometry_model": "각 균열은 중심 center_xyz_m, 단위 법선 normal_xyz, 반지름 radius_m 인 원판",
            "inclusion_rule": "원판의 경계구(중심, 반지름)가 도메인과 만나면 포함 — 도메인 밖에 중심이 있어도 도메인을 자르는 균열은 들어간다",
            "labels": {
                "observed": "관측 절리선에서 복원한 균열",
                "unobserved": "역산 파라미터로 확률적으로 생성한 균열 (관측 막장면에 절리선을 남기는 것은 제거)",
            },
            "counts": {
                "total": len(fractures),
                "observed": len(vis_in),
                "unobserved": len(hid_in),
                "observed_outside_domain": len(visible_all) - len(vis_in),
                "tunnel_wall_traces": len(wall_traces),
                "fractures_daylighting_on_wall": n_frac_on_wall,
            },
            "set_parameters": set_params,
            "stochastic_radius_max_m": args.rmax_local,
            "seed": args.seed,
            "caveats": [
                f"stochastic_generated=false 인 절리군은 확률 생성 배경이 없다. "
                f"관측에서 복원한 균열만 들어 있으므로 그 절리군의 균열 밀도는 실제보다 낮다.",
                f"확률 생성 균열의 반지름은 {args.rmax_local:g} m 에서 잘랐다. 그보다 큰 균열은 이 파일에 없다.",
                f"도메인 앞면(x={x0:g} m)은 마지막 관측 막장면이다. 그 면에 절리선을 남기는 "
                f"확률 생성 균열은 제거했으므로 x={x0:g} m 부근의 unobserved 밀도가 낮게 나타난다. "
                f"그 자리는 observed 균열이 대신한다.",
                "observed 균열의 reconstruction 값이 orientation_only 이면 방향은 절리선에서 정했지만 "
                "반지름은 직접 결정되지 않아 사후분포에서 추정한 것이다.",
            ],
            "tunnel_wall_traces": (
                "기굴착 구간 터널 벽면(천장·측벽)에 나타나는 복원 균열의 절리선. "
                "복원 균열에서 파생된 표시용 자료이며 역산 입력이 아니다."),
        },
        "fractures": fractures,
        "tunnel_wall_traces": wall_traces,
    }

    out = args.out or (pdir / "export" / f"dfn_domain_x{x0:g}-{x1:g}_halo{args.halo:g}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    print(f"[out] {out}  ({out.stat().st_size/1e6:.2f} MB, 균열 {len(fractures):,}개)")


if __name__ == "__main__":
    main()
