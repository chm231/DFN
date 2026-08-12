# =============================================================================
# 이 파일의 역할:
#   제6장 v2 범위(2026-08-07 결정)에 맞춘 절리선 데이터셋 생성기.
#   요철 막장면 mesh 없이, 막장면을 x=const **평면**으로 두고 균열 원판과의 교선을
#   해석적으로 계산한 뒤 **터널 단면 다각형으로 정확히 클리핑**한다.
#
#   기존 export_setwise_3d_traces(요철 mesh 교차)와의 차이:
#     - 관측면이 삼각 mesh가 아니라 다각형 그 자체 -> 관측면적 A_obs = 다각형 면적.
#       (mesh 방식은 격자 이산화로 경계가 계단형이 되어 다각형 대비 면적이 작아지고,
#        그 결과 관측면적 기준과 모델 창 기준이 어긋나는 문제가 있었다:
#        grid_step 0.2/0.1/0.05 m 에서 각각 -3.85 / -1.97 / -0.94 %)
#     - polyline 굴곡이 없으므로 3점법 법선 추정이 불가능/불필요.
#       방향은 외부 제공 3D 법선(벤치마크에서는 fracture 참값 법선)을 그대로 사용한다.
#
# 주요 입력:
#   --dfn-h5      : DFN export HDF5 (/fractures/centers, normals, radii, set_id)
#   --tunnel-dat  : 터널 단면 다각형 (HDF5에 없을 때)
#   --face-x      : 막장면 x 위치 목록 [m] (기본 0 1 2 3)
#   --lmin        : 최소 절리선 길이 [m]. 0이면 필터 없음(기본). 관측 단계 QC 기준.
#
# 주요 출력:
#   trace_dataset_3d.h5 / .csv — export_setwise_3d_traces 와 동일 스키마
#   (polyline은 양 끝점 2점으로 채워 하위 호환 유지)
#
# 핵심 처리 흐름:
#   1) DFN 로드 후 면별로 |중심-면거리| <= r 인 균열만 후보로 추림
#   2) 원판∩평면 교선의 중점과 방향(면내 yz)·참 현길이를 해석식으로 계산
#   3) 교선을 터널 단면 다각형으로 클리핑 -> 가시 길이·절단등급(0/1/2)
#   4) lmin 미만 절리선 제거 후 HDF5/CSV 기록
# =============================================================================
import argparse
import csv
import os
import sys

import h5py
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dfn_analysis.export_setwise_3d_traces import (
    load_hdf5_dfn,
    load_tunnel_polygon_from_dat,
    signed_polygon_area,
)


# 원판(중심 c, 법선 n, 반지름 r)과 평면 x=face_x 의 교선을 면내 좌표로 계산한다.
#   반환: (중점_yz, 단위방향_yz, 참현길이, 유효마스크)
#   기하: 평면 법선 e_x 와 원판 법선 n 의 교선 방향은 d = e_x × n (yz 성분만 남음).
#         원판 중심에서 교선까지의 면내 수직거리 h = |Δx| / sinφ,  sinφ = ||(n_y,n_z)||.
#         참 현길이 = 2√(r² − h²).
def disc_plane_chords(centers, normals, radii, face_x):
    dx = face_x - centers[:, 0]
    sin_phi = np.sqrt(np.clip(normals[:, 1] ** 2 + normals[:, 2] ** 2, 0.0, 1.0))
    ok = sin_phi > 1e-9
    h = np.full(len(radii), np.inf)
    h[ok] = np.abs(dx[ok]) / sin_phi[ok]
    valid = ok & (h <= radii)
    half = np.zeros(len(radii))
    half[valid] = np.sqrt(np.maximum(radii[valid] ** 2 - h[valid] ** 2, 0.0))
    true_len = 2.0 * half

    # 교선 방향(면내 단위벡터): e_x × n = (0, -n_z, n_y) 의 yz 성분을 정규화.
    dir_yz = np.column_stack([-normals[:, 2], normals[:, 1]])
    norm = np.linalg.norm(dir_yz, axis=1)
    dir_ok = norm > 1e-12
    dir_yz[dir_ok] /= norm[dir_ok, None]

    # 교선 중점: 원판면 안에서 중심으로부터 교선까지 수직으로 t 만큼 이동한 점의 yz 성분.
    #   원판면 내에서 x가 증가하는 단위방향 û = (e_x − n_x·n)/sinφ 이고,
    #   c + t·û 의 x좌표가 face_x 가 되려면 t = Δx/sinφ.
    #   û 의 yz 성분은 −n_x(n_y, n_z)/sinφ 이므로
    #     mid_yz = c_yz − (Δx·n_x/sin²φ)·(n_y, n_z)
    #   (estimate_p32_mc_calibrated 의 unit-P32 forward MC 와 동일한 공식)
    t = np.zeros(len(radii))
    t[ok] = dx[ok] / np.maximum(sin_phi[ok] ** 2, 1e-12)
    mid_yz = centers[:, 1:3] - (t * normals[:, 0])[:, None] * normals[:, 1:3]

    valid &= dir_ok
    return mid_yz, dir_yz, true_len, valid


# 선분(중점 m, 단위방향 d, 반길이 h)을 볼록 다각형으로 클리핑해 **끝점까지** 돌려준다.
#   CCW 다각형에서 각 변 a→b 의 내부 방향 법선은 n=(-e_y, e_x) 이고 내부 조건은 (p-a)·n ≥ 0.
#   p = m + s·d 로 두면 A + s·B ≥ 0  (A=(m-a)·n, B=d·n) → s 구간을 좁혀 나간다.
#   반환: (p0_yz, p1_yz, 가시길이, 절단등급 0/1/2, 유효마스크)
#   절단등급 = 관측창 경계에 의해 잘린 끝점 개수(원래 현 끝 ±h 가 살아남았는지로 판정).
def clip_chords_to_polygon(mid_yz, dir_yz, true_len, polygon_yz, eps=1e-9):
    h = 0.5 * true_len
    s_lo, s_hi = -h.copy(), h.copy()
    ok = true_len > 0.0
    a = polygon_yz
    e = np.roll(polygon_yz, -1, axis=0) - polygon_yz
    n = np.column_stack([-e[:, 1], e[:, 0]])           # 내부 방향 법선 (CCW)
    for k in range(len(a)):
        A = (mid_yz - a[k]) @ n[k]
        B = dir_yz @ n[k]
        s_bound = np.where(np.abs(B) > eps, -A / np.where(np.abs(B) > eps, B, 1.0), 0.0)
        enter = B > eps                                 # s ≥ s_bound
        leave = B < -eps                                # s ≤ s_bound
        parallel_out = (np.abs(B) <= eps) & (A < 0.0)   # 변 밖에서 평행 → 교차 없음
        s_lo = np.where(enter, np.maximum(s_lo, s_bound), s_lo)
        s_hi = np.where(leave, np.minimum(s_hi, s_bound), s_hi)
        ok &= ~parallel_out
    ok &= s_hi > s_lo
    vis = np.where(ok, s_hi - s_lo, 0.0)
    p0 = mid_yz + s_lo[:, None] * dir_yz
    p1 = mid_yz + s_hi[:, None] * dir_yz
    tol = 1e-7
    cls = ((s_lo > -h + tol).astype(np.int32) + (s_hi < h - tol).astype(np.int32))
    return p0, p1, vis, cls, ok


# 절리선 끝점이 관측창(터널 단면 다각형) 경계에 놓였는지로 끝점 유형을 판정한다.
#   경계 위(허용오차 tol 이내) -> "tunnel_boundary" (관측창에 의해 잘린 끝점)
#   그 외                      -> "disc_boundary"   (균열 원판 자체의 가장자리)
# 복원 단계는 disc_boundary 끝점만 경계 원적합(determined 반지름)에 사용한다.
def classify_endpoints(pts_yz, polygon_yz, tol=1e-6):
    pts = np.asarray(pts_yz, dtype=np.float64)
    a = polygon_yz
    b = np.roll(polygon_yz, -1, axis=0)
    seg = b - a                                        # (E,2)
    seg_len2 = np.maximum(np.sum(seg * seg, axis=1), 1e-30)
    # 각 점에서 모든 변까지의 최단거리 (선분 투영 후 클램프)
    d = pts[:, None, :] - a[None, :, :]                # (N,E,2)
    t = np.clip(np.sum(d * seg[None, :, :], axis=2) / seg_len2[None, :], 0.0, 1.0)
    proj = a[None, :, :] + t[:, :, None] * seg[None, :, :]
    dist = np.min(np.linalg.norm(pts[:, None, :] - proj, axis=2), axis=1)
    return np.array([b"tunnel_boundary" if v <= tol else b"disc_boundary" for v in dist],
                    dtype="S32")


def main():
    ap = argparse.ArgumentParser(
        description="평면 막장면 × 터널 단면 다각형 기준 절리선 데이터셋 생성 (v2 범위)")
    ap.add_argument("--dfn-h5", required=True, help="DFN export HDF5")
    ap.add_argument("--tunnel-dat", default=None, help="터널 단면 다각형 .dat (HDF5에 없을 때)")
    ap.add_argument("--face-x", nargs="+", type=float, default=[0.0, 1.0, 2.0, 3.0],
                    help="막장면 x 위치 목록 [m]")
    ap.add_argument("--lmin", type=float, default=0.0,
                    help="최소 절리선 길이 [m] (QC 기준). 0이면 필터 없음")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    data = load_hdf5_dfn(args.dfn_h5)
    poly = data["poly_yz"]
    if poly is None:
        if not args.tunnel_dat:
            raise ValueError("HDF5에 터널 다각형이 없습니다. --tunnel-dat 을 지정하세요.")
        poly = load_tunnel_polygon_from_dat(args.tunnel_dat)
    if signed_polygon_area(poly) < 0.0:
        poly = poly[::-1].copy()
    poly = np.asarray(poly, dtype=np.float64)
    poly_area = abs(0.5 * (np.dot(poly[:, 0], np.roll(poly[:, 1], -1))
                           - np.dot(poly[:, 1], np.roll(poly[:, 0], -1))))

    centers = np.asarray(data["centers"], dtype=np.float64)
    normals = np.asarray(data["normals"], dtype=np.float64)
    radii = np.asarray(data["radii"], dtype=np.float64)
    set_ids = np.asarray(data["set_ids"], dtype=np.int32)

    rec = {k: [] for k in ("set_id", "face_id", "face_x", "p0", "p1", "length",
                           "censor", "normal", "radius", "fid")}
    print(f"[*] 평면 막장면 {len(args.face_x)}매 | 다각형 면적 {poly_area:.4f} m² "
          f"| 총 관측면적 {poly_area*len(args.face_x):.4f} m²")

    for fi, fx in enumerate(args.face_x, start=1):
        # 면과 만날 수 있는 후보만 추림(|Δx| <= r).
        cand = np.abs(centers[:, 0] - fx) <= radii
        c, n, r, s = centers[cand], normals[cand], radii[cand], set_ids[cand]
        idx = np.nonzero(cand)[0]
        mid, dvec, tlen, valid = disc_plane_chords(c, n, r, fx)
        if not np.any(valid):
            continue
        mid, dvec, tlen = mid[valid], dvec[valid], tlen[valid]
        n_v, r_v, s_v, idx_v = n[valid], r[valid], s[valid], idx[valid]

        p0_yz, p1_yz, vis_len, cls, ok_clip = clip_chords_to_polygon(mid, dvec, tlen, poly)
        acc = ok_clip & (vis_len >= args.lmin if args.lmin > 0 else vis_len > 0.0)
        if not np.any(acc):
            continue
        p0_yz, p1_yz, vis_len, cls = p0_yz[acc], p1_yz[acc], vis_len[acc], cls[acc]
        n_v, r_v, s_v, idx_v = n_v[acc], r_v[acc], s_v[acc], idx_v[acc]

        z = np.full((len(vis_len), 1), fx)
        rec["p0"].append(np.hstack([z, p0_yz]))
        rec["p1"].append(np.hstack([z, p1_yz]))
        rec["set_id"].append(s_v)
        rec["face_id"].append(np.full(len(vis_len), fi, dtype=np.int32))
        rec["face_x"].append(np.full(len(vis_len), fx))
        rec["length"].append(vis_len)
        rec["censor"].append(cls)
        rec["normal"].append(n_v)
        rec["radius"].append(r_v)
        rec["fid"].append(idx_v.astype(np.int32))
        print(f"    - Face {fi:03d} @ x={fx:.2f} m | 후보 {int(cand.sum()):,} "
              f"→ 교차 {int(valid.sum()):,} → 채택 {len(vis_len):,}")

    cat = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in rec.items()}
    n_tr = len(cat["length"])
    print(f"[*] 총 {n_tr}개 절리선 (lmin={args.lmin} m)")
    for sid in np.unique(cat["set_id"]):
        m = cat["set_id"] == sid
        print(f"    - Set {int(sid)}: {int(m.sum())} traces, "
              f"총길이 {cat['length'][m].sum():.3f} m, "
              f"P21 = {cat['length'][m].sum()/(poly_area*len(args.face_x)):.4f} m/m²")

    h5_path = os.path.join(args.outdir, "trace_dataset_3d.h5")
    # polyline은 양 끝점 2점으로 채워 기존 스키마와 하위 호환을 유지한다.
    pl = np.empty((2 * n_tr, 3), dtype=np.float32)
    pl[0::2], pl[1::2] = cat["p0"], cat["p1"]
    with h5py.File(h5_path, "w") as f:
        g = f.create_group("traces")
        g.create_dataset("trace_id", data=np.arange(n_tr, dtype=np.int32))
        g.create_dataset("fracture_id", data=cat["fid"].astype(np.int32))
        g.create_dataset("component_id", data=cat["fid"].astype(np.int32))
        g.create_dataset("set_id", data=cat["set_id"].astype(np.uint16))
        g.create_dataset("face_id", data=cat["face_id"].astype(np.uint16))
        g.create_dataset("face_x_m", data=cat["face_x"].astype(np.float32))
        g.create_dataset("face_mesh_name", data=np.array(
            [f"flat_face_{int(i):06d}".encode() for i in cat["face_id"]], dtype="S64"))
        g.create_dataset("p0_xyz", data=cat["p0"].astype(np.float32))
        g.create_dataset("p1_xyz", data=cat["p1"].astype(np.float32))
        g.create_dataset("observed_length_m", data=cat["length"].astype(np.float32))
        g.create_dataset("censoring_class", data=cat["censor"].astype(np.uint8))
        g.create_dataset("radius_m", data=cat["radius"].astype(np.float32))
        g.create_dataset("trace_normal_xyz", data=cat["normal"].astype(np.float32))
        g.create_dataset("trace_normal_valid", data=np.ones(n_tr, dtype=np.uint8))
        g.create_dataset("trace_normal_quality", data=np.ones(n_tr, dtype=np.float32))
        g.create_dataset("trace_normal_reason", data=np.array(
            [b"external_flat_face"] * n_tr, dtype="S32"))
        g.create_dataset("polyline_vertex_start",
                         data=(2 * np.arange(n_tr)).astype(np.int32))
        g.create_dataset("polyline_vertex_count",
                         data=np.full(n_tr, 2, dtype=np.int32))
        g.create_dataset("polyline_vertices_xyz", data=pl)
        g.create_dataset("is_closed_loop", data=np.zeros(n_tr, dtype=np.uint8))
        g.create_dataset("n_raw_segments", data=np.ones(n_tr, dtype=np.int32))
        # 끝점 유형: 관측창(다각형) 경계에 닿아 잘린 끝점은 tunnel_boundary,
        # 그렇지 않으면 균열 원판 자체의 가장자리이므로 disc_boundary.
        # (복원 단계는 disc_boundary 끝점만 경계 원적합에 사용한다.)
        g.create_dataset("p0_endpoint_type", data=classify_endpoints(cat["p0"][:, 1:3], poly))
        g.create_dataset("p1_endpoint_type", data=classify_endpoints(cat["p1"][:, 1:3], poly))
        m = f.create_group("meta")
        m.create_dataset("tunnel_poly_yz", data=poly.astype(np.float32))
        m.create_dataset("face_x_positions_m",
                         data=np.asarray(args.face_x, dtype=np.float32))
        m.create_dataset("observation_area_m2",
                         data=np.array([poly_area * len(args.face_x)], dtype=np.float32))
        m.create_dataset("lmin_applied_m", data=np.array([args.lmin], dtype=np.float32))
        # set별 메타(rmin 규약 등)는 하위 추정기가 /meta/set_ids 순서로 조회하므로
        # 반드시 set 단위 배열(set_meta_ids)을 써야 한다(per-fracture set_ids 아님).
        for src_key, dst_key in (("generation_rmin", "generation_rmin"),
                                 ("generation_rmax", "generation_rmax"),
                                 ("set_meta_ids", "set_ids"),
                                 ("set_generation_rmin", "set_generation_rmin"),
                                 ("set_effective_rmin", "set_effective_rmin"),
                                 ("set_table_r0", "set_table_r0")):
            v = data.get(src_key)
            if v is None:
                continue
            m.create_dataset(dst_key, data=np.atleast_1d(np.asarray(v)))
    print(f"[*] HDF5 written: {h5_path}")

    csv_path = os.path.join(args.outdir, "trace_dataset_3d.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["trace_id", "fracture_id", "set_id", "face_id", "face_x_m",
                    "p0_x", "p0_y", "p0_z", "p1_x", "p1_y", "p1_z",
                    "observed_length_m", "censoring_class", "radius_m",
                    "nx", "ny", "nz"])
        for i in range(n_tr):
            w.writerow([i, int(cat["fid"][i]), int(cat["set_id"][i]),
                        int(cat["face_id"][i]), f"{cat['face_x'][i]:.4f}",
                        *[f"{v:.6f}" for v in cat["p0"][i]],
                        *[f"{v:.6f}" for v in cat["p1"][i]],
                        f"{cat['length'][i]:.6f}", int(cat["censor"][i]),
                        f"{cat['radius'][i]:.6f}",
                        *[f"{v:.6f}" for v in cat["normal"][i]]])
    print(f"[*] CSV written: {csv_path}")


if __name__ == "__main__":
    main()
