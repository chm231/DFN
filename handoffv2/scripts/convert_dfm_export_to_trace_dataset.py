# -*- coding: utf-8 -*-
"""DFM Viewer export(실측 막장면 절리 매핑) → 파이프라인 trace dataset 변환기.

입력: DFM_Export 폴더 (막장면별 번호 폴더 01..NN)
  <face>/Export 2D DS Data/2D_trace_data.csv   절리선 양끝점(이상화 평면 위, 월드좌표)
  <face>/Export 3D DS Data/3D_disk_data.csv    절리면 패치 법선(월드좌표, 2D csv와 행 대응)
  <face>/Export 3D DS Data/3D_tunnel_face_point_cloud.ply   막장면 점군 (관측창 산정용)

변환 규약 (좌표계: x = 터널 진행축, y/z = 막장면 내):
  1) 전역 회전: 면 평면 법선(SVD 적합)의 평균을 +x(굴진방향)로 정렬, 연직은 z 유지.
     면별 잔여 기울기(수 도)는 좌표를 왜곡하지 않도록 실제 3D 좌표를 그대로 저장한다.
  2) set_id: 원자료 DS 번호는 면별 클러스터 번호라 전역 절리군이 아니다.
     (면, DS) 단위 평균 법선(축성)을 계층 군집(평균연결, 축간각)으로 묶어
     전역 절리군 번호를 다시 부여한다(절리선 수 내림차순 1..K).
  3) 관측면적: 면별 점군을 면 내 좌표로 투영해 점유격자(occupancy grid) 면적 합산.
     관측창 다각형(/meta/tunnel_poly_yz)은 면적 중앙값 면의 convex hull로 대표한다.
  4) 끝점 censoring: 끝점이 자기 면 hull 경계에서 edge-tol 이내면 tunnel_boundary
     (관측창에 잘림), 그 외 disc_boundary (균열 자체 끝).
  5) 반경 참값 없음 → radius_m = NaN, fracture_id = 연번(참값 아님).

출력: <outdir>/trace_dataset/trace_dataset_3d.h5 + .csv  (export_flat_face_traces 스키마)
      <outdir>/dfn_export_for_python.h5  (관측 법선 + 관측창 다각형만 담은 보조 h5;
        조건부 생성 단계의 /tunnel/poly_YZ, P32 방향계수 참고용 /fractures/normals)
      <outdir>/conversion_diagnostics.json / set_mapping.csv  (면·set 진단표)

실행 예:
  python scripts/convert_dfm_export_to_trace_dataset.py \
    --dfm-dir demodata/DFM_Export/DFM_Export --outdir demo_output/dfm_demo
"""
import argparse
import csv
import json
import os
import sys

# cp949 콘솔에서도 한글 출력이 죽지 않도록 utf-8 강제 (run_benchmark_demo.py와 동일)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

import h5py
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import ConvexHull
from scipy.spatial.distance import squareform


def read_trace_csvs(face_dir):
    """면 폴더에서 2D 끝점(평면화)과 3D 패치 법선을 행 대응으로 읽는다."""
    p2 = os.path.join(face_dir, "Export 2D DS Data", "2D_trace_data.csv")
    p3 = os.path.join(face_dir, "Export 3D DS Data", "3D_disk_data.csv")
    ds, p0_w, p1_w, n_w = [], [], [], []
    with open(p2, newline="") as f2, open(p3, newline="") as f3:
        for r2, r3 in zip(csv.DictReader(f2), csv.DictReader(f3)):
            if r2["DS"] != r3["DS"]:
                raise ValueError(f"2D/3D DS 행 대응이 깨졌습니다: {face_dir}")
            ds.append(int(r2["DS"]))
            p0_w.append([float(r2[f"TraceP1_flat{c}"]) for c in "XYZ"])
            p1_w.append([float(r2[f"TraceP2_flat{c}"]) for c in "XYZ"])
            n_w.append([float(r3[f"Normal{c}"]) for c in "XYZ"])
    return np.array(ds), np.array(p0_w), np.array(p1_w), np.array(n_w)


def read_face_cloud_xyz(face_dir):
    """ASCII PLY 점군에서 xyz만 읽는다."""
    path = os.path.join(face_dir, "Export 3D DS Data", "3D_tunnel_face_point_cloud.ply")
    with open(path, "r") as f:
        n_vert, line_no = 0, 0
        for line in f:
            line_no += 1
            if line.startswith("element vertex"):
                n_vert = int(line.split()[-1])
            if line.strip() == "end_header":
                break
    data = np.loadtxt(path, skiprows=line_no, usecols=(0, 1, 2), max_rows=n_vert)
    return data


def fit_plane(points_xyz):
    """SVD 평면 적합 → (단위법선, 중심)."""
    c = points_xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(points_xyz - c, full_matrices=False)
    return vt[2] / np.linalg.norm(vt[2]), c


def axial_mean(normals_xyz):
    """축성(±동일시) 평균 방향: 산포행렬 최대 고유벡터."""
    n = normals_xyz / np.linalg.norm(normals_xyz, axis=1, keepdims=True)
    _, v = np.linalg.eigh(n.T @ n)
    return v[:, -1]


def occupancy_area(points_2d, cell):
    """점유격자 면적 [m^2]."""
    ij = np.floor(points_2d / cell).astype(np.int64)
    return len({(int(a), int(b)) for a, b in ij}) * cell * cell


def hull_polygon(points_2d):
    """convex hull 꼭짓점(CCW)."""
    h = ConvexHull(points_2d)
    return points_2d[h.vertices]


def dist_to_polygon_edges(pts_2d, poly_2d):
    """각 점에서 다각형 변까지의 최단거리."""
    a = poly_2d
    b = np.roll(poly_2d, -1, axis=0)
    seg = b - a
    seg_len2 = np.maximum((seg * seg).sum(axis=1), 1e-30)
    d = pts_2d[:, None, :] - a[None, :, :]
    t = np.clip((d * seg[None, :, :]).sum(axis=2) / seg_len2[None, :], 0.0, 1.0)
    proj = a[None, :, :] + t[:, :, None] * seg[None, :, :]
    return np.linalg.norm(pts_2d[:, None, :] - proj, axis=2).min(axis=1)


def main():
    # ------------------------------------------------------------------------
    # [CLI 인자 요약] — 자세한 도움말: --help
    #   --dfm-dir     : 필수 — DFM_Export 폴더 (01..NN 면 폴더 포함)
    #   --outdir      : 필수
    #   --grid        : 기본 0.05 — 관측면적 점유격자 셀 [m]
    #   --edge-tol    : 기본 0.15 — 끝점 censoring 판정: 면 경계까지 거리 임계 [m]
    #   --cluster-cut : 기본 30.0 — 전역 절리군 군집화 절단 각도 [deg]
    # ------------------------------------------------------------------------
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dfm-dir", required=True, help="DFM_Export 폴더 (01..NN 면 폴더 포함)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--grid", type=float, default=0.05, help="관측면적 점유격자 셀 [m]")
    ap.add_argument("--edge-tol", type=float, default=0.15,
                    help="끝점 censoring 판정: 면 경계까지 거리 임계 [m]")
    ap.add_argument("--cluster-cut", type=float, default=30.0,
                    help="전역 절리군 군집화 절단 각도 [deg]")
    args = ap.parse_args()

    faces = sorted(d for d in os.listdir(args.dfm_dir)
                   if os.path.isdir(os.path.join(args.dfm_dir, d)) and d.isdigit())
    print(f"[*] 막장면 폴더 {len(faces)}개: {faces}")

    # ---- 1) 면별 로드 + 평면 적합 ----
    per_face = {}
    for f in faces:
        fdir = os.path.join(args.dfm_dir, f)
        ds, p0_w, p1_w, n_w = read_trace_csvs(fdir)
        n_face, c_face = fit_plane(np.vstack([p0_w, p1_w]))
        per_face[f] = dict(ds=ds, p0_w=p0_w, p1_w=p1_w, n_w=n_w,
                           n_face=n_face, c_face=c_face)

    # 터널축(+x) = 면 '중심 궤적'의 선형 적합 방향(굴진 방향으로 부호 통일).
    # 주의: 면 법선 평균을 쓰면 안 된다 — 실측 막장면은 굴진방향에 대해 계통적으로
    # 기울어 굴착될 수 있어(본 데이터 8.2°), 법선 기준 정렬은 단면 중심을 x를 따라
    # 드리프트시켜 관측창과 절리선이 어긋난다. 중심 궤적이 실제 선형이다.
    cs = np.array([per_face[f]["c_face"] for f in faces])
    advance = cs[-1] - cs[0]
    advance /= np.linalg.norm(advance)
    _, _, vt_c = np.linalg.svd(cs - cs.mean(axis=0), full_matrices=False)
    x_hat = vt_c[0] / np.linalg.norm(vt_c[0])
    if x_hat @ advance < 0:
        x_hat = -x_hat
    z_hat = np.array([0.0, 0.0, 1.0]) - (np.array([0.0, 0.0, 1.0]) @ x_hat) * x_hat
    z_hat /= np.linalg.norm(z_hat)
    y_hat = np.cross(z_hat, x_hat)
    R = np.vstack([x_hat, y_hat, z_hat])  # world → pipeline: p' = R @ p
    print(f"[*] 터널축(x̂, world) = [{x_hat[0]:+.4f},{x_hat[1]:+.4f},{x_hat[2]:+.4f}]")

    # ---- 2) 전역 절리군 군집화: (면, DS) 단위 평균 법선 ----
    units, unit_vecs, unit_sizes = [], [], []
    for f in faces:
        d = per_face[f]
        for ds_id in sorted(set(d["ds"].tolist())):
            m = d["ds"] == ds_id
            units.append((f, ds_id))
            unit_vecs.append(axial_mean(d["n_w"][m]))
            unit_sizes.append(int(m.sum()))
    V = np.array(unit_vecs)
    ang = np.degrees(np.arccos(np.clip(np.abs(V @ V.T), 0.0, 1.0)))
    np.fill_diagonal(ang, 0.0)
    lab = fcluster(linkage(squareform(ang, checks=False), method="average"),
                   args.cluster_cut, criterion="distance")
    # 절리선 수 내림차순으로 전역 set 번호 부여
    counts = {}
    for k, n in zip(lab, unit_sizes):
        counts[k] = counts.get(k, 0) + n
    order = sorted(counts, key=lambda k: -counts[k])
    relabel = {k: i + 1 for i, k in enumerate(order)}
    unit_to_set = {u: relabel[k] for u, k in zip(units, lab)}
    print(f"[*] 전역 절리군 {len(order)}개 (군집 절단 {args.cluster_cut}°):")
    map_rows = []
    for k in order:
        gid = relabel[k]
        members = [u for u, kk in zip(units, lab) if kk == k]
        n_tr = counts[k]
        m_vec_raw = [V[i] for i, kk in enumerate(lab) if kk == k]
        ref = m_vec_raw[0]
        m_vec = axial_mean(np.array([v if v @ ref >= 0 else -v for v in m_vec_raw]))
        print(f"    - Set {gid}: {n_tr:5d} traces, {len({u[0] for u in members}):2d}개 면, "
              f"평균 법선(world)=[{m_vec[0]:+.2f},{m_vec[1]:+.2f},{m_vec[2]:+.2f}]")
        for u in members:
            map_rows.append(dict(face=u[0], source_ds=u[1], global_set=gid,
                                 n_traces=unit_sizes[units.index(u)]))

    # ---- 3) 면별 관측창 (점군 점유면적 + hull) ----
    face_meta = {}
    for f in faces:
        cloud_w = read_face_cloud_xyz(os.path.join(args.dfm_dir, f))
        cloud_p = cloud_w @ R.T
        yz = cloud_p[:, 1:3]
        area = occupancy_area(yz, args.grid)
        poly = hull_polygon(yz)
        face_x = float(np.median(cloud_p[:, 0]))
        face_meta[f] = dict(area=area, poly=poly, face_x=face_x,
                            hull_area=float(ConvexHull(yz).volume))
        print(f"    - 면 {f}: x={face_x:+7.2f} m, 점유면적 {area:6.2f} m², "
              f"hull {face_meta[f]['hull_area']:6.2f} m², 점 {len(yz):,}")

    # 대표 관측창: hull 면적 중앙값 면
    med_face = sorted(faces, key=lambda f: face_meta[f]["hull_area"])[len(faces) // 2]
    window_poly = face_meta[med_face]["poly"].astype(np.float64)
    # CCW 보장
    if 0.5 * np.sum(window_poly[:, 0] * np.roll(window_poly[:, 1], -1)
                    - window_poly[:, 1] * np.roll(window_poly[:, 0], -1)) < 0:
        window_poly = window_poly[::-1].copy()
    total_area = float(sum(face_meta[f]["area"] for f in faces))
    print(f"[*] 대표 관측창 = 면 {med_face} hull ({face_meta[med_face]['hull_area']:.2f} m²) | "
          f"총 관측면적(점유격자) = {total_area:.2f} m²")

    # ---- 4) trace 레코드 조립 ----
    rec = {k: [] for k in ("set_id", "face_id", "face_x", "p0", "p1", "length",
                           "censor", "normal", "e0", "e1")}
    for fi, f in enumerate(faces, start=1):
        d = per_face[f]
        p0 = d["p0_w"] @ R.T
        p1 = d["p1_w"] @ R.T
        nr = d["n_w"] @ R.T
        gset = np.array([unit_to_set[(f, int(s))] for s in d["ds"]], dtype=np.int32)
        length = np.linalg.norm(p1 - p0, axis=1)
        poly = face_meta[f]["poly"]
        d0 = dist_to_polygon_edges(p0[:, 1:3], poly)
        d1 = dist_to_polygon_edges(p1[:, 1:3], poly)
        e0 = d0 <= args.edge_tol
        e1 = d1 <= args.edge_tol
        rec["set_id"].append(gset)
        rec["face_id"].append(np.full(len(gset), fi, dtype=np.int32))
        rec["face_x"].append(np.full(len(gset), face_meta[f]["face_x"]))
        rec["p0"].append(p0)
        rec["p1"].append(p1)
        rec["length"].append(length)
        rec["censor"].append((e0.astype(np.int32) + e1.astype(np.int32)))
        rec["normal"].append(nr)
        rec["e0"].append(e0)
        rec["e1"].append(e1)
    cat = {k: np.concatenate(v) for k, v in rec.items()}
    n_tr = len(cat["length"])
    print(f"[*] 총 절리선 {n_tr}개")
    for sid in np.unique(cat["set_id"]):
        m = cat["set_id"] == sid
        print(f"    - Set {int(sid)}: {int(m.sum())} traces, 총길이 {cat['length'][m].sum():.1f} m, "
              f"P21 = {cat['length'][m].sum() / total_area:.4f} m/m²")

    # ---- 5) HDF5/CSV 저장 (export_flat_face_traces 스키마) ----
    tr_dir = os.path.join(args.outdir, "trace_dataset")
    os.makedirs(tr_dir, exist_ok=True)
    h5_path = os.path.join(tr_dir, "trace_dataset_3d.h5")
    pl = np.empty((2 * n_tr, 3), dtype=np.float32)
    pl[0::2], pl[1::2] = cat["p0"], cat["p1"]
    fid = np.arange(n_tr, dtype=np.int32)  # 참값 균열 ID 없음 → 연번
    ep = lambda e: np.array([b"tunnel_boundary" if v else b"disc_boundary" for v in e],
                            dtype="S32")
    with h5py.File(h5_path, "w") as f:
        g = f.create_group("traces")
        g.create_dataset("trace_id", data=fid)
        g.create_dataset("fracture_id", data=fid)
        g.create_dataset("component_id", data=fid)
        g.create_dataset("set_id", data=cat["set_id"].astype(np.uint16))
        g.create_dataset("face_id", data=cat["face_id"].astype(np.uint16))
        g.create_dataset("face_x_m", data=cat["face_x"].astype(np.float32))
        g.create_dataset("face_mesh_name", data=np.array(
            [f"dfm_face_{int(i):06d}".encode() for i in cat["face_id"]], dtype="S64"))
        g.create_dataset("p0_xyz", data=cat["p0"].astype(np.float32))
        g.create_dataset("p1_xyz", data=cat["p1"].astype(np.float32))
        g.create_dataset("observed_length_m", data=cat["length"].astype(np.float32))
        g.create_dataset("censoring_class", data=cat["censor"].astype(np.uint8))
        g.create_dataset("radius_m", data=np.full(n_tr, np.nan, dtype=np.float32))
        g.create_dataset("trace_normal_xyz", data=cat["normal"].astype(np.float32))
        g.create_dataset("trace_normal_valid", data=np.ones(n_tr, dtype=np.uint8))
        g.create_dataset("trace_normal_quality", data=np.ones(n_tr, dtype=np.float32))
        g.create_dataset("trace_normal_reason", data=np.array(
            [b"external_dfm_patch"] * n_tr, dtype="S32"))
        g.create_dataset("polyline_vertex_start", data=(2 * np.arange(n_tr)).astype(np.int32))
        g.create_dataset("polyline_vertex_count", data=np.full(n_tr, 2, dtype=np.int32))
        g.create_dataset("polyline_vertices_xyz", data=pl)
        g.create_dataset("is_closed_loop", data=np.zeros(n_tr, dtype=np.uint8))
        g.create_dataset("p0_endpoint_type", data=ep(cat["e0"]))
        g.create_dataset("p1_endpoint_type", data=ep(cat["e1"]))
        m = f.create_group("meta")
        m.create_dataset("tunnel_poly_yz", data=window_poly.astype(np.float32))
        m.create_dataset("face_x_positions_m", data=np.array(
            [face_meta[f_]["face_x"] for f_ in faces], dtype=np.float32))
        m.create_dataset("observation_area_m2", data=np.array([total_area], dtype=np.float32))
        m.create_dataset("lmin_applied_m", data=np.array([0.0], dtype=np.float32))
        # 이중좌표계: p_world = p_pipeline @ R (신규 필드 — 기존 리더는 무시해도 무방)
        m.create_dataset("world_to_pipeline_rotation", data=R.astype(np.float64))
    print(f"[*] HDF5 written: {h5_path}")

    csv_path = os.path.join(tr_dir, "trace_dataset_3d.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["trace_id", "fracture_id", "set_id", "face_id", "face_x_m",
                    "p0_x", "p0_y", "p0_z", "p1_x", "p1_y", "p1_z",
                    "observed_length_m", "censoring_class", "radius_m",
                    "nx", "ny", "nz"])
        for i in range(n_tr):
            w.writerow([i, i, int(cat["set_id"][i]), int(cat["face_id"][i]),
                        f"{cat['face_x'][i]:.4f}",
                        *[f"{v:.6f}" for v in cat["p0"][i]],
                        *[f"{v:.6f}" for v in cat["p1"][i]],
                        f"{cat['length'][i]:.6f}", int(cat["censor"][i]), "nan",
                        *[f"{v:.6f}" for v in cat["normal"][i]]])
    print(f"[*] CSV written: {csv_path}")

    # ---- 6) 보조 DFN h5 (관측 법선 + 관측창; 조건부/P32 단계 입력) ----
    aux_path = os.path.join(args.outdir, "dfn_export_for_python.h5")
    mid = 0.5 * (cat["p0"] + cat["p1"])
    with h5py.File(aux_path, "w") as f:
        g = f.create_group("fractures")
        g.create_dataset("centers", data=mid.astype(np.float32))
        g.create_dataset("normals", data=cat["normal"].astype(np.float32))
        g.create_dataset("radii", data=(0.5 * cat["length"]).astype(np.float32))
        g.create_dataset("set_id", data=cat["set_id"].astype(np.int32))
        t = f.create_group("tunnel")
        t.create_dataset("poly_YZ", data=window_poly.astype(np.float64))
        m = f.create_group("meta")
        m.create_dataset("source", data=np.bytes_("dfm_export_observed_traces_only"))
    print(f"[*] 보조 DFN HDF5 written: {aux_path} (관측 절리선 기반 — 생성 DFN 아님)")

    # ---- 7) 진단 출력 ----
    with open(os.path.join(args.outdir, "set_mapping.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["face", "source_ds", "global_set", "n_traces"])
        w.writeheader()
        w.writerows(sorted(map_rows, key=lambda r: (r["face"], r["source_ds"])))
    diag = dict(
        dfm_dir=os.path.abspath(args.dfm_dir), n_faces=len(faces), n_traces=int(n_tr),
        tunnel_axis_world=[round(float(v), 6) for v in x_hat],
        # 이중좌표계: 파이프라인 좌표 = R @ world (순수 회전, 평행이동 없음).
        # world 복원: p_world = R.T @ p_pipeline  (행벡터로는 p_pipeline @ R)
        coordinate_system=dict(
            convention="pipeline = R @ world (rotation only, no translation); "
                       "x=굴진축(면 중심 궤적), z≈연직",
            world_to_pipeline_rotation=[[float(v) for v in row] for row in R],
            translation_world=[0.0, 0.0, 0.0],
        ),
        face_center_world={f_: [round(float(v), 4) for v in per_face[f_]["c_face"]]
                           for f_ in faces},
        face_normal_world={f_: [round(float(v), 6) for v in (
            per_face[f_]["n_face"] if per_face[f_]["n_face"] @ x_hat >= 0
            else -per_face[f_]["n_face"])] for f_ in faces},
        face_x_m={f_: round(face_meta[f_]["face_x"], 3) for f_ in faces},
        face_area_m2={f_: round(face_meta[f_]["area"], 2) for f_ in faces},
        observation_area_m2=round(total_area, 2), window_face=med_face,
        grid_cell_m=args.grid, edge_tol_m=args.edge_tol,
        cluster_cut_deg=args.cluster_cut,
        set_trace_counts={int(s): int((cat["set_id"] == s).sum())
                          for s in np.unique(cat["set_id"])},
        censoring_share=float((cat["censor"] > 0).mean()),
    )
    with open(os.path.join(args.outdir, "conversion_diagnostics.json"), "w",
              encoding="utf-8") as fh:
        json.dump(diag, fh, ensure_ascii=False, indent=2)
    print("[*] 진단 파일: set_mapping.csv, conversion_diagnostics.json")


if __name__ == "__main__":
    main()
