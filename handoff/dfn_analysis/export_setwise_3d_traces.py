# =============================================================================
# 파일 역할:
#   이 스크립트는 3D DFN(fracture disc 집합)과 rough face mesh(굴착면 메쉬)를
#   교차시켜, 각 절리군(set)별로 굴착면 위에 나타나는 3D trace(교선)를 추출하고
#   trace 데이터셋(CSV/HDF5)으로 내보낸다.
#
# 주요 입력:
#   --input        : DFN fracture(centers/normals/radii/set_id)와 터널 메타를
#                    담은 HDF5 파일.
#   --tunnel-dat   : (선택) HDF5에 터널 폴리곤이 없을 때 사용하는 DAT 단면 파일.
#   --rough-mesh-h5: (선택) rough face mesh 컬렉션(/rough_faces 또는 /rough_face)
#                    이 별도 HDF5에 있을 때 지정.
#
# 주요 출력:
#   trace_dataset_3d.csv : trace별 끝점/길이/censoring 등 표 형태 레코드.
#   trace_dataset_3d.h5  : 위와 동일 정보 + polyline vertex, trace normal 등을
#                          담은 HDF5(다운스트림 분석용).
#
# 핵심 처리 흐름:
#   1) HDF5에서 DFN과 터널 정보를 읽는다(load_hdf5_dfn).
#   2) rough face mesh 컬렉션을 읽어 face별 삼각형/경계 정보를 미리 계산
#      한다(load_rough_face_collection_from_h5 → precompute_face_mesh).
#   3) 각 face마다 bbox/plane 필터로 후보 fracture를 줄이고
#      (filter_fractures_for_face), disc-mesh 교차 세그먼트를 만든 뒤
#      (intersect_disc_with_face_mesh_segments), connected component 단위로
#      trace를 추출한다(extract_trace_components).
#   4) trace 레코드를 만들고(build_rows_rough_faces) CSV/HDF5로 저장한 뒤
#      절리군별 통계를 출력한다(print_summary).
# =============================================================================
import argparse
import csv
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from dfn_analysis.estimate_fisher_kappa import (
    estimate_fisher_k_axial,
    estimate_trace_normal_3pt,
)
from dfn_analysis.estimate_mean_orientation import normal_to_trend_plunge_ned

# trace_dataset_3d.csv 컬럼 설명:
# - trace_id:
#   CSV 전체에서 각 trace row에 부여한 고유 번호
# - face_id:
#   몇 번째 rough face mesh에서 추출된 trace인지 나타내는 번호
# - face_x_m:
#   해당 face의 기준 x 좌표(m)
# - fracture_id:
#   이 trace를 만든 원래 DFN fracture disc의 인덱스
# - set_id:
#   해당 fracture가 속한 절리군 번호
# - component_id:
#   같은 fracture가 같은 face에서 여러 disconnected trace component를 만들 때의 로컬 순번
# - p0_x, p0_y, p0_z:
#   trace 한쪽 끝점 p0의 전역 3D 좌표(m)
# - p1_x, p1_y, p1_z:
#   trace 다른 쪽 끝점 p1의 전역 3D 좌표(m)
# - observed_length_m:
#   rough face mesh 위에서 실제로 관측된 trace component 길이(m)
# - censoring_class:
#   trace 양 끝이 mesh/window 경계에서 잘렸는지 나타내는 분류
#   0=안 잘림, 1=한쪽 잘림, 2=양쪽 잘림
# - is_closed_loop:
#   열린 선분이 아니라 폐곡선 component인지 여부
#   0=open trace, 1=closed loop
# - n_raw_segments:
#   triangle-plane 교차에서 나온 raw segment가 몇 개 연결되어 이 component를 이루는지
# - p0_endpoint_type:
#   p0 끝점의 유형 (예: mesh_boundary, disc_boundary, interior)
# - p1_endpoint_type:
#   p1 끝점의 유형 (예: mesh_boundary, disc_boundary, interior)
# - face_mesh_name:
#   trace를 만든 rough face mesh 이름


# DFN HDF5에서 trace 계산에 필요한 fracture 기하와 터널/생성 메타를 읽어 dict로 반환한다.
#   인자:  h5_path - 입력 DFN HDF5 경로.
#   반환:  centers/normals/radii/set_ids와 터널 폴리곤, x 범위, 생성 반경 등을 담은 dict.
def load_hdf5_dfn(h5_path: str) -> dict:
    """trace 계산에 필요한 DFN과 터널 기본 정보만 읽는다."""
    with h5py.File(h5_path, "r") as f:
        # centers/normals를 읽고, (3, N) 형태로 저장돼 있으면 (N, 3)으로 전치한다.
        raw_c = f["/fractures/centers"][:]
        raw_n = f["/fractures/normals"][:]
        centers = raw_c.T if raw_c.shape[0] == 3 and raw_c.shape[0] < raw_c.shape[1] else raw_c
        normals = raw_n.T if raw_n.shape[0] == 3 and raw_n.shape[0] < raw_n.shape[1] else raw_n
        # 반경은 1D로 펴고, set_id는 없으면 모두 1번 절리군으로 채운다.
        radii = f["/fractures/radii"][:].ravel()
        set_ids = (
            f["/fractures/set_id"][:].ravel().astype(np.uint16)
            if "/fractures/set_id" in f
            else np.ones(len(radii), dtype=np.uint16)
        )

        # 터널 단면 폴리곤(YZ 평면)을 읽는다. 대문자/소문자 키를 모두 지원하고,
        # (2, N)이면 (N, 2)로 전치한다.
        poly_yz = None
        if "/tunnel/poly_YZ" in f:
            raw_p = f["/tunnel/poly_YZ"][:]
            poly_yz = raw_p.T if raw_p.shape[0] == 2 and raw_p.shape[0] < raw_p.shape[1] else raw_p
        elif "/tunnel/poly_yz" in f:
            raw_p = f["/tunnel/poly_yz"][:]
            poly_yz = raw_p.T if raw_p.shape[0] == 2 and raw_p.shape[0] < raw_p.shape[1] else raw_p

        # 터널 x 범위, crop box, 생성 반경 범위, 절리군별 반경 테이블 등 메타를 있으면 읽는다.
        x_start = float(f["/meta/x_start"][()]) if "/meta/x_start" in f else None
        x_end = float(f["/meta/x_end"][()]) if "/meta/x_end" in f else None
        crop_box = f["/meta/crop_box"][:].ravel() if "/meta/crop_box" in f else None
        generation_rmin = float(np.asarray(f["/meta/generation_rmin"][()]).ravel()[0]) if "/meta/generation_rmin" in f else None
        generation_rmax = float(np.asarray(f["/meta/generation_rmax"][()]).ravel()[0]) if "/meta/generation_rmax" in f else None
        set_meta_ids = f["/meta/set_ids"][:].ravel().astype(np.int32) if "/meta/set_ids" in f else None
        set_table_r0 = f["/meta/set_table_r0"][:].ravel().astype(np.float64) if "/meta/set_table_r0" in f else None
        set_generation_rmin = f["/meta/set_generation_rmin"][:].ravel().astype(np.float64) if "/meta/set_generation_rmin" in f else None
        set_effective_rmin = f["/meta/set_effective_rmin"][:].ravel().astype(np.float64) if "/meta/set_effective_rmin" in f else None

    # 이후 파이프라인에서 쓰기 좋은 float64 타입으로 정리해 dict로 반환한다.
    return {
        "centers": centers.astype(np.float64),
        "normals": normals.astype(np.float64),
        "radii": radii.astype(np.float64),
        "set_ids": set_ids,
        "poly_yz": poly_yz.astype(np.float64) if poly_yz is not None else None,
        "x_start": x_start,
        "x_end": x_end,
        "crop_box": crop_box.astype(np.float64) if crop_box is not None else None,
        "generation_rmin": generation_rmin,
        "generation_rmax": generation_rmax,
        "set_meta_ids": set_meta_ids,
        "set_table_r0": set_table_r0,
        "set_generation_rmin": set_generation_rmin,
        "set_effective_rmin": set_effective_rmin,
    }


# HDF5 group에서 스칼라 값을 안전하게 읽는 헬퍼.
#   인자:  group - h5py 그룹, key - 데이터셋 이름, default - 없을 때 반환할 기본값.
#   반환:  bytes는 문자열로 디코딩하고, 배열이면 첫 원소를 반환한다.
def _read_scalar(group: h5py.Group, key: str, default: Any) -> Any:
    if key not in group:
        return default
    value = group[key][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if np.ndim(value) > 0:
        return value.ravel()[0]
    return value


# rough face mesh 컬렉션을 HDF5에서 읽어 face 리스트(dict)로 반환한다.
#   인자:  h5_path - rough face가 담긴 HDF5 경로.
#   반환:  각 face의 face_id/face_x/source_name/정점/삼각형을 담은 dict 리스트.
#   비고:  신규 /rough_faces(다중 face), 구 /rough_face(단일), 최소 /mesh 스키마를 모두 지원.
def load_rough_face_collection_from_h5(h5_path: str) -> List[dict]:
    """새 /rough_faces 스키마와 기존 단일 /rough_face 스키마를 모두 읽는다."""
    with h5py.File(h5_path, "r") as f:
        # 신규 스키마: /rough_faces 아래 여러 face를 이름 순으로 순회하며 수집한다.
        if "rough_faces" in f:
            rough_faces = []
            faces_grp = f["rough_faces"]
            for face_name in sorted(faces_grp.keys()):
                grp = faces_grp[face_name]
                meta = grp["meta"] if "meta" in grp else None
                face_id = int(_read_scalar(meta, "face_id", len(rough_faces) + 1)) if meta else len(rough_faces) + 1
                face_x = float(_read_scalar(meta, "face_x", 0.0)) if meta else 0.0
                source_name = str(_read_scalar(meta, "source_name", face_name)) if meta else face_name
                rough_faces.append(
                    {
                        "face_id": face_id,
                        "face_x": face_x,
                        "source_name": source_name,
                        "vertices_xyz": grp["mesh/vertices_xyz"][:].astype(np.float64),
                        "triangles": grp["mesh/triangles"][:].astype(np.int32),
                    }
                )
            return rough_faces

        # 구 스키마: 단일 /rough_face 를 face 하나로 감싸서 반환한다.
        if "rough_face" in f:
            grp = f["rough_face"]
            meta = grp["meta"] if "meta" in grp else None
            base_x = float(_read_scalar(meta, "base_x", 0.0)) if meta else 0.0
            return [
                {
                    "face_id": 1,
                    "face_x": base_x,
                    "source_name": "rough_face",
                    "vertices_xyz": grp["mesh/vertices_xyz"][:].astype(np.float64),
                    "triangles": grp["mesh/triangles"][:].astype(np.int32),
                }
            ]

        # 최소 스키마: 최상위 /mesh 만 있는 경우도 face 하나로 처리한다.
        if "mesh" in f:
            meta = f["meta"] if "meta" in f else None
            base_x = float(_read_scalar(meta, "base_x", 0.0)) if meta else 0.0
            return [
                {
                    "face_id": 1,
                    "face_x": base_x,
                    "source_name": "mesh",
                    "vertices_xyz": f["mesh/vertices_xyz"][:].astype(np.float64),
                    "triangles": f["mesh/triangles"][:].astype(np.int32),
                }
            ]

    raise ValueError(f"Could not find rough face collection in: {h5_path}")


# DAT 파일에서 터널 단면 폴리곤을 파싱해 (N, 2) YZ 좌표 배열로 반환한다.
#   인자:  dat_path - 터널 단면 DAT 경로, scale - 단위 변환 계수(기본 mm→m).
#   반환:  각 행이 [y, z](m)인 폴리곤 정점 배열.
def load_tunnel_polygon_from_dat(dat_path: str, scale: float = 0.001) -> np.ndarray:
    """DAT 형식 터널 단면을 읽고 mm를 m로 변환한다."""
    poly_y = []
    poly_z = []
    # 각 줄에서 (y, z) 형태 좌표쌍을 정규식으로 뽑아 scale을 곱해 저장한다.
    with open(dat_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(r"\(\s*([\d\.-]+),\s*([\d\.-]+)\)", line)
            if not match:
                continue
            poly_y.append(float(match.group(1)) * scale)
            poly_z.append(float(match.group(2)) * scale)

    if not poly_y:
        raise ValueError(f"Failed to parse tunnel polygon from: {dat_path}")

    return np.column_stack([poly_y, poly_z]).astype(np.float64)


# 폴리곤의 부호 있는 면적(shoelace)을 계산한다.
#   인자:  poly_yz - (N, 2) YZ 정점 배열.
#   반환:  부호 있는 면적. 부호로 정점 진행 방향(CCW/CW)을 판정하는 데 쓴다.
def signed_polygon_area(poly_yz: np.ndarray) -> float:
    y = poly_yz[:, 0]
    z = poly_yz[:, 1]
    return 0.5 * float(np.dot(y, np.roll(z, -1)) - np.dot(z, np.roll(y, -1)))


# 삼각형 하나와 fracture가 놓인 무한 평면의 교선 세그먼트(두 점)를 구한다.
#   인자:  v0/v1/v2 - 삼각형 정점(xyz), center_xyz/normal_xyz - fracture 평면 정의.
#   반환:  교선 끝점 두 개의 튜플, 교차하지 않으면 None.
def intersect_triangle_with_plane(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, center_xyz: np.ndarray, normal_xyz: np.ndarray
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """삼각형과 fracture plane의 교선 세그먼트를 구한다."""
    # 각 정점의 평면에 대한 부호 있는 거리(법선 방향 투영)를 계산한다.
    d0 = np.dot(v0 - center_xyz, normal_xyz)
    d1 = np.dot(v1 - center_xyz, normal_xyz)
    d2 = np.dot(v2 - center_xyz, normal_xyz)
    # 세 정점이 모두 같은 쪽(전부 +, 또는 전부 -)이면 평면과 교차하지 않는다.
    if (d0 > 1e-9 and d1 > 1e-9 and d2 > 1e-9) or (d0 < -1e-9 and d1 < -1e-9 and d2 < -1e-9):
        return None

    # 세 변을 검사해 부호가 바뀌는 지점(교차점) 또는 평면 위의 정점을 모은다.
    pts = []
    if (d0 > 0 and d1 < 0) or (d0 < 0 and d1 > 0):
        t = -d0 / (d1 - d0)
        pts.append(v0 + t * (v1 - v0))
    elif abs(d0) < 1e-9:
        pts.append(v0)

    if (d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0):
        t = -d1 / (d2 - d1)
        pts.append(v1 + t * (v2 - v1))
    elif abs(d1) < 1e-9:
        pts.append(v1)

    if (d2 > 0 and d0 < 0) or (d2 < 0 and d0 > 0):
        t = -d2 / (d0 - d2)
        pts.append(v2 + t * (v0 - v2))
    elif abs(d2) < 1e-9:
        pts.append(v2)

    # 정점 공유 등으로 생기는 중복 교차점을 제거하고, 서로 다른 두 점이 있으면 세그먼트로 반환.
    unique_pts = []
    for p in pts:
        if not any(np.linalg.norm(p - up) < 1e-8 for up in unique_pts):
            unique_pts.append(p)
    if len(unique_pts) >= 2:
        return unique_pts[0], unique_pts[1]
    return None


# 평면-삼각형 교선 세그먼트를 반경 radius의 fracture disc(원) 안쪽으로만 잘라낸다.
#   인자:  e0/e1 - 세그먼트 끝점(xyz), center_xyz/radius - disc 중심과 반경.
#   반환:  disc 안에 들어오는 부분 세그먼트(두 점), 교차가 없으면 None.
def clip_segment_to_disc(
    e0: np.ndarray, e1: np.ndarray, center_xyz: np.ndarray, radius: float
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """plane-triangle 세그먼트를 fracture disc 안으로 자른다."""
    # 세그먼트를 매개변수 직선으로 보고 |P(t)-center|=radius 인 t를 이차방정식으로 푼다.
    direction = e1 - e0
    a = np.dot(direction, direction)
    # 길이가 거의 0인 세그먼트(퇴화)는 점이 disc 안에 있으면 그대로 반환한다.
    if a < 1e-12:
        if np.linalg.norm(e0 - center_xyz) <= radius:
            return e0, e1
        return None
    # 이차방정식 계수를 세워 판별식을 계산한다. 음수면 직선이 disc를 지나지 않는다.
    diff = e0 - center_xyz
    b = 2.0 * np.dot(direction, diff)
    c_quad = np.dot(diff, diff) - radius * radius
    discriminant = b * b - 4.0 * a * c_quad
    if discriminant < 0:
        return None
    # 두 교차 매개변수 t를 구한 뒤, 원래 세그먼트 구간 [0, 1]과 교집합을 취한다.
    sqrt_disc = np.sqrt(discriminant)
    t_min = (-b - sqrt_disc) / (2.0 * a)
    t_max = (-b + sqrt_disc) / (2.0 * a)
    t0 = max(0.0, t_min)
    t1 = min(1.0, t_max)
    # 교집합이 비었으면(진입/이탈 구간이 세그먼트 밖) disc 안 부분이 없다.
    if t0 > t1 + 1e-9:
        return None
    return e0 + t0 * direction, e0 + t1 * direction


# 좌표를 tol 격자에 반올림해 정수 튜플 키로 만든다(부동소수 오차로 같은 점이 갈라지는 것을 방지).
#   인자:  point_xyz - 3D 좌표, tol - 격자 크기.
#   반환:  (i, j, k) 정수 튜플 키.
def _quantize_point(point_xyz: np.ndarray, tol: float) -> Tuple[int, int, int]:
    return tuple(np.round(point_xyz / tol).astype(np.int64).tolist())


# 점과 선분 사이 최단 거리를 계산한다(끝점 유형 판정 시 mesh 경계와의 거리 측정에 사용).
#   인자:  point_xyz - 대상 점, seg0_xyz/seg1_xyz - 선분 끝점.
#   반환:  점에서 선분까지의 최단 거리(m).
def point_to_segment_distance(point_xyz: np.ndarray, seg0_xyz: np.ndarray, seg1_xyz: np.ndarray) -> float:
    segment = seg1_xyz - seg0_xyz
    seg_len_sq = float(np.dot(segment, segment))
    if seg_len_sq < 1e-18:
        return float(np.linalg.norm(point_xyz - seg0_xyz))
    t = float(np.dot(point_xyz - seg0_xyz, segment) / seg_len_sq)
    t = min(1.0, max(0.0, t))
    closest = seg0_xyz + t * segment
    return float(np.linalg.norm(point_xyz - closest))


# 하나의 connected component(그래프)를 따라가며 정점을 연결 순서대로 이어 polyline을 만든다.
#   인자:  component_nodes - 이 component의 노드 키들, graph - 인접 그래프, point_map - 키→좌표.
#   반환:  연결 순서대로 정렬된 정점들의 (M, 3) 배열.
def build_component_polyline_xyz(
    component_nodes: Sequence[Tuple[int, int, int]],
    graph: Dict[Tuple[int, int, int], set],
    point_map: Dict[Tuple[int, int, int], np.ndarray],
) -> np.ndarray:
    """connected trace component의 vertex들을 연결 순서대로 복원한다."""
    # 차수 1인 끝점(선분의 끝)에서 시작한다. 폐곡선이면 임의의 최소 노드에서 시작한다.
    degree_one_nodes = [node for node in component_nodes if len(graph[node]) == 1]
    if degree_one_nodes:
        start_key = min(degree_one_nodes)
    else:
        start_key = min(component_nodes)

    polyline_keys = [start_key]
    used_edges = set()
    current = start_key
    prev = None

    # 현재 노드에서 아직 안 쓴 간선을 따라 다음 노드로 이동하며 polyline을 확장한다.
    while True:
        # 우선 직전 노드로 되돌아가지 않는 미사용 간선을 고른다.
        next_key = None
        for candidate in sorted(graph[current]):
            edge_key = (current, candidate) if current < candidate else (candidate, current)
            if edge_key in used_edges:
                continue
            if prev is not None and candidate == prev:
                continue
            next_key = candidate
            break
        # 그런 간선이 없으면(막다른 곳) 남은 미사용 간선 아무거나 사용한다.
        if next_key is None:
            for candidate in sorted(graph[current]):
                edge_key = (current, candidate) if current < candidate else (candidate, current)
                if edge_key not in used_edges:
                    next_key = candidate
                    break
        # 더 갈 간선이 없으면 순회 종료.
        if next_key is None:
            break

        used_edges.add((current, next_key) if current < next_key else (next_key, current))
        polyline_keys.append(next_key)
        prev, current = current, next_key

    # 시작/끝이 다르지만 남은 간선 없이 폐곡선이면 시작점을 붙여 닫아준다.
    if polyline_keys[0] != polyline_keys[-1]:
        remaining_edges = set()
        for node in component_nodes:
            for nxt in graph[node]:
                remaining_edges.add((node, nxt) if node < nxt else (nxt, node))
        if used_edges != remaining_edges and not degree_one_nodes:
            polyline_keys.append(polyline_keys[0])

    # 정렬된 노드 키 순서를 실제 3D 좌표 배열로 변환해 반환한다.
    return np.vstack([point_map[key] for key in polyline_keys]).astype(np.float64)


# disc-mesh 교차로 나온 세그먼트 집합을 그래프로 묶어, 연결된 component(=개별 trace)별로
# 끝점 p0/p1, 관측 길이, censoring 분류 등을 계산해 dict 리스트로 반환한다.
#   인자:  segments - (p0, p1) 세그먼트 리스트, center_xyz/radius - 이 fracture disc 정의,
#          boundary_segments_xyz - mesh 경계 선분들, tol/eps_* - 격자·경계 판정 허용오차.
#   반환:  각 trace component의 끝점/polyline/길이/끝점유형 등을 담은 dict 리스트.
def extract_trace_components(
    segments: List[Tuple[np.ndarray, np.ndarray]],
    center_xyz: np.ndarray,
    radius: float,
    boundary_segments_xyz: np.ndarray,
    tol: float = 1e-5,
    eps_disc: float = 5e-3,
    eps_mesh: float = 5e-3,
) -> List[dict]:
    """세그먼트 집합에서 connected component별 p0/p1와 길이를 추출한다."""
    if not segments:
        return []

    # 세그먼트 끝점을 양자화 키로 하는 무방향 그래프를 만든다: graph=인접, edge_lengths=간선 길이.
    graph: Dict[Tuple[int, int, int], set] = {}
    point_map: Dict[Tuple[int, int, int], np.ndarray] = {}
    edge_lengths: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], float] = {}

    # 각 세그먼트를 노드 두 개와 간선 하나로 그래프에 등록한다.
    for p0_xyz, p1_xyz in segments:
        k0 = _quantize_point(p0_xyz, tol)
        k1 = _quantize_point(p1_xyz, tol)
        point_map.setdefault(k0, p0_xyz)
        point_map.setdefault(k1, p1_xyz)
        graph.setdefault(k0, set()).add(k1)
        graph.setdefault(k1, set()).add(k0)
        edge_key = (k0, k1) if k0 < k1 else (k1, k0)
        edge_lengths[edge_key] = float(np.linalg.norm(p1_xyz - p0_xyz))

    # 방문하지 않은 노드에서 DFS로 연결 요소(component)를 하나씩 찾아 처리한다.
    components = []
    visited = set()
    for start_key in graph:
        if start_key in visited:
            continue
        # DFS로 이 component에 속하는 모든 노드를 수집한다.
        stack = [start_key]
        component_nodes = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component_nodes.add(node)
            for nxt in graph[node]:
                if nxt not in visited:
                    stack.append(nxt)

        # component 내 간선을 모아 관측 길이(observed_length)를 합산한다(간선 중복 제거).
        component_edges = set()
        observed_length = 0.0
        for node in component_nodes:
            for nxt in graph[node]:
                edge_key = (node, nxt) if node < nxt else (nxt, node)
                if edge_key not in component_edges:
                    component_edges.add(edge_key)
                    observed_length += edge_lengths[edge_key]

        # 차수 1 노드가 없으면 폐곡선(closed loop)으로 표시한다.
        degree_one_nodes = [node for node in component_nodes if len(graph[node]) == 1]
        is_closed_loop = 1 if len(degree_one_nodes) == 0 else 0

        # 끝점 p0/p1 선택: 열린 trace면 두 끝점을, 폐곡선이면 가장 멀리 떨어진 두 점을 쓴다.
        if len(degree_one_nodes) >= 2:
            p0_key, p1_key = degree_one_nodes[0], degree_one_nodes[1]
        else:
            node_list = list(component_nodes)
            p0_key = node_list[0]
            p1_key = max(
                node_list,
                key=lambda key: np.linalg.norm(point_map[key] - point_map[p0_key]),
            )

        p0_xyz = point_map[p0_key]
        p1_xyz = point_map[p1_key]

        # 끝점 유형을 판정하는 내부 함수: mesh 경계 근처면 mesh_boundary, disc 원주 근처면
        # disc_boundary, 둘 다 아니면 interior(잘리지 않은 실제 끝)로 분류한다.
        def endpoint_type(point_xyz: np.ndarray, point_key: Tuple[int, int, int]) -> str:
            dist_to_disc_boundary = abs(np.linalg.norm(point_xyz - center_xyz) - radius)
            if len(boundary_segments_xyz):
                min_boundary_dist = min(
                    point_to_segment_distance(point_xyz, segment_xyz[0], segment_xyz[1])
                    for segment_xyz in boundary_segments_xyz
                )
            else:
                min_boundary_dist = float("inf")
            if min_boundary_dist < eps_mesh:
                return "mesh_boundary"
            if dist_to_disc_boundary < eps_disc:
                return "disc_boundary"
            return "interior"

        # 두 끝점 유형을 판정하고, mesh 경계에서 잘린 끝의 개수로 censoring_class(0/1/2)를 정한다.
        p0_type = endpoint_type(p0_xyz, p0_key)
        p1_type = endpoint_type(p1_xyz, p1_key)
        censoring_class = int(p0_type == "mesh_boundary") + int(p1_type == "mesh_boundary")
        # component의 정점들을 연결 순서대로 이은 polyline을 만든다.
        polyline_xyz = build_component_polyline_xyz(
            component_nodes=sorted(component_nodes),
            graph=graph,
            point_map=point_map,
        )

        # 이 trace component의 결과를 dict로 담는다.
        components.append(
            {
                "p0_xyz": p0_xyz,
                "p1_xyz": p1_xyz,
                "polyline_xyz": polyline_xyz,
                "observed_length_m": observed_length,
                "censoring_class": min(censoring_class, 2),
                "is_closed_loop": is_closed_loop,
                "n_raw_segments": len(component_edges),
                "p0_endpoint_type": p0_type,
                "p1_endpoint_type": p1_type,
            }
        )

    return components


# 한 face mesh에 대해 fracture마다 반복 사용될 삼각형 좌표/bbox/경계 정보를 미리 계산한다.
#   인자:  face_mesh - 정점/삼각형/face 메타를 담은 dict, tol - 경계 노드 양자화 허용오차.
#   반환:  삼각형별 정점(v0/v1/v2), bbox, mesh 경계 선분/노드 등을 담은 face context dict.
def precompute_face_mesh(face_mesh: dict, tol: float = 1e-8) -> dict:
    """face mesh에서 반복 사용될 삼각형/경계 정보를 한 번만 계산한다."""
    # 삼각형별 정점 좌표와 각 삼각형/전체 mesh의 bounding box를 구한다.
    vertices_xyz = face_mesh["vertices_xyz"].astype(np.float64)
    triangles = face_mesh["triangles"].astype(np.int32)
    v0 = vertices_xyz[triangles[:, 0]]
    v1 = vertices_xyz[triangles[:, 1]]
    v2 = vertices_xyz[triangles[:, 2]]
    tri_min = np.minimum(v0, np.minimum(v1, v2))
    tri_max = np.maximum(v0, np.maximum(v1, v2))
    mesh_bbox_min = np.min(vertices_xyz, axis=0)
    mesh_bbox_max = np.max(vertices_xyz, axis=0)

    # 각 변이 몇 개의 삼각형에 공유되는지 센다(한 번만 등장하는 변이 mesh 경계).
    edge_counts: Dict[Tuple[int, int], int] = {}
    for tri in triangles:
        t0, t1, t2 = int(tri[0]), int(tri[1]), int(tri[2])
        edges = [
            (t0, t1) if t0 < t1 else (t1, t0),
            (t1, t2) if t1 < t2 else (t2, t1),
            (t2, t0) if t2 < t0 else (t0, t2),
        ]
        for edge in edges:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    # 공유 횟수 1인 변만 골라 경계 정점과 경계 선분(끝점 유형 판정용)을 모은다.
    boundary_vertex_ids = set()
    boundary_segments_xyz = []
    for edge, count in edge_counts.items():
        if count == 1:
            boundary_vertex_ids.update(edge)
            boundary_segments_xyz.append(vertices_xyz[list(edge)].astype(np.float64))

    # 경계 정점을 양자화 키로, mesh bbox의 8개 꼭짓점을 미리 만들어 둔다(plane 필터에 사용).
    boundary_nodes = {_quantize_point(vertices_xyz[idx], tol) for idx in boundary_vertex_ids}
    bbox_corners = np.array(
        [
            [x, y, z]
            for x in [mesh_bbox_min[0], mesh_bbox_max[0]]
            for y in [mesh_bbox_min[1], mesh_bbox_max[1]]
            for z in [mesh_bbox_min[2], mesh_bbox_max[2]]
        ],
        dtype=np.float64,
    )

    return {
        "face_id": int(face_mesh["face_id"]),
        "face_x": float(face_mesh["face_x"]),
        "source_name": face_mesh["source_name"],
        "vertices_xyz": vertices_xyz,
        "triangles": triangles,
        "v0": v0,
        "v1": v1,
        "v2": v2,
        "tri_min": tri_min,
        "tri_max": tri_max,
        "mesh_bbox_min": mesh_bbox_min,
        "mesh_bbox_max": mesh_bbox_max,
        "bbox_corners": bbox_corners,
        "boundary_nodes": boundary_nodes,
        "boundary_segments_xyz": np.asarray(boundary_segments_xyz, dtype=np.float64),
    }


# 이 face와 교차할 가능성이 있는 fracture만 두 단계 필터로 빠르게 추려낸다.
#   인자:  data - DFN dict, face_ctx - precompute_face_mesh가 만든 face context.
#   반환:  후보 fracture의 인덱스 배열.
def filter_fractures_for_face(data: dict, face_ctx: dict) -> np.ndarray:
    """mesh bbox와 plane-range를 이용해 fracture 후보를 미리 줄인다."""
    centers = data["centers"]
    radii = data["radii"]
    normals = data["normals"]

    # 1차: fracture bounding box와 mesh bounding box가 겹치는 것만 남긴다.
    f_min = centers - radii[:, None]
    f_max = centers + radii[:, None]
    bbox_overlap = np.all((f_max >= face_ctx["mesh_bbox_min"]) & (f_min <= face_ctx["mesh_bbox_max"]), axis=1)
    candidate_ids = np.where(bbox_overlap)[0]
    if len(candidate_ids) == 0:
        return candidate_ids

    # 2차: mesh bbox 8개 꼭짓점이 fracture 평면 양쪽에 걸쳐 있는(평면이 bbox를 관통) 것만 남긴다.
    candidate_centers = centers[candidate_ids]
    candidate_normals = normals[candidate_ids]
    dots = np.einsum("fij,fj->fi", face_ctx["bbox_corners"][None, :, :] - candidate_centers[:, None, :], candidate_normals)
    plane_hits_bbox = (np.min(dots, axis=1) <= 0.0) & (np.max(dots, axis=1) >= 0.0)
    return candidate_ids[plane_hits_bbox]


# fracture disc 하나를 rough face mesh와 교차시켜, disc 안으로 잘린 교선 세그먼트들을 만든다.
#   인자:  center_xyz/normal_xyz/radius - fracture disc, face_ctx - face context.
#   반환:  (p0, p1) 세그먼트 리스트(disc 안쪽으로 clip되고 너무 짧은 것은 제외).
def intersect_disc_with_face_mesh_segments(
    center_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    radius: float,
    face_ctx: dict,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """rough face mesh와 fracture disc 교차 세그먼트를 생성한다."""
    # disc bbox와 겹치는 삼각형만 후보로 추린다.
    f_min = center_xyz - radius
    f_max = center_xyz + radius
    overlap = np.all((face_ctx["tri_max"] >= f_min) & (face_ctx["tri_min"] <= f_max), axis=1)
    candidate_indices = np.where(overlap)[0]

    # 후보 삼각형마다 평면 교선을 구하고 disc 안으로 잘라, 충분히 긴 세그먼트만 모은다.
    segments = []
    for idx in candidate_indices:
        plane_intersect = intersect_triangle_with_plane(
            face_ctx["v0"][idx],
            face_ctx["v1"][idx],
            face_ctx["v2"][idx],
            center_xyz,
            normal_xyz,
        )
        if plane_intersect is None:
            continue
        clipped = clip_segment_to_disc(plane_intersect[0], plane_intersect[1], center_xyz, radius)
        if clipped is not None and np.linalg.norm(clipped[1] - clipped[0]) >= 1e-3:
            segments.append(clipped)
    return segments


# trace 레코드를 trace_dataset_3d.csv로 저장한다(파일 상단 컬럼 설명 참고).
#   인자:  rows - trace 레코드 dict 리스트, csv_path - 출력 CSV 경로.
#   반환:  없음(파일 기록). fieldnames에 없는 키는 무시(extrasaction="ignore")된다.
def write_csv(rows: Sequence[dict], csv_path: str) -> None:
    fieldnames = [
        "trace_id",
        "face_id",
        "face_x_m",
        "fracture_id",
        "radius_m",
        "set_id",
        "component_id",
        "p0_x",
        "p0_y",
        "p0_z",
        "p1_x",
        "p1_y",
        "p1_z",
        "observed_length_m",
        "censoring_class",
        "is_closed_loop",
        "n_raw_segments",
        "p0_endpoint_type",
        "p1_endpoint_type",
        "face_mesh_name",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# trace 레코드와 메타를 trace_dataset_3d.h5로 저장한다(CSV보다 풍부한 필드 포함).
#   인자:  rows - trace 레코드, poly_yz - 터널 폴리곤, face_x - face x 위치 배열,
#          h5_path - 출력 경로, generation_*/set_* - 생성 반경·절리군 메타(선택).
#   반환:  없음(HDF5 기록). /traces 그룹과 /meta 그룹을 만든다.
def write_hdf5(
    rows: Sequence[dict],
    poly_yz: np.ndarray,
    face_x: np.ndarray,
    h5_path: str,
    generation_rmin: Optional[float] = None,
    generation_rmax: Optional[float] = None,
    set_meta_ids: Optional[np.ndarray] = None,
    set_table_r0: Optional[np.ndarray] = None,
    set_generation_rmin: Optional[np.ndarray] = None,
    set_effective_rmin: Optional[np.ndarray] = None,
) -> None:
    # 각 필드를 rows에서 뽑아 컬럼별 numpy 배열로 정리한다(빈 rows도 안전하게 처리).
    p0 = np.array([[r["p0_x"], r["p0_y"], r["p0_z"]] for r in rows], dtype=np.float32) if rows else np.zeros((0, 3), dtype=np.float32)
    p1 = np.array([[r["p1_x"], r["p1_y"], r["p1_z"]] for r in rows], dtype=np.float32) if rows else np.zeros((0, 3), dtype=np.float32)
    set_ids = np.array([r["set_id"] for r in rows], dtype=np.uint16)
    face_ids = np.array([r["face_id"] for r in rows], dtype=np.uint16)
    fracture_ids = np.array([r["fracture_id"] for r in rows], dtype=np.int32)
    radius_m = np.array([r["radius_m"] for r in rows], dtype=np.float32) if rows else np.zeros((0,), dtype=np.float32)
    trace_ids = np.array([r["trace_id"] for r in rows], dtype=np.int32)
    component_ids = np.array([r["component_id"] for r in rows], dtype=np.int32)
    censoring = np.array([r["censoring_class"] for r in rows], dtype=np.uint8)
    observed_length = np.array([r["observed_length_m"] for r in rows], dtype=np.float32)
    face_x_values = np.array([r["face_x_m"] for r in rows], dtype=np.float32)
    is_closed_loop = np.array([r["is_closed_loop"] for r in rows], dtype=np.uint8)
    n_raw_segments = np.array([r["n_raw_segments"] for r in rows], dtype=np.int32)
    p0_endpoint_type = np.array([r["p0_endpoint_type"].encode("utf-8") for r in rows], dtype="S32")
    p1_endpoint_type = np.array([r["p1_endpoint_type"].encode("utf-8") for r in rows], dtype="S32")
    face_mesh_name = np.array([r["face_mesh_name"].encode("utf-8") for r in rows], dtype="S64")
    trace_normal_xyz = np.array(
        [
            r["trace_normal_xyz"] if r["trace_normal_xyz"] is not None else np.array([np.nan, np.nan, np.nan], dtype=np.float32)
            for r in rows
        ],
        dtype=np.float32,
    ) if rows else np.zeros((0, 3), dtype=np.float32)
    trace_normal_valid = np.array([r["trace_normal_valid"] for r in rows], dtype=np.uint8)
    trace_normal_quality = np.array([r["trace_normal_quality"] for r in rows], dtype=np.float32)
    trace_normal_reason = np.array([r["trace_normal_reason"].encode("utf-8") for r in rows], dtype="S32")
    # 모든 trace의 polyline 정점을 하나의 배열로 이어붙이고, trace별 시작 offset/개수를 계산한다.
    polyline_vertex_counts = np.array([len(r["polyline_xyz"]) for r in rows], dtype=np.int32)
    polyline_vertex_starts = np.zeros(len(rows), dtype=np.int32)
    if len(rows):
        polyline_vertex_starts[1:] = np.cumsum(polyline_vertex_counts[:-1], dtype=np.int32)
    if rows:
        polyline_vertices_xyz = np.vstack([r["polyline_xyz"] for r in rows]).astype(np.float32)
    else:
        polyline_vertices_xyz = np.zeros((0, 3), dtype=np.float32)

    # /traces 그룹에 trace별 컬럼 데이터셋을 기록한다.
    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("traces")
        grp.create_dataset("trace_id", data=trace_ids)
        grp.create_dataset("fracture_id", data=fracture_ids)
        grp.create_dataset("radius_m", data=radius_m)
        grp.create_dataset("set_id", data=set_ids)
        grp.create_dataset("face_id", data=face_ids)
        grp.create_dataset("face_x_m", data=face_x_values)
        grp.create_dataset("component_id", data=component_ids)
        grp.create_dataset("observed_length_m", data=observed_length)
        grp.create_dataset("censoring_class", data=censoring)
        grp.create_dataset("is_closed_loop", data=is_closed_loop)
        grp.create_dataset("n_raw_segments", data=n_raw_segments)
        grp.create_dataset("p0_endpoint_type", data=p0_endpoint_type)
        grp.create_dataset("p1_endpoint_type", data=p1_endpoint_type)
        grp.create_dataset("face_mesh_name", data=face_mesh_name)
        grp.create_dataset("p0_xyz", data=p0)
        grp.create_dataset("p1_xyz", data=p1)
        grp.create_dataset("trace_normal_xyz", data=trace_normal_xyz)
        grp.create_dataset("trace_normal_valid", data=trace_normal_valid)
        grp.create_dataset("trace_normal_quality", data=trace_normal_quality)
        grp.create_dataset("trace_normal_reason", data=trace_normal_reason)
        grp.create_dataset("polyline_vertex_start", data=polyline_vertex_starts)
        grp.create_dataset("polyline_vertex_count", data=polyline_vertex_counts)
        grp.create_dataset("polyline_vertices_xyz", data=polyline_vertices_xyz)

        # /meta 그룹에 터널 폴리곤, face x 위치, 생성 반경·절리군 메타(있는 것만)를 기록한다.
        meta = f.create_group("meta")
        meta.create_dataset("tunnel_poly_yz", data=poly_yz.astype(np.float32))
        meta.create_dataset("face_x_positions_m", data=face_x.astype(np.float32))
        if generation_rmin is not None:
            meta.create_dataset("generation_rmin", data=np.array([generation_rmin], dtype=np.float32))
        if generation_rmax is not None:
            meta.create_dataset("generation_rmax", data=np.array([generation_rmax], dtype=np.float32))
        if set_meta_ids is not None:
            meta.create_dataset("set_ids", data=np.asarray(set_meta_ids, dtype=np.int32))
        if set_table_r0 is not None:
            meta.create_dataset("set_table_r0", data=np.asarray(set_table_r0, dtype=np.float32))
        if set_generation_rmin is not None:
            meta.create_dataset("set_generation_rmin", data=np.asarray(set_generation_rmin, dtype=np.float32))
        if set_effective_rmin is not None:
            meta.create_dataset("set_effective_rmin", data=np.asarray(set_effective_rmin, dtype=np.float32))


# 모든 face x fracture 조합을 순회하며 trace를 추출해 전체 trace 레코드 리스트를 만든다.
#   인자:  data - DFN dict, face_contexts - precompute된 face context 리스트.
#   반환:  CSV/HDF5로 저장할 trace 레코드 dict 리스트(trace_id는 전체에서 1부터 연속 부여).
def build_rows_rough_faces(data: dict, face_contexts: Sequence[dict]) -> List[dict]:
    """rough mode trace 레코드 생성."""
    rows = []
    trace_id = 1
    total_fractures = len(data["radii"])

    # face 하나씩 처리한다. 먼저 이 face와 교차 가능한 fracture 후보를 필터로 줄인다.
    for face_ctx in face_contexts:
        t0 = time.perf_counter()
        candidate_fracture_ids = filter_fractures_for_face(data, face_ctx)
        t1 = time.perf_counter()

        n_fractures_with_segments = 0
        n_trace_components = 0
        intersection_time = 0.0
        graph_time = 0.0

        # 후보 fracture마다 disc-mesh 교차 세그먼트를 만든다(없으면 건너뜀).
        for fracture_id in candidate_fracture_ids:
            t_intersect_start = time.perf_counter()
            segments = intersect_disc_with_face_mesh_segments(
                data["centers"][fracture_id],
                data["normals"][fracture_id],
                float(data["radii"][fracture_id]),
                face_ctx,
            )
            intersection_time += time.perf_counter() - t_intersect_start
            if not segments:
                continue

            # 세그먼트를 connected component(개별 trace) 단위로 묶어 끝점/길이 등을 추출한다.
            n_fractures_with_segments += 1
            t_graph_start = time.perf_counter()
            components = extract_trace_components(
                segments=segments,
                center_xyz=data["centers"][fracture_id],
                radius=float(data["radii"][fracture_id]),
                boundary_segments_xyz=face_ctx["boundary_segments_xyz"],
            )
            graph_time += time.perf_counter() - t_graph_start

            # 각 trace component마다 3점 기반 법선을 추정하고 한 개의 레코드 행으로 만든다.
            for component_id, seg in enumerate(components):
                normal_est = estimate_trace_normal_3pt(seg["polyline_xyz"])
                rows.append(
                    {
                        "trace_id": trace_id,
                        "face_id": int(face_ctx["face_id"]),
                        "face_x_m": float(face_ctx["face_x"]),
                        "fracture_id": int(fracture_id),
                        "radius_m": float(data["radii"][fracture_id]),
                        "set_id": int(data["set_ids"][fracture_id]),
                        "component_id": component_id,
                        "p0_x": float(seg["p0_xyz"][0]),
                        "p0_y": float(seg["p0_xyz"][1]),
                        "p0_z": float(seg["p0_xyz"][2]),
                        "p1_x": float(seg["p1_xyz"][0]),
                        "p1_y": float(seg["p1_xyz"][1]),
                        "p1_z": float(seg["p1_xyz"][2]),
                        "observed_length_m": float(seg["observed_length_m"]),
                        "censoring_class": int(seg["censoring_class"]),
                        "is_closed_loop": int(seg["is_closed_loop"]),
                        "n_raw_segments": int(seg["n_raw_segments"]),
                        "p0_endpoint_type": seg["p0_endpoint_type"],
                        "p1_endpoint_type": seg["p1_endpoint_type"],
                        "trace_normal_xyz": None if normal_est["normal"] is None else normal_est["normal"].astype(np.float64),
                        "trace_normal_valid": int(normal_est["valid"]),
                        "trace_normal_quality": float(normal_est["quality"]),
                        "trace_normal_reason": normal_est["reason"],
                        "polyline_xyz": seg["polyline_xyz"].astype(np.float64),
                        "face_mesh_name": face_ctx["source_name"],
                    }
                )
                trace_id += 1
                n_trace_components += 1

        # face별 처리 통계(삼각형 수, 후보/교차 fracture 수, trace 수, 단계별 소요시간)를 출력한다.
        print(
            f"[*] Face {face_ctx['face_id']:03d} @ x={face_ctx['face_x']:.2f} m | "
            f"triangles={len(face_ctx['triangles']):,}, fractures_total={total_fractures:,}, "
            f"fractures_candidate_after_bbox={len(candidate_fracture_ids):,}, "
            f"fractures_with_segments={n_fractures_with_segments:,}, trace_components={n_trace_components:,}, "
            f"time_filter={t1 - t0:.3f}s, time_intersection={intersection_time:.3f}s, time_endpoint_graph={graph_time:.3f}s"
        )

    return rows


# 추출된 trace를 절리군(set)별로 묶어 개수/총길이/Fisher kappa/평균 방향 등 통계를 출력한다.
#   인자:  rows - trace 레코드 리스트.
#   반환:  없음(콘솔 출력만).
def print_summary(rows: Sequence[dict]) -> None:
    print(f"[*] Exported {len(rows):,} clipped 3D traces.")
    if not rows:
        return
    # 등장하는 절리군 id를 정렬해 하나씩 통계를 계산한다.
    set_ids = sorted({row["set_id"] for row in rows})
    for set_id in set_ids:
        # 이 set의 trace 개수와 관측 총 길이, 유효 법선만 모아 Fisher kappa(방향 집중도)를 추정한다.
        set_rows = [row for row in rows if row["set_id"] == set_id]
        total_length = sum(row["observed_length_m"] for row in set_rows)
        valid_normals = [
            row["trace_normal_xyz"]
            for row in set_rows
            if row["trace_normal_valid"] and row["trace_normal_xyz"] is not None
        ]
        fisher_stats = estimate_fisher_k_axial(np.vstack(valid_normals)) if valid_normals else {
            "valid": False,
            "kappa": np.nan,
            "mean_normal": None,
            "n": 0,
            "resultant_length": 0.0,
        }
        valid_count = sum(int(row["trace_normal_valid"]) for row in set_rows)
        mean_normal = fisher_stats["mean_normal"]
        
        # 평균 법선이 유효하면 벡터 표기와 함께 trend/plunge(주향·경사 방향) 각도로 변환해 표시한다.
        mean_normal_text = "None"
        trend_plunge_text = "None"
        if mean_normal is not None and isinstance(mean_normal, np.ndarray):
            mean_normal_text = (
                f"[{mean_normal[0]:+.4f}, {mean_normal[1]:+.4f}, {mean_normal[2]:+.4f}]"
            )
            trend, plunge = normal_to_trend_plunge_ned(mean_normal)
            if trend is not None and plunge is not None:
                trend_plunge_text = f"{trend:05.1f}° / {plunge:04.1f}°"

        # kappa를 안전하게 float로 변환(비유한 값 처리)한 뒤 set별 한 줄 요약을 출력한다.
        kappa_val = fisher_stats["kappa"]
        kappa_float = float(kappa_val) if (isinstance(kappa_val, (int, float)) or isinstance(kappa_val, np.number)) else np.nan
        kappa_text = f"{kappa_float:.3f}" if np.isfinite(kappa_float) else str(kappa_float)
        print(
            f"    - Set {set_id}: {len(set_rows):,} traces, observed total length = {total_length:.3f} m, "
            f"valid_normals = {valid_count:,}, Fisher kappa = {kappa_text}, "
            f"mean_normal = {mean_normal_text}, Trend/Plunge = {trend_plunge_text}"
        )


# CLI 진입점: 인자를 파싱하고 DFN/터널/rough face를 읽어 trace를 추출한 뒤 CSV/HDF5로 저장한다.
#   인자:  없음(argparse로 커맨드라인 인자 처리).
#   반환:  없음(파일 출력 및 콘솔 요약).
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export per-set 3D trace datasets from face-wise rough face mesh collections."
    )
    parser.add_argument("--input", required=True, help="Input HDF5 DFN file")
    parser.add_argument("--outdir", default="storage/output/trace_dataset_collection", help="Output directory")
    parser.add_argument("--tunnel-dat", help="Optional tunnel polygon .dat file when HDF5 has no tunnel polygon")
    parser.add_argument("--rough-mesh-h5", help="Optional HDF5 containing /rough_faces or legacy /rough_face")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # DFN과 터널 폴리곤을 읽는다. HDF5에 폴리곤이 없으면 --tunnel-dat에서 읽어야 한다.
    data = load_hdf5_dfn(args.input)
    poly_yz = data["poly_yz"]

    if poly_yz is None:
        if not args.tunnel_dat:
            raise ValueError("Tunnel polygon not found in HDF5. Provide --tunnel-dat.")
        poly_yz = load_tunnel_polygon_from_dat(args.tunnel_dat)

    # 폴리곤 진행 방향을 CCW로 정규화한다(면적 부호가 음수면 뒤집는다).
    if signed_polygon_area(poly_yz) < 0.0:
        poly_yz = poly_yz[::-1].copy()

    # rough face 컬렉션은 별도 파일(--rough-mesh-h5) 또는 입력 HDF5에서 읽는다.
    rough_face_source = args.rough_mesh_h5 if args.rough_mesh_h5 else args.input
    try:
        rough_faces = load_rough_face_collection_from_h5(rough_face_source)
    except ValueError as exc:
        raise ValueError(
            f"Rough face collection is required. Could not find /rough_faces or /rough_face in: {rough_face_source}"
        ) from exc

    # 각 face의 삼각형/경계 정보를 미리 계산하고, 모든 face에 대해 trace 레코드를 만든다.
    print(f"[*] Using face-wise rough face collection from: {rough_face_source}")
    face_contexts = [precompute_face_mesh(face) for face in rough_faces]
    rows = build_rows_rough_faces(data, face_contexts)
    face_x = np.array([ctx["face_x"] for ctx in face_contexts], dtype=np.float64)

    # 결과를 CSV와 HDF5로 저장한다.
    csv_path = os.path.join(args.outdir, "trace_dataset_3d.csv")
    h5_path = os.path.join(args.outdir, "trace_dataset_3d.h5")
    write_csv(rows, csv_path)
    write_hdf5(
        rows,
        poly_yz,
        face_x,
        h5_path,
        data.get("generation_rmin"),
        data.get("generation_rmax"),
        data.get("set_meta_ids"),
        data.get("set_table_r0"),
        data.get("set_generation_rmin"),
        data.get("set_effective_rmin"),
    )

    # 절리군별 통계와 출력 경로를 콘솔에 요약한다.
    print_summary(rows)
    print(f"[*] CSV written to: {csv_path}")
    print(f"[*] HDF5 written to: {h5_path}")


if __name__ == "__main__":
    main()
