import argparse
import csv
import os
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

# trace_dataset_3d.csv 컬럼 설명:
# - trace_id:
#   CSV 전체에서 각 trace row에 부여한 고유 번호
# - face_id:
#   몇 번째 rough face mesh 또는 flat face에서 추출된 trace인지 나타내는 번호
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
#   rough face mesh 또는 flat face 위에서 실제로 관측된 trace component 길이(m)
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
#   trace를 만든 face mesh 이름 또는 flat mode 식별자


def load_hdf5_dfn(h5_path: str) -> dict:
    """trace 계산에 필요한 DFN과 터널 기본 정보만 읽는다."""
    with h5py.File(h5_path, "r") as f:
        raw_c = f["/fractures/centers"][:]
        raw_n = f["/fractures/normals"][:]
        centers = raw_c.T if raw_c.shape[0] == 3 and raw_c.shape[0] < raw_c.shape[1] else raw_c
        normals = raw_n.T if raw_n.shape[0] == 3 and raw_n.shape[0] < raw_n.shape[1] else raw_n
        radii = f["/fractures/radii"][:].ravel()
        set_ids = (
            f["/fractures/set_id"][:].ravel().astype(np.uint16)
            if "/fractures/set_id" in f
            else np.ones(len(radii), dtype=np.uint16)
        )

        poly_yz = None
        if "/tunnel/poly_YZ" in f:
            raw_p = f["/tunnel/poly_YZ"][:]
            poly_yz = raw_p.T if raw_p.shape[0] == 2 and raw_p.shape[0] < raw_p.shape[1] else raw_p
        elif "/tunnel/poly_yz" in f:
            raw_p = f["/tunnel/poly_yz"][:]
            poly_yz = raw_p.T if raw_p.shape[0] == 2 and raw_p.shape[0] < raw_p.shape[1] else raw_p

        x_start = float(f["/meta/x_start"][()]) if "/meta/x_start" in f else None
        x_end = float(f["/meta/x_end"][()]) if "/meta/x_end" in f else None
        crop_box = f["/meta/crop_box"][:].ravel() if "/meta/crop_box" in f else None

    return {
        "centers": centers.astype(np.float64),
        "normals": normals.astype(np.float64),
        "radii": radii.astype(np.float64),
        "set_ids": set_ids,
        "poly_yz": poly_yz.astype(np.float64) if poly_yz is not None else None,
        "x_start": x_start,
        "x_end": x_end,
        "crop_box": crop_box.astype(np.float64) if crop_box is not None else None,
    }


def _read_scalar(group: h5py.Group, key: str, default) -> object:
    if key not in group:
        return default
    value = group[key][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if np.ndim(value) > 0:
        return value.ravel()[0]
    return value


def load_rough_face_collection_from_h5(h5_path: str) -> List[dict]:
    """새 /rough_faces 스키마와 기존 단일 /rough_face 스키마를 모두 읽는다."""
    with h5py.File(h5_path, "r") as f:
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


def load_tunnel_polygon_from_dat(dat_path: str, scale: float = 0.001) -> np.ndarray:
    """DAT 형식 터널 단면을 읽고 mm를 m로 변환한다."""
    poly_y = []
    poly_z = []
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


def signed_polygon_area(poly_yz: np.ndarray) -> float:
    y = poly_yz[:, 0]
    z = poly_yz[:, 1]
    return 0.5 * float(np.dot(y, np.roll(z, -1)) - np.dot(z, np.roll(y, -1)))


def point_on_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> bool:
    ab = b - a
    ap = point - a
    cross = ab[0] * ap[1] - ab[1] * ap[0]
    if abs(cross) > tol:
        return False
    dot = np.dot(ap, ab)
    if dot < -tol:
        return False
    if dot - np.dot(ab, ab) > tol:
        return False
    return True


def point_in_polygon(point_yz: np.ndarray, poly_yz: np.ndarray) -> bool:
    y, z = float(point_yz[0]), float(point_yz[1])
    inside = False
    n = len(poly_yz)
    for i in range(n):
        a = poly_yz[i]
        b = poly_yz[(i + 1) % n]
        if point_on_segment(point_yz, a, b):
            return True
        yi, zi = a
        yj, zj = b
        intersects = ((zi > z) != (zj > z)) and (y < (yj - yi) * (z - zi) / (zj - zi + 1e-15) + yi)
        if intersects:
            inside = not inside
    return inside


def segment_intersection_2d(
    p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray, tol: float = 1e-9
) -> Optional[Tuple[float, np.ndarray]]:
    r = p1 - p0
    s = q1 - q0
    rxs = r[0] * s[1] - r[1] * s[0]
    qp = q0 - p0
    qpxr = qp[0] * r[1] - qp[1] * r[0]
    if abs(rxs) <= tol and abs(qpxr) <= tol:
        return None
    if abs(rxs) <= tol:
        return None
    t = (qp[0] * s[1] - qp[1] * s[0]) / rxs
    u = (qp[0] * r[1] - qp[1] * r[0]) / rxs
    if -tol <= t <= 1.0 + tol and -tol <= u <= 1.0 + tol:
        point = p0 + np.clip(t, 0.0, 1.0) * r
        return float(np.clip(t, 0.0, 1.0)), point
    return None


def clip_segment_to_polygon(
    p0_yz: np.ndarray, p1_yz: np.ndarray, poly_yz: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    t_values = []
    if point_in_polygon(p0_yz, poly_yz):
        t_values.append((0.0, p0_yz.copy(), True))
    if point_in_polygon(p1_yz, poly_yz):
        t_values.append((1.0, p1_yz.copy(), True))
    for i in range(len(poly_yz)):
        q0 = poly_yz[i]
        q1 = poly_yz[(i + 1) % len(poly_yz)]
        hit = segment_intersection_2d(p0_yz, p1_yz, q0, q1)
        if hit is not None:
            t_values.append((hit[0], hit[1], False))
    if not t_values:
        return []
    t_values.sort(key=lambda item: item[0])
    unique = []
    for t_val, point, is_inside in t_values:
        if unique and abs(t_val - unique[-1][0]) < 1e-8:
            continue
        unique.append((t_val, point, is_inside))
    clipped = []
    for idx in range(len(unique) - 1):
        t0, p0, _ = unique[idx]
        t1, p1, _ = unique[idx + 1]
        if t1 - t0 < 1e-8:
            continue
        mid_t = 0.5 * (t0 + t1)
        mid = p0_yz + mid_t * (p1_yz - p0_yz)
        if point_in_polygon(mid, poly_yz):
            n_clipped = int(t0 > 1e-8) + int(t1 < 1.0 - 1e-8)
            clipped.append((p0.copy(), p1.copy(), n_clipped))
    if not clipped and len(unique) == 1 and point_in_polygon(unique[0][1], poly_yz):
        clipped.append((unique[0][1].copy(), unique[0][1].copy(), 0))
    return clipped


def intersect_disc_with_face(
    center_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    radius: float,
    face_x: float,
    poly_yz: np.ndarray,
) -> List[dict]:
    """기존 flat face용 analytic trace 계산."""
    cx, cy, cz = center_xyz
    nx, ny, nz = normal_xyz
    yz_norm_sq = ny * ny + nz * nz
    if yz_norm_sq < 1e-12:
        return []
    dist_to_line = abs(face_x - cx) / np.sqrt(yz_norm_sq)
    if dist_to_line > radius + 1e-12:
        return []
    c_rhs = nx * (cx - face_x) + ny * cy + nz * cz
    factor = (ny * cy + nz * cz - c_rhs) / yz_norm_sq
    chord_mid_yz = np.array([cy - ny * factor, cz - nz * factor], dtype=np.float64)
    chord_half_len = float(np.sqrt(max(radius * radius - dist_to_line * dist_to_line, 0.0)))
    chord_dir_yz = np.array([-nz, ny], dtype=np.float64) / np.sqrt(yz_norm_sq)
    full_p0_yz = chord_mid_yz - chord_half_len * chord_dir_yz
    full_p1_yz = chord_mid_yz + chord_half_len * chord_dir_yz
    clipped_segments = clip_segment_to_polygon(full_p0_yz, full_p1_yz, poly_yz)
    rows = []
    for component_id, (clip_p0_yz, clip_p1_yz, n_clipped) in enumerate(clipped_segments):
        clip_p0_xyz = np.array([face_x, clip_p0_yz[0], clip_p0_yz[1]], dtype=np.float64)
        clip_p1_xyz = np.array([face_x, clip_p1_yz[0], clip_p1_yz[1]], dtype=np.float64)
        observed_length = float(np.linalg.norm(clip_p1_xyz - clip_p0_xyz))
        rows.append(
            {
                "component_id": component_id,
                "p0_xyz": clip_p0_xyz,
                "p1_xyz": clip_p1_xyz,
                "observed_length_m": observed_length,
                "censoring_class": min(n_clipped, 2),
                "is_closed_loop": 0,
                "n_raw_segments": 1,
                "p0_endpoint_type": "mesh_boundary" if n_clipped > 0 else "disc_boundary",
                "p1_endpoint_type": "mesh_boundary" if n_clipped > 1 else "disc_boundary",
            }
        )
    return rows


def intersect_triangle_with_plane(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, center_xyz: np.ndarray, normal_xyz: np.ndarray
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """삼각형과 fracture plane의 교선 세그먼트를 구한다."""
    d0 = np.dot(v0 - center_xyz, normal_xyz)
    d1 = np.dot(v1 - center_xyz, normal_xyz)
    d2 = np.dot(v2 - center_xyz, normal_xyz)
    if (d0 > 1e-9 and d1 > 1e-9 and d2 > 1e-9) or (d0 < -1e-9 and d1 < -1e-9 and d2 < -1e-9):
        return None

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

    unique_pts = []
    for p in pts:
        if not any(np.linalg.norm(p - up) < 1e-8 for up in unique_pts):
            unique_pts.append(p)
    if len(unique_pts) >= 2:
        return unique_pts[0], unique_pts[1]
    return None


def clip_segment_to_disc(
    e0: np.ndarray, e1: np.ndarray, center_xyz: np.ndarray, radius: float
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """plane-triangle 세그먼트를 fracture disc 안으로 자른다."""
    direction = e1 - e0
    a = np.dot(direction, direction)
    if a < 1e-12:
        if np.linalg.norm(e0 - center_xyz) <= radius:
            return e0, e1
        return None
    diff = e0 - center_xyz
    b = 2.0 * np.dot(direction, diff)
    c_quad = np.dot(diff, diff) - radius * radius
    discriminant = b * b - 4.0 * a * c_quad
    if discriminant < 0:
        return None
    sqrt_disc = np.sqrt(discriminant)
    t_min = (-b - sqrt_disc) / (2.0 * a)
    t_max = (-b + sqrt_disc) / (2.0 * a)
    t0 = max(0.0, t_min)
    t1 = min(1.0, t_max)
    if t0 > t1 + 1e-9:
        return None
    return e0 + t0 * direction, e0 + t1 * direction


def _quantize_point(point_xyz: np.ndarray, tol: float) -> Tuple[int, int, int]:
    return tuple(np.round(point_xyz / tol).astype(np.int64).tolist())


def extract_trace_components(
    segments: List[Tuple[np.ndarray, np.ndarray]],
    center_xyz: np.ndarray,
    radius: float,
    boundary_nodes: set,
    tol: float = 1e-5,
    eps_disc: float = 5e-3,
) -> List[dict]:
    """세그먼트 집합에서 connected component별 p0/p1와 길이를 추출한다."""
    if not segments:
        return []

    graph: Dict[Tuple[int, int, int], set] = {}
    point_map: Dict[Tuple[int, int, int], np.ndarray] = {}
    edge_lengths: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], float] = {}

    for p0_xyz, p1_xyz in segments:
        k0 = _quantize_point(p0_xyz, tol)
        k1 = _quantize_point(p1_xyz, tol)
        point_map.setdefault(k0, p0_xyz)
        point_map.setdefault(k1, p1_xyz)
        graph.setdefault(k0, set()).add(k1)
        graph.setdefault(k1, set()).add(k0)
        edge_key = tuple(sorted([k0, k1]))
        edge_lengths[edge_key] = float(np.linalg.norm(p1_xyz - p0_xyz))

    components = []
    visited = set()
    for start_key in graph:
        if start_key in visited:
            continue
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

        component_edges = set()
        observed_length = 0.0
        for node in component_nodes:
            for nxt in graph[node]:
                edge_key = tuple(sorted([node, nxt]))
                if edge_key not in component_edges:
                    component_edges.add(edge_key)
                    observed_length += edge_lengths[edge_key]

        degree_one_nodes = [node for node in component_nodes if len(graph[node]) == 1]
        is_closed_loop = 1 if len(degree_one_nodes) == 0 else 0

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

        def endpoint_type(point_xyz: np.ndarray, point_key: Tuple[int, int, int]) -> str:
            dist_to_disc_boundary = abs(np.linalg.norm(point_xyz - center_xyz) - radius)
            if point_key in boundary_nodes:
                return "mesh_boundary"
            if dist_to_disc_boundary < eps_disc:
                return "disc_boundary"
            return "interior"

        p0_type = endpoint_type(p0_xyz, p0_key)
        p1_type = endpoint_type(p1_xyz, p1_key)
        censoring_class = int(p0_type == "mesh_boundary") + int(p1_type == "mesh_boundary")

        components.append(
            {
                "p0_xyz": p0_xyz,
                "p1_xyz": p1_xyz,
                "observed_length_m": observed_length,
                "censoring_class": min(censoring_class, 2),
                "is_closed_loop": is_closed_loop,
                "n_raw_segments": len(component_edges),
                "p0_endpoint_type": p0_type,
                "p1_endpoint_type": p1_type,
            }
        )

    return components


def precompute_face_mesh(face_mesh: dict, tol: float = 1e-8) -> dict:
    """face mesh에서 반복 사용될 삼각형/경계 정보를 한 번만 계산한다."""
    vertices_xyz = face_mesh["vertices_xyz"].astype(np.float64)
    triangles = face_mesh["triangles"].astype(np.int32)
    v0 = vertices_xyz[triangles[:, 0]]
    v1 = vertices_xyz[triangles[:, 1]]
    v2 = vertices_xyz[triangles[:, 2]]
    tri_min = np.minimum(v0, np.minimum(v1, v2))
    tri_max = np.maximum(v0, np.maximum(v1, v2))
    mesh_bbox_min = np.min(vertices_xyz, axis=0)
    mesh_bbox_max = np.max(vertices_xyz, axis=0)

    edge_counts: Dict[Tuple[int, int], int] = {}
    for tri in triangles:
        edges = [
            tuple(sorted((int(tri[0]), int(tri[1])))),
            tuple(sorted((int(tri[1]), int(tri[2])))),
            tuple(sorted((int(tri[2]), int(tri[0])))),
        ]
        for edge in edges:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    boundary_vertex_ids = set()
    for edge, count in edge_counts.items():
        if count == 1:
            boundary_vertex_ids.update(edge)

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
    }


def filter_fractures_for_face(data: dict, face_ctx: dict) -> np.ndarray:
    """mesh bbox와 plane-range를 이용해 fracture 후보를 미리 줄인다."""
    centers = data["centers"]
    radii = data["radii"]
    normals = data["normals"]

    f_min = centers - radii[:, None]
    f_max = centers + radii[:, None]
    bbox_overlap = np.all((f_max >= face_ctx["mesh_bbox_min"]) & (f_min <= face_ctx["mesh_bbox_max"]), axis=1)
    candidate_ids = np.where(bbox_overlap)[0]
    if len(candidate_ids) == 0:
        return candidate_ids

    candidate_centers = centers[candidate_ids]
    candidate_normals = normals[candidate_ids]
    dots = np.einsum("fij,fj->fi", face_ctx["bbox_corners"][None, :, :] - candidate_centers[:, None, :], candidate_normals)
    plane_hits_bbox = (np.min(dots, axis=1) <= 0.0) & (np.max(dots, axis=1) >= 0.0)
    return candidate_ids[plane_hits_bbox]


def intersect_disc_with_face_mesh_segments(
    center_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    radius: float,
    face_ctx: dict,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """rough face mesh와 fracture disc 교차 세그먼트를 생성한다."""
    f_min = center_xyz - radius
    f_max = center_xyz + radius
    overlap = np.all((face_ctx["tri_max"] >= f_min) & (face_ctx["tri_min"] <= f_max), axis=1)
    candidate_indices = np.where(overlap)[0]

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


def resolve_face_range(data: dict, args: argparse.Namespace) -> np.ndarray:
    """flat mode에서 사용할 x=const face 위치들을 정한다."""
    if args.face_x_csv:
        return np.array([float(x.strip()) for x in args.face_x_csv.split(",") if x.strip()], dtype=np.float64)
    if args.x_start is not None and args.x_end is not None:
        return np.arange(args.x_start, args.x_end + 1e-9, args.face_step, dtype=np.float64)
    if data["x_start"] is not None and data["x_end"] is not None:
        return np.arange(data["x_start"], data["x_end"] + 1e-9, args.face_step, dtype=np.float64)
    if data["crop_box"] is not None:
        return np.arange(data["crop_box"][0], data["crop_box"][1] + 1e-9, args.face_step, dtype=np.float64)
    xmin = float(np.min(data["centers"][:, 0]))
    xmax = float(np.max(data["centers"][:, 0]))
    return np.arange(xmin, xmax + 1e-9, args.face_step, dtype=np.float64)


def write_csv(rows: Sequence[dict], csv_path: str) -> None:
    fieldnames = [
        "trace_id",
        "face_id",
        "face_x_m",
        "fracture_id",
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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_hdf5(rows: Sequence[dict], poly_yz: np.ndarray, face_x: np.ndarray, h5_path: str) -> None:
    p0 = np.array([[r["p0_x"], r["p0_y"], r["p0_z"]] for r in rows], dtype=np.float32) if rows else np.zeros((0, 3), dtype=np.float32)
    p1 = np.array([[r["p1_x"], r["p1_y"], r["p1_z"]] for r in rows], dtype=np.float32) if rows else np.zeros((0, 3), dtype=np.float32)
    set_ids = np.array([r["set_id"] for r in rows], dtype=np.uint16)
    face_ids = np.array([r["face_id"] for r in rows], dtype=np.uint16)
    fracture_ids = np.array([r["fracture_id"] for r in rows], dtype=np.int32)
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

    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("traces")
        grp.create_dataset("trace_id", data=trace_ids)
        grp.create_dataset("fracture_id", data=fracture_ids)
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

        meta = f.create_group("meta")
        meta.create_dataset("tunnel_poly_yz", data=poly_yz.astype(np.float32))
        meta.create_dataset("face_x_positions_m", data=face_x.astype(np.float32))


def build_rows_flat_faces(data: dict, poly_yz: np.ndarray, face_x: np.ndarray) -> List[dict]:
    """flat mode trace 레코드 생성."""
    rows = []
    trace_id = 1
    for face_idx, x_face in enumerate(face_x, start=1):
        face_trace_count = 0
        for fracture_id in range(len(data["radii"])):
            components = intersect_disc_with_face(
                data["centers"][fracture_id],
                data["normals"][fracture_id],
                float(data["radii"][fracture_id]),
                float(x_face),
                poly_yz,
            )
            for seg in components:
                rows.append(
                    {
                        "trace_id": trace_id,
                        "face_id": int(face_idx),
                        "face_x_m": float(x_face),
                        "fracture_id": int(fracture_id),
                        "set_id": int(data["set_ids"][fracture_id]),
                        "component_id": int(seg["component_id"]),
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
                        "face_mesh_name": "flat_face",
                    }
                )
                trace_id += 1
                face_trace_count += 1
        print(f"[*] Flat face {face_idx}/{len(face_x)} @ x={x_face:.2f} m -> {face_trace_count} traces")
    return rows


def build_rows_rough_faces(data: dict, face_contexts: Sequence[dict]) -> List[dict]:
    """rough mode trace 레코드 생성."""
    rows = []
    trace_id = 1
    total_fractures = len(data["radii"])

    for face_ctx in face_contexts:
        t0 = time.perf_counter()
        candidate_fracture_ids = filter_fractures_for_face(data, face_ctx)
        t1 = time.perf_counter()

        n_fractures_with_segments = 0
        n_trace_components = 0
        intersection_time = 0.0
        graph_time = 0.0

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

            n_fractures_with_segments += 1
            t_graph_start = time.perf_counter()
            components = extract_trace_components(
                segments=segments,
                center_xyz=data["centers"][fracture_id],
                radius=float(data["radii"][fracture_id]),
                boundary_nodes=face_ctx["boundary_nodes"],
            )
            graph_time += time.perf_counter() - t_graph_start

            for component_id, seg in enumerate(components):
                rows.append(
                    {
                        "trace_id": trace_id,
                        "face_id": int(face_ctx["face_id"]),
                        "face_x_m": float(face_ctx["face_x"]),
                        "fracture_id": int(fracture_id),
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
                        "face_mesh_name": face_ctx["source_name"],
                    }
                )
                trace_id += 1
                n_trace_components += 1

        print(
            f"[*] Face {face_ctx['face_id']:03d} @ x={face_ctx['face_x']:.2f} m | "
            f"triangles={len(face_ctx['triangles']):,}, fractures_total={total_fractures:,}, "
            f"fractures_candidate_after_bbox={len(candidate_fracture_ids):,}, "
            f"fractures_with_segments={n_fractures_with_segments:,}, trace_components={n_trace_components:,}, "
            f"time_filter={t1 - t0:.3f}s, time_intersection={intersection_time:.3f}s, time_endpoint_graph={graph_time:.3f}s"
        )

    return rows


def print_summary(rows: Sequence[dict]) -> None:
    print(f"[*] Exported {len(rows):,} clipped 3D traces.")
    if not rows:
        return
    set_ids = sorted({row["set_id"] for row in rows})
    for set_id in set_ids:
        set_rows = [row for row in rows if row["set_id"] == set_id]
        total_length = sum(row["observed_length_m"] for row in set_rows)
        print(f"    - Set {set_id}: {len(set_rows):,} traces, observed total length = {total_length:.3f} m")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export per-set 3D trace datasets from flat faces or face-wise rough face mesh collections."
    )
    parser.add_argument("--input", required=True, help="Input HDF5 DFN file")
    parser.add_argument("--outdir", default="storage/output/trace_dataset_collection", help="Output directory")
    parser.add_argument("--tunnel-dat", help="Optional tunnel polygon .dat file when HDF5 has no tunnel polygon")
    parser.add_argument("--rough-mesh-h5", help="Optional HDF5 containing /rough_faces or legacy /rough_face")
    parser.add_argument("--face-step", type=float, default=3.0, help="Face spacing along X (flat mode only)")
    parser.add_argument("--x-start", type=float, help="Face range start (flat mode only)")
    parser.add_argument("--x-end", type=float, help="Face range end (flat mode only)")
    parser.add_argument("--face-x-csv", help="Explicit comma-separated face X positions for flat mode")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    data = load_hdf5_dfn(args.input)
    poly_yz = data["poly_yz"]

    if poly_yz is None:
        if not args.tunnel_dat:
            raise ValueError("Tunnel polygon not found in HDF5. Provide --tunnel-dat.")
        poly_yz = load_tunnel_polygon_from_dat(args.tunnel_dat)

    if signed_polygon_area(poly_yz) < 0.0:
        poly_yz = poly_yz[::-1].copy()

    rough_faces = None
    rough_face_source = args.rough_mesh_h5 if args.rough_mesh_h5 else args.input
    try:
        rough_faces = load_rough_face_collection_from_h5(rough_face_source)
    except ValueError:
        rough_faces = None

    if rough_faces:
        print(f"[*] Using face-wise rough face collection from: {rough_face_source}")
        face_contexts = [precompute_face_mesh(face) for face in rough_faces]
        rows = build_rows_rough_faces(data, face_contexts)
        face_x = np.array([ctx["face_x"] for ctx in face_contexts], dtype=np.float64)
    else:
        print("[*] Rough face collection not found. Falling back to flat x=const face mode.")
        face_x = resolve_face_range(data, args)
        rows = build_rows_flat_faces(data, poly_yz, face_x)

    csv_path = os.path.join(args.outdir, "trace_dataset_3d.csv")
    h5_path = os.path.join(args.outdir, "trace_dataset_3d.h5")
    write_csv(rows, csv_path)
    write_hdf5(rows, poly_yz, face_x, h5_path)

    print_summary(rows)
    print(f"[*] CSV written to: {csv_path}")
    print(f"[*] HDF5 written to: {h5_path}")


if __name__ == "__main__":
    main()
