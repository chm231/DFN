import argparse
import csv
import os
import re
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np


def load_hdf5_dfn(h5_path: str) -> dict:
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


def load_tunnel_polygon_from_dat(dat_path: str, scale: float = 0.001) -> np.ndarray:
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
    full_length = 2.0 * chord_half_len

    clipped_segments = clip_segment_to_polygon(full_p0_yz, full_p1_yz, poly_yz)
    rows = []
    for clip_p0_yz, clip_p1_yz, n_clipped in clipped_segments:
        clip_p0_xyz = np.array([face_x, clip_p0_yz[0], clip_p0_yz[1]], dtype=np.float64)
        clip_p1_xyz = np.array([face_x, clip_p1_yz[0], clip_p1_yz[1]], dtype=np.float64)
        observed_length = float(np.linalg.norm(clip_p1_xyz - clip_p0_xyz))
        rows.append(
            {
                "p0_xyz": clip_p0_xyz,
                "p1_xyz": clip_p1_xyz,
                "full_p0_xyz": np.array([face_x, full_p0_yz[0], full_p0_yz[1]], dtype=np.float64),
                "full_p1_xyz": np.array([face_x, full_p1_yz[0], full_p1_yz[1]], dtype=np.float64),
                "observed_length_m": observed_length,
                "full_length_m": full_length,
                "censoring_class": min(n_clipped, 2),
            }
        )
    return rows


def resolve_face_range(data: dict, args: argparse.Namespace) -> np.ndarray:
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
        "fracture_id",
        "set_id",
        "face_id",
        "face_x_m",
        "observed_length_m",
        "full_length_m",
        "censoring_class",
        "p0_x",
        "p0_y",
        "p0_z",
        "p1_x",
        "p1_y",
        "p1_z",
        "full_p0_x",
        "full_p0_y",
        "full_p0_z",
        "full_p1_x",
        "full_p1_y",
        "full_p1_z",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_hdf5(rows: Sequence[dict], poly_yz: np.ndarray, face_x: np.ndarray, h5_path: str) -> None:
    p0 = np.array([[r["p0_x"], r["p0_y"], r["p0_z"]] for r in rows], dtype=np.float32)
    p1 = np.array([[r["p1_x"], r["p1_y"], r["p1_z"]] for r in rows], dtype=np.float32)
    full_p0 = np.array([[r["full_p0_x"], r["full_p0_y"], r["full_p0_z"]] for r in rows], dtype=np.float32)
    full_p1 = np.array([[r["full_p1_x"], r["full_p1_y"], r["full_p1_z"]] for r in rows], dtype=np.float32)
    set_ids = np.array([r["set_id"] for r in rows], dtype=np.uint16)
    face_ids = np.array([r["face_id"] for r in rows], dtype=np.uint16)
    fracture_ids = np.array([r["fracture_id"] for r in rows], dtype=np.int32)
    trace_ids = np.array([r["trace_id"] for r in rows], dtype=np.int32)
    censoring = np.array([r["censoring_class"] for r in rows], dtype=np.uint8)
    observed_length = np.array([r["observed_length_m"] for r in rows], dtype=np.float32)
    full_length = np.array([r["full_length_m"] for r in rows], dtype=np.float32)
    face_x_values = np.array([r["face_x_m"] for r in rows], dtype=np.float32)

    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("traces")
        grp.create_dataset("trace_id", data=trace_ids)
        grp.create_dataset("fracture_id", data=fracture_ids)
        grp.create_dataset("set_id", data=set_ids)
        grp.create_dataset("face_id", data=face_ids)
        grp.create_dataset("face_x_m", data=face_x_values)
        grp.create_dataset("observed_length_m", data=observed_length)
        grp.create_dataset("full_length_m", data=full_length)
        grp.create_dataset("censoring_class", data=censoring)
        grp.create_dataset("p0_xyz", data=p0)
        grp.create_dataset("p1_xyz", data=p1)
        grp.create_dataset("full_p0_xyz", data=full_p0)
        grp.create_dataset("full_p1_xyz", data=full_p1)

        meta = f.create_group("meta")
        meta.create_dataset("tunnel_poly_yz", data=poly_yz.astype(np.float32))
        meta.create_dataset("face_x_positions_m", data=face_x.astype(np.float32))


def build_rows(data: dict, poly_yz: np.ndarray, face_x: np.ndarray) -> List[dict]:
    rows = []
    trace_id = 1
    for face_idx, x_face in enumerate(face_x, start=1):
        for fracture_id in range(len(data["radii"])):
            segments = intersect_disc_with_face(
                data["centers"][fracture_id],
                data["normals"][fracture_id],
                float(data["radii"][fracture_id]),
                float(x_face),
                poly_yz,
            )
            for seg in segments:
                row = {
                    "trace_id": trace_id,
                    "fracture_id": int(fracture_id),
                    "set_id": int(data["set_ids"][fracture_id]),
                    "face_id": int(face_idx),
                    "face_x_m": float(x_face),
                    "observed_length_m": float(seg["observed_length_m"]),
                    "full_length_m": float(seg["full_length_m"]),
                    "censoring_class": int(seg["censoring_class"]),
                    "p0_x": float(seg["p0_xyz"][0]),
                    "p0_y": float(seg["p0_xyz"][1]),
                    "p0_z": float(seg["p0_xyz"][2]),
                    "p1_x": float(seg["p1_xyz"][0]),
                    "p1_y": float(seg["p1_xyz"][1]),
                    "p1_z": float(seg["p1_xyz"][2]),
                    "full_p0_x": float(seg["full_p0_xyz"][0]),
                    "full_p0_y": float(seg["full_p0_xyz"][1]),
                    "full_p0_z": float(seg["full_p0_xyz"][2]),
                    "full_p1_x": float(seg["full_p1_xyz"][0]),
                    "full_p1_y": float(seg["full_p1_xyz"][1]),
                    "full_p1_z": float(seg["full_p1_xyz"][2]),
                }
                rows.append(row)
                trace_id += 1
    return rows


def print_summary(rows: Sequence[dict]) -> None:
    print(f"[*] Exported {len(rows):,} clipped 3D traces.")
    if not rows:
        return
    set_ids = sorted({row["set_id"] for row in rows})
    for set_id in set_ids:
        set_rows = [row for row in rows if row["set_id"] == set_id]
        total_length = sum(row["observed_length_m"] for row in set_rows)
        print(
            f"    - Set {set_id}: {len(set_rows):,} traces, "
            f"observed total length = {total_length:.3f} m"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export per-set 3D trace datasets where DFN discs intersect tunnel face polygons."
    )
    parser.add_argument("--input", required=True, help="Input HDF5 DFN file")
    parser.add_argument("--outdir", default="storage/output/trace_dataset_collection", help="Output directory")
    parser.add_argument("--tunnel-dat", help="Optional tunnel polygon .dat file. Used when HDF5 has no /tunnel/poly_YZ.")
    parser.add_argument("--face-step", type=float, default=3.0, help="Face spacing along X (m)")
    parser.add_argument("--x-start", type=float, help="Face range start (m)")
    parser.add_argument("--x-end", type=float, help="Face range end (m)")
    parser.add_argument("--face-x-csv", help="Explicit comma-separated face X positions, e.g. 0,3,6,9")
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

    face_x = resolve_face_range(data, args)
    rows = build_rows(data, poly_yz, face_x)

    csv_path = os.path.join(args.outdir, "trace_dataset_3d.csv")
    h5_path = os.path.join(args.outdir, "trace_dataset_3d.h5")
    write_csv(rows, csv_path)
    write_hdf5(rows, poly_yz, face_x, h5_path)
    print_summary(rows)
    print(f"[*] CSV written to: {csv_path}")
    print(f"[*] HDF5 written to: {h5_path}")


if __name__ == "__main__":
    main()
