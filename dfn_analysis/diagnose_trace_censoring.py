import argparse
import csv
import os
from typing import List, Optional, Sequence

import h5py
import numpy as np


def load_trace_rows_from_h5(h5_path: str) -> List[dict]:
    rows: List[dict] = []
    with h5py.File(h5_path, "r") as f:
        if "traces" not in f:
            raise ValueError(f"Could not find /traces in: {h5_path}")
        grp = f["traces"]
        for idx in range(len(grp["set_id"])):
            rows.append(
                {
                    "set_id": int(grp["set_id"][idx]),
                    "observed_length_m": float(grp["observed_length_m"][idx]),
                    "censoring_class": int(grp["censoring_class"][idx]),
                }
            )
    return rows


def load_trace_rows_from_csv(csv_path: str) -> List[dict]:
    rows: List[dict] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"set_id", "observed_length_m", "censoring_class"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV fields: {sorted(missing)}")
        for row in reader:
            rows.append(
                {
                    "set_id": int(row["set_id"]),
                    "observed_length_m": float(row["observed_length_m"]),
                    "censoring_class": int(row["censoring_class"]),
                }
            )
    return rows


def _read_scalar(group: h5py.Group, key: str, default):
    if key not in group:
        return default
    value = group[key][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if np.ndim(value) > 0:
        return value.ravel()[0]
    return value


def load_rough_face_meshes(h5_path: str) -> List[dict]:
    with h5py.File(h5_path, "r") as f:
        if "rough_faces" in f:
            faces = []
            for idx, face_name in enumerate(sorted(f["rough_faces"].keys()), start=1):
                grp = f["rough_faces"][face_name]
                meta = grp["meta"] if "meta" in grp else None
                face_id = int(_read_scalar(meta, "face_id", idx)) if meta else idx
                faces.append(
                    {
                        "face_id": face_id,
                        "vertices_xyz": grp["mesh/vertices_xyz"][:].astype(np.float64),
                        "triangles": grp["mesh/triangles"][:].astype(np.int32),
                    }
                )
            return faces
        if "rough_face" in f:
            grp = f["rough_face"]
            return [{"face_id": 1, "vertices_xyz": grp["mesh/vertices_xyz"][:], "triangles": grp["mesh/triangles"][:]}]
        if "mesh" in f:
            return [{"face_id": 1, "vertices_xyz": f["mesh/vertices_xyz"][:], "triangles": f["mesh/triangles"][:]}]
    raise ValueError(f"Could not find rough face mesh in: {h5_path}")


def triangle_area_sum(vertices_xyz: np.ndarray, triangles: np.ndarray) -> float:
    vertices_xyz = vertices_xyz.astype(np.float64)
    triangles = triangles.astype(np.int32)
    v0 = vertices_xyz[triangles[:, 0]]
    v1 = vertices_xyz[triangles[:, 1]]
    v2 = vertices_xyz[triangles[:, 2]]
    return 0.5 * float(np.sum(np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)))


def observation_area_from_h5(h5_path: str) -> float:
    return float(sum(triangle_area_sum(face["vertices_xyz"], face["triangles"]) for face in load_rough_face_meshes(h5_path)))


def build_summary_rows(rows: Sequence[dict], observation_area_m2: Optional[float]) -> List[dict]:
    out_rows = []
    for set_id in sorted({int(row["set_id"]) for row in rows}):
        set_rows = [row for row in rows if int(row["set_id"]) == set_id]
        n_total = len(set_rows)
        n_uncensored = int(sum(int(row["censoring_class"]) == 0 for row in set_rows))
        n_one_end = int(sum(int(row["censoring_class"]) == 1 for row in set_rows))
        n_two_end = int(sum(int(row["censoring_class"]) == 2 for row in set_rows))
        total_length = float(sum(float(row["observed_length_m"]) for row in set_rows))
        n_censored = n_one_end + n_two_end
        out_rows.append(
            {
                "set_id": set_id,
                "n_total": n_total,
                "n_uncensored": n_uncensored,
                "n_one_end_censored": n_one_end,
                "n_two_end_censored": n_two_end,
                "censoring_ratio_total": (n_censored / n_total) if n_total else float("nan"),
                "total_trace_length": total_length,
                "P21_observed": (total_length / observation_area_m2) if observation_area_m2 else float("nan"),
            }
        )
    return out_rows


def write_summary_csv(rows: Sequence[dict], csv_path: str) -> None:
    fieldnames = [
        "set_id",
        "n_total",
        "n_uncensored",
        "n_one_end_censored",
        "n_two_end_censored",
        "censoring_ratio_total",
        "total_trace_length",
        "P21_observed",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose set-wise trace censoring and observed P21.")
    parser.add_argument("--trace-h5", help="Input trace HDF5")
    parser.add_argument("--trace-csv", help="Input trace CSV")
    parser.add_argument("--rough-mesh-h5", help="Optional rough face mesh HDF5 for P21 area")
    parser.add_argument("--outdir", default="storage/output/trace_censoring_diagnostics")
    args = parser.parse_args()

    if bool(args.trace_h5) == bool(args.trace_csv):
        raise ValueError("Provide exactly one of --trace-h5 or --trace-csv.")

    rows = load_trace_rows_from_h5(args.trace_h5) if args.trace_h5 else load_trace_rows_from_csv(args.trace_csv)
    observation_area_m2 = observation_area_from_h5(args.rough_mesh_h5) if args.rough_mesh_h5 else None
    summary_rows = build_summary_rows(rows, observation_area_m2)
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "trace_censoring_by_set.csv")
    write_summary_csv(summary_rows, csv_path)

    if observation_area_m2 is not None:
        print(f"[*] observation_area_m2={observation_area_m2:.6f}")
    for row in summary_rows:
        print(
            f"Set {row['set_id']}: n_total={row['n_total']}, n_uncensored={row['n_uncensored']}, "
            f"n_one_end_censored={row['n_one_end_censored']}, n_two_end_censored={row['n_two_end_censored']}, "
            f"censoring_ratio_total={row['censoring_ratio_total']:.4f}, "
            f"total_trace_length={row['total_trace_length']:.4f}, P21_observed={row['P21_observed']:.6f}"
        )
    print(f"[*] CSV written to: {csv_path}")


if __name__ == "__main__":
    main()
