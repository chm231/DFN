import argparse
import csv
import os
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np


def _read_scalar(group: h5py.Group, key: str, default):
    if key not in group:
        return default
    value = group[key][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if np.ndim(value) > 0:
        return value.ravel()[0]
    return value


def load_trace_rows_from_h5(h5_path: str) -> List[dict]:
    rows: List[dict] = []
    with h5py.File(h5_path, "r") as f:
        if "traces" not in f:
            raise ValueError(f"Could not find /traces in: {h5_path}")
        grp = f["traces"]
        n_rows = len(grp["trace_id"])
        for idx in range(n_rows):
            rows.append(
                {
                    "set_id": int(grp["set_id"][idx]),
                    "observed_length_m": float(grp["observed_length_m"][idx]),
                    "censoring_class": int(grp["censoring_class"][idx]),
                    "trace_normal_valid": int(grp["trace_normal_valid"][idx]),
                }
            )
    return rows


def load_trace_rows_from_csv(csv_path: str) -> List[dict]:
    rows: List[dict] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "set_id": int(row["set_id"]),
                    "observed_length_m": float(row["observed_length_m"]),
                    "censoring_class": int(row["censoring_class"]),
                    "trace_normal_valid": int(row["trace_normal_valid"]),
                }
            )
    return rows


def load_rough_face_collection_from_h5(h5_path: str) -> List[dict]:
    with h5py.File(h5_path, "r") as f:
        if "rough_faces" in f:
            rough_faces = []
            faces_grp = f["rough_faces"]
            for face_name in sorted(faces_grp.keys()):
                grp = faces_grp[face_name]
                meta = grp["meta"] if "meta" in grp else None
                face_id = int(_read_scalar(meta, "face_id", len(rough_faces) + 1)) if meta else len(rough_faces) + 1
                rough_faces.append(
                    {
                        "face_id": face_id,
                        "vertices_xyz": grp["mesh/vertices_xyz"][:].astype(np.float64),
                        "triangles": grp["mesh/triangles"][:].astype(np.int32),
                    }
                )
            return rough_faces

        if "rough_face" in f:
            grp = f["rough_face"]
            return [
                {
                    "face_id": 1,
                    "vertices_xyz": grp["mesh/vertices_xyz"][:].astype(np.float64),
                    "triangles": grp["mesh/triangles"][:].astype(np.int32),
                }
            ]

        if "mesh" in f:
            return [
                {
                    "face_id": 1,
                    "vertices_xyz": f["mesh/vertices_xyz"][:].astype(np.float64),
                    "triangles": f["mesh/triangles"][:].astype(np.int32),
                }
            ]

    raise ValueError(f"Could not find rough face collection in: {h5_path}")


def triangle_area_sum(vertices_xyz: np.ndarray, triangles: np.ndarray) -> float:
    v0 = vertices_xyz[triangles[:, 0]]
    v1 = vertices_xyz[triangles[:, 1]]
    v2 = vertices_xyz[triangles[:, 2]]
    return 0.5 * float(np.sum(np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)))


def compute_total_observation_area(rough_faces: Sequence[dict]) -> float:
    return float(sum(triangle_area_sum(face["vertices_xyz"], face["triangles"]) for face in rough_faces))


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def _safe_median(values: np.ndarray) -> float:
    return float(np.median(values)) if len(values) else float("nan")


def build_summary_rows(rows: Sequence[dict], observation_area_m2: Optional[float]) -> List[dict]:
    set_ids = sorted({int(row["set_id"]) for row in rows})
    summary_rows: List[dict] = []
    for set_id in set_ids:
        set_rows = [row for row in rows if int(row["set_id"]) == set_id]
        uncensored = np.asarray(
            [float(row["observed_length_m"]) for row in set_rows if int(row["censoring_class"]) == 0],
            dtype=np.float64,
        )
        censored = np.asarray(
            [float(row["observed_length_m"]) for row in set_rows if int(row["censoring_class"]) > 0],
            dtype=np.float64,
        )
        total_length = float(sum(float(row["observed_length_m"]) for row in set_rows))
        n_total = len(set_rows)
        n_uncensored = int(sum(int(row["censoring_class"]) == 0 for row in set_rows))
        n_one_end_censored = int(sum(int(row["censoring_class"]) == 1 for row in set_rows))
        n_two_end_censored = int(sum(int(row["censoring_class"]) == 2 for row in set_rows))
        n_censored = n_one_end_censored + n_two_end_censored
        n_valid_normal = int(sum(int(row["trace_normal_valid"]) == 1 for row in set_rows))

        summary_rows.append(
            {
                "set_id": set_id,
                "n_total": n_total,
                "n_uncensored": n_uncensored,
                "n_one_end_censored": n_one_end_censored,
                "n_two_end_censored": n_two_end_censored,
                "censoring_ratio_total": (n_censored / n_total) if n_total else float("nan"),
                "mean_length_uncensored": _safe_mean(uncensored),
                "mean_length_censored": _safe_mean(censored),
                "median_length_uncensored": _safe_median(uncensored),
                "median_length_censored": _safe_median(censored),
                "total_trace_length": total_length,
                "P21_observed": (total_length / observation_area_m2) if observation_area_m2 and observation_area_m2 > 0.0 else float("nan"),
                "trace_normal_valid_ratio": (n_valid_normal / n_total) if n_total else float("nan"),
            }
        )
    return summary_rows


def write_summary_csv(summary_rows: Sequence[dict], csv_path: str) -> None:
    fieldnames = [
        "set_id",
        "n_total",
        "n_uncensored",
        "n_one_end_censored",
        "n_two_end_censored",
        "censoring_ratio_total",
        "mean_length_uncensored",
        "mean_length_censored",
        "median_length_uncensored",
        "median_length_censored",
        "total_trace_length",
        "P21_observed",
        "trace_normal_valid_ratio",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def print_summary_table(summary_rows: Sequence[dict], observation_area_m2: Optional[float]) -> None:
    print("[*] Set-wise trace statistics")
    if observation_area_m2 is not None:
        print(f"    observation_area_m2 = {observation_area_m2:.6f}")
    for row in summary_rows:
        print(
            f"    - Set {row['set_id']}: "
            f"n_total={row['n_total']}, n_uncensored={row['n_uncensored']}, "
            f"n_one_end_censored={row['n_one_end_censored']}, n_two_end_censored={row['n_two_end_censored']}, "
            f"censoring_ratio_total={row['censoring_ratio_total']:.4f}, "
            f"mean_length_uncensored={row['mean_length_uncensored']:.4f}, "
            f"mean_length_censored={row['mean_length_censored']:.4f}, "
            f"median_length_uncensored={row['median_length_uncensored']:.4f}, "
            f"median_length_censored={row['median_length_censored']:.4f}, "
            f"total_trace_length={row['total_trace_length']:.4f}, "
            f"P21_observed={row['P21_observed']:.6f}, "
            f"trace_normal_valid_ratio={row['trace_normal_valid_ratio']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize set-wise 3D trace statistics.")
    parser.add_argument("--trace-h5", help="Input trace HDF5 created by export_setwise_3d_traces.py")
    parser.add_argument("--trace-csv", help="Input trace CSV created by export_setwise_3d_traces.py")
    parser.add_argument("--rough-mesh-h5", help="Rough face collection HDF5 used to compute observed P21 area")
    parser.add_argument(
        "--out-csv",
        default="storage/output/trace_dataset_collection/setwise_trace_statistics.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    if bool(args.trace_h5) == bool(args.trace_csv):
        raise ValueError("Provide exactly one of --trace-h5 or --trace-csv.")

    rows = load_trace_rows_from_h5(args.trace_h5) if args.trace_h5 else load_trace_rows_from_csv(args.trace_csv)
    observation_area_m2 = None
    if args.rough_mesh_h5:
        rough_faces = load_rough_face_collection_from_h5(args.rough_mesh_h5)
        observation_area_m2 = compute_total_observation_area(rough_faces)

    summary_rows = build_summary_rows(rows, observation_area_m2)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    write_summary_csv(summary_rows, args.out_csv)
    print_summary_table(summary_rows, observation_area_m2)
    print(f"[*] CSV written to: {args.out_csv}")


if __name__ == "__main__":
    main()
