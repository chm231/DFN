r"""
실행 예시:

```powershell
$env:PYTHONPATH="."
python dfn_analysis\plot_3d_traces_on_rough_faces.py `
  --rough-mesh-h5 storage\output\rough_face_mesh_collection\synthetic_rough_face_collection.h5 `
  --trace-h5 storage\output\trace_dataset_collection\trace_dataset_3d.h5 `
  --outdir storage\output\trace_visualization_collection
```
"""

import argparse
import os
from typing import Dict, List

import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import Normalize


def _read_scalar(group: h5py.Group, key: str, default):
    if key not in group:
        return default
    value = group[key][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if np.ndim(value) > 0:
        return value.ravel()[0]
    return value


def load_rough_face_collection_from_h5(h5_path: str) -> List[dict]:
    """rough face collection HDF5에서 face mesh 목록을 읽는다."""
    with h5py.File(h5_path, "r") as f:
        if "rough_faces" not in f:
            raise ValueError(f"Could not find /rough_faces in: {h5_path}")

        faces = []
        for face_name in sorted(f["rough_faces"].keys()):
            grp = f["rough_faces"][face_name]
            meta = grp["meta"] if "meta" in grp else None
            face_id = int(_read_scalar(meta, "face_id", len(faces) + 1)) if meta else len(faces) + 1
            face_x = float(_read_scalar(meta, "face_x", 0.0)) if meta else 0.0
            source_name = str(_read_scalar(meta, "source_name", face_name)) if meta else face_name
            faces.append(
                {
                    "face_id": face_id,
                    "face_x": face_x,
                    "source_name": source_name,
                    "vertices_xyz": grp["mesh/vertices_xyz"][:].astype(np.float64),
                    "triangles": grp["mesh/triangles"][:].astype(np.int32),
                }
            )
    return faces


def load_trace_rows(trace_h5_path: str) -> List[dict]:
    """trace HDF5를 읽고 polyline vertex까지 복원한다."""
    rows = []
    with h5py.File(trace_h5_path, "r") as f:
        if "traces" not in f:
            raise ValueError(f"Could not find /traces in: {trace_h5_path}")

        grp = f["traces"]
        n_rows = len(grp["trace_id"])
        p0_xyz_all = grp["p0_xyz"][:].astype(np.float64)
        p1_xyz_all = grp["p1_xyz"][:].astype(np.float64)
        polyline_vertices_all = grp["polyline_vertices_xyz"][:].astype(np.float64) if "polyline_vertices_xyz" in grp else None
        polyline_vertex_starts = grp["polyline_vertex_start"][:].astype(np.int64) if "polyline_vertex_start" in grp else None
        polyline_vertex_counts = grp["polyline_vertex_count"][:].astype(np.int64) if "polyline_vertex_count" in grp else None

        for idx in range(n_rows):
            if polyline_vertices_all is not None and polyline_vertex_starts is not None and polyline_vertex_counts is not None:
                start = int(polyline_vertex_starts[idx])
                count = int(polyline_vertex_counts[idx])
                polyline_xyz = polyline_vertices_all[start : start + count]
            else:
                polyline_xyz = np.vstack([p0_xyz_all[idx], p1_xyz_all[idx]]).astype(np.float64)

            p0_endpoint_type = grp["p0_endpoint_type"][idx]
            p1_endpoint_type = grp["p1_endpoint_type"][idx]
            face_mesh_name = grp["face_mesh_name"][idx]
            rows.append(
                {
                    "trace_id": int(grp["trace_id"][idx]),
                    "face_id": int(grp["face_id"][idx]),
                    "face_x_m": float(grp["face_x_m"][idx]),
                    "fracture_id": int(grp["fracture_id"][idx]),
                    "set_id": int(grp["set_id"][idx]),
                    "component_id": int(grp["component_id"][idx]),
                    "p0_xyz": p0_xyz_all[idx],
                    "p1_xyz": p1_xyz_all[idx],
                    "polyline_xyz": polyline_xyz,
                    "observed_length_m": float(grp["observed_length_m"][idx]),
                    "censoring_class": int(grp["censoring_class"][idx]),
                    "is_closed_loop": int(grp["is_closed_loop"][idx]),
                    "n_raw_segments": int(grp["n_raw_segments"][idx]),
                    "p0_endpoint_type": p0_endpoint_type.decode("utf-8") if isinstance(p0_endpoint_type, bytes) else str(p0_endpoint_type),
                    "p1_endpoint_type": p1_endpoint_type.decode("utf-8") if isinstance(p1_endpoint_type, bytes) else str(p1_endpoint_type),
                    "face_mesh_name": face_mesh_name.decode("utf-8") if isinstance(face_mesh_name, bytes) else str(face_mesh_name),
                }
            )
    return rows


def build_set_color_map(set_ids: List[int]) -> Dict[int, tuple]:
    """절리군별 일관된 색상을 만든다."""
    unique_set_ids = sorted(set(set_ids))
    cmap = plt.get_cmap("tab10")
    return {set_id: cmap(idx % 10) for idx, set_id in enumerate(unique_set_ids)}


def plot_face_overlay(
    out_png: str,
    face_mesh: dict,
    trace_rows: List[dict],
    set_color_map: Dict[int, tuple],
) -> None:
    """face 하나의 rough mesh와 trace overlay를 저장한다."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_title(f"Face {face_mesh['face_id']:03d} Traces on Rough Mesh")

    vertices_xyz = face_mesh["vertices_xyz"]
    triangles = face_mesh["triangles"]
    tri = mtri.Triangulation(vertices_xyz[:, 1], vertices_xyz[:, 2], triangles)
    tri_roughness = np.mean(vertices_xyz[triangles, 0] - face_mesh["face_x"], axis=1)
    max_abs_roughness = max(float(np.max(np.abs(tri_roughness))), 1e-9)
    surface = ax.plot_trisurf(
        vertices_xyz[:, 0],
        vertices_xyz[:, 1],
        vertices_xyz[:, 2],
        triangles=tri.triangles,
        cmap="Greys",
        linewidth=0.08,
        edgecolor="none",
        alpha=0.55,
    )
    surface.set_array(tri_roughness)
    surface.set_norm(Normalize(vmin=-max_abs_roughness, vmax=max_abs_roughness))

    for row in trace_rows:
        color = set_color_map[row["set_id"]]
        polyline_xyz = row["polyline_xyz"]
        ax.plot(
            polyline_xyz[:, 0],
            polyline_xyz[:, 1],
            polyline_xyz[:, 2],
            color=color,
            linewidth=2.0,
            alpha=0.95,
        )

    plotted_sets = sorted({row["set_id"] for row in trace_rows})
    legend_handles = [
        plt.Line2D([0], [0], color=set_color_map[set_id], lw=2.5, label=f"Set {set_id}")
        for set_id in plotted_sets
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim(float(np.min(vertices_xyz[:, 0])) - 0.1, float(np.max(vertices_xyz[:, 0])) + 0.1)
    ax.set_ylim(float(np.min(vertices_xyz[:, 1])), float(np.max(vertices_xyz[:, 1])))
    ax.set_zlim(float(np.min(vertices_xyz[:, 2])), float(np.max(vertices_xyz[:, 2])))

    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_collection_overview(
    out_png: str,
    rough_faces: List[dict],
    trace_rows: List[dict],
    set_color_map: Dict[int, tuple],
) -> None:
    """모든 face와 trace를 한 장에 겹쳐서 저장한다."""
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_title("3D Traces on Synthetic Rough Face Mesh Collection")

    for face_mesh in rough_faces:
        vertices_xyz = face_mesh["vertices_xyz"]
        triangles = face_mesh["triangles"]
        tri = mtri.Triangulation(vertices_xyz[:, 1], vertices_xyz[:, 2], triangles)
        ax.plot_trisurf(
            vertices_xyz[:, 0],
            vertices_xyz[:, 1],
            vertices_xyz[:, 2],
            triangles=tri.triangles,
            color="lightgray",
            linewidth=0.05,
            edgecolor="none",
            alpha=0.18,
        )

    for row in trace_rows:
        color = set_color_map[row["set_id"]]
        polyline_xyz = row["polyline_xyz"]
        ax.plot(
            polyline_xyz[:, 0],
            polyline_xyz[:, 1],
            polyline_xyz[:, 2],
            color=color,
            linewidth=1.4,
            alpha=0.95,
        )

    all_vertices = np.vstack([face["vertices_xyz"] for face in rough_faces if len(face["vertices_xyz"]) > 0])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim(float(np.min(all_vertices[:, 0])) - 0.2, float(np.max(all_vertices[:, 0])) + 0.2)
    ax.set_ylim(float(np.min(all_vertices[:, 1])), float(np.max(all_vertices[:, 1])))
    ax.set_zlim(float(np.min(all_vertices[:, 2])), float(np.max(all_vertices[:, 2])))

    plotted_sets = sorted({row["set_id"] for row in trace_rows})
    legend_handles = [
        plt.Line2D([0], [0], color=set_color_map[set_id], lw=2.0, label=f"Set {set_id}")
        for set_id in plotted_sets
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize extracted 3D traces on top of the synthetic rough face mesh collection."
    )
    parser.add_argument("--rough-mesh-h5", required=True, help="Rough face mesh collection HDF5")
    parser.add_argument("--trace-h5", required=True, help="trace_dataset_3d.h5 path")
    parser.add_argument("--outdir", default="storage/output/trace_visualization_collection", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rough_faces = load_rough_face_collection_from_h5(args.rough_mesh_h5)
    trace_rows = load_trace_rows(args.trace_h5)
    set_color_map = build_set_color_map([row["set_id"] for row in trace_rows])

    for face_mesh in rough_faces:
        face_rows = [row for row in trace_rows if row["face_id"] == face_mesh["face_id"]]
        out_png = os.path.join(args.outdir, f"trace_overlay_face_{face_mesh['face_id']:06d}.png")
        plot_face_overlay(
            out_png=out_png,
            face_mesh=face_mesh,
            trace_rows=face_rows,
            set_color_map=set_color_map,
        )
        print(f"[*] Saved face overlay: {out_png}")

    overview_png = os.path.join(args.outdir, "trace_overlay_collection_overview.png")
    plot_collection_overview(
        out_png=overview_png,
        rough_faces=rough_faces,
        trace_rows=trace_rows,
        set_color_map=set_color_map,
    )
    print(f"[*] Saved collection overview: {overview_png}")


if __name__ == "__main__":
    main()
