"""Interactive 3D visualisation of the conditional DFN with tunnel faces.

Renders the discs in ``conditional_dfn.csv`` (visible + hidden) as oriented
circular polygons in global 3D space, together with the observation faces
(tunnel cross-section windows at each face x). Coloured by fracture set.

Visualisation only — no geometry is computed or changed here (CLAUDE.md §12).

Coordinate convention: x = East = tunnel advance, y = North, z = Up.
Discs are clipped to a local box around the tunnel so the picture stays
readable (the full DFN spans hundreds of metres).

Usage
-----
    python dfn_analysis/visualize_conditional_dfn_3d.py            # interactive window + screenshot
    python dfn_analysis/visualize_conditional_dfn_3d.py --no-window  # screenshot only (headless)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Tuple

import numpy as np
import h5py
import pyvista as pv

REPO = Path(__file__).resolve().parent.parent

# Per-set colours (sets 1,2,3,5 are the reconstructed/conditioned powerlaw sets)
SET_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 5: "#d62728"}
FACE_COLOR = "#888888"


# ----------------------------------------------------------------------
def load_conditional_dfn(path: Path):
    """Load discs from conditional_dfn.csv into parallel numpy arrays."""
    centers, normals, radii, set_ids, sources = [], [], [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            centers.append([float(row["cx"]), float(row["cy"]), float(row["cz"])])
            normals.append([float(row["nx"]), float(row["ny"]), float(row["nz"])])
            radii.append(float(row["radius"]))
            set_ids.append(int(row["set_id"]))
            sources.append(row["source"])
    return (np.array(centers), np.array(normals), np.array(radii),
            np.array(set_ids), np.array(sources))


def in_box(centers: np.ndarray, box: dict) -> np.ndarray:
    """Boolean mask of disc centres inside the local viewing box."""
    return (
        (centers[:, 0] >= box["xmin"]) & (centers[:, 0] <= box["xmax"])
        & (centers[:, 1] >= box["ymin"]) & (centers[:, 1] <= box["ymax"])
        & (centers[:, 2] >= box["zmin"]) & (centers[:, 2] <= box["zmax"])
    )


def _disc_basis(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two orthonormal in-plane vectors for each unit normal (vectorised)."""
    n = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    ref = np.zeros_like(n)
    ref[:, 2] = 1.0
    ref[np.abs(n[:, 2]) > 0.95] = [1.0, 0.0, 0.0]
    u = np.cross(ref, n)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(n, u)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return u, v


def build_disc_mesh(
    centers: np.ndarray, normals: np.ndarray, radii: np.ndarray, n_pts: int = 24
) -> pv.PolyData:
    """Combine oriented circular discs into a single PolyData of polygon faces."""
    if len(centers) == 0:
        return pv.PolyData()
    u, v = _disc_basis(normals)
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    ct, st = np.cos(theta), np.sin(theta)
    # rim[i,k] = c[i] + r[i]*(cos θk * u[i] + sin θk * v[i])
    rim = (centers[:, None, :]
           + radii[:, None, None] * (ct[None, :, None] * u[:, None, :]
                                     + st[None, :, None] * v[:, None, :]))
    N = len(centers)
    pts = rim.reshape(N * n_pts, 3)
    idx = np.arange(N)[:, None] * n_pts + np.arange(n_pts)[None, :]
    faces = np.hstack([np.full((N, 1), n_pts), idx]).astype(np.int64).ravel()
    return pv.PolyData(pts, faces)


def build_face_polygon(poly_yz: np.ndarray, xf: float) -> pv.PolyData:
    """Tunnel observation window (y-z polygon) placed at plane x = xf."""
    pts = np.column_stack([np.full(len(poly_yz), xf), poly_yz[:, 0], poly_yz[:, 1]])
    face = np.hstack([[len(poly_yz)], np.arange(len(poly_yz))]).astype(np.int64)
    return pv.PolyData(pts, face)


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pipeline-dir", type=Path,
                    default=REPO / "storage/output/pipeline_test_laxemar")
    ap.add_argument("--box-x", type=float, nargs=2, default=[-8.0, 11.0],
                    help="Local box x-range [m].")
    ap.add_argument("--box-half-yz", type=float, default=3.0,
                    help="Local box half-extent in y and z about the window [m].")
    ap.add_argument("--min-radius", type=float, default=1.5,
                    help="Only show HIDDEN discs with radius >= this [m] "
                         "(declutter; visible discs always shown). Set 0 for all.")
    ap.add_argument("--no-window", action="store_true",
                    help="Render off-screen and only save the screenshot.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Screenshot path (default: <pipeline>/conditional_hidden/conditional_dfn_3d.png).")
    args = ap.parse_args()

    pdir = args.pipeline_dir
    dfn_csv = pdir / "conditional_hidden/conditional_dfn.csv"
    out_png = args.out or (pdir / "conditional_hidden/conditional_dfn_3d.png")

    centers, normals, radii, set_ids, sources = load_conditional_dfn(dfn_csv)

    # Tunnel window + observed face x-locations
    with h5py.File(pdir / "dfn_export_for_python.h5", "r") as f:
        poly_yz = np.array(f["tunnel/poly_YZ"])
    face_xs = sorted({round(float(r["face_x_m"]), 3)
                      for r in csv.DictReader(open(pdir / "trace_dataset/trace_dataset_3d.csv"))})

    ymin, ymax = poly_yz[:, 0].min(), poly_yz[:, 0].max()
    zmin, zmax = poly_yz[:, 1].min(), poly_yz[:, 1].max()
    box = dict(
        xmin=args.box_x[0], xmax=args.box_x[1],
        ymin=ymin - args.box_half_yz, ymax=ymax + args.box_half_yz,
        zmin=zmin - args.box_half_yz, zmax=zmax + args.box_half_yz,
    )
    in_local = in_box(centers, box)
    # Show all visible discs in the box; declutter hidden by a radius threshold.
    small_hidden = (sources == "hidden") & (radii < args.min_radius)
    mask = in_local & ~small_hidden
    n_dropped = int((in_local & small_hidden).sum())
    print(f"discs total={len(centers):,}  in local box={in_local.sum():,}  shown={mask.sum():,}  "
          f"(hidden dropped by r<{args.min_radius} m: {n_dropped:,})")
    print(f"  box x[{box['xmin']},{box['xmax']}] y[{box['ymin']:.1f},{box['ymax']:.1f}] "
          f"z[{box['zmin']:.1f},{box['zmax']:.1f}]")

    plotter = pv.Plotter(off_screen=args.no_window, window_size=(1400, 900))
    plotter.set_background("white")

    # Discs are drawn as circle OUTLINES (wireframe) coloured by set, so the
    # tunnel faces stay visible through them. Visible (reconstructed) discs are
    # bold and opaque; hidden (stochastic) discs are thin and faint.
    for s, color in SET_COLORS.items():
        sel_h = mask & (set_ids == s) & (sources == "hidden")
        if sel_h.any():
            mesh = build_disc_mesh(centers[sel_h], normals[sel_h], radii[sel_h])
            plotter.add_mesh(mesh, color=color, style="wireframe",
                             line_width=1, opacity=0.35)
        sel_v = mask & (set_ids == s) & (sources == "visible")
        if sel_v.any():
            mesh = build_disc_mesh(centers[sel_v], normals[sel_v], radii[sel_v])
            plotter.add_mesh(mesh, color=color, style="wireframe",
                             line_width=3, opacity=1.0, label=f"Set {s}")
        else:
            plotter.add_mesh(pv.PolyData(np.zeros((1, 3))), color=color,
                             point_size=1, label=f"Set {s}")

    # Tunnel observation faces (the only filled surfaces — spatial reference)
    for xf in face_xs:
        face = build_face_polygon(poly_yz, xf)
        plotter.add_mesh(face, color=FACE_COLOR, opacity=0.35,
                         show_edges=True, edge_color="black", line_width=3)
    plotter.add_mesh(build_face_polygon(poly_yz, face_xs[0]), color=FACE_COLOR,
                     opacity=0.0, label="Tunnel face")

    n_vis = int((mask & (sources == "visible")).sum())
    n_hid = int((mask & (sources == "hidden")).sum())
    plotter.add_text(
        f"Conditional DFN (local box)\nvisible={n_vis}  hidden={n_hid}  faces={len(face_xs)}",
        position="upper_left", font_size=11, color="black",
    )
    plotter.add_legend(bcolor="white", border=True)
    plotter.add_axes(xlabel="x (advance)", ylabel="y (North)", zlabel="z (Up)")
    mid_x = (box["xmin"] + box["xmax"]) / 2
    plotter.camera_position = [
        (mid_x + 14, box["ymin"] - 28, box["zmax"] + 12),  # camera: front-side-above
        (mid_x, 0.0, 0.0),                                 # focal: tunnel centre
        (0, 0, 1),                                          # up = z
    ]

    if args.no_window:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(out_png))
        print(f"Screenshot written to {out_png}")
    else:
        # Interactive: showing then screenshotting on close is unreliable
        # (render context is gone once the window closes). Use --no-window to
        # save an image.
        plotter.show()


if __name__ == "__main__":
    main()
