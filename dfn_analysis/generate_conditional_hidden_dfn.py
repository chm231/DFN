"""Conditional hidden DFN generation (MVP: remove-and-resample).

SKB DFN-R style geometric conditioning. Combines
  * VISIBLE discs   — directly reconstructed from observed traces (kept as-is)
  * HIDDEN discs     — stochastically generated from INVERTED set parameters,
                       with any disc that intersects an observed face removed
                       (that region is already explained by the visible discs).

Output = visible ∪ hidden, plus a SKB Fig. 4-1 style comparison figure
(observed traces = blue, conditioned traces = red) and P21/P32 diagnostics.

Conditioning is LOCAL: hidden discs are generated only in a box around the
observed faces (conditioning is inherently a local operation; the full 250 m
domain holds ~1.8e7 fractures and is neither needed nor tractable here).

Ground truth (dfn_export_for_python.h5) is used for VALIDATION ONLY — never
inside the generation recipe (project policy D013).

Coordinate convention (from the generator): x = East = tunnel advance,
y = North, z = Up. Observation faces are planes of constant x.

Reused, unmodified: sampling functions from ``dfn generator v1/python/generate_dfn.py``.

Usage
-----
    python dfn_analysis/generate_conditional_hidden_dfn.py \
        --pipeline-dir storage/output/pipeline_test_laxemar
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
OBSERVED_COLOR = "tab:blue"
CONDITIONED_COLOR = "tab:red"
# 대상 set은 하드코딩하지 않고 역산 파라미터가 존재하는 set에서 자동 유도한다
# (예: Laxemar는 Set4=지수분포라 kr 미보유 → 자동 제외; Forsmark는 5개 모두 powerlaw → 전부 포함).
# --sets / --exclude-sets 로 수동 지정 가능. per-set 분포타입은 params["dist_type"](기본 powerlaw).


# ----------------------------------------------------------------------
# Reuse the generator's sampling functions (single source of truth)
# ----------------------------------------------------------------------
def _load_generator():
    path = REPO / "dfn generator v1" / "python" / "generate_dfn.py"
    spec = importlib.util.spec_from_file_location("generate_dfn", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


GEN = _load_generator()


# ----------------------------------------------------------------------
# Geometry: disc ∩ face plane, segment ∩ convex polygon window
# ----------------------------------------------------------------------
def disc_face_chord(
    center: np.ndarray, normal: np.ndarray, radius: float, xf: float, tol: float = 1e-9
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Full chord of a disc cut by the plane x = xf, or None if it misses."""
    n = normal / np.linalg.norm(normal)
    ex = np.array([1.0, 0.0, 0.0])
    dist = float(center[0] - xf)  # signed distance along +x
    if abs(dist) >= radius - tol:
        return None
    half = math.sqrt(max(radius**2 - dist**2, 0.0))
    chord_dir = np.cross(n, ex)
    m = np.linalg.norm(chord_dir)
    if m < tol:  # disc normal ∥ x → disc plane parallel to face, no line
        return None
    chord_dir /= m
    n_face_proj = ex - (ex @ n) * n  # in-disc component of face normal
    mm = np.linalg.norm(n_face_proj)
    if mm < tol:
        return None
    n_face_proj /= mm
    foot = center - dist * ex
    d_in = float((foot - center) @ n_face_proj)
    mid = center + d_in * n_face_proj
    return mid - half * chord_dir, mid + half * chord_dir


def _ccw_polygon(poly: np.ndarray) -> np.ndarray:
    """Return polygon vertices in counter-clockwise order."""
    x, y = poly[:, 0], poly[:, 1]
    area2 = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    return poly if area2 >= 0 else poly[::-1].copy()


def clip_segment_to_convex_polygon(
    p0_yz: np.ndarray, p1_yz: np.ndarray, poly_ccw: np.ndarray, tol: float = 1e-12
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Cyrus-Beck clip of a 2D segment against a convex CCW polygon (y-z)."""
    d = p1_yz - p0_yz
    t_enter, t_leave = 0.0, 1.0
    n = len(poly_ccw)
    for i in range(n):
        a = poly_ccw[i]
        b = poly_ccw[(i + 1) % n]
        edge = b - a
        inward = np.array([-edge[1], edge[0]])  # left normal (interior side, CCW)
        c0 = float(inward @ (p0_yz - a))
        cd = float(inward @ d)
        if abs(cd) < tol:
            if c0 < 0:
                return None  # parallel and outside this edge
            continue
        t = -c0 / cd
        if cd > 0:
            t_enter = max(t_enter, t)
        else:
            t_leave = min(t_leave, t)
        if t_enter > t_leave:
            return None
    return p0_yz + t_enter * d, p0_yz + t_leave * d


def visible_trace_on_face(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    xf: float,
    poly_ccw: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return the window-clipped trace (3D endpoints) of a disc on face x=xf."""
    chord = disc_face_chord(center, normal, radius, xf)
    if chord is None:
        return None
    p0, p1 = chord
    clipped = clip_segment_to_convex_polygon(p0[1:3], p1[1:3], poly_ccw)
    if clipped is None:
        return None
    q0_yz, q1_yz = clipped
    if np.linalg.norm(q1_yz - q0_yz) < 1e-6:
        return None
    q0 = np.array([xf, q0_yz[0], q0_yz[1]])
    q1 = np.array([xf, q1_yz[0], q1_yz[1]])
    return q0, q1


# ----------------------------------------------------------------------
# Input loaders
# ----------------------------------------------------------------------
def load_visible_discs(path: Path, keep_adoptions: Tuple[str, ...]) -> List[dict]:
    """Reconstructed discs kept as the visible (observed) part of the DFN."""
    discs = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["adoption"] not in keep_adoptions:
                continue
            discs.append(dict(
                set_id=int(row["set_id"]),
                center=np.array([float(row["cx"]), float(row["cy"]), float(row["cz"])]),
                normal=np.array([float(row["nx"]), float(row["ny"]), float(row["nz"])]),
                radius=float(row["radius"]),
                source="visible",
                adoption=row["adoption"],
            ))
    return discs


def load_observed_traces(
    path: Path,
) -> Tuple[Dict[float, List[Tuple[np.ndarray, np.ndarray]]], Dict[int, float]]:
    """Observed traces grouped by face x, plus total observed length per set.

    The per-set length lets the P21 comparison be restricted to the sets that
    are actually reconstructed/conditioned (Set 4 is exponential and excluded,
    so including it on the observed side only would bias the comparison).
    """
    by_face: Dict[float, List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    len_by_set: Dict[int, float] = defaultdict(float)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            xf = round(float(row["face_x_m"]), 3)
            p0 = np.array([float(row["p0_x"]), float(row["p0_y"]), float(row["p0_z"])])
            p1 = np.array([float(row["p1_x"]), float(row["p1_y"]), float(row["p1_z"])])
            by_face[xf].append((p0, p1))
            len_by_set[int(row["set_id"])] += float(np.linalg.norm(p1 - p0))
    return by_face, len_by_set


def _fisher_kappa(Rbar: float) -> float:
    Rbar = min(Rbar, 0.999999)
    return Rbar * (3.0 - Rbar**2) / (1.0 - Rbar**2)


def set_orientation_from_discs(discs: List[dict], set_id: int) -> Optional[Tuple[np.ndarray, float]]:
    """Mean pole + Fisher kappa for a set, estimated from reconstructed normals."""
    normals = np.array([d["normal"] for d in discs if d["set_id"] == set_id])
    if len(normals) < 2:
        return None
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    ref = normals[0]
    signs = np.sign(normals @ ref)
    signs[signs == 0] = 1.0
    aligned = normals * signs[:, None]
    mean_vec = aligned.mean(axis=0)
    Rbar = float(np.linalg.norm(mean_vec))
    if Rbar < 1e-6:
        return None
    return mean_vec / Rbar, _fisher_kappa(Rbar)


def load_inverted_params(kr_csv: Path, p32_csv: Path) -> Dict[int, dict]:
    """Per-set inverted kr, P32, effective rmin, generation rmax."""
    params: Dict[int, dict] = {}
    with open(kr_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            sid = int(row["set_id"])
            params.setdefault(sid, {})
            params[sid]["kr"] = float(row["kr_hat"])
            params[sid]["rmin"] = float(row["set_effective_generation_rmin"])
            params[sid]["rmax_gen"] = float(row["generation_rmax"])
    with open(p32_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            sid = int(row["set_id"])
            params.setdefault(sid, {})
            params[sid]["P32"] = float(row["P32_hat"])
    # 분포 타입 기본값(powerlaw); 외부 config로 set별 override 가능(예: 지수분포)
    for sid in params:
        params[sid].setdefault("dist_type", "powerlaw")
    return params


# ----------------------------------------------------------------------
# Hidden generation + conditioning
# ----------------------------------------------------------------------
def generate_hidden_discs(
    params: Dict[int, dict],
    visible_discs: List[dict],
    box: dict,
    rmax_local: float,
    seed: int,
    target_sets: List[int],
) -> List[dict]:
    """Generate unconditioned stochastic discs per set inside the local box.

    target_sets는 하드코딩이 아니라 상위(main)에서 데이터로부터 유도해 전달한다.
    per-set 분포타입(params["dist_type"])에 따라 powerlaw/exponential 크기분포로 생성한다.
    """
    V = box["dx"] * box["dy"] * box["dz"]
    hidden: List[dict] = []
    for sid in target_sets:
        p = params.get(sid)
        if p is None or "P32" not in p:
            continue
        dist_type = p.get("dist_type", "powerlaw")
        # 크기분포 구성 (powerlaw는 kr, exponential은 r0 필요)
        if dist_type == "exponential":
            r0 = p.get("r0")
            if r0 is None:
                print(f"  [set {sid}] exponential set needs r0 (via --config); skipped")
                continue
            size_dist = {"type": "exponential", "r0": float(r0),
                         "rmin": p.get("rmin", 0.5), "rmax": rmax_local}
            size_desc = f"exp r0={float(r0):.2f}"
        else:  # powerlaw
            if "kr" not in p:
                print(f"  [set {sid}] powerlaw set has no kr; skipped")
                continue
            size_dist = {"type": "powerlaw", "kr": p["kr"], "rmin": p["rmin"], "rmax": rmax_local}
            size_desc = f"kr={p['kr']:.2f}"
        # 방향(orientation)은 복원 disc에서 추정
        ori = set_orientation_from_discs(visible_discs, sid)
        if ori is None:
            print(f"  [set {sid}] skipped hidden gen: insufficient orientation evidence")
            continue
        mean_n, kappa = ori
        N = GEN.compute_num_fractures_from_P32(p["P32"], size_dist, V)
        if N <= 0:
            continue
        base = seed + sid * 1000
        radii = GEN.sample_radius(size_dist, N, seed=base)
        normals = GEN.sample_fisher_normals(mean_n, kappa, N, seed=base + 1)
        strike_u, dip_u = GEN.normal_to_strike_dip_basis_vectorized(normals)
        centers = GEN.sample_centers_from_surface_points(
            box, radii, strike_u, dip_u, "area_uniform", seed=base + 2
        )
        for j in range(N):
            hidden.append(dict(
                set_id=sid, center=centers[j], normal=normals[j],
                radius=float(radii[j]), source="hidden", adoption="stochastic",
            ))
        print(f"  [set {sid}] hidden generated: N={N:,}  (P32={p['P32']:.3f}, {size_desc}, "
              f"rmin={size_dist['rmin']:.2f}, kappa={kappa:.1f})")
    return hidden


def remove_face_intersecting(
    discs: List[dict], face_xs: List[float], poly_ccw: np.ndarray
) -> Tuple[List[dict], int]:
    """Drop discs that produce a visible trace on any observed face."""
    kept, removed = [], 0
    for d in discs:
        intersects = any(
            visible_trace_on_face(d["center"], d["normal"], d["radius"], xf, poly_ccw) is not None
            for xf in face_xs
        )
        if intersects:
            removed += 1
        else:
            kept.append(d)
    return kept, removed


# ----------------------------------------------------------------------
# Diagnostics + figure
# ----------------------------------------------------------------------
def _polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def conditioned_traces(
    visible: List[dict], face_xs: List[float], poly_ccw: np.ndarray
) -> Dict[float, List[Tuple[np.ndarray, np.ndarray]]]:
    """Traces the VISIBLE discs leave on each face (the conditioned traces)."""
    by_face: Dict[float, List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for d in visible:
        for xf in face_xs:
            seg = visible_trace_on_face(d["center"], d["normal"], d["radius"], xf, poly_ccw)
            if seg is not None:
                by_face[xf].append(seg)
    return by_face


def print_diagnostics(
    face_xs, poly_ccw, observed, cond, obs_len_by_set, n_visible, n_hidden_gen,
    n_hidden_removed, n_hidden_kept, params, target_sets, visible=None,
):
    window_area = _polygon_area(poly_ccw) * len(face_xs)
    cond_len = sum(np.linalg.norm(p1 - p0) for segs in cond.values() for p0, p1 in segs)
    n_obs = sum(len(s) for s in observed.values())
    n_cond = sum(len(s) for s in cond.values())

    # set별 복원(visible) 재생성 길이 (P21 을 set 별로 공정 비교하기 위함)
    cond_len_by_set: Dict[int, float] = defaultdict(float)
    for d in (visible or []):
        for xf in face_xs:
            seg = visible_trace_on_face(d["center"], d["normal"], d["radius"], xf, poly_ccw)
            if seg is not None:
                cond_len_by_set[int(d["set_id"])] += float(np.linalg.norm(seg[1] - seg[0]))

    obs_len_all = sum(obs_len_by_set.values())
    # Conditioned traces come only from the conditioned sets, so the fair P21
    # comparison restricts the observed side to the same sets.
    ts = set(target_sets)
    obs_len_matched = sum(L for s, L in obs_len_by_set.items() if s in ts)
    p21_obs_all = obs_len_all / window_area if window_area else 0.0
    p21_obs_matched = obs_len_matched / window_area if window_area else 0.0
    p21_cond = cond_len / window_area if window_area else 0.0
    p32_target = sum(params[s]["P32"] for s in target_sets if s in params and "P32" in params[s])

    print("-" * 64)
    print("Conditional hidden DFN - diagnostics")
    print("-" * 64)
    print(f"  observation faces        : {face_xs}")
    print(f"  visible discs (kept)     : {n_visible}")
    print(f"  hidden discs generated   : {n_hidden_gen:,}")
    print(f"  hidden removed (face-hit) : {n_hidden_removed:,}")
    print(f"  hidden discs kept        : {n_hidden_kept:,}")
    print(f"  total conditional discs  : {n_visible + n_hidden_kept:,}")
    print("  " + "-" * 40)
    print(f"  observed traces (blue)   : {n_obs}  | P21(all sets)      = {p21_obs_all:.3f} 1/m")
    print(f"  conditioned traces (red) : {n_cond}  | P21               = {p21_cond:.3f} 1/m")
    print("  " + "-" * 40)
    print(f"  set-matched comparison (conditioned sets {sorted(target_sets)}):")
    print(f"    observed  P21 (matched): {p21_obs_matched:.3f} 1/m")
    print(f"    conditioned P21        : {p21_cond:.3f} 1/m")
    if p21_obs_matched > 0:
        err = (p21_cond - p21_obs_matched) / p21_obs_matched * 100
        print(f"    P21 error (matched)    : {err:+.1f} %")
    print(f"  target P32 (sets {sorted(target_sets)}) : {p32_target:.3f} 1/m")

    # set별 P21 (관측 vs 복원 visible) — set 을 섞지 않는 공정 비교.
    # 혼합 'matched' 비교는 conditioned(전 set visible)와 observed(target set만)의
    # 비대칭 때문에 왜곡될 수 있어, set 별 표를 권장 지표로 함께 출력한다.
    if visible is not None:
        ts = set(target_sets)
        all_sets = sorted(set(obs_len_by_set) | set(cond_len_by_set))
        print("  " + "-" * 40)
        print("  per-set P21 [1/m]  (관측 vs 복원 visible):")
        print(f"    {'set':>4} {'observed':>9} {'reconstructed':>13} {'err':>8}  role")
        for s in all_sets:
            po = obs_len_by_set.get(s, 0.0) / window_area if window_area else 0.0
            pc = cond_len_by_set.get(s, 0.0) / window_area if window_area else 0.0
            role = "conditioned(visible+hidden)" if s in ts else "visible-only"
            err_s = f"{(pc - po) / po * 100:+.1f}%" if po > 0 else "  n/a"
            print(f"    {s:>4} {po:>9.3f} {pc:>13.3f} {err_s:>8}  {role}")
        if p21_obs_all > 0:
            err_all = (p21_cond - p21_obs_all) / p21_obs_all * 100
            print(f"    {'ALL':>4} {p21_obs_all:>9.3f} {p21_cond:>13.3f} {err_all:>+7.1f}%  "
                  f"(대칭: 전 set vs 전 set)")
    print("-" * 64)


def plot_fig(observed, cond, face_xs, poly_ccw, hidden_kept, out_path, seed, show_hidden=False):
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(111, projection="3d")

    ymin, ymax = poly_ccw[:, 0].min(), poly_ccw[:, 0].max()
    zmin, zmax = poly_ccw[:, 1].min(), poly_ccw[:, 1].max()

    # Observation window on each face
    for xf in face_xs:
        loop = np.vstack([poly_ccw, poly_ccw[0]])
        ax.plot(np.full(len(loop), xf), loop[:, 0], loop[:, 1], color="0.6", lw=0.8, alpha=0.7)

    # Optional: hidden disc centers near the window slab (to show volume filling)
    if show_hidden and hidden_kept:
        hc = np.array([d["center"] for d in hidden_kept])
        m = ((hc[:, 1] >= ymin - 1) & (hc[:, 1] <= ymax + 1)
             & (hc[:, 2] >= zmin - 1) & (hc[:, 2] <= zmax + 1))
        ax.scatter(hc[m, 0], hc[m, 1], hc[m, 2], s=2, color="0.8", alpha=0.2)

    for segs in observed.values():
        for p0, p1 in segs:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    color=OBSERVED_COLOR, lw=1.3, alpha=0.9)
    for segs in cond.values():
        for p0, p1 in segs:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    color=CONDITIONED_COLOR, lw=1.3, alpha=0.9)

    ax.set_xlabel("x — tunnel advance [m]")
    ax.set_ylabel("y — North [m]")
    ax.set_zlabel("z — Up [m]")
    ax.set_title(f"Observed (blue) vs conditioned (red) traces on tunnel faces  (seed={seed})")

    # Zoom to the tunnel window slab so traces fill the frame
    ax.set_xlim(min(face_xs) - 0.5, max(face_xs) + 0.5)
    ax.set_ylim(ymin - 0.5, ymax + 0.5)
    ax.set_zlim(zmin - 0.5, zmax + 0.5)
    ax.set_box_aspect((max(face_xs) - min(face_xs) + 1, ymax - ymin + 1, zmax - zmin + 1))
    ax.view_init(elev=18, azim=-72)
    ax.legend(handles=[
        Line2D([0], [0], color=OBSERVED_COLOR, lw=1.5, label="Observed traces"),
        Line2D([0], [0], color=CONDITIONED_COLOR, lw=1.5, label="Conditioned traces (visible discs)"),
        Line2D([0], [0], color="0.6", lw=1.0, label="Observation window"),
    ], loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Figure written to {out_path}")


def write_dfn_csv(visible, hidden_kept, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "set_id", "cx", "cy", "cz", "nx", "ny", "nz", "radius", "adoption"])
        for d in visible + hidden_kept:
            c, n = d["center"], d["normal"]
            w.writerow([d["source"], d["set_id"], c[0], c[1], c[2],
                        n[0], n[1], n[2], d["radius"], d["adoption"]])
    print(f"Conditional DFN written to {out_path}")


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pipeline-dir", type=Path,
                    default=REPO / "storage/output/pipeline_test_laxemar")
    ap.add_argument("--rmax-local", type=float, default=10.0,
                    help="Upper radius cutoff for local hidden generation [m].")
    ap.add_argument("--margin", type=float, default=None,
                    help="Local box margin around faces/window [m]. Default = rmax-local.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-adoptions", default="deterministic_disc,orientation_only",
                    help="Comma list of reconstructed-disc adoptions to keep as visible.")
    ap.add_argument("--show-hidden", action="store_true",
                    help="Overlay hidden disc centers near the window (default off).")
    ap.add_argument("--sets", nargs="+", type=int, default=None,
                    help="Sets to condition (default: all sets with inverted params).")
    ap.add_argument("--exclude-sets", nargs="+", type=int, default=[],
                    help="Sets to exclude from conditioning.")
    ap.add_argument("--config", default=None,
                    help="Optional dataset JSON: per-set dist_type/r0 (e.g. exponential sets).")
    args = ap.parse_args()

    pdir = args.pipeline_dir
    margin = args.margin if args.margin is not None else args.rmax_local

    # --- Load inputs ---
    keep = tuple(a.strip() for a in args.keep_adoptions.split(","))
    visible = load_visible_discs(pdir / "reconstruct/reconstructed_discs.csv", keep)
    observed, obs_len_by_set = load_observed_traces(pdir / "trace_dataset/trace_dataset_3d.csv")
    params = load_inverted_params(pdir / "kr/kr_summary_by_set.csv", pdir / "p32/p32_summary.csv")

    # per-set 분포타입/r0 override (지수분포 set 등)
    if args.config:
        import json
        cfg = json.load(open(args.config, encoding="utf-8"))
        for sid_str, s in cfg.get("sets", {}).items():
            sid = int(sid_str)
            params.setdefault(sid, {})
            if "dist_type" in s:
                params[sid]["dist_type"] = str(s["dist_type"])
            if "r0" in s:
                params[sid]["r0"] = float(s["r0"])

    # 대상 set: 지정 없으면 역산 파라미터가 있는 모든 set (하드코딩 (1,2,3,5) 제거)
    exclude = set(args.exclude_sets)
    target_sets = args.sets if args.sets is not None else sorted(params.keys())
    target_sets = [s for s in target_sets if s not in exclude]
    print(f"Target sets (conditioned): {target_sets}")

    face_xs = sorted(observed.keys())
    with h5py.File(pdir / "dfn_export_for_python.h5", "r") as f:
        poly = np.array(f["tunnel/poly_YZ"])
    poly_ccw = _ccw_polygon(poly)

    # --- Local conditioning box around the observed faces ---
    ymin, ymax = poly_ccw[:, 0].min(), poly_ccw[:, 0].max()
    zmin, zmax = poly_ccw[:, 1].min(), poly_ccw[:, 1].max()
    box = dict(
        x0=min(face_xs) - margin, dx=(max(face_xs) - min(face_xs)) + 2 * margin,
        y0=ymin - margin, dy=(ymax - ymin) + 2 * margin,
        z0=zmin - margin, dz=(zmax - zmin) + 2 * margin,
    )
    print(f"Local conditioning box: x[{box['x0']:.1f},{box['x0']+box['dx']:.1f}] "
          f"y[{box['y0']:.1f},{box['y0']+box['dy']:.1f}] z[{box['z0']:.1f},{box['z0']+box['dz']:.1f}]")
    print(f"Visible discs kept ({keep}): {len(visible)}")

    # --- Generate + condition ---
    hidden_all = generate_hidden_discs(params, visible, box, args.rmax_local, args.seed, target_sets)
    hidden_kept, n_removed = remove_face_intersecting(hidden_all, face_xs, poly_ccw)

    # --- Conditioned traces (red) = visible discs re-projected onto faces ---
    cond = conditioned_traces(visible, face_xs, poly_ccw)

    # --- Diagnostics + outputs ---
    print_diagnostics(face_xs, poly_ccw, observed, cond, obs_len_by_set, len(visible),
                      len(hidden_all), n_removed, len(hidden_kept), params, target_sets,
                      visible=visible)
    out_dir = pdir / "conditional_hidden"
    write_dfn_csv(visible, hidden_kept, out_dir / "conditional_dfn.csv")
    plot_fig(observed, cond, face_xs, poly_ccw, hidden_kept,
             out_dir / "observed_vs_conditioned_traces.png", args.seed, args.show_hidden)


if __name__ == "__main__":
    main()
