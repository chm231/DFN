"""
diagnose_unit_p32_importance_oracle.py
======================================
Validates the unit-P32 importance-sampling calibration factor C by comparing it
against a brute-force small-domain oracle simulation.

Oracle method
-------------
For each (site, set_id):

  1. Draw a small number of representative faces from face_x_positions
     (``--n-oracle-faces``, default 5).

  2. For each oracle face, place P32 = 1 m2/m3 worth of fractures inside a
     finite influence volume centred on the face.

     Number of fractures to place = n_fractures_per_face (CLI arg).
     The IS weight compensates: weight = fracture_number_density * V_proposal / N_frac.

  3. For each fracture:
     - sample radius r from the population distribution (unbiased)
     - sample normal from Fisher(mean_pole, kappa)
     - sample centre uniformly in influence volume
     - compute disc-face intersection chord
     - clip chord to tunnel polygon
     - accumulate IS-weighted visible_length

  4. C_bruteforce_replicate = sum(weights * visible_lengths) / total_observation_area

  5. Run ``--n-oracle-replicates`` independent replicates; report mean, std, 95% CI.

Comparison
----------
  importance_bruteforce_ratio = C_importance / C_bruteforce_mean

  oracle_pass     : 0.90 <= ratio <= 1.10
  oracle_marginal : 0.80 <= ratio < 0.90  OR  1.10 < ratio <= 1.25
  oracle_failed   : otherwise

Outputs
-------
  storage/output/p32_mc_calibrated_effective_rmin/unit_p32_oracle_check/
    unit_p32_importance_vs_bruteforce_oracle.csv
    unit_p32_importance_vs_bruteforce_oracle_report.md
"""

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dfn_analysis.estimate_p32_mc_calibrated import (
    estimate_unit_p32_forward_mc,
    load_face_x_positions,
    radius_moments,
    read_csv,
    sample_population_radius,
    to_float,
    write_csv,
)
from dfn_analysis.estimate_radius_powerlaw_window_mc import (
    SITE_FISHER_PARAMS,
    clip_segments_to_convex_polygon_vectorized,
    mean_pole_from_trend_plunge,
    normals_to_trace_directions_yz,
    sample_fisher_normals,
)
from dfn_analysis.summarize_setwise_trace_statistics import (
    compute_total_observation_area,
    load_rough_face_collection_from_h5,
)


# ---------------------------------------------------------------------------
# Target sets for oracle validation
# ---------------------------------------------------------------------------
ORACLE_TARGETS = [
    ("laxemar", 1),
    ("laxemar", 2),
    ("forsmark", 2),
    ("forsmark", 5),
]

DEFAULT_TRACE_H5 = {
    "forsmark": "storage/output/traces_forsmark_rmin0p5/setwise_3d_traces.h5",
    "laxemar": "storage/output/traces_laxemar_rmin0p5/setwise_3d_traces.h5",
}
DEFAULT_ROUGH_MESH_H5 = (
    "storage/output/rough_face_mesh_collection/synthetic_rough_face_collection.h5"
)
DEFAULT_KR_CSV = "storage/output/final_kr_recovery_summary_effective_rmin.csv"
DEFAULT_FULL_UNIT_CSV = (
    "storage/output/p32_mc_calibrated_effective_rmin/full_unit_p32/p32_full_unit_summary.csv"
)
DEFAULT_OUTDIR = (
    "storage/output/p32_mc_calibrated_effective_rmin/unit_p32_oracle_check"
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def load_polygon_from_h5(trace_h5: str) -> np.ndarray:
    import h5py
    with h5py.File(trace_h5, "r") as f:
        if "meta" in f and "tunnel_poly_yz" in f["meta"]:
            return np.asarray(f["meta"]["tunnel_poly_yz"][:], dtype=np.float64)
    raise ValueError(f"tunnel_poly_yz not found in {trace_h5}")


# ---------------------------------------------------------------------------
# Brute-force oracle: one face, one replicate
# ---------------------------------------------------------------------------

def bruteforce_one_face(
    site: str,
    set_id: int,
    kr: float,
    rmin: float,
    rmax: float,
    polygon_yz: np.ndarray,
    face_x: float,
    total_observation_area: float,
    domain_margin_m: float,
    n_fractures_per_face: int,
    rng: np.random.Generator,
) -> float:
    """
    Brute-force estimate of C contribution from one face.

    Uses the same IS weight formula as estimate_unit_p32_forward_mc so the
    two estimators are directly comparable.  The only difference is that the
    population radius is drawn from the true (unbiased) distribution rather
    than from the size-biased importance proposal.

    Returns: sum(weights * visible_lengths) / total_observation_area
    """
    bbox_min = np.min(polygon_yz, axis=0)
    bbox_max = np.max(polygon_yz, axis=0)
    bbox_w = float(bbox_max[0] - bbox_min[0])
    bbox_h = float(bbox_max[1] - bbox_min[1])

    params = SITE_FISHER_PARAMS.get(site, {}).get(set_id)
    if params is None:
        return float("nan")
    trend, plunge, kappa = params
    mean_pole = mean_pole_from_trend_plunge(trend, plunge)

    _, mean_r2 = radius_moments(site, set_id, kr, rmin, rmax)
    if not np.isfinite(mean_r2) or mean_r2 <= 0.0:
        return float("nan")
    fracture_number_density = 1.0 / (math.pi * mean_r2)

    # --- Sample fracture population (unbiased radius) -----------------------
    radii = sample_population_radius(
        site, set_id, kr, rmin, rmax, n_fractures_per_face, rng
    )
    normals = sample_fisher_normals(mean_pole, kappa, n_fractures_per_face, rng)
    directions_yz, valid = normals_to_trace_directions_yz(normals)

    if not np.any(valid):
        return 0.0

    radii = radii[valid]
    normals = normals[valid]
    directions_yz = directions_yz[valid]

    sin_theta = np.sqrt(np.clip(1.0 - normals[:, 0] ** 2, 0.0, 1.0))
    valid_theta = sin_theta > 1e-8
    if not np.any(valid_theta):
        return 0.0

    radii = radii[valid_theta]
    normals = normals[valid_theta]
    directions_yz = directions_yz[valid_theta]
    sin_theta = sin_theta[valid_theta]

    x_half = radii * sin_theta
    proposal_w = bbox_w + 4.0 * radii
    proposal_h = bbox_h + 4.0 * radii
    proposal_volume = 2.0 * x_half * proposal_w * proposal_h

    # --- Fracture centres uniformly in influence volume --------------------
    # Match the proposal box used in estimate_unit_p32_forward_mc exactly
    center_x = rng.uniform(face_x - x_half, face_x + x_half)
    center_y = rng.uniform(
        bbox_min[0] - 2.0 * radii, bbox_max[0] + 2.0 * radii
    )
    center_z = rng.uniform(
        bbox_min[1] - 2.0 * radii, bbox_max[1] + 2.0 * radii
    )
    centers_yz = np.column_stack([center_y, center_z])

    # --- Disc-face intersection --------------------------------------------
    dx = face_x - center_x
    chord_offsets = np.abs(dx) / sin_theta
    intersecting = chord_offsets <= radii + 1e-10
    if not np.any(intersecting):
        return 0.0

    normals_i = normals[intersecting]
    directions_yz_i = directions_yz[intersecting]
    radii_i = radii[intersecting]
    sin_theta_i = sin_theta[intersecting]
    chord_offsets_i = chord_offsets[intersecting]
    centers_yz_i = centers_yz[intersecting]
    proposal_volume_i = proposal_volume[intersecting]
    dx_i = dx[intersecting]

    chord_lengths = 2.0 * np.sqrt(
        np.maximum(radii_i ** 2 - chord_offsets_i ** 2, 0.0)
    )
    t = dx_i / np.maximum(sin_theta_i ** 2, 1e-12)
    line_midpoints_yz = centers_yz_i + np.column_stack(
        [
            -t * normals_i[:, 0] * normals_i[:, 1],
            -t * normals_i[:, 0] * normals_i[:, 2],
        ]
    )

    # --- Polygon clipping --------------------------------------------------
    visible_lengths, classes = clip_segments_to_convex_polygon_vectorized(
        line_midpoints_yz,
        directions_yz_i,
        chord_lengths,
        polygon_yz,
    )
    accepted = (classes >= 0) & (visible_lengths > 0.0)
    if not np.any(accepted):
        return 0.0

    weights = fracture_number_density * proposal_volume_i[accepted] / n_fractures_per_face
    return float(np.sum(weights * visible_lengths[accepted])) / total_observation_area


# ---------------------------------------------------------------------------
# Oracle runner: all replicates for one set
# ---------------------------------------------------------------------------

def run_oracle(
    site: str,
    set_id: int,
    kr: float,
    rmin: float,
    rmax: float,
    polygon_yz: np.ndarray,
    face_x_positions: np.ndarray,
    total_observation_area: float,
    n_oracle_faces: int,
    n_oracle_replicates: int,
    n_fractures_per_face: int,
    domain_margin_m: float,
    rng_seed: int,
) -> dict:
    """Run brute-force oracle replicates and return statistics."""
    all_faces = np.asarray(face_x_positions, dtype=np.float64)
    n_total_faces = len(all_faces)
    n_faces_used = min(n_oracle_faces, n_total_faces)

    replicate_values: List[float] = []
    for rep in range(n_oracle_replicates):
        rng = np.random.default_rng(rng_seed + rep * 997)
        face_indices = rng.choice(n_total_faces, size=n_faces_used, replace=False)
        selected_faces = all_faces[face_indices]

        total_c = 0.0
        for face_x in selected_faces:
            c_face = bruteforce_one_face(
                site=site,
                set_id=set_id,
                kr=kr,
                rmin=rmin,
                rmax=rmax,
                polygon_yz=polygon_yz,
                face_x=float(face_x),
                total_observation_area=total_observation_area,
                domain_margin_m=domain_margin_m,
                n_fractures_per_face=n_fractures_per_face,
                rng=rng,
            )
            if np.isfinite(c_face):
                total_c += c_face

        # Scale to full face count
        if n_faces_used > 0:
            replicate_values.append(total_c * n_total_faces / n_faces_used)

    if not replicate_values:
        return {
            "C_bruteforce_mean": float("nan"),
            "C_bruteforce_std": float("nan"),
            "C_bruteforce_ci_low": float("nan"),
            "C_bruteforce_ci_high": float("nan"),
            "n_faces_used": n_faces_used,
            "n_oracle_replicates": n_oracle_replicates,
        }

    vals = np.asarray(replicate_values, dtype=np.float64)
    return {
        "C_bruteforce_mean": float(np.mean(vals)),
        "C_bruteforce_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "C_bruteforce_ci_low": float(np.percentile(vals, 2.5)),
        "C_bruteforce_ci_high": float(np.percentile(vals, 97.5)),
        "n_faces_used": n_faces_used,
        "n_oracle_replicates": n_oracle_replicates,
    }


# ---------------------------------------------------------------------------
# Oracle status
# ---------------------------------------------------------------------------

def classify_oracle_status(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "oracle_failed"
    if 0.90 <= ratio <= 1.10:
        return "oracle_pass"
    if (0.80 <= ratio < 0.90) or (1.10 < ratio <= 1.25):
        return "oracle_marginal"
    return "oracle_failed"


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def build_markdown_report(rows: List[dict], mc_samples: int, mc_replicates: int) -> str:
    lines = [
        "# Unit-P32 Importance Sampling vs Brute-Force Oracle Check",
        "",
        "## Purpose",
        "Validate that the importance-sampling unit-P32 calibration factor **C_importance**",
        "matches the directly simulated **C_bruteforce** from a brute-force small-domain",
        "oracle population.  If the estimators agree (ratio ~1), the IS implementation",
        "is unbiased and observed deviations (e.g. Laxemar Set 2 C_ratio = 1.48) are",
        "attributable to geometry / process differences rather than IS weight errors.",
        "",
        f"**IS settings**: mc_samples = {mc_samples}, mc_replicates = {mc_replicates}",
        "",
        "## Results",
        "",
        "| site | set_id | C_importance | C_bruteforce_mean | C_bruteforce_std | ratio | oracle_status |",
        "|------|--------|-------------|-------------------|-----------------|-------|---------------|",
    ]
    for row in rows:
        def _fmt(v: object, fmt: str = ".5f") -> str:
            try:
                return format(float(v), fmt)  # type: ignore[arg-type]
            except Exception:
                return str(v)

        lines.append(
            f"| {row.get('site','')} | {row.get('set_id','')} "
            f"| {_fmt(row.get('C_importance'))} "
            f"| {_fmt(row.get('C_bruteforce_mean'))} "
            f"| {_fmt(row.get('C_bruteforce_std'))} "
            f"| {_fmt(row.get('importance_bruteforce_ratio'), '.4f')} "
            f"| {row.get('oracle_status','')} |"
        )

    lines += [
        "",
        "## Oracle Status Legend",
        "- **oracle_pass**: ratio in [0.90, 1.10] — IS estimator unbiased",
        "- **oracle_marginal**: ratio in [0.80, 0.90) or (1.10, 1.25] — minor systematic deviation",
        "- **oracle_failed**: ratio outside [0.80, 1.25] — IS weight or proposal volume problem",
        "",
        "## Interpretation",
        "",
    ]

    for row in rows:
        site = row.get("site", "")
        set_id = row.get("set_id", "")
        status = row.get("oracle_status", "")
        try:
            ratio_f = float(row.get("importance_bruteforce_ratio", "nan"))
        except Exception:
            ratio_f = float("nan")

        lines.append(f"### {str(site).capitalize()} Set {set_id}")
        if status == "oracle_pass":
            lines.append(
                f"  ratio = {ratio_f:.4f} -> **oracle_pass**.  "
                "IS estimator agrees with brute-force.  "
                "Calibration factor C is unbiased for this set."
            )
        elif status == "oracle_marginal":
            lines.append(
                f"  ratio = {ratio_f:.4f} -> **oracle_marginal**.  "
                "Small systematic deviation.  "
                "Recommend re-run with larger --n-fractures-per-face for confirmation."
            )
        else:
            lines.append(
                f"  ratio = {ratio_f:.4f} -> **oracle_failed**.  "
                "Significant bias in IS estimator or proposal volume.  "
                "Review estimate_unit_p32_forward_mc importance weights."
            )
        lines.append("")

    lines += [
        "## Next Steps",
        "",
        "1. If all targets oracle_pass: proceed to unit_p32_mc_replicates = 100 re-run.",
        "2. If Laxemar Set 2 oracle_pass: C_ratio = 1.48 reflects genuine geometry/process",
        "   difference, not an estimator bug; investigate trace geometry.",
        "3. If any target oracle_failed: debug importance weights in estimate_unit_p32_forward_mc.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose unit-P32 importance sampling vs brute-force oracle."
    )
    parser.add_argument(
        "--full-unit-csv",
        default=DEFAULT_FULL_UNIT_CSV,
        help="Full-scale unit-P32 summary CSV (contains C_importance per set).",
    )
    parser.add_argument(
        "--trace-h5-forsmark",
        default=DEFAULT_TRACE_H5["forsmark"],
    )
    parser.add_argument(
        "--trace-h5-laxemar",
        default=DEFAULT_TRACE_H5["laxemar"],
    )
    parser.add_argument(
        "--rough-mesh-h5",
        default=DEFAULT_ROUGH_MESH_H5,
    )
    parser.add_argument(
        "--kr-csv",
        default=DEFAULT_KR_CSV,
    )
    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
    )
    parser.add_argument(
        "--n-oracle-faces",
        type=int,
        default=5,
        help="Number of faces sampled per oracle replicate.",
    )
    parser.add_argument(
        "--n-oracle-replicates",
        type=int,
        default=30,
        help="Number of brute-force replicates.",
    )
    parser.add_argument(
        "--n-fractures-per-face",
        type=int,
        default=20000,
        help="Number of fractures placed per face per brute-force replicate.",
    )
    parser.add_argument(
        "--domain-margin-m",
        type=float,
        default=5.0,
        help="Extra margin beyond tunnel bbox for brute-force domain (metres).",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=50000,
        help="MC samples for IS estimator (must match full run).",
    )
    parser.add_argument(
        "--mc-replicates",
        type=int,
        default=50,
        help="MC replicates for IS estimator (must match full run).",
    )
    parser.add_argument(
        "--recompute-importance",
        action="store_true",
        help="Recompute C_importance instead of reading from --full-unit-csv.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=770000,
        help="Base RNG seed for oracle replicates.",
    )
    parser.add_argument(
        "--site",
        choices=["forsmark", "laxemar"],
        default=None,
        help="Filter oracle targets to a single site.",
    )
    parser.add_argument(
        "--target-set",
        type=int,
        nargs="+",
        default=None,
        help="Filter oracle targets to specific set IDs (e.g. --target-set 2).",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # -- Load kr values -------------------------------------------------------
    kr_rows_all = read_csv(args.kr_csv)
    kr_map: Dict[Tuple[str, int], dict] = {
        (str(r["site"]), int(r["set_id"])): r for r in kr_rows_all
    }

    # -- Load C_importance from full-scale CSV --------------------------------
    full_unit_map: Dict[Tuple[str, int], dict] = {}
    if not args.recompute_importance and os.path.isfile(args.full_unit_csv):
        for r in read_csv(args.full_unit_csv):
            full_unit_map[(str(r["site"]), int(r["set_id"]))] = r
        print(f"[*] Loaded C_importance from: {args.full_unit_csv}")
    else:
        print("[*] recompute_importance=True or CSV missing; will recompute C_importance.")

    # -- Load rough mesh area (shared) ----------------------------------------
    rough_faces = load_rough_face_collection_from_h5(args.rough_mesh_h5)
    total_observation_area = compute_total_observation_area(rough_faces)
    print(f"[*] Total observation area: {total_observation_area:.3f} m2")

    # -- Build active target list (apply --site / --target-set filter) --------
    active_targets = [
        (s, sid) for s, sid in ORACLE_TARGETS
        if (args.site is None or s == args.site)
        and (args.target_set is None or sid in args.target_set)
    ]
    if not active_targets:
        print("[!] No oracle targets match --site / --target-set filter.")
        sys.exit(1)
    active_sites = {s for s, _ in active_targets}

    # -- Per-site polygon and face positions (only needed sites) ---------------
    polygon_cache: Dict[str, np.ndarray] = {}
    face_x_cache: Dict[str, np.ndarray] = {}
    for site in ("forsmark", "laxemar"):
        if site not in active_sites:
            continue
        trace_h5 = (
            args.trace_h5_forsmark if site == "forsmark" else args.trace_h5_laxemar
        )
        polygon_cache[site] = load_polygon_from_h5(trace_h5)
        face_x_cache[site] = load_face_x_positions(trace_h5)
        n_total = len(face_x_cache[site])
        print(
            f"[*] {site}: polygon vertices={len(polygon_cache[site])}, "
            f"faces={n_total}"
        )
        if args.n_oracle_faces >= n_total:
            print(
                f"  [*] --n-oracle-faces {args.n_oracle_faces} >= available faces {n_total}; "
                f"all {n_total} faces will be used each replicate."
            )

    # -- Main loop ------------------------------------------------------------
    out_rows: List[dict] = []
    for site, set_id in active_targets:
        print(f"\n[*] Processing {site} Set {set_id} ...")
        kr_row = kr_map.get((site, set_id), {})
        kr_used = to_float(kr_row, "kr_hat")
        set_rmin = to_float(kr_row, "set_likelihood_rmin")
        set_effective_rmin = to_float(kr_row, "set_effective_generation_rmin")
        rmax = 250.0

        if not np.isfinite(kr_used):
            print(f"  [!] kr_used not finite for {site} Set {set_id} — skipping.")
            continue
        if not np.isfinite(set_rmin):
            print(f"  [!] set_rmin not finite for {site} Set {set_id} — skipping.")
            continue

        polygon_yz = polygon_cache[site]
        face_x_positions = face_x_cache[site]

        # -- C_importance -----------------------------------------------------
        full_row = full_unit_map.get((site, set_id), {})
        if not args.recompute_importance and full_row:
            c_imp = to_float(full_row, "calibration_factor_C")
            print(f"  C_importance (from CSV) = {c_imp:.6f}")
        else:
            print(f"  Recomputing C_importance ...")
            imp_result = estimate_unit_p32_forward_mc(
                site=site,
                set_id=set_id,
                kr=kr_used,
                rmin=set_rmin,
                rmax=rmax,
                polygon_yz=polygon_yz,
                face_x_positions=face_x_positions,
                total_observation_area=total_observation_area,
                mc_samples=args.mc_samples,
                mc_replicates=args.mc_replicates,
                rng_seed=260000 + set_id * 100,
                window_mode="polygon",
            )
            c_imp = float(imp_result["calibration_factor_C"])
            print(f"  C_importance (recomputed) = {c_imp:.6f}")

        # -- C_bruteforce oracle ----------------------------------------------
        print(
            f"  Running brute-force oracle: {args.n_oracle_replicates} replicates x "
            f"{args.n_oracle_faces} faces x {args.n_fractures_per_face} fractures/face ..."
        )
        oracle = run_oracle(
            site=site,
            set_id=set_id,
            kr=kr_used,
            rmin=set_rmin,
            rmax=rmax,
            polygon_yz=polygon_yz,
            face_x_positions=face_x_positions,
            total_observation_area=total_observation_area,
            n_oracle_faces=args.n_oracle_faces,
            n_oracle_replicates=args.n_oracle_replicates,
            n_fractures_per_face=args.n_fractures_per_face,
            domain_margin_m=args.domain_margin_m,
            rng_seed=args.rng_seed + set_id * 1000,
        )
        c_bf = oracle["C_bruteforce_mean"]
        print(
            f"  C_bruteforce_mean = {c_bf:.6f}  "
            f"(std={oracle['C_bruteforce_std']:.6f})"
        )

        # -- Ratio and classification -----------------------------------------
        if np.isfinite(c_imp) and np.isfinite(c_bf) and c_bf > 0.0:
            ratio = c_imp / c_bf
        else:
            ratio = float("nan")
        status = classify_oracle_status(ratio)
        print(f"  ratio = {ratio:.4f}  ->  {status}")

        # -- Notes -----------------------------------------------------------
        notes_parts = [
            f"mc_samples={args.mc_samples}",
            f"mc_replicates={args.mc_replicates}",
            f"n_oracle_replicates={args.n_oracle_replicates}",
            f"n_fractures_per_face={args.n_fractures_per_face}",
        ]
        if site == "laxemar" and set_id == 2:
            notes_parts.append(
                "laxemar_set2_c_ratio=1.48_marginal; "
                "oracle_pass=IS_unbiased_geometry_difference; "
                "oracle_failed=IS_weight_error"
            )
        if site == "forsmark" and set_id == 2:
            notes_parts.append("forsmark_set2_kr_systematic_bias_provisional")

        out_rows.append(
            {
                "site": site,
                "set_id": set_id,
                "kr_used": kr_used,
                "set_effective_generation_rmin": set_effective_rmin,
                "face_subset": f"{args.n_oracle_faces}_of_{len(face_x_positions)}",
                "n_faces": args.n_oracle_faces,
                "p32_unit_value": 1.0,
                "C_importance": c_imp,
                "C_bruteforce_mean": oracle["C_bruteforce_mean"],
                "C_bruteforce_std": oracle["C_bruteforce_std"],
                "C_bruteforce_ci_low": oracle["C_bruteforce_ci_low"],
                "C_bruteforce_ci_high": oracle["C_bruteforce_ci_high"],
                "importance_bruteforce_ratio": ratio,
                "oracle_status": status,
                "notes": "; ".join(notes_parts),
            }
        )

    if not out_rows:
        print("[!] No oracle rows produced — check input files and kr CSV.")
        sys.exit(1)

    # -- Write outputs --------------------------------------------------------
    csv_path = os.path.join(
        args.outdir, "unit_p32_importance_vs_bruteforce_oracle.csv"
    )
    md_path = os.path.join(
        args.outdir, "unit_p32_importance_vs_bruteforce_oracle_report.md"
    )

    write_csv(out_rows, csv_path)
    print(f"\n[*] Oracle CSV written to: {csv_path}")

    md_content = build_markdown_report(out_rows, args.mc_samples, args.mc_replicates)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[*] Oracle report written to: {md_path}")

    # -- Console summary ------------------------------------------------------
    print("\n" + "=" * 68)
    print("Oracle check summary")
    print("=" * 68)
    print(
        f"{'site':<12} {'set_id':<8} {'C_imp':>10} {'C_bf':>10} "
        f"{'ratio':>8} {'status':<22}"
    )
    for row in out_rows:
        def _s(v: object, fmt: str = ".5f") -> str:
            try:
                return format(float(v), fmt)  # type: ignore[arg-type]
            except Exception:
                return "nan"

        print(
            f"{row['site']:<12} {row['set_id']:<8} "
            f"{_s(row['C_importance']):>10} "
            f"{_s(row['C_bruteforce_mean']):>10} "
            f"{_s(row['importance_bruteforce_ratio'], '.4f'):>8} "
            f"{row['oracle_status']:<22}"
        )
    print("=" * 68)


if __name__ == "__main__":
    main()
