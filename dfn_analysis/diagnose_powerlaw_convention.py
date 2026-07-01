import argparse
import csv
import os
import numpy as np
import h5py
from scipy.optimize import minimize_scalar

SITE_TABLE_R0 = {
    "forsmark": {1: 0.28, 2: 0.25, 3: 0.14, 4: 0.15, 5: 0.25},
    "laxemar": {1: 0.328, 2: 0.977, 3: 0.858, 4: 4.0, 5: 0.400},
}
SITE_DIST_TYPE = {
    "forsmark": {1: "powerlaw", 2: "powerlaw", 3: "powerlaw", 4: "powerlaw", 5: "powerlaw"},
    "laxemar": {1: "powerlaw", 2: "powerlaw", 3: "powerlaw", 4: "exponential", 5: "powerlaw"},
}

# Nominals
FORSMARK_KR_TRUE = {1: 2.88, 2: 3.02, 3: 2.81, 4: 2.95, 5: 2.92}
LAXEMAR_KR_TRUE = {1: 2.85, 2: 3.04, 3: 3.01, 5: 3.6}


def log_likelihood_truncated_powerlaw(alpha: float, r: np.ndarray, rmin: float, rmax: float) -> float:
    n = len(r)
    if n == 0:
        return -np.inf
    
    if abs(alpha - 1.0) < 1e-7:
        log_c = -np.log(rmin) - np.log(np.log(rmax / rmin))
    else:
        denom = rmin**(1.0 - alpha) - rmax**(1.0 - alpha)
        if denom <= 0:
            return -np.inf
        log_c = np.log(abs(alpha - 1.0)) - np.log(denom)
        
    return n * log_c - alpha * np.sum(np.log(r))


def estimate_alpha_mle(r: np.ndarray, rmin: float, rmax: float) -> float:
    """Estimates the PDF exponent alpha of the truncated power law."""
    res = minimize_scalar(
        lambda a: -log_likelihood_truncated_powerlaw(a, r, rmin, rmax),
        bounds=(1.01, 15.0),
        method="bounded"
    )
    return float(res.x) if res.success else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the power-law exponent convention of DFN generator.")
    parser.add_argument("--input", default="storage/data/dfn_export_for_python.h5", help="Input HDF5 DFN file")
    parser.add_argument("--outdir", default="storage/output/powerlaw_convention_diagnostics", help="Output directory")
    parser.add_argument("--site", choices=["forsmark", "laxemar"], default="forsmark", help="Site name")
    parser.add_argument("--estimator-rmin-used", type=float, default=0.5, help="Estimator/global rmin to audit against generated support.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"[*] Loading DFN raw radii from: {args.input}")
    with h5py.File(args.input, "r") as f:
        radii = f["/fractures/radii"][:].ravel().astype(np.float64)
        set_ids = f["/fractures/set_id"][:].ravel().astype(np.int32) if "/fractures/set_id" in f else np.ones(len(radii), dtype=np.int32)
        global_generation_rmin = float(np.asarray(f["/meta/generation_rmin"][()]).ravel()[0]) if "/meta/generation_rmin" in f else float(args.estimator_rmin_used)
    
    kr_table_map = FORSMARK_KR_TRUE if args.site == "forsmark" else LAXEMAR_KR_TRUE
    
    summary_rows = []
    
    print("\n" + "=" * 80)
    print("                      DFN RAW RADIUS CONVENTION AUDIT")
    print("=" * 80)
    
    for set_id, kr_table in sorted(kr_table_map.items()):
        mask = set_ids == set_id
        set_radii = radii[mask]
        n_gen = len(set_radii)
        if n_gen == 0:
            print(f"Set {set_id}: No fractures found.")
            continue
            
        rmin_act = float(np.min(set_radii))
        rmax_act = float(np.max(set_radii))
        table_r0 = float(SITE_TABLE_R0[args.site][set_id])
        dist_type = SITE_DIST_TYPE[args.site][set_id]
        effective_generation_rmin = max(global_generation_rmin, table_r0) if dist_type == "powerlaw" else float(global_generation_rmin)
        if args.estimator_rmin_used < effective_generation_rmin - 1e-6:
            rmin_support_status = "estimator_lower_than_generated_support"
        elif abs(args.estimator_rmin_used - effective_generation_rmin) <= 1e-6:
            rmin_support_status = "matched"
        else:
            rmin_support_status = "estimator_higher_than_generated_support"
        p50 = float(np.percentile(set_radii, 50))
        p90 = float(np.percentile(set_radii, 90))
        p99 = float(np.percentile(set_radii, 99))
        
        # Estimate alpha_mle
        alpha_mle = estimate_alpha_mle(set_radii, rmin_act, rmax_act)
        
        # Convention A: f(r) ~ r^-(kr+1) => kr = alpha - 1
        kr_mle_A = alpha_mle - 1.0
        err_A = kr_mle_A - kr_table
        
        # Convention B: f(r) ~ r^-kr => kr = alpha
        kr_mle_B = alpha_mle
        err_B = kr_mle_B - kr_table
        
        if abs(err_A) < abs(err_B):
            preferred = "A"
            status = "consistent" if abs(err_A) < 0.1 else "offset"
        else:
            preferred = "B"
            status = "consistent" if abs(err_B) < 0.1 else "offset"
            
        print(f"\n[Set {set_id}] (n = {n_gen:,} fractures)")
        print(f"  - Radius Range  : [{rmin_act:.3f}, {rmax_act:.3f}] (m)")
        print(f"  - Percentiles   : p50={p50:.3f}, p90={p90:.3f}, p99={p99:.3f} (m)")
        print(f"  - Nominal kr    : {kr_table:.3f}")
        print(f"  - Est alpha_mle : {alpha_mle:.3f}")
        print(f"  - Convention A  : kr_mle_A={kr_mle_A:.3f} (error={err_A:+.3f})")
        print(f"  - Convention B  : kr_mle_B={kr_mle_B:.3f} (error={err_B:+.3f})")
        print(f"  - Preferred     : Convention {preferred} ({status})")
        
        summary_rows.append({
            "site": args.site,
            "rmin": rmin_act,
            "rmax": rmax_act,
            "set_id": set_id,
            "global_generation_rmin": global_generation_rmin,
            "table_r0": table_r0,
            "effective_generation_rmin": effective_generation_rmin,
            "observed_radius_min": float("nan"),
            "generated_radius_min": rmin_act,
            "estimator_rmin_used": float(args.estimator_rmin_used),
            "rmin_support_status": rmin_support_status,
            "kr_table": kr_table,
            "n_fractures_generated": n_gen,
            "radius_min": rmin_act,
            "radius_p50": p50,
            "radius_p90": p90,
            "radius_p99": p99,
            "kr_mle_convention_A": kr_mle_A,
            "kr_error_convention_A": err_A,
            "kr_mle_convention_B": kr_mle_B,
            "kr_error_convention_B": err_B,
            "preferred_convention": preferred,
            "convention_status": status
        })
        
    print("=" * 80 + "\n")
    
    csv_path = os.path.join(args.outdir, "generated_radius_convention_by_set.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
            
    print(f"[*] Diagnostics CSV written to: {csv_path}")


if __name__ == "__main__":
    main()
