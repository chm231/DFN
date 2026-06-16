"""Estimate P21, P30, and P32 from trace observations and fitted radius distributions."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _expected_radius_square(radius_info: Dict[str, object]) -> float:
    distribution = radius_info["radius_distribution"]
    dist_type = distribution["type"]
    mean = float(distribution.get("mean", np.nan))
    std = float(distribution.get("std", np.nan))
    if dist_type == "fixed_bound_truncated_power_law":
        fit_result = radius_info.get("fit_result")
        if fit_result is None:
            return np.nan
        alpha = float(fit_result.alpha_pdf_exponent)
        r_min = float(fit_result.r_min_m)
        r_max = float(fit_result.r_max_m)
        if abs(alpha - 3.0) <= 1e-8:
            c_r = 1.0 / ((1.0 / r_min) - (1.0 / r_max))
            return float(c_r * np.log(r_max / r_min))
        c_r = (
            1.0 / np.log(r_max / r_min)
            if abs(alpha - 1.0) <= 1e-8
            else (1.0 - alpha) / (r_max ** (1.0 - alpha) - r_min ** (1.0 - alpha))
        )
        exponent = 3.0 - alpha
        return float(c_r * (r_max**exponent - r_min**exponent) / exponent)
    if np.isfinite(std):
        return float(std**2 + mean**2)
    return mean**2 if np.isfinite(mean) else np.nan


def estimate_intensity_parameters(
    corrected_df: pd.DataFrame,
    set_stats_df: pd.DataFrame,
    correction_summary: Dict[int, Dict[str, object]],
    radius_distributions: Dict[int, Dict[str, object]],
) -> Dict[int, Dict[str, object]]:
    """Estimate intensity parameters by set, with explicit warnings for weak assumptions."""
    stats_map = {int(row["set_id"]): row for _, row in set_stats_df.iterrows()}
    results: Dict[int, Dict[str, object]] = {}
    for set_id, group in corrected_df.groupby("set_id", sort=True):
        stats_row = stats_map[int(set_id)]
        area = float(stats_row["observation_window_area"])
        corrected_p21 = float(group["corrected_length_yz"].sum() / area) if np.isfinite(area) and area > 0 else np.nan
        observed_p21 = float(stats_row["observed_P21"])
        summary = correction_summary[int(set_id)]
        q_set = float(summary["q_set"]) if summary.get("q_set") is not None and np.isfinite(summary.get("q_set")) else np.nan

        warnings: List[str] = list(summary.get("warnings", []))
        if np.isfinite(q_set) and q_set > 1e-8:
            orientation_corrected_p21 = float(corrected_p21 / q_set)
            method = "orientation_corrected_p21_proxy"
        else:
            orientation_corrected_p21 = corrected_p21
            method = "uncorrected_p21_proxy"
            warnings.append("Orientation correction was not applied to intensity.")

        radius_info = radius_distributions[int(set_id)]
        e_pi_r2 = float(np.pi * _expected_radius_square(radius_info))
        if not np.isfinite(e_pi_r2) or e_pi_r2 <= 0:
            estimated_p30 = np.nan
            estimated_p32 = orientation_corrected_p21
            warnings.append("P30 could not be estimated reliably from the radius model.")
        else:
            estimated_p32 = orientation_corrected_p21
            estimated_p30 = float(estimated_p32 / e_pi_r2)

        results[int(set_id)] = {
            "set_id": int(set_id),
            "observed_P21": observed_p21,
            "corrected_P21": corrected_p21,
            "orientation_corrected_P21": orientation_corrected_p21,
            "estimated_P32": estimated_p32,
            "estimated_P30": estimated_p30,
            "method": method,
            "warnings": warnings,
        }
    return results
