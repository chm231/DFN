"""Estimate fixed-bound TPL fracture-radius distributions from trace observations.

This estimator assumes each joint set follows a fixed-bound truncated power-law radius model:

    R ~ TPL(alpha, 1 m, 250 m)

where ``alpha`` is the PDF exponent, not the CCDF exponent, and

    p_R(r) ∝ r^(-alpha)

Observed trace length is treated as chord length, not as a direct radius sample.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from trace_analysis.fixed_bound_tpl import DEFAULT_R_MAX_M, DEFAULT_R_MIN_M
from trace_analysis.trace_likelihood import fit_alpha_ideal


def estimate_radius_distributions(
    trace_df: pd.DataFrame,
    joint_set_col: str = "set_id",
    length_col: str = "length_yz",
    censor_col: str | None = "censor_label",
    detection_limit_m: float | None = None,
    r_min_m: float = DEFAULT_R_MIN_M,
    r_max_m: float = DEFAULT_R_MAX_M,
) -> Dict[int, Dict[str, object]]:
    """Estimate per-set fixed-bound TPL radius distributions from trace-length likelihood.

    The returned ``alpha_pdf_exponent`` refers to the PDF exponent of the radius model,
    not the CCDF exponent.
    """
    fit_results = fit_alpha_ideal(
        traces=trace_df,
        joint_set_col=joint_set_col,
        length_col=length_col,
        censor_col=censor_col,
        detection_limit=detection_limit_m,
        r_min=r_min_m,
        r_max=r_max_m,
    )
    results: Dict[int, Dict[str, object]] = {}
    for set_id, result in fit_results.items():
        results[int(set_id)] = {
            "set_id": int(set_id),
            "radius_distribution": {
                "type": "fixed_bound_truncated_power_law",
                "alpha_pdf_exponent": result.alpha_pdf_exponent,
                "r_min_m": result.r_min_m,
                "r_max_m": result.r_max_m,
                "mean": result.expected_R_m,
                "R50_m": result.R50_m,
                "R80_m": result.R80_m,
                "R90_m": result.R90_m,
                "R95_m": result.R95_m,
            },
            "diameter_distribution": {
                "type": "fixed_bound_truncated_power_law",
                "d_min_m": result.d_min_m,
                "d_max_m": result.d_max_m,
            },
            "alpha_pdf_exponent": result.alpha_pdf_exponent,
            "rho": result.rho,
            "method": "ideal_infinite_plane_trace_likelihood",
            "warnings": result.warnings,
            "diagnostics": result.diagnostics,
            "fit_result": result,
        }
    return results
