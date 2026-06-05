"""Estimate fracture-radius distributions from corrected trace-length distributions."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

try:
    from trace_analysis.trace_distribution_correction import select_best_distribution
except ImportError:
    from trace_distribution_correction import select_best_distribution


def estimate_radius_distributions(corrected_df: pd.DataFrame) -> Dict[int, Dict[str, object]]:
    """Estimate set-wise radius distributions using corrected trace lengths as baseline chords."""
    results: Dict[int, Dict[str, object]] = {}
    for set_id, group in corrected_df.groupby("set_id", sort=True):
        radius_samples = 0.5 * group["corrected_length_yz"].to_numpy(dtype=float)
        distribution = select_best_distribution(radius_samples, lower_bound=max(np.min(radius_samples), 1e-6))
        empirical = {
            "mean": float(np.mean(radius_samples)),
            "std": float(np.std(radius_samples, ddof=0)),
            "r_min": float(np.min(radius_samples)),
            "r_max": float(np.max(radius_samples)),
            "count": int(len(radius_samples)),
        }
        results[int(set_id)] = {
            "set_id": int(set_id),
            "radius_distribution": {
                "type": distribution["type"],
                "params": distribution["params"],
                "mean": empirical["mean"],
                "std": empirical["std"],
                "r_min": empirical["r_min"],
                "r_max": empirical["r_max"],
            },
            "diameter_distribution": {
                "mean": float(2.0 * empirical["mean"]),
                "std": float(2.0 * empirical["std"]),
            },
            "method": "simple_chord_approximation",
            "warnings": [
                "Radius estimates use corrected trace length / 2 as a conservative baseline."
            ],
        }
    return results
