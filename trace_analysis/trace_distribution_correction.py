"""Bias correction for observed trace-length distributions."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import expon, lognorm, pareto


def _safe_normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return None
    return vector / norm


def select_best_distribution(samples: np.ndarray, lower_bound: float) -> Dict[str, float | str | List[float]]:
    """Fit several candidate distributions and select the best by AIC."""
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return {
            "type": "empirical",
            "params": [],
            "aic": np.nan,
            "bic": np.nan,
        }
    if len(samples) < 3:
        return {
            "type": "empirical",
            "params": [],
            "aic": np.nan,
            "bic": np.nan,
        }

    fits: List[Dict[str, float | str | List[float]]] = []
    try:
        s, _, scale = lognorm.fit(samples, floc=0)
        log_lik = float(np.sum(lognorm.logpdf(samples, s=s, scale=scale)))
        fits.append(
            {
                "type": "lognormal",
                "params": [float(np.log(scale)), float(s)],
                "aic": float(2 * 2 - 2 * log_lik),
                "bic": float(np.log(len(samples)) * 2 - 2 * log_lik),
                "log_likelihood": log_lik,
            }
        )
    except Exception:
        pass

    try:
        _, scale = expon.fit(samples, floc=0)
        log_lik = float(np.sum(expon.logpdf(samples, scale=scale)))
        fits.append(
            {
                "type": "exponential",
                "params": [float(scale)],
                "aic": float(2 * 1 - 2 * log_lik),
                "bic": float(np.log(len(samples)) * 1 - 2 * log_lik),
                "log_likelihood": log_lik,
            }
        )
    except Exception:
        pass

    try:
        positive_lower = max(lower_bound, 1e-6)
        alpha, _, scale = pareto.fit(samples, floc=0, fscale=positive_lower)
        log_lik = float(np.sum(pareto.logpdf(samples, b=alpha, scale=scale)))
        fits.append(
            {
                "type": "pareto",
                "params": [float(alpha), float(scale)],
                "aic": float(2 * 2 - 2 * log_lik),
                "bic": float(np.log(len(samples)) * 2 - 2 * log_lik),
                "log_likelihood": log_lik,
            }
        )
    except Exception:
        pass

    if not fits:
        return {
            "type": "empirical",
            "params": [],
            "aic": np.nan,
            "bic": np.nan,
        }
    return min(fits, key=lambda item: float(item["aic"]))


def _estimate_censoring_offsets(lengths: np.ndarray, censoring: np.ndarray, min_trace_length: float) -> Tuple[float, float]:
    uncensored = lengths[censoring == 0]
    base = float(np.median(uncensored)) if len(uncensored) else float(np.median(lengths))
    if not np.isfinite(base) or base <= 0:
        base = max(float(min_trace_length), 1e-3)
    one_side = max(float(min_trace_length), 0.5 * base)
    two_side = max(float(min_trace_length), 1.0 * base)
    return one_side, two_side


def _mean_normal_for_set(group: pd.DataFrame) -> np.ndarray | None:
    normal_cols = ["normal_x", "normal_y", "normal_z"]
    if not set(normal_cols).issubset(group.columns):
        return None
    normals = group[normal_cols].dropna().to_numpy(dtype=float)
    if len(normals) == 0:
        return None
    unit_normals = []
    for normal in normals:
        unit = _safe_normalize(normal)
        if unit is not None:
            unit_normals.append(unit)
    if not unit_normals:
        return None
    stacked = np.vstack(unit_normals)
    mean_vector = np.mean(stacked, axis=0)
    return _safe_normalize(mean_vector)


def correct_trace_distributions(
    qc_df: pd.DataFrame,
    min_trace_length: float,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, object]]]:
    """
    Apply a conservative per-set trace correction using censoring classes and model selection.

    The corrected distribution is still trace-based and uses Y-Z projected trace lengths.
    """
    corrected_frames = []
    summary: Dict[int, Dict[str, object]] = {}
    face_normal = np.array([1.0, 0.0, 0.0], dtype=float)

    for set_id, group in qc_df.groupby("set_id", sort=True):
        work = group.copy()
        work = work[work["valid_length"]]
        if work.empty:
            summary[int(set_id)] = {
                "warnings": ["No valid traces were available for correction."],
                "distribution": {"type": "empirical", "params": [], "aic": np.nan, "bic": np.nan},
            }
            continue

        censoring = work["censoring_class"].to_numpy(dtype=int) if "censoring_class" in work else np.zeros(len(work), dtype=int)
        lengths = work["length_yz"].to_numpy(dtype=float)
        one_side_offset, two_side_offset = _estimate_censoring_offsets(lengths, censoring, min_trace_length)
        corrected = lengths.copy()
        corrected[censoring == 1] = corrected[censoring == 1] + one_side_offset
        corrected[censoring == 2] = corrected[censoring == 2] + two_side_offset
        if min_trace_length > 0:
            corrected = corrected[corrected >= min_trace_length]
            work = work.loc[work["length_yz"] >= min_trace_length].copy()

        if work.empty:
            continue

        work["corrected_length_yz"] = corrected
        work["correction_offset"] = corrected - work["length_yz"].to_numpy(dtype=float)
        work["truncation_threshold"] = float(min_trace_length)
        distribution = select_best_distribution(corrected, lower_bound=max(min_trace_length, 1e-6))
        work["corrected_distribution_type"] = distribution["type"]

        mean_normal = _mean_normal_for_set(work)
        q_set = np.nan
        warnings: List[str] = []
        if mean_normal is None:
            warnings.append("Orientation bias correction skipped because set normal data are unavailable.")
        else:
            q_set = float(np.linalg.norm(np.cross(face_normal, mean_normal)))
            if q_set < 0.2:
                warnings.append(
                    "Orientation bias factor q_set is very small; intensity estimates are likely unstable."
                )

        work["orientation_bias_factor"] = q_set
        corrected_frames.append(work)
        summary[int(set_id)] = {
            "distribution": distribution,
            "one_side_offset": float(one_side_offset),
            "two_side_offset": float(two_side_offset),
            "q_set": q_set,
            "mean_normal": mean_normal.tolist() if mean_normal is not None else None,
            "warnings": warnings,
        }

    corrected_df = pd.concat(corrected_frames, ignore_index=True) if corrected_frames else pd.DataFrame()
    return corrected_df, summary
