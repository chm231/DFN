"""Forward-validation helpers for fixed-bound TPL trace estimation results."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def summarize_trace_validation(
    observed_lengths_m: np.ndarray,
    simulated_lengths_m: np.ndarray,
    observed_censoring: np.ndarray | None = None,
    simulated_censoring: np.ndarray | None = None,
) -> Dict[str, float]:
    """Return lightweight forward-validation metrics for observed vs simulated trace samples."""
    observed = np.asarray(observed_lengths_m, dtype=float)
    simulated = np.asarray(simulated_lengths_m, dtype=float)
    summary = {
        "observed_trace_count": float(len(observed)),
        "simulated_trace_count": float(len(simulated)),
        "observed_l50_m": float(np.quantile(observed, 0.50)) if len(observed) else np.nan,
        "simulated_l50_m": float(np.quantile(simulated, 0.50)) if len(simulated) else np.nan,
        "observed_l90_m": float(np.quantile(observed, 0.90)) if len(observed) else np.nan,
        "simulated_l90_m": float(np.quantile(simulated, 0.90)) if len(simulated) else np.nan,
        "observed_l95_m": float(np.quantile(observed, 0.95)) if len(observed) else np.nan,
        "simulated_l95_m": float(np.quantile(simulated, 0.95)) if len(simulated) else np.nan,
        "observed_P20": float(len(observed)),
        "simulated_P20": float(len(simulated)),
        "observed_P21_proxy": float(np.sum(observed)),
        "simulated_P21_proxy": float(np.sum(simulated)),
    }
    if observed_censoring is not None and simulated_censoring is not None:
        obs = pd.Series(observed_censoring).value_counts(normalize=True)
        sim = pd.Series(simulated_censoring).value_counts(normalize=True)
        for label in sorted(set(obs.index).union(sim.index)):
            summary[f"observed_{label}_ratio"] = float(obs.get(label, 0.0))
            summary[f"simulated_{label}_ratio"] = float(sim.get(label, 0.0))
    return summary
