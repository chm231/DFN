import math

import numpy as np

from dfn_analysis.diagnose_trace_length_km import (
    classify_event_indicators,
    compare_with_mc,
    empirical_survival,
    kaplan_meier_curve,
    percentile_from_survival,
    summarize_subset,
)


def test_km_equals_empirical_without_censoring() -> None:
    lengths = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    events = np.array([1, 1, 1, 1], dtype=np.int32)
    raw_grid, raw_surv = empirical_survival(lengths)
    km_grid, km_surv, _, _, _ = kaplan_meier_curve(lengths, events)
    assert np.allclose(raw_grid, km_grid)
    assert np.allclose(raw_surv, km_surv)


def test_km_tail_is_not_smaller_than_raw_with_right_censoring() -> None:
    lengths = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    events = np.array([1, 1, 0, 0], dtype=np.int32)
    raw_grid, raw_surv = empirical_survival(lengths)
    km_grid, km_surv, _, _, _ = kaplan_meier_curve(lengths, events)
    assert np.allclose(raw_grid, km_grid)
    assert np.all(km_surv >= raw_surv - 1e-12)
    assert percentile_from_survival(km_grid, km_surv, 0.90) >= np.percentile(lengths, 90)


def test_class_0_exact_and_class_1_2_censored() -> None:
    censoring = np.array([0, 1, 2, 0], dtype=np.int32)
    indicators = classify_event_indicators(censoring)
    assert indicators.tolist() == [1, 0, 0, 1]


def test_lmin_filter_reduces_trace_count() -> None:
    rows = [
        {"set_id": 1, "observed_length_m": 0.1, "censoring_class": 0},
        {"set_id": 1, "observed_length_m": 0.3, "censoring_class": 1},
        {"set_id": 1, "observed_length_m": 0.8, "censoring_class": 0},
    ]
    _, summary_low = summarize_subset(rows, "laxemar", 1, 0.1)
    _, summary_high = summarize_subset(rows, "laxemar", 1, 0.5)
    assert summary_low["n_traces"] == 3
    assert summary_high["n_traces"] == 1


def test_empty_subset_is_handled_gracefully() -> None:
    rows = [{"set_id": 2, "observed_length_m": 0.2, "censoring_class": 0}]
    curve_rows, summary = summarize_subset(rows, "forsmark", 1, 0.5)
    assert curve_rows == []
    assert summary["n_traces"] == 0
    assert summary["km_status"] == "no_traces"
    assert math.isnan(summary["km_p90"])


def test_compare_with_mc_filters_survival_mode() -> None:
    curve_rows = [
        {"site": "forsmark", "set_id": 5, "lmin_fit": 0.5, "length": 1.0, "km_survival": 0.8},
        {"site": "forsmark", "set_id": 5, "lmin_fit": 0.5, "length": 2.0, "km_survival": 0.1},
    ]
    mc_rows = [
        {
            "site": "forsmark",
            "set_id": 5,
            "lmin_fit": 0.5,
            "length": 1.0,
            "mc_survival": 0.8,
            "survival_mode": "mc_km_emulated_survival",
        },
        {
            "site": "forsmark",
            "set_id": 5,
            "lmin_fit": 0.5,
            "length": 2.0,
            "mc_survival": 0.1,
            "survival_mode": "mc_km_emulated_survival",
        },
        {
            "site": "forsmark",
            "set_id": 5,
            "lmin_fit": 0.5,
            "length": 1.0,
            "mc_survival": 0.3,
            "survival_mode": "mc_observed_visible_survival",
        },
        {
            "site": "forsmark",
            "set_id": 5,
            "lmin_fit": 0.5,
            "length": 2.0,
            "mc_survival": 0.05,
            "survival_mode": "mc_observed_visible_survival",
        },
    ]
    _, comparison_rows = compare_with_mc(curve_rows, mc_rows, "mc_km_emulated_survival")
    assert len(comparison_rows) == 1
    assert comparison_rows[0]["mc_survival_mode"] == "mc_km_emulated_survival"
    assert comparison_rows[0]["mc_km_consistency_status"] == "mc_consistent_with_km"
