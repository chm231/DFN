import math

from dfn_analysis.estimate_p32_mc_calibrated import (
    CALIBRATION_FACTOR_MODE_PROXY,
    infer_calibration_factor_mode,
    mean_intersection_chord_length,
    radius_moments,
)


def test_current_calibration_mode_is_explicit_proxy() -> None:
    assert infer_calibration_factor_mode() == CALIBRATION_FACTOR_MODE_PROXY


def test_large_window_no_clipping_limit_matches_orientation_factor() -> None:
    site = "laxemar"
    set_id = 1
    kr = 2.85
    rmin = 0.5
    rmax = 250.0
    orientation_factor = 0.899496816267244

    mean_r, mean_r2 = radius_moments(site, set_id, kr, rmin, rmax)
    intersection_intensity_per_unit_p32 = (2.0 * orientation_factor * mean_r) / (math.pi * mean_r2)
    mean_chord = mean_intersection_chord_length(site, set_id, kr, rmin, rmax)
    calibration_limit = intersection_intensity_per_unit_p32 * mean_chord

    assert math.isclose(calibration_limit, orientation_factor, rel_tol=1e-10, abs_tol=1e-10)
