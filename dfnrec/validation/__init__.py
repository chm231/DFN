"""dfnrec.validation — Synthetic data generation and metrics."""
from dfnrec.validation.generator import SyntheticDFNGenerator, GroundTruth
from dfnrec.validation.metrics import (
    association_precision_recall,
    plane_normal_angular_error,
    radius_map_relative_error,
    p32_error,
    non_observation_violation_count,
    compare_dfn_parameters,
)

__all__ = [
    "SyntheticDFNGenerator",
    "GroundTruth",
    "association_precision_recall",
    "plane_normal_angular_error",
    "radius_map_relative_error",
    "p32_error",
    "non_observation_violation_count",
    "compare_dfn_parameters",
]
