"""Trace-level QC metrics for measured tunnel-face traces."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from trace_analysis.load_measured_traces import TraceMeta
    from trace_analysis.trace_reconstruction_unified import FaceTrace
except ImportError:
    from load_measured_traces import TraceMeta
    from trace_reconstruction_unified import FaceTrace


def build_trace_qc_dataframe(
    traces: List[FaceTrace],
    trace_meta: Dict[int, TraceMeta],
    original_3d_points: Dict[int, Tuple[np.ndarray, np.ndarray]],
    min_trace_length: float,
    x_tolerance: float = 0.0,
) -> pd.DataFrame:
    """Build a QC dataframe using Y-Z statistics while preserving 3D roughness metadata."""
    records = []
    for trace in traces:
        meta = trace_meta[trace.trace_id]
        p0_xyz, p1_xyz = original_3d_points[trace.trace_id]
        dy = float(p1_xyz[1] - p0_xyz[1])
        dz = float(p1_xyz[2] - p0_xyz[2])
        dx = float(p1_xyz[0] - p0_xyz[0])
        length_yz = float(np.sqrt(dy**2 + dz**2))
        length_3d = float(np.sqrt(dx**2 + dy**2 + dz**2))
        theta_yz_deg = float(np.degrees(np.arctan2(dz, dy)) % 180.0)
        length_ratio = float(length_3d / max(length_yz, 1e-12))
        valid_x_tolerance = True
        if meta.dx is not None and np.isfinite(meta.dx):
            bound = float(meta.dx) + float(x_tolerance)
            valid_x_tolerance = (
                abs(meta.rough_offset_p0) <= bound and abs(meta.rough_offset_p1) <= bound
            )

        record = {
            "trace_id": trace.trace_id,
            "face_id": trace.face_id,
            "set_id": trace.set_id,
            "x_face_nominal": meta.x_face_nominal,
            "x_mid_rough": meta.x_mid_rough,
            "x_span": meta.x_span,
            "rough_offset_p0": meta.rough_offset_p0,
            "rough_offset_p1": meta.rough_offset_p1,
            "rough_offset_mid": meta.rough_offset_mid,
            "p0_x": p0_xyz[0],
            "p0_y": p0_xyz[1],
            "p0_z": p0_xyz[2],
            "p1_x": p1_xyz[0],
            "p1_y": p1_xyz[1],
            "p1_z": p1_xyz[2],
            "length_yz": length_yz,
            "length_3d": length_3d,
            "length_ratio_3d_to_yz": length_ratio,
            "theta_yz_deg": theta_yz_deg,
            "valid_length": bool(length_yz >= min_trace_length),
            "valid_x_tolerance": bool(valid_x_tolerance),
            "confidence": meta.confidence,
            "parent_fracture_id": meta.parent_fracture_id,
        }
        if meta.dx is not None:
            record["dx"] = meta.dx
        for key, value in meta.extra.items():
            if key not in record:
                record[key] = value
        records.append(record)

    qc_df = pd.DataFrame.from_records(records)
    if not qc_df.empty:
        qc_df["rough_length_warning"] = qc_df["length_ratio_3d_to_yz"] > 1.05
    return qc_df
