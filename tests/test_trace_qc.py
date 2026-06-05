import numpy as np
import pytest

from trace_analysis.load_measured_traces import TraceMeta
from trace_analysis.trace_qc import build_trace_qc_dataframe
from trace_analysis.trace_reconstruction_unified import FaceTrace


def test_trace_qc_computes_lengths_and_theta():
    trace = FaceTrace(
        face_id=1,
        trace_id=1,
        x_face=0.0,
        p0_y=1.0,
        p0_z=1.0,
        p1_y=2.0,
        p1_z=5.0,
    )
    trace.set_id = 2
    meta = {
        1: TraceMeta(
            trace_id=1,
            face_id=1,
            set_id=2,
            x_face_nominal=0.0,
            x_mid_rough=-0.045,
            x_span=0.11,
            rough_offset_p0=0.01,
            rough_offset_p1=-0.10,
            rough_offset_mid=-0.045,
        )
    }
    original = {
        1: (
            np.array([0.01, 1.0, 1.0]),
            np.array([-0.10, 2.0, 5.0]),
        )
    }
    qc = build_trace_qc_dataframe([trace], meta, original, min_trace_length=0.1)
    row = qc.iloc[0]
    assert row["length_yz"] == pytest.approx(np.sqrt(17.0))
    assert row["length_3d"] == pytest.approx(np.sqrt(17.0 + 0.11**2))
    assert row["theta_yz_deg"] == pytest.approx(np.degrees(np.arctan2(4.0, 1.0)) % 180.0)
