import pandas as pd
import pytest

from trace_analysis.load_measured_traces import load_measured_traces


def test_loader_rejects_missing_required_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"trace_id": [1], "face_id": [1]}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        load_measured_traces(str(csv_path))


def test_loader_distinguishes_nominal_and_mid_rough(tmp_path):
    csv_path = tmp_path / "traces.csv"
    pd.DataFrame(
        [
            {
                "trace_id": 1,
                "face_id": 1,
                "p0_x": 0.0,
                "p0_y": 0.0,
                "p0_z": 0.0,
                "p1_x": 0.2,
                "p1_y": 1.0,
                "p1_z": 0.0,
                "set_id": 1,
                "x_face": 0.1,
            },
            {
                "trace_id": 2,
                "face_id": 1,
                "p0_x": 0.0,
                "p0_y": 0.0,
                "p0_z": 1.0,
                "p1_x": 0.2,
                "p1_y": 1.0,
                "p1_z": 1.0,
                "set_id": 1,
                "x_face": 0.1,
            },
        ]
    ).to_csv(csv_path, index=False)

    traces, meta, _, _ = load_measured_traces(str(csv_path), x_face_mode="mid_rough")
    assert len(traces) == 2
    assert meta[1].x_face_nominal == pytest.approx(0.1)
    assert meta[1].x_mid_rough == pytest.approx(0.1)
