import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_set_parameter_estimation_runner_creates_outputs(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    traces_csv = tmp_path / "sample_traces.csv"
    polygon_csv = tmp_path / "sample_tunnel_polygon.csv"
    stations_csv = tmp_path / "sample_face_stations.csv"
    output_dir = tmp_path / "outputs"

    pd.DataFrame(
        [
            {
                "trace_id": 1,
                "face_id": 1,
                "p0_x": 0.01,
                "p0_y": 0.0,
                "p0_z": 0.0,
                "p1_x": -0.10,
                "p1_y": 2.0,
                "p1_z": 0.0,
                "set_id": 1,
                "parent_fracture_id": 10,
                "normal_x": 0.7,
                "normal_y": 0.7,
                "normal_z": 0.0,
            },
            {
                "trace_id": 2,
                "face_id": 1,
                "p0_x": 0.02,
                "p0_y": 0.0,
                "p0_z": 2.0,
                "p1_x": -0.08,
                "p1_y": 1.0,
                "p1_z": 2.0,
                "set_id": 1,
                "parent_fracture_id": 11,
                "normal_x": 0.7,
                "normal_y": 0.7,
                "normal_z": 0.0,
            },
            {
                "trace_id": 3,
                "face_id": 2,
                "p0_x": 1.02,
                "p0_y": 0.0,
                "p0_z": 0.0,
                "p1_x": 0.96,
                "p1_y": 3.0,
                "p1_z": 0.0,
                "set_id": 2,
                "parent_fracture_id": 20,
                "normal_x": 0.6,
                "normal_y": 0.0,
                "normal_z": 0.8,
            },
        ]
    ).to_csv(traces_csv, index=False)

    pd.DataFrame(
        [
            {"face_id": 1, "x_face_nominal": 0.0, "dx": 0.2},
            {"face_id": 2, "x_face_nominal": 1.0, "dx": 0.2},
        ]
    ).to_csv(stations_csv, index=False)

    pd.DataFrame(
        [
            {"y": 0.0, "z": 0.0},
            {"y": 3.0, "z": 0.0},
            {"y": 3.0, "z": 3.0},
            {"y": 0.0, "z": 3.0},
        ]
    ).to_csv(polygon_csv, index=False)

    command = [
        sys.executable,
        "trace_analysis/run_set_trace_parameter_estimation.py",
        "--traces-csv",
        str(traces_csv),
        "--face-stations-csv",
        str(stations_csv),
        "--tunnel-polygon",
        str(polygon_csv),
        "--output-dir",
        str(output_dir),
        "--x-face-mode",
        "nominal",
        "--min-trace-length",
        "0.1",
        "--boundary-tol",
        "0.05",
    ]
    subprocess.run(command, check=True, cwd=str(repo_root))
    assert (output_dir / "trace_qc.csv").exists()
    assert (output_dir / "set_observed_statistics.csv").exists()
    assert (output_dir / "set_corrected_trace_distribution.csv").exists()
    assert (output_dir / "set_radius_distribution.json").exists()
    assert (output_dir / "set_intensity_parameters.json").exists()
    assert (output_dir / "set_dfn_params.json").exists()
    assert (output_dir / "run_summary.md").exists()
    assert (output_dir / "validation_parent_fracture_summary.csv").exists()

    with open(output_dir / "set_dfn_params.json", "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert "metadata" in payload
    assert "sets" in payload
