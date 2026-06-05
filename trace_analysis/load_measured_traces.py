"""Load measured tunnel-face traces and preserve original 3D endpoint metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from trace_analysis.trace_reconstruction_unified import FaceTrace
except ImportError:
    from trace_reconstruction_unified import FaceTrace


REQUIRED_TRACE_COLUMNS = [
    "trace_id",
    "face_id",
    "p0_x",
    "p0_y",
    "p0_z",
    "p1_x",
    "p1_y",
    "p1_z",
    "set_id",
]

OPTIONAL_TRACE_COLUMNS = [
    "x_face",
    "x_face_nominal",
    "dx",
    "parent_fracture_id",
    "direction_type",
    "trace_angle_deg",
    "dip_deg",
    "dip_direction_deg",
    "normal_x",
    "normal_y",
    "normal_z",
    "confidence",
]


@dataclass
class TraceMeta:
    """Stores original 3D endpoint metadata and nominal-face interpretation."""

    trace_id: int
    face_id: int
    set_id: int
    x_face_nominal: float
    x_mid_rough: float
    x_span: float
    rough_offset_p0: float
    rough_offset_p1: float
    rough_offset_mid: float
    dx: Optional[float] = None
    parent_fracture_id: Optional[int] = None
    confidence: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_TRACE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Measured trace CSV is missing required columns: "
            + ", ".join(missing)
        )


def _validate_numeric_columns(df: pd.DataFrame, columns: List[str]) -> None:
    for column in columns:
        if not np.isfinite(pd.to_numeric(df[column], errors="coerce")).all():
            raise ValueError(
                f"Column '{column}' contains non-numeric or non-finite values."
            )


def _load_face_stations(face_stations_csv: Optional[str]) -> pd.DataFrame:
    if not face_stations_csv:
        return pd.DataFrame(columns=["face_id", "x_face_nominal", "dx"])
    stations = pd.read_csv(face_stations_csv)
    required = ["face_id", "x_face_nominal"]
    missing = [col for col in required if col not in stations.columns]
    if missing:
        raise ValueError(
            "Face stations CSV is missing required columns: "
            + ", ".join(missing)
        )
    if "dx" not in stations.columns:
        stations["dx"] = np.nan
    _validate_numeric_columns(stations, ["face_id", "x_face_nominal", "dx"])
    if stations["face_id"].duplicated().any():
        raise ValueError("Face stations CSV contains duplicate face_id values.")
    stations["face_id"] = stations["face_id"].astype(int)
    return stations


def _resolve_nominal_face_positions(
    df: pd.DataFrame,
    x_face_mode: str,
    face_stations: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()
    df["x_mid_rough"] = 0.5 * (df["p0_x"] + df["p1_x"])

    if "x_face_nominal" in df.columns:
        df["x_face_nominal"] = pd.to_numeric(df["x_face_nominal"], errors="coerce")
    else:
        df["x_face_nominal"] = np.nan
    if "dx" not in df.columns:
        df["dx"] = np.nan

    if not face_stations.empty:
        station_map = face_stations.set_index("face_id")[["x_face_nominal", "dx"]]
        df = df.merge(
            station_map,
            left_on="face_id",
            right_index=True,
            how="left",
            suffixes=("", "_station"),
        )
        fill_mask = df["x_face_nominal"].isna()
        df.loc[fill_mask, "x_face_nominal"] = df.loc[fill_mask, "x_face_nominal_station"]
        fill_dx_mask = df["dx"].isna()
        df.loc[fill_dx_mask, "dx"] = df.loc[fill_dx_mask, "dx_station"]
        df = df.drop(columns=["x_face_nominal_station", "dx_station"])

    if "x_face" in df.columns:
        df["x_face"] = pd.to_numeric(df["x_face"], errors="coerce")

    if df["x_face_nominal"].isna().any():
        missing_nominal = df["x_face_nominal"].isna()
        if x_face_mode == "nominal":
            if "x_face" not in df.columns:
                raise ValueError(
                    "CSV lacks 'x_face_nominal' and 'x_face'. "
                    "Provide x_face_nominal or use --face-stations-csv."
                )
            df.loc[missing_nominal, "x_face_nominal"] = df.loc[missing_nominal, "x_face"]
        elif x_face_mode == "mid_rough":
            if "x_face" in df.columns:
                nominal_by_face = df.groupby("face_id")["x_face"].transform("median")
            else:
                nominal_by_face = df.groupby("face_id")["x_mid_rough"].transform("median")
            df.loc[missing_nominal, "x_face_nominal"] = nominal_by_face[missing_nominal]
        else:
            raise ValueError(f"Unsupported x_face_mode: {x_face_mode}")

    if df["x_face_nominal"].isna().any():
        missing_face_ids = sorted(df.loc[df["x_face_nominal"].isna(), "face_id"].unique())
        raise ValueError(
            "Failed to resolve x_face_nominal for face_id values: "
            + ", ".join(str(v) for v in missing_face_ids)
        )
    return df


def load_measured_traces(
    traces_csv: str,
    x_face_mode: str = "nominal",
    face_stations_csv: Optional[str] = None,
    min_trace_length: float = 1e-9,
) -> Tuple[List[FaceTrace], Dict[int, TraceMeta], Dict[int, Tuple[np.ndarray, np.ndarray]], List[str]]:
    """
    Load measured traces from CSV, preserving 3D endpoints while creating Y-Z traces.

    Returns:
        traces: list[FaceTrace]
        trace_meta: dict[trace_id, TraceMeta]
        original_3d_points: dict[trace_id, (p0_xyz, p1_xyz)]
        warnings: loader warnings
    """
    df = pd.read_csv(traces_csv)
    _validate_required_columns(df)
    _validate_numeric_columns(df, REQUIRED_TRACE_COLUMNS)

    face_stations = _load_face_stations(face_stations_csv)
    df = _resolve_nominal_face_positions(df, x_face_mode=x_face_mode, face_stations=face_stations)

    if df["trace_id"].duplicated().any():
        dupes = sorted(df.loc[df["trace_id"].duplicated(), "trace_id"].unique())
        raise ValueError(
            "Measured trace CSV contains duplicate trace_id values: "
            + ", ".join(str(v) for v in dupes[:10])
        )

    for key in ["face_id", "set_id", "trace_id"]:
        float_values = pd.to_numeric(df[key], errors="coerce")
        if not np.all(np.equal(float_values, np.floor(float_values))):
            raise ValueError(f"Column '{key}' must contain integer-like values.")
        df[key] = float_values.astype(int)

    warnings: List[str] = []
    traces: List[FaceTrace] = []
    trace_meta: Dict[int, TraceMeta] = {}
    original_3d_points: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for _, row in df.iterrows():
        p0_xyz = np.array([row["p0_x"], row["p0_y"], row["p0_z"]], dtype=float)
        p1_xyz = np.array([row["p1_x"], row["p1_y"], row["p1_z"]], dtype=float)
        length_yz = float(np.linalg.norm(p1_xyz[1:] - p0_xyz[1:]))
        trace_id = int(row["trace_id"])
        if length_yz <= min_trace_length:
            warnings.append(
                f"Trace {trace_id} was excluded because length_yz={length_yz:.6f} is too small."
            )
            continue

        confidence = float(row["confidence"]) if "confidence" in row and pd.notna(row["confidence"]) else 1.0
        parent_fracture_id = None
        if "parent_fracture_id" in row and pd.notna(row["parent_fracture_id"]):
            parent_fracture_id = int(row["parent_fracture_id"])

        x_face_nominal = float(row["x_face_nominal"])
        x_mid_rough = float(0.5 * (row["p0_x"] + row["p1_x"]))
        dx_val = float(row["dx"]) if "dx" in row and pd.notna(row["dx"]) else None

        face_trace = FaceTrace(
            face_id=int(row["face_id"]),
            trace_id=trace_id,
            x_face=x_face_nominal,
            p0_y=float(row["p0_y"]),
            p0_z=float(row["p0_z"]),
            p1_y=float(row["p1_y"]),
            p1_z=float(row["p1_z"]),
            confidence=confidence,
            parent_fracture_id=parent_fracture_id,
        )
        face_trace.set_id = int(row["set_id"])
        traces.append(face_trace)

        extra = {}
        for column in OPTIONAL_TRACE_COLUMNS:
            if column in row and pd.notna(row[column]):
                extra[column] = row[column]

        trace_meta[trace_id] = TraceMeta(
            trace_id=trace_id,
            face_id=int(row["face_id"]),
            set_id=int(row["set_id"]),
            x_face_nominal=x_face_nominal,
            x_mid_rough=x_mid_rough,
            x_span=float(abs(row["p1_x"] - row["p0_x"])),
            rough_offset_p0=float(row["p0_x"] - x_face_nominal),
            rough_offset_p1=float(row["p1_x"] - x_face_nominal),
            rough_offset_mid=float(x_mid_rough - x_face_nominal),
            dx=dx_val,
            parent_fracture_id=parent_fracture_id,
            confidence=confidence,
            extra=extra,
        )
        original_3d_points[trace_id] = (p0_xyz, p1_xyz)

    return traces, trace_meta, original_3d_points, warnings
