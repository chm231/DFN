"""Data contract models for observed traces on tunnel faces."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any
import numpy as np


class CensorType(str, Enum):
    """How the trace endpoint is terminated.

    NATURAL  : endpoint is the actual fracture edge (disc rim)
    CLIPPED  : endpoint is cut by the observation window boundary
    UNKNOWN  : cannot determine from field observation
    """
    NATURAL = "NATURAL"
    CLIPPED = "CLIPPED"
    UNKNOWN = "UNKNOWN"


@dataclass
class Trace:
    """A single observed trace line segment on a tunnel face.

    All endpoint coordinates are in global 3D [x, y, z] (metres).
    face-local [u, v] projections are optional convenience fields.

    Coordinate convention
    ---------------------
      x = tunnel advance direction
      y = horizontal lateral
      z = vertical
    """

    # --- Identity ---
    trace_id: str
    """Unique identifier, e.g. 'F001_T003'."""
    face_id: str
    """Which face this trace was observed on."""
    set_id: Optional[str] = None
    """Fracture set label assigned during mapping (e.g. 'S1', 'S2'). None if unlabelled."""

    # --- Endpoints in 3D global coordinates ---
    p0_xyz: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """First endpoint [x, y, z] in metres."""
    p1_xyz: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Second endpoint [x, y, z] in metres."""

    # --- Optional polyline (for curved or segmented traces) ---
    polyline_xyz: Optional[List[List[float]]] = None
    """If the trace is not a straight segment, provide a polyline [N×3].
    When provided, p0_xyz and p1_xyz should equal the first and last points."""

    # --- Endpoint face-local UV (optional convenience) ---
    p0_uv: Optional[List[float]] = None
    """p0 in face-local UV [u, v] metres."""
    p1_uv: Optional[List[float]] = None
    """p1 in face-local UV [u, v] metres."""

    # --- Censoring ---
    censor_p0: CensorType = CensorType.UNKNOWN
    """Censoring type at p0."""
    censor_p1: CensorType = CensorType.UNKNOWN
    """Censoring type at p1."""

    # --- Quality ---
    measurement_sigma: float = 0.05
    """Positional measurement noise std dev for both endpoints [m]."""

    # --- Optional orientation from field measurement ---
    trend_deg: Optional[float] = None
    """Apparent dip direction (trend) of the fracture measured at this trace [deg]."""
    plunge_deg: Optional[float] = None
    """Apparent dip (plunge) of the fracture measured at this trace [deg]."""

    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    @property
    def observed_length(self) -> float:
        """Straight-line (or polyline arc) length [m]."""
        if self.polyline_xyz is not None and len(self.polyline_xyz) >= 2:
            pts = np.asarray(self.polyline_xyz, dtype=float)
            return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        p0 = np.asarray(self.p0_xyz, dtype=float)
        p1 = np.asarray(self.p1_xyz, dtype=float)
        return float(np.linalg.norm(p1 - p0))

    @property
    def midpoint_xyz(self) -> List[float]:
        p0 = np.asarray(self.p0_xyz, dtype=float)
        p1 = np.asarray(self.p1_xyz, dtype=float)
        return ((p0 + p1) / 2.0).tolist()

    @property
    def direction_xyz(self) -> List[float]:
        """Unit direction vector along the trace (p0 → p1)."""
        p0 = np.asarray(self.p0_xyz, dtype=float)
        p1 = np.asarray(self.p1_xyz, dtype=float)
        d = p1 - p0
        norm = np.linalg.norm(d)
        if norm < 1e-10:
            return [0.0, 0.0, 0.0]
        return (d / norm).tolist()

    @property
    def n_clipped_endpoints(self) -> int:
        return sum(1 for c in (self.censor_p0, self.censor_p1) if c == CensorType.CLIPPED)

    @property
    def is_contained(self) -> bool:
        """True when both endpoints are NATURAL (fully observed trace)."""
        return self.censor_p0 == CensorType.NATURAL and self.censor_p1 == CensorType.NATURAL

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["censor_p0"] = self.censor_p0.value
        d["censor_p1"] = self.censor_p1.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trace":
        d = dict(d)
        d["censor_p0"] = CensorType(d.get("censor_p0", "UNKNOWN"))
        d["censor_p1"] = CensorType(d.get("censor_p1", "UNKNOWN"))
        return cls(**d)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_json(cls, s: str) -> "Trace":
        return cls.from_dict(json.loads(s))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if len(self.p0_xyz) != 3:
            raise ValueError("p0_xyz must have 3 components")
        if len(self.p1_xyz) != 3:
            raise ValueError("p1_xyz must have 3 components")
        if self.polyline_xyz is not None:
            for pt in self.polyline_xyz:
                if len(pt) != 3:
                    raise ValueError("Each polyline point must have 3 components")
        if self.measurement_sigma <= 0:
            raise ValueError(f"measurement_sigma must be positive, got {self.measurement_sigma}")
        if self.observed_length < 1e-6:
            raise ValueError(f"Trace '{self.trace_id}' has zero length")
