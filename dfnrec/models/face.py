"""Data contract models for tunnel face geometry."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class Face:
    """A single excavation face (막장면).

    Coordinate convention
    ---------------------
    All XYZ coordinates are in the **global tunnel frame**:
      x = tunnel advance direction
      y = horizontal lateral
      z = vertical

    The observation window is defined as a closed polygon in face-local UV
    coordinates (u = axis_u direction projected on face plane, v = axis_v
    direction projected on face plane).
    """

    # --- Identity ---
    face_id: str
    """Unique identifier, e.g. 'F001'."""
    order_index: int
    """Sequential excavation order (0, 1, 2, 3, …). F0 < F1 < F2 < F3."""

    # --- Geometry ---
    origin_xyz: List[float]
    """A reference point on the face plane [x, y, z] in metres."""
    normal_xyz: List[float]
    """Outward unit normal vector of the face [nx, ny, nz].
    By convention this points in the +x (advance) direction."""
    axis_u_xyz: List[float]
    """Unit vector along the u-axis of the face-local coordinate system."""
    axis_v_xyz: List[float]
    """Unit vector along the v-axis of the face-local coordinate system."""

    # --- Observation window polygon in face-local UV (metres) ---
    observation_window_polygon_uv: List[List[float]]
    """Closed polygon vertices [[u0,v0], …] defining the observable region.
    Need not repeat first vertex."""

    # --- Optional masks ---
    visibility_polygon_uv: Optional[List[List[float]]] = None
    """Subset of observation window that was actually visible (clear of
    muck, water, etc.). None means full window is visible."""
    occlusion_polygon_uv: Optional[List[List[float]]] = None
    """Region explicitly blocked (muck pile, equipment). None = no occlusion."""

    # --- Detection parameters ---
    L_min: float = 0.1
    """Minimum observable trace length [m]. Default 0.1 m."""
    sigma_xyz: float = 0.05
    """Positional measurement noise std dev [m]. Default 0.05 m."""

    # --- Rough face support ---
    x_nominal: Optional[float] = None
    """Nominal x-coordinate of the face plane [m]. Used when the actual face
    is rough. If None, origin_xyz[0] is used."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Free-form metadata (date, excavation method, operator, etc.)."""

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    def normal_np(self) -> np.ndarray:
        return np.asarray(self.normal_xyz, dtype=float)

    def origin_np(self) -> np.ndarray:
        return np.asarray(self.origin_xyz, dtype=float)

    def axis_u_np(self) -> np.ndarray:
        return np.asarray(self.axis_u_xyz, dtype=float)

    def axis_v_np(self) -> np.ndarray:
        return np.asarray(self.axis_v_xyz, dtype=float)

    def face_x(self) -> float:
        """Return the x-coordinate of the face (nominal or origin)."""
        if self.x_nominal is not None:
            return self.x_nominal
        return float(self.origin_xyz[0])

    def window_area(self) -> float:
        """Area of the observation window polygon [m²] (shoelace formula)."""
        pts = np.asarray(self.observation_window_polygon_uv, dtype=float)
        n = len(pts)
        if n < 3:
            return 0.0
        x, y = pts[:, 0], pts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Face":
        return cls(**d)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_json(cls, s: str) -> "Face":
        return cls.from_dict(json.loads(s))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if len(self.origin_xyz) != 3:
            raise ValueError("origin_xyz must have 3 components")
        if len(self.normal_xyz) != 3:
            raise ValueError("normal_xyz must have 3 components")
        if len(self.axis_u_xyz) != 3:
            raise ValueError("axis_u_xyz must have 3 components")
        if len(self.axis_v_xyz) != 3:
            raise ValueError("axis_v_xyz must have 3 components")
        if len(self.observation_window_polygon_uv) < 3:
            raise ValueError("observation_window_polygon_uv must have ≥ 3 vertices")
        for v in self.observation_window_polygon_uv:
            if len(v) != 2:
                raise ValueError("Each vertex of observation_window_polygon_uv must be [u, v]")
        if self.L_min <= 0:
            raise ValueError(f"L_min must be positive, got {self.L_min}")
        if self.sigma_xyz <= 0:
            raise ValueError(f"sigma_xyz must be positive, got {self.sigma_xyz}")
        # Normalise vectors (in-place)
        for attr in ("normal_xyz", "axis_u_xyz", "axis_v_xyz"):
            v = np.asarray(getattr(self, attr), dtype=float)
            norm = np.linalg.norm(v)
            if norm < 1e-10:
                raise ValueError(f"{attr} is a zero vector")
            setattr(self, attr, (v / norm).tolist())
