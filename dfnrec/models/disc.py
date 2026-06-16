"""Data contract models for reconstructed fracture discs."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any
import numpy as np


class ReliabilityClass(str, Enum):
    """Quality classification for a reconstructed disc.

    Class A : n_faces ≥ 3 AND plane_fit_rms < 0.10 m AND censoring < 50 %
    Class B : n_faces ≥ 2 AND plane_fit_rms < 0.20 m                       (no A)
    Class C : n_faces = 1  (single-face disc; radius is poorly constrained)
    Class D : censoring_dominance ≥ 80 % (radius unconstrained — lower bound only)
    """
    A = "A"
    B = "B"
    C = "C"
    D = "D"


@dataclass
class ReconstructedDisc:
    """A fracture disc reconstructed from trace observations.

    Disc geometry is stored in global 3D coordinates (x=tunnel axis,
    y=lateral, z=vertical).

    source is always ``"observed_reconstructed"`` for this class.
    Use :class:`GeneratedHiddenFracture` for stochastic discs.
    """

    # --- Identity ---
    disc_id: str
    """Unique identifier, e.g. 'D_F001F002_001'."""
    set_id: Optional[str] = None
    """Fracture set label. None if uncategorised."""
    source: str = "observed_reconstructed"
    """Always 'observed_reconstructed'. Read-only — do not change."""

    # --- Which traces contributed ---
    contributing_trace_ids: List[str] = field(default_factory=list)
    """List of trace_ids used to fit this disc."""
    contributing_face_ids: List[str] = field(default_factory=list)
    """List of face_ids from which traces were taken."""

    # --- Plane geometry ---
    center_xyz: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """MAP-estimated disc centre [x, y, z] in metres."""
    normal_xyz: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    """Unit normal vector of the disc plane [nx, ny, nz]."""
    trend_deg: Optional[float] = None
    """Dip direction (trend) of the disc [degrees from North, 0–360]."""
    plunge_deg: Optional[float] = None
    """Dip angle (plunge) of the disc [degrees from horizontal, 0–90]."""

    # --- Size ---
    radius_m: float = 1.0
    """MAP-estimated disc radius [m]."""
    radius_std_m: Optional[float] = None
    """Laplace posterior std dev of radius [m]. None if not computed."""
    radius_lower_bound_m: Optional[float] = None
    """Minimum radius inferred from trace half-lengths (censored case)."""

    # --- Plane fit quality ---
    plane_fit_rms_m: Optional[float] = None
    """RMS of endpoint residuals to fitted plane [m]."""
    n_faces_observed: int = 1
    """Number of distinct faces contributing traces."""
    censoring_dominance: Optional[float] = None
    """Fraction of endpoints that are CLIPPED (0–1)."""
    censoring_dominance_flag: bool = False
    """True when censoring_dominance ≥ 0.80 (radius is a lower bound only)."""

    # --- Classification ---
    reliability_class: ReliabilityClass = ReliabilityClass.C
    """Overall quality class A/B/C/D (see ReliabilityClass docstring)."""

    # --- Posterior covariance (optional) ---
    center_covariance_3x3: Optional[List[List[float]]] = None
    """3×3 Laplace posterior covariance of centre coordinates [m²]."""

    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    def normal_np(self) -> np.ndarray:
        return np.asarray(self.normal_xyz, dtype=float)

    def center_np(self) -> np.ndarray:
        return np.asarray(self.center_xyz, dtype=float)

    def area_m2(self) -> float:
        """Disc area [m²]."""
        return math.pi * self.radius_m ** 2

    # ------------------------------------------------------------------
    # Reliability auto-classification
    # ------------------------------------------------------------------
    @classmethod
    def classify_reliability(
        cls,
        n_faces: int,
        plane_fit_rms: Optional[float],
        censoring_dominance: Optional[float],
    ) -> ReliabilityClass:
        """Determine reliability class from quality indicators.

        Parameters
        ----------
        n_faces : int
            Number of distinct faces contributing traces.
        plane_fit_rms : float or None
            RMS residual of the plane fit [m].
        censoring_dominance : float or None
            Fraction of clipped endpoints [0, 1].
        """
        cd = censoring_dominance if censoring_dominance is not None else 0.0
        rms = plane_fit_rms if plane_fit_rms is not None else float("inf")

        if cd >= 0.80:
            return ReliabilityClass.D
        if n_faces >= 3 and rms < 0.10 and cd < 0.50:
            return ReliabilityClass.A
        if n_faces >= 2 and rms < 0.20:
            return ReliabilityClass.B
        if n_faces == 1:
            return ReliabilityClass.C
        return ReliabilityClass.B  # n_faces>=2, rms might be poor

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["reliability_class"] = self.reliability_class.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReconstructedDisc":
        d = dict(d)
        d["reliability_class"] = ReliabilityClass(d.get("reliability_class", "C"))
        return cls(**d)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_json(cls, s: str) -> "ReconstructedDisc":
        return cls.from_dict(json.loads(s))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.source != "observed_reconstructed":
            raise ValueError(
                f"ReconstructedDisc.source must be 'observed_reconstructed', got '{self.source}'"
            )
        if len(self.center_xyz) != 3:
            raise ValueError("center_xyz must have 3 components")
        if len(self.normal_xyz) != 3:
            raise ValueError("normal_xyz must have 3 components")
        if self.radius_m <= 0:
            raise ValueError(f"radius_m must be positive, got {self.radius_m}")
        # Normalise normal
        n = np.asarray(self.normal_xyz, dtype=float)
        norm = np.linalg.norm(n)
        if norm < 1e-10:
            raise ValueError("normal_xyz is a zero vector")
        self.normal_xyz = (n / norm).tolist()
