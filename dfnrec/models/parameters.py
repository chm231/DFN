"""Data contracts for DFN parameter sets and generated hidden fractures."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any
import numpy as np


class SizeModel(str, Enum):
    """Fracture size distribution model."""
    POWER_LAW = "POWER_LAW"
    """Truncated power-law: f(r) ∝ r^{-alpha} for r in [r_min, r_max]."""
    EXPONENTIAL = "EXPONENTIAL"
    """Exponential: f(r) ∝ exp(-lambda * r) for r >= r_min."""
    LOG_UNIFORM = "LOG_UNIFORM"
    """Log-uniform: f(r) ∝ 1/r for r in [r_min, r_max]."""


@dataclass
class FractureSetOrientation:
    """Fisher distribution parameters for a fracture set's orientation."""
    set_id: str
    mean_trend_deg: float
    """Mean pole trend [degrees, 0–360]."""
    mean_plunge_deg: float
    """Mean pole plunge [degrees, 0–90]."""
    kappa: float
    """Fisher concentration parameter (κ ≥ 0). κ=0 → isotropic."""
    n_discs_used: int = 0
    """Number of reconstructed discs used in MLE."""
    kappa_ci_95: Optional[List[float]] = None
    """95 % bootstrap CI for kappa [lo, hi]."""
    orientation_bias_corrected: bool = False
    """Whether orientation-bias (sampling area correction) was applied."""


@dataclass
class FractureSetSizeIntensity:
    """Size distribution and intensity parameters for a fracture set."""
    set_id: str

    # --- Size model ---
    size_model: SizeModel = SizeModel.POWER_LAW
    """Which size distribution model was fitted."""

    # Power-law parameters
    k_r: Optional[float] = None
    """CCDF exponent (k_r = alpha - 1 for PDF exponent alpha).
    For POWER_LAW: PDF ∝ r^{-(k_r+1)}."""
    r_min: Optional[float] = None
    """Minimum radius used in fitting [m]."""
    r_max: Optional[float] = None
    """Maximum radius used in fitting [m]. None = unbounded in data."""

    # Exponential parameter
    lambda_exp: Optional[float] = None
    """Exponential rate [1/m]. For EXPONENTIAL model only."""

    # --- Intensity ---
    P32_total: Optional[float] = None
    """Total fracture area per unit volume (all radii) [m²/m³]."""
    P32_eff: Optional[float] = None
    """Effective P32 for r >= r_min [m²/m³]."""
    P30: Optional[float] = None
    """Number of fracture centres per unit volume [1/m³]."""
    n0: Optional[float] = None
    """Fracture centre density (same as P30 in Poisson model) [1/m³]."""

    # --- Derived / diagnostic ---
    P21_observed: Optional[float] = None
    """Observed P21 (total trace length / observation area) [m/m²]."""
    P21_simulated: Optional[float] = None
    """Simulated P21 from fitted parameters [m/m²]."""
    P20_observed: Optional[float] = None
    """Observed P20 (trace count / observation area) [1/m²]."""
    N_traces_observed: Optional[int] = None
    """Number of observed traces used in fitting."""
    N_expected: Optional[float] = None
    """Expected number of traces from fitted parameters (validation)."""
    C_s: Optional[float] = None
    """Orientation correction factor (mean |n · m_face| over faces)."""

    n_discs_used: int = 0
    """Number of reconstructed discs used in size/intensity fitting."""

    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    def mean_area_m2(self) -> Optional[float]:
        """E[pi r^2] under the fitted size model [m²]."""
        if self.size_model == SizeModel.POWER_LAW:
            if self.k_r is None or self.r_min is None:
                return None
            alpha = self.k_r + 1.0  # PDF exponent
            r0 = self.r_min
            r1 = self.r_max or (r0 * 1e3)
            if alpha <= 3.0:
                # E[r^2] = int r^2 * C * r^{-alpha} dr = C * int r^{2-alpha} dr
                # Normalisation C and integral from r0 to r1
                if abs(3.0 - alpha) < 1e-9:
                    er2 = math.log(r1 / r0) / (1.0 / r0 - 1.0 / r1)
                else:
                    exp = 3.0 - alpha
                    er2 = (r1**exp - r0**exp) / (exp * (r0**(1-alpha) - r1**(1-alpha)) / (1-alpha))
            else:
                er2 = r0**2 * (alpha - 1) / (alpha - 3)
            return math.pi * er2
        return None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["size_model"] = self.size_model.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FractureSetSizeIntensity":
        d = dict(d)
        d["size_model"] = SizeModel(d.get("size_model", "POWER_LAW"))
        return cls(**d)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_json(cls, s: str) -> "FractureSetSizeIntensity":
        return cls.from_dict(json.loads(s))


@dataclass
class DFNParameterSet:
    """Complete DFN parameter set for all fracture sets."""
    orientation: Dict[str, FractureSetOrientation] = field(default_factory=dict)
    """Keyed by set_id."""
    size_intensity: Dict[str, FractureSetSizeIntensity] = field(default_factory=dict)
    """Keyed by set_id."""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def set_ids(self) -> List[str]:
        return sorted(set(list(self.orientation.keys()) + list(self.size_intensity.keys())))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "orientation": {k: v.__dict__ for k, v in self.orientation.items()},
            "size_intensity": {k: v.to_dict() for k, v in self.size_intensity.items()},
            "metadata": self.metadata,
        }

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)


@dataclass
class GeneratedHiddenFracture:
    """A stochastically generated fracture for the unobserved domain.

    source is always ``"conditional_stochastic"``.
    """

    # --- Identity ---
    disc_id: str
    set_id: str
    source: str = "conditional_stochastic"
    realization_id: int = 0
    """Which Monte Carlo realization produced this fracture."""

    # --- Geometry ---
    center_xyz: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    normal_xyz: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    radius_m: float = 1.0
    trend_deg: Optional[float] = None
    plunge_deg: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GeneratedHiddenFracture":
        return cls(**d)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_json(cls, s: str) -> "GeneratedHiddenFracture":
        return cls.from_dict(json.loads(s))

    def __post_init__(self) -> None:
        if self.source != "conditional_stochastic":
            raise ValueError(
                f"GeneratedHiddenFracture.source must be 'conditional_stochastic', got '{self.source}'"
            )
        if self.radius_m <= 0:
            raise ValueError(f"radius_m must be positive, got {self.radius_m}")
        n = np.asarray(self.normal_xyz, dtype=float)
        norm = np.linalg.norm(n)
        if norm < 1e-10:
            raise ValueError("normal_xyz is a zero vector")
        self.normal_xyz = (n / norm).tolist()
