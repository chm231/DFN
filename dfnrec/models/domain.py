"""Data contract for the composed DFN domain model."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

from dfnrec.models.disc import ReconstructedDisc
from dfnrec.models.parameters import GeneratedHiddenFracture, DFNParameterSet


@dataclass
class Diagnostics:
    """Summary diagnostics for a composed domain."""
    n_observed_discs: int = 0
    """Number of observed-reconstructed discs."""
    n_hidden_fractures: int = 0
    """Number of conditional-stochastic fractures."""
    realization_id: int = 0
    """Which Monte Carlo realization this domain represents."""

    # Per-set intensity comparison
    p32_target: Dict[str, float] = field(default_factory=dict)
    """Target P32 from inversion, keyed by set_id [m²/m³]."""
    p32_apparent: Dict[str, float] = field(default_factory=dict)
    """Apparent P32 in the final domain, keyed by set_id [m²/m³]."""
    p32_relative_error: Dict[str, float] = field(default_factory=dict)
    """(apparent - target) / target, keyed by set_id."""

    warnings: List[str] = field(default_factory=list)
    """Any diagnostics warnings (e.g. P32 mismatch > 20 %)."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)


@dataclass
class DomainGeometry:
    """Bounding box of the model domain [m]."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def volume_m3(self) -> float:
        return (
            (self.x_max - self.x_min)
            * (self.y_max - self.y_min)
            * (self.z_max - self.z_min)
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DomainGeometry":
        return cls(**d)


@dataclass
class DomainModel:
    """Final composed DFN domain.

    Contains both observed (hard data) and hidden (stochastic) fractures,
    clearly tagged by their ``source`` attribute.
    """

    domain_id: str = "domain_001"
    domain_geometry: Optional[DomainGeometry] = None

    # --- Hard data (source = "observed_reconstructed") ---
    observed_discs: List[ReconstructedDisc] = field(default_factory=list)

    # --- Stochastic (source = "conditional_stochastic") ---
    hidden_fractures: List[GeneratedHiddenFracture] = field(default_factory=list)

    # --- Inversion parameters used ---
    dfn_params: Optional[DFNParameterSet] = None

    diagnostics: Optional[Diagnostics] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def all_fracture_count(self) -> int:
        return len(self.observed_discs) + len(self.hidden_fractures)

    def observed_count_by_set(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in self.observed_discs:
            key = d.set_id or "unknown"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def hidden_count_by_set(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for f in self.hidden_fractures:
            counts[f.set_id] = counts.get(f.set_id, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "domain_geometry": self.domain_geometry.to_dict() if self.domain_geometry else None,
            "observed_discs": [d.to_dict() for d in self.observed_discs],
            "hidden_fractures": [f.to_dict() for f in self.hidden_fractures],
            "dfn_params": self.dfn_params.to_dict() if self.dfn_params else None,
            "diagnostics": self.diagnostics.to_dict() if self.diagnostics else None,
            "metadata": self.metadata,
        }

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DomainModel":
        geom = DomainGeometry.from_dict(d["domain_geometry"]) if d.get("domain_geometry") else None
        obs = [ReconstructedDisc.from_dict(x) for x in d.get("observed_discs", [])]
        hid = [GeneratedHiddenFracture.from_dict(x) for x in d.get("hidden_fractures", [])]
        return cls(
            domain_id=d.get("domain_id", "domain_001"),
            domain_geometry=geom,
            observed_discs=obs,
            hidden_fractures=hid,
            dfn_params=None,  # full DFNParameterSet deserialisation omitted for brevity
            diagnostics=None,
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, s: str) -> "DomainModel":
        return cls.from_dict(json.loads(s))
