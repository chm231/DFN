"""Synthetic DFN generator for pipeline validation.

Generates a ground-truth DFN (fracture discs with known parameters) and
the corresponding observed traces on 4 sequential tunnel faces.

Usage
-----
>>> gen = SyntheticDFNGenerator(seed=42)
>>> gt = gen.generate(n_sets=2, n_faces=4)
>>> gt.faces            # List[Face]
>>> gt.traces           # List[Trace]
>>> gt.true_discs       # List[ReconstructedDisc] — ground truth discs
>>> gt.dfn_params       # DFNParameterSet — true parameters
"""
from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np

from dfnrec.models import (
    Face,
    Trace,
    CensorType,
    ReconstructedDisc,
    ReliabilityClass,
    SizeModel,
    FractureSetOrientation,
    FractureSetSizeIntensity,
    DFNParameterSet,
)
from dfnrec.geometry.disc_trace import predicted_visible_trace
from dfnrec.geometry.vector import normalize, normal_from_trend_plunge, trend_plunge_from_normal


@dataclass
class GroundTruth:
    """Container for a synthetic DFN ground-truth dataset."""
    faces: List[Face]
    traces: List[Trace]
    true_discs: List[ReconstructedDisc]
    dfn_params: DFNParameterSet
    domain_bounds: Dict[str, float]  # x/y/z min/max
    seed: int


class SyntheticDFNGenerator:
    """Generate synthetic tunnel-face trace data from known DFN parameters.

    Fracture geometry model
    -----------------------
    * Orientation : Fisher distribution (mean pole + kappa).
    * Size        : Truncated power-law f(r) ∝ r^{-alpha} with r in [r_min, r_max].
    * Centres     : Homogeneous Poisson process (uniform) in domain box.
    * Intensity   : P32_total controls number of discs.

    Coordinate convention
    ---------------------
    x = tunnel advance, y = lateral, z = vertical.
    """

    # Default set configurations [trend_deg, plunge_deg, kappa, k_r, r_min, r_max, P32]
    DEFAULT_SETS = [
        dict(set_id="S1", trend=10,  plunge=80, kappa=20, k_r=2.5, r_min=0.5, r_max=15.0, P32=0.6),
        dict(set_id="S2", trend=120, plunge=20, kappa=15, k_r=3.0, r_min=0.5, r_max=15.0, P32=0.4),
    ]

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def generate(
        self,
        n_sets: int = 2,
        n_faces: int = 4,
        face_spacing: float = 2.0,
        face_half_size: float = 3.0,
        domain_ahead: float = 6.0,
        domain_half_y: float = 5.0,
        domain_half_z: float = 4.0,
        noise_sigma: float = 0.02,
        L_min: float = 0.1,
    ) -> GroundTruth:
        """Generate faces, discs, and traces.

        Parameters
        ----------
        n_sets : int
            Number of fracture sets (≤ len(DEFAULT_SETS)).
        n_faces : int
            Number of sequential tunnel faces.
        face_spacing : float
            x-distance between consecutive faces [m].
        face_half_size : float
            Half-width of square observation window [m].
        domain_ahead : float
            Domain extent in x beyond the last face [m].
        domain_half_y, domain_half_z : float
            Domain half-extents in y and z [m].
        noise_sigma : float
            Gaussian noise on trace endpoint positions [m].
        L_min : float
            Minimum visible trace length [m].
        """
        set_configs = self.DEFAULT_SETS[:n_sets]

        # --- Build faces ---
        faces = self._make_faces(n_faces, face_spacing, face_half_size, L_min)
        x_first = faces[0].face_x()
        x_last = faces[-1].face_x()

        # --- Domain bounds ---
        domain = dict(
            x_min=x_first - face_spacing,
            x_max=x_last + domain_ahead,
            y_min=-domain_half_y,
            y_max=domain_half_y,
            z_min=-domain_half_z,
            z_max=domain_half_z,
        )

        # --- Generate discs per set ---
        all_discs: List[ReconstructedDisc] = []
        orientation_params: Dict[str, FractureSetOrientation] = {}
        size_intensity_params: Dict[str, FractureSetSizeIntensity] = {}

        for cfg in set_configs:
            discs, ori_param, si_param = self._generate_set(cfg, domain)
            all_discs.extend(discs)
            orientation_params[cfg["set_id"]] = ori_param
            size_intensity_params[cfg["set_id"]] = si_param

        dfn_params = DFNParameterSet(
            orientation=orientation_params,
            size_intensity=size_intensity_params,
        )

        # --- Generate traces ---
        traces = self._generate_traces(all_discs, faces, noise_sigma, L_min)

        return GroundTruth(
            faces=faces,
            traces=traces,
            true_discs=all_discs,
            dfn_params=dfn_params,
            domain_bounds=domain,
            seed=self.seed,
        )

    # ------------------------------------------------------------------
    def _make_faces(
        self,
        n_faces: int,
        spacing: float,
        half_size: float,
        L_min: float,
    ) -> List[Face]:
        faces = []
        window = [
            [-half_size, -half_size],
            [half_size, -half_size],
            [half_size, half_size],
            [-half_size, half_size],
        ]
        for i in range(n_faces):
            x = float(i * spacing)
            faces.append(Face(
                face_id=f"F{i+1:03d}",
                order_index=i,
                origin_xyz=[x, 0.0, 0.0],
                normal_xyz=[1.0, 0.0, 0.0],
                axis_u_xyz=[0.0, 1.0, 0.0],
                axis_v_xyz=[0.0, 0.0, 1.0],
                observation_window_polygon_uv=window,
                L_min=L_min,
            ))
        return faces

    def _generate_set(
        self,
        cfg: dict,
        domain: dict,
    ) -> Tuple[List[ReconstructedDisc], FractureSetOrientation, FractureSetSizeIntensity]:
        """Generate fractures for one set via Poisson process."""
        sid = cfg["set_id"]
        trend, plunge, kappa = cfg["trend"], cfg["plunge"], cfg["kappa"]
        k_r, r_min, r_max, P32 = cfg["k_r"], cfg["r_min"], cfg["r_max"], cfg["P32"]

        # Domain volume
        Lx = domain["x_max"] - domain["x_min"]
        Ly = domain["y_max"] - domain["y_min"]
        Lz = domain["z_max"] - domain["z_min"]
        V = Lx * Ly * Lz

        # Mean radius under truncated power-law
        alpha = k_r + 1.0
        if alpha > 2.0:
            r0a, r1a = r_min ** (2 - alpha), r_max ** (2 - alpha)
            r0b, r1b = r_min ** (1 - alpha), r_max ** (1 - alpha)
            mean_r2 = (r0a - r1a) / (2 - alpha) / ((r0b - r1b) / (1 - alpha))
        else:
            mean_r2 = r_min * r_max  # fallback for alpha <= 2

        mean_area = math.pi * mean_r2
        # n0 = P32 / mean_area
        n0 = P32 / max(mean_area, 1e-9)
        N_expected = n0 * V
        N = int(self.rng.poisson(max(N_expected, 1)))

        discs = []
        for j in range(N):
            # Centre: uniform in domain
            cx = self.rng.uniform(domain["x_min"], domain["x_max"])
            cy = self.rng.uniform(domain["y_min"], domain["y_max"])
            cz = self.rng.uniform(domain["z_min"], domain["z_max"])

            # Orientation: Fisher distribution around mean pole
            normal = self._sample_fisher(trend, plunge, kappa)

            # Radius: truncated power-law via rejection sampling
            radius = self._sample_tpl(k_r, r_min, r_max)

            t, p = trend_plunge_from_normal(normal)
            disc = ReconstructedDisc(
                disc_id=f"GT_{sid}_{j:04d}",
                set_id=sid,
                source="observed_reconstructed",
                center_xyz=[cx, cy, cz],
                normal_xyz=normal.tolist(),
                radius_m=float(radius),
                trend_deg=t,
                plunge_deg=p,
                reliability_class=ReliabilityClass.A,
                n_faces_observed=0,  # will be updated when traced
                metadata={"ground_truth": True},
            )
            discs.append(disc)

        ori_param = FractureSetOrientation(
            set_id=sid,
            mean_trend_deg=float(trend),
            mean_plunge_deg=float(plunge),
            kappa=float(kappa),
            n_discs_used=N,
        )
        si_param = FractureSetSizeIntensity(
            set_id=sid,
            size_model=SizeModel.POWER_LAW,
            k_r=float(k_r),
            r_min=float(r_min),
            r_max=float(r_max),
            P32_total=float(P32),
            P32_eff=float(P32),
            P30=float(n0),
            n0=float(n0),
            N_expected=float(N_expected),
            n_discs_used=N,
        )
        return discs, ori_param, si_param

    def _generate_traces(
        self,
        discs: List[ReconstructedDisc],
        faces: List[Face],
        noise_sigma: float,
        L_min: float,
    ) -> List[Trace]:
        traces = []
        for face in faces:
            window = np.asarray(face.observation_window_polygon_uv, dtype=float)
            for disc in discs:
                result = predicted_visible_trace(
                    center_xyz=np.asarray(disc.center_xyz),
                    normal_xyz=np.asarray(disc.normal_xyz),
                    radius_m=disc.radius_m,
                    face_origin_xyz=np.asarray(face.origin_xyz),
                    face_normal_xyz=np.asarray(face.normal_xyz),
                    face_axis_u_xyz=np.asarray(face.axis_u_xyz),
                    face_axis_v_xyz=np.asarray(face.axis_v_xyz),
                    observation_window_uv=window,
                )

                if not result.chord_exists or result.visible_length < L_min:
                    continue

                # Add noise to endpoints
                p0 = np.asarray(result.visible_p0_xyz, dtype=float)
                p1 = np.asarray(result.visible_p1_xyz, dtype=float)
                if noise_sigma > 0:
                    p0 += self.rng.normal(0, noise_sigma, 3)
                    p1 += self.rng.normal(0, noise_sigma, 3)

                # Censoring: if clipped by window → CLIPPED, else NATURAL
                if result.clipped_by_window:
                    # Determine which endpoint is clipped
                    # (both clipped if chord much longer than window clip)
                    c0 = CensorType.CLIPPED if result.clipped_by_window else CensorType.NATURAL
                    c1 = CensorType.CLIPPED if result.clipped_by_window else CensorType.NATURAL
                    # More precise: check if endpoint moved
                    c0 = (CensorType.CLIPPED
                           if abs(np.linalg.norm(p0 - np.asarray(result.chord_p0_xyz))) > 0.01
                           else CensorType.NATURAL)
                    c1 = (CensorType.CLIPPED
                           if abs(np.linalg.norm(p1 - np.asarray(result.chord_p1_xyz))) > 0.01
                           else CensorType.NATURAL)
                else:
                    c0 = CensorType.NATURAL
                    c1 = CensorType.NATURAL

                t_id = f"{face.face_id}_{disc.disc_id}"
                trace = Trace(
                    trace_id=t_id,
                    face_id=face.face_id,
                    set_id=disc.set_id,
                    p0_xyz=p0.tolist(),
                    p1_xyz=p1.tolist(),
                    censor_p0=c0,
                    censor_p1=c1,
                    measurement_sigma=noise_sigma if noise_sigma > 0 else 0.001,
                    metadata={"source_disc_id": disc.disc_id, "ground_truth": True},
                )
                traces.append(trace)

        return traces

    # ------------------------------------------------------------------
    def _sample_fisher(self, trend_deg: float, plunge_deg: float, kappa: float) -> np.ndarray:
        """Sample a unit vector from Fisher distribution with given mean pole."""
        mean_pole = normal_from_trend_plunge(trend_deg, plunge_deg)
        # Sample via Wood (1994) algorithm
        xi = self.rng.uniform(0, 1)
        cos_theta = 1.0 + math.log(xi + (1 - xi) * math.exp(-2 * kappa)) / kappa if kappa > 1e-6 else 1.0 - 2 * xi
        sin_theta = math.sqrt(max(1 - cos_theta**2, 0.0))
        phi = self.rng.uniform(0, 2 * math.pi)
        # Local frame: z-axis = mean_pole
        z = mean_pole
        # Arbitrary perpendicular direction
        if abs(z[0]) < 0.9:
            tmp = np.array([1.0, 0.0, 0.0])
        else:
            tmp = np.array([0.0, 1.0, 0.0])
        x = normalize(np.cross(z, tmp))
        y = np.cross(z, x)
        v = sin_theta * (math.cos(phi) * x + math.sin(phi) * y) + cos_theta * z
        return normalize(v)

    def _sample_tpl(self, k_r: float, r_min: float, r_max: float) -> float:
        """Sample from truncated power-law F(r) = 1 - (r/r_min)^{-k_r} (CCDF).

        PDF: f(r) = k_r * r_min^{k_r} * r^{-(k_r+1)} for r in [r_min, r_max].
        Inverse CDF: r = r_min * (1 - u*(1-(r_min/r_max)^k_r))^{-1/k_r}.
        """
        u = self.rng.uniform(0, 1)
        lo = r_min ** k_r
        hi = r_max ** k_r
        # CDF: F(r) = (r^k_r - r_min^k_r) / (r_max^k_r - r_min^k_r)   [for POWER_LAW with PDF ∝ r^{k_r-1}?]
        # Use: r^k_r = lo + u*(hi - lo) → r = (lo + u*(hi-lo))^(1/k_r)
        # This is the standard truncated Pareto CCDF inverse
        return float((lo + u * (hi - lo)) ** (1.0 / k_r))
