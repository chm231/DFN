"""Conditional sampler for hidden stochastic DFN fractures.

This module generates stochastic fractures that are consistent with observed data,
specifically by filtering out generated candidates that would have produced a
visible trace on the excavation faces (non-observation constraint).
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from dfnrec.models import (
    Face,
    DomainModel,
    DFNParameterSet,
    GeneratedHiddenFracture,
    ReconstructedDisc,
    SizeModel,
)
from dfnrec.geometry.vector import normal_from_trend_plunge, trend_plunge_from_normal, normalize
from dfnrec.geometry.disc_trace import predicted_visible_trace


def sample_fisher_pole(mean_normal: np.ndarray, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """Sample a unit pole normal from Fisher distribution centered at mean_normal."""
    mu = normalize(np.asarray(mean_normal, dtype=float))

    # 1. Sample relative to Z axis
    u = rng.uniform(0.0, 1.0)
    if kappa < 1e-6:
        # Isotropic: uniform on sphere
        z = rng.uniform(-1.0, 1.0)
        phi = rng.uniform(0.0, 2.0 * math.pi)
        v = np.array([
            math.sqrt(max(1.0 - z**2, 0.0)) * math.cos(phi),
            math.sqrt(max(1.0 - z**2, 0.0)) * math.sin(phi),
            z
        ])
    else:
        # Fisher pole distribution (Langevin-von Mises-Fisher)
        # cos(theta) inverse CDF sampling
        exp_term = 0.0 if kappa > 25.0 else math.exp(-2.0 * kappa)
        cos_theta = 1.0 + (1.0 / kappa) * math.log(1.0 - u + u * exp_term)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        sin_theta = math.sqrt(max(1.0 - cos_theta**2, 0.0))
        phi = rng.uniform(0.0, 2.0 * math.pi)
        v = np.array([
            sin_theta * math.cos(phi),
            sin_theta * math.sin(phi),
            cos_theta
        ])

    # 2. Rotate Z axis to mu
    if abs(mu[2]) > 0.999:
        return v if mu[2] > 0 else -v
    else:
        u1 = normalize(np.array([-mu[1], mu[0], 0.0]))
        u2 = np.cross(mu, u1)
        u3 = mu
        return v[0] * u1 + v[1] * u2 + v[2] * u3


def sample_radius(
    size_model: SizeModel,
    k_r: float,
    r_min: float,
    r_max: float,
    lambda_exp: float,
    rng: np.random.Generator,
) -> float:
    """Sample radius from size distribution model."""
    u = rng.uniform(0.0, 1.0)
    if size_model == SizeModel.POWER_LAW:
        alpha = k_r + 1.0
        if abs(alpha - 1.0) < 1e-9:
            return r_min * ((r_max / r_min) ** u)
        else:
            term = r_min**(1.0 - alpha) + u * (r_max**(1.0 - alpha) - r_min**(1.0 - alpha))
            return term**(1.0 / (1.0 - alpha))
    elif size_model == SizeModel.EXPONENTIAL:
        return r_min - math.log(max(1.0 - u, 1e-15)) / lambda_exp
    elif size_model == SizeModel.LOG_UNIFORM:
        return r_min * ((r_max / r_min) ** u)
    else:
        return r_min


def sample_hidden_dfn(
    domain: DomainModel,
    dfn_params: DFNParameterSet,
    faces: List[Face],
    seed: Optional[int] = None,
    realization_id: int = 0,
    L_min: float = 0.1,
) -> List[GeneratedHiddenFracture]:
    """Sample hidden stochastic fractures in the domain volume.

    Generates fractures according to DFNParameterSet and filters out
    those that violate the non-observation condition on the given faces.
    """
    if domain.domain_geometry is None:
        raise ValueError("domain.domain_geometry must be specified for stochastic generation")

    geom = domain.domain_geometry
    V = geom.volume_m3()
    if V <= 0:
        raise ValueError(f"Domain volume must be positive, got {V}")

    rng = np.random.default_rng(seed)
    hidden_fractures: List[GeneratedHiddenFracture] = []
    observed_discs = domain.observed_discs

    for set_id in dfn_params.set_ids():
        ori = dfn_params.orientation.get(set_id)
        if ori is None:
            continue
        mean_normal = normal_from_trend_plunge(ori.mean_trend_deg, ori.mean_plunge_deg)
        kappa = ori.kappa

        si = dfn_params.size_intensity.get(set_id)
        if si is None:
            continue

        n0 = si.n0 or si.P30
        if n0 is None or n0 <= 0:
            continue

        n_candidates = rng.poisson(n0 * V)

        set_hidden_count = 0
        for _ in range(n_candidates):
            # 1. Sample Center
            cx = rng.uniform(geom.x_min, geom.x_max)
            cy = rng.uniform(geom.y_min, geom.y_max)
            cz = rng.uniform(geom.z_min, geom.z_max)
            center = np.array([cx, cy, cz])

            # 2. Sample Normal
            normal = sample_fisher_pole(mean_normal, kappa, rng)
            trend, plunge = trend_plunge_from_normal(normal)

            # 3. Sample Radius
            k_r = si.k_r if si.k_r is not None else 2.5
            r_min = si.r_min if si.r_min is not None else 0.5
            r_max = si.r_max if si.r_max is not None else 30.0
            lambda_exp = si.lambda_exp if si.lambda_exp is not None else 1.0
            radius = sample_radius(si.size_model, k_r, r_min, r_max, lambda_exp, rng)

            # 4. Duplicate Check with Observed Discs
            is_dup = False
            for d in observed_discs:
                if d.set_id != set_id:
                    continue
                dist_c = np.linalg.norm(center - d.center_np())
                if dist_c < 0.05 * d.radius_m:
                    ang = np.arccos(np.clip(abs(np.dot(normal, d.normal_np())), 0.0, 1.0))
                    if ang < 0.1:  # ~5.7 deg
                        if abs(radius - d.radius_m) / d.radius_m < 0.1:
                            is_dup = True
                            break
            if is_dup:
                continue

            # 5. Non-Observation Test (Conditional rejection)
            violation = False
            for face in faces:
                result = predicted_visible_trace(
                    center_xyz=center,
                    normal_xyz=normal,
                    radius_m=radius,
                    face_origin_xyz=np.asarray(face.origin_xyz),
                    face_normal_xyz=np.asarray(face.normal_xyz),
                    face_axis_u_xyz=np.asarray(face.axis_u_xyz),
                    face_axis_v_xyz=np.asarray(face.axis_v_xyz),
                    observation_window_uv=np.asarray(face.observation_window_polygon_uv),
                )
                if result.visible_length >= L_min:
                    violation = True
                    break
            if violation:
                continue

            # 6. Accept
            set_hidden_count += 1
            disc_id = f"H_{set_id}_{realization_id:03d}_{set_hidden_count:04d}"
            hidden_fractures.append(
                GeneratedHiddenFracture(
                    disc_id=disc_id,
                    set_id=set_id,
                    center_xyz=center.tolist(),
                    normal_xyz=normal.tolist(),
                    radius_m=radius,
                    trend_deg=trend,
                    plunge_deg=plunge,
                    realization_id=realization_id,
                )
            )

    return hidden_fractures
