"""Domain composer: combine reconstructed and stochastic fractures into a DomainModel."""
from __future__ import annotations

import math
from typing import List, Optional, Dict, Any
import numpy as np

from dfnrec.models import (
    DomainModel,
    DomainGeometry,
    ReconstructedDisc,
    GeneratedHiddenFracture,
    DFNParameterSet,
    Diagnostics,
)


def compose_domain(
    reconstructed: List[ReconstructedDisc],
    hidden: List[GeneratedHiddenFracture],
    domain_geom: DomainGeometry,
    dfn_params: Optional[DFNParameterSet] = None,
    realization_id: int = 0,
) -> DomainModel:
    """Compose the final DFN domain by combining observed and hidden fractures.

    Applies boundary clipping (center-based) and computes P32 diagnostics.
    """
    # 1. Boundary clipping: filter both sets to keep only fractures with centers inside domain
    def inside_domain(center: List[float]) -> bool:
        x, y, z = center
        return (
            domain_geom.x_min <= x <= domain_geom.x_max
            and domain_geom.y_min <= y <= domain_geom.y_max
            and domain_geom.z_min <= z <= domain_geom.z_max
        )

    clipped_reconstructed = [d for d in reconstructed if inside_domain(d.center_xyz)]
    clipped_hidden = [h for h in hidden if inside_domain(h.center_xyz)]

    # 2. Duplicate check between reconstructed and hidden
    final_hidden: List[GeneratedHiddenFracture] = []
    for h in clipped_hidden:
        is_dup = False
        h_center = np.asarray(h.center_xyz)
        h_normal = np.asarray(h.normal_xyz)
        for d in clipped_reconstructed:
            if d.set_id != h.set_id:
                continue
            dist_c = np.linalg.norm(h_center - d.center_np())
            if dist_c < 0.05 * d.radius_m:
                ang = np.arccos(np.clip(abs(np.dot(h_normal, d.normal_np())), 0.0, 1.0))
                if ang < 0.1:  # ~5.7 deg
                    if abs(h.radius_m - d.radius_m) / d.radius_m < 0.1:
                        is_dup = True
                        break
        if not is_dup:
            final_hidden.append(h)

    # 3. Compute diagnostics: apparent P32 vs target P32
    V = domain_geom.volume_m3()

    p32_target: Dict[str, float] = {}
    p32_apparent: Dict[str, float] = {}
    p32_rel_err: Dict[str, float] = {}
    warnings: List[str] = []

    # Calculate apparent P32 per set
    set_ids = set()
    for d in clipped_reconstructed:
        if d.set_id:
            set_ids.add(d.set_id)
    for h in final_hidden:
        if h.set_id:
            set_ids.add(h.set_id)

    if dfn_params:
        for sid in dfn_params.set_ids():
            set_ids.add(sid)

    for sid in sorted(set_ids):
        # target P32
        target = 0.0
        if dfn_params and sid in dfn_params.size_intensity:
            target = dfn_params.size_intensity[sid].P32_total or 0.0
        p32_target[sid] = target

        # apparent area inside domain
        area_rec = sum(d.area_m2() for d in clipped_reconstructed if d.set_id == sid)
        area_hid = sum(math.pi * h.radius_m**2 for h in final_hidden if h.set_id == sid)
        total_area = area_rec + area_hid
        apparent = total_area / max(V, 1e-9)
        p32_apparent[sid] = apparent

        if target > 0.0:
            rel_err = (apparent - target) / target
            p32_rel_err[sid] = rel_err
            if abs(rel_err) > 0.20:
                warnings.append(
                    f"Set {sid}: Apparent P32 ({apparent:.4f}) deviates from target ({target:.4f}) by {rel_err * 100:.1f}%"
                )
        else:
            p32_rel_err[sid] = 0.0

    diag = Diagnostics(
        n_observed_discs=len(clipped_reconstructed),
        n_hidden_fractures=len(final_hidden),
        realization_id=realization_id,
        p32_target=p32_target,
        p32_apparent=p32_apparent,
        p32_relative_error=p32_rel_err,
        warnings=warnings,
    )

    return DomainModel(
        domain_id=f"composed_{realization_id}",
        domain_geometry=domain_geom,
        observed_discs=clipped_reconstructed,
        hidden_fractures=final_hidden,
        dfn_params=dfn_params,
        diagnostics=diag,
    )
