"""DFN inversion and reconstruction pipeline entry point."""
from __future__ import annotations

import logging
from typing import List, Optional

from dfnrec.models import (
    Face,
    Trace,
    DomainModel,
    DomainGeometry,
    DFNParameterSet,
)
from dfnrec.reconstruction import (
    build_candidate_graph,
    select_non_overlapping_tracks,
    estimate_disc_map,
)
from dfnrec.orientation import estimate_fisher_orientation
from dfnrec.size_intensity.p32_estimator import estimate_size_model, estimate_p32
from dfnrec.hidden_dfn import sample_hidden_dfn
from dfnrec.composer import compose_domain

logger = logging.getLogger(__name__)


def run_pipeline(
    faces: List[Face],
    traces: List[Trace],
    domain_geom: Optional[DomainGeometry] = None,
    seed: Optional[int] = None,
    log_bf_threshold: float = -20.0,
    min_faces: int = 1,
    r_min: float = 0.5,
    r_max: float = 30.0,
) -> DomainModel:
    """Run the complete 3D DFN reconstruction and parameter inversion pipeline.

    Steps
    -----
    1. Reconstruct observed discs from trace tracks across faces.
    2. Group reconstructed discs and traces by fracture set (set_id).
    3. Invert/estimate DFN parameters (orientation, size, intensity) for each set.
    4. Sample conditional hidden stochastic DFN in the domain volume.
    5. Compose and crop the final DFN domain model.
    """
    faces_dict = {f.face_id: f for f in faces}
    L_min = faces[0].L_min if faces else 0.1

    # Step 1: Reconstruct observed discs
    logger.info("Step 1: Reconstructing observed discs from traces...")
    edges = build_candidate_graph(traces, faces_dict, log_bf_threshold=log_bf_threshold)
    tracks = select_non_overlapping_tracks(traces, edges, min_faces=min_faces)

    observed_discs = []
    for i, track in enumerate(tracks):
        disc_prefix = f"D_{i:04d}"
        disc = estimate_disc_map(track, faces_dict, disc_id_prefix=disc_prefix)
        if disc is not None:
            observed_discs.append(disc)

    logger.info(f"Reconstructed {len(observed_discs)} discs from {len(traces)} traces.")

    # Step 2: DFN Parameter Inversion
    logger.info("Step 2: Performing DFN parameter inversion...")
    orientation_results = {}
    size_intensity_results = {}

    # Find unique set IDs
    set_ids = sorted(list(set(t.set_id for t in traces if t.set_id is not None)))
    if not set_ids:
        set_ids = ["S1"]
        for t in traces:
            t.set_id = "S1"
        for d in observed_discs:
            d.set_id = "S1"

    for set_id in set_ids:
        # Orientation Estimation
        logger.info(f"Estimating orientation for set {set_id}...")
        ori = estimate_fisher_orientation(observed_discs, set_id, faces=faces)
        if ori is None:
            logger.warning(f"Insufficient discs to estimate orientation for set {set_id}. Using default.")
            from dfnrec.models import FractureSetOrientation
            ori = FractureSetOrientation(
                set_id=set_id,
                mean_trend_deg=0.0,
                mean_plunge_deg=90.0,
                kappa=10.0,
                orientation_bias_corrected=False,
            )
        orientation_results[set_id] = ori

        # Size & Intensity Estimation
        logger.info(f"Estimating size and intensity for set {set_id}...")
        traces_of_set = [t for t in traces if t.set_id == set_id]
        discs_of_set = [d for d in observed_discs if d.set_id == set_id]

        alpha, r_min_used = estimate_size_model(traces_of_set, set_id, r_min=r_min, r_max=r_max, L_min=L_min)
        si = estimate_p32(
            traces=traces_of_set,
            faces=faces,
            orientation_result=ori,
            alpha=alpha,
            r_min=r_min_used,
            r_max=r_max,
            L_min=L_min,
            discs=discs_of_set,
        )
        size_intensity_results[set_id] = si

    dfn_params = DFNParameterSet(
        orientation=orientation_results,
        size_intensity=size_intensity_results,
    )

    # Step 3: Default Domain Geometry if not provided
    if domain_geom is None:
        x_coords = [f.origin_xyz[0] for f in faces]
        if not x_coords:
            x_coords = [0.0, 10.0]
        x_min = min(x_coords) - 2.0
        x_max = max(x_coords) + 6.0

        domain_geom = DomainGeometry(
            x_min=x_min,
            x_max=x_max,
            y_min=-5.0,
            y_max=5.0,
            z_min=-5.0,
            z_max=5.0,
        )

    # Step 4: Sample Conditional Hidden Stochastic DFN
    logger.info("Step 4: Sampling conditional hidden DFN...")
    temp_domain = DomainModel(
        domain_id="temp_domain",
        domain_geometry=domain_geom,
        observed_discs=observed_discs,
    )

    hidden_fractures = sample_hidden_dfn(
        domain=temp_domain,
        dfn_params=dfn_params,
        faces=faces,
        seed=seed,
        L_min=L_min,
    )

    # Step 5: Compose final DomainModel
    logger.info("Step 5: Composing final domain model...")
    final_domain = compose_domain(
        reconstructed=observed_discs,
        hidden=hidden_fractures,
        domain_geom=domain_geom,
        dfn_params=dfn_params,
        realization_id=0,
    )

    return final_domain
