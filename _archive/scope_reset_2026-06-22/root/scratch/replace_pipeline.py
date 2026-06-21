import os

file_path = r"dfnrec/pipeline.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace import
target_import = """from dfnrec.models import (
    Face,
    Trace,
    DomainModel,
    DomainGeometry,
    DFNParameterSet,
)"""

replacement_import = """from dfnrec.models import (
    Face,
    Trace,
    DomainModel,
    DomainGeometry,
    DFNParameterSet,
    ReconstructedDisc,
)"""

# Replace signature
target_sig = """def run_pipeline(
    faces: List[Face],
    traces: List[Trace],
    domain_geom: Optional[DomainGeometry] = None,
    seed: Optional[int] = None,
    log_bf_threshold: float = -20.0,
    min_faces: int = 1,
    r_min: float = 0.5,
    r_max: float = 30.0,
) -> DomainModel:"""

replacement_sig = """def run_pipeline(
    faces: List[Face],
    traces: List[Trace],
    domain_geom: Optional[DomainGeometry] = None,
    seed: Optional[int] = None,
    log_bf_threshold: float = -20.0,
    min_faces: int = 1,
    r_min: float = 0.5,
    r_max: float = 30.0,
    orientation_source: str = "raw_normals",
) -> DomainModel:"""

# Replace orientation loop
target_loop = """    for set_id in set_ids:
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
        orientation_results[set_id] = ori"""

replacement_loop = """    for set_id in set_ids:
        # Orientation Estimation
        logger.info(f"Estimating orientation for set {set_id} with source={orientation_source}...")
        
        if orientation_source == "raw_normals":
            # Direct estimate from trace trend/plunge
            traces_of_set = [t for t in traces if t.set_id == set_id and t.trend_deg is not None and t.plunge_deg is not None]
            if len(traces_of_set) >= 2:
                from dfnrec.geometry.vector import normal_from_trend_plunge
                temp_discs = []
                for idx, t in enumerate(traces_of_set):
                    normal = normal_from_trend_plunge(t.trend_deg, t.plunge_deg)
                    temp_discs.append(ReconstructedDisc(
                        disc_id=f"temp_trace_{idx}",
                        set_id=set_id,
                        source="observed_reconstructed",
                        normal_xyz=normal.tolist(),
                        trend_deg=t.trend_deg,
                        plunge_deg=t.plunge_deg,
                        radius_m=1.0,
                        normal_source="raw_measured",
                        n_faces_observed=1
                    ))
                ori = estimate_fisher_orientation(
                    temp_discs,
                    set_id,
                    faces=faces,
                    exclude_single_face_fallback=False
                )
            else:
                ori = estimate_fisher_orientation(observed_discs, set_id, faces=faces, exclude_single_face_fallback=True)
        elif orientation_source == "reconstructed_multi_face":
            # Use only multi_face_svd
            multi_discs = [d for d in observed_discs if getattr(d, "normal_source", None) == "multi_face_svd"]
            ori = estimate_fisher_orientation(multi_discs, set_id, faces=faces, exclude_single_face_fallback=False)
        else:  # reconstructed_all
            # Use all discs (except single_face_fallback by default rule)
            ori = estimate_fisher_orientation(observed_discs, set_id, faces=faces, exclude_single_face_fallback=True)

        if ori is None:
            logger.warning(f"Insufficient data to estimate orientation for set {set_id}. Using default.")
            from dfnrec.models import FractureSetOrientation
            ori = FractureSetOrientation(
                set_id=set_id,
                mean_trend_deg=0.0,
                mean_plunge_deg=90.0,
                kappa=10.0,
                orientation_bias_corrected=False,
            )
        orientation_results[set_id] = ori"""

# Normalise newlines
content_norm = content.replace("\r\n", "\n")
target_import_norm = target_import.replace("\r\n", "\n")
replacement_import_norm = replacement_import.replace("\r\n", "\n")
target_sig_norm = target_sig.replace("\r\n", "\n")
replacement_sig_norm = replacement_sig.replace("\r\n", "\n")
target_loop_norm = target_loop.replace("\r\n", "\n")
replacement_loop_norm = replacement_loop.replace("\r\n", "\n")

if target_import_norm in content_norm and target_sig_norm in content_norm and target_loop_norm in content_norm:
    content_norm = content_norm.replace(target_import_norm, replacement_import_norm)
    content_norm = content_norm.replace(target_sig_norm, replacement_sig_norm)
    content_norm = content_norm.replace(target_loop_norm, replacement_loop_norm)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content_norm)
    print("SUCCESS: pipeline.py updated successfully!")
else:
    print("ERROR: One or more targets not found exactly in file content!")
