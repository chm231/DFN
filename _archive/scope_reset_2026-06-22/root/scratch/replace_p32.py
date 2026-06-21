import os

file_path = r"dfnrec/size_intensity/p32_estimator.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target_p32 = """def estimate_p32(
    traces: List[Trace],
    faces: List[Face],
    orientation_result: Optional[FractureSetOrientation],
    alpha: float,
    r_min: float,
    r_max: float = 30.0,
    L_min: float = 0.1,
    discs: Optional[List[ReconstructedDisc]] = None,
    size_model: str = "POWER_LAW",
) -> FractureSetSizeIntensity:
    \"\"\"Estimate P32 and intensity parameters for a fracture set.

    Parameters
    ----------
    traces : list of Trace for this set
    faces : list of Face
    orientation_result : FractureSetOrientation or None
    alpha : float
        Power-law PDF exponent (k_r = alpha - 1) or Exponential rate (lambda).
    r_min : float
        Estimated r_min_mle.
    r_max : float
    L_min : float
    discs : list of ReconstructedDisc or None
    size_model : str
        POWER_LAW or EXPONENTIAL.
    \"\"\"
    sid = orientation_result.set_id if orientation_result else "unknown"

    # Observed P21 = total trace length / total face area
    total_trace_length = sum(t.observed_length for t in traces)
    total_face_area = sum(f.window_area() for f in faces)
    P21_obs = total_trace_length / max(total_face_area, 1e-9)
    P20_obs = len(traces) / max(total_face_area, 1e-9)

    # Orientation factor C_s = ||n x m_face|| (sin theta)
    if orientation_result is not None:
        from dfnrec.geometry.vector import normal_from_trend_plunge
        mean_normal = normal_from_trend_plunge(
            orientation_result.mean_trend_deg,
            orientation_result.mean_plunge_deg,
        )
        C_s = _orientation_factor(mean_normal, faces)
    else:
        C_s = 0.5  # isotropic default

    C_s_safe = max(C_s, 0.01)

    # Compute mean area E[pi r^2]
    if size_model == "POWER_LAW":
        mean_r2 = _mean_r2_power_law(alpha, r_min, r_max)
    else:
        mean_r2 = _mean_r2_exponential(alpha, r_min)
    mean_area = math.pi * mean_r2

    # P32_eff is estimated via exact stereology relation: P32 = P21 / Cs
    P32_eff = P21_obs / C_s_safe

    # Correct/Scale up P32_total using the area integral ratio from r_min_mle to the target r_min
    # Note: S4 exponential target range starts from 0.0 in the DFN generator config
    r_target_min = 0.0 if size_model == "EXPONENTIAL" else 0.5
    
    if size_model == "POWER_LAW":
        num_area = _area_integral_power_law(r_max, alpha) - _area_integral_power_law(r_min, alpha)
        den_area = _area_integral_power_law(r_max, alpha) - _area_integral_power_law(r_target_min, alpha)
    else:
        num_area = _area_integral_exponential(r_max, alpha) - _area_integral_exponential(r_min, alpha)
        den_area = _area_integral_exponential(r_max, alpha) - _area_integral_exponential(r_target_min, alpha)
        
    F_area = num_area / max(den_area, 1e-9)
    # P32_total is the scaled total area density in the target range
    P32_total = P32_eff / max(F_area, 0.01)

    # Number density n0 = P32_total / mean_area
    n0 = P32_total / max(mean_area, 1e-9)

    # Simulated trace length density P21_sim = P32_eff * Cs
    P21_sim = P32_eff * C_s_safe

    return FractureSetSizeIntensity(
        set_id=sid,
        size_model=SizeModel.POWER_LAW if size_model == "POWER_LAW" else SizeModel.EXPONENTIAL,
        k_r=alpha - 1.0 if size_model == "POWER_LAW" else alpha,
        r_min=r_min,
        r_max=r_max,
        lambda_exp=alpha if size_model == "EXPONENTIAL" else None,
        P32_total=P32_total,
        P32_eff=P32_eff,
        P30=n0,
        n0=n0,
        P21_observed=P21_obs,
        P21_simulated=P21_sim,
        P20_observed=P20_obs,
        N_traces_observed=len(traces),
        C_s=C_s,
        n_discs_used=len(discs) if discs else 0,
        metadata={
            "alpha": alpha,
            "mean_area_m2": mean_area,
            "F_area": F_area,
        },
    )"""

replacement_p32 = """def estimate_p32(
    traces: List[Trace],
    faces: List[Face],
    orientation_result: Optional[FractureSetOrientation],
    alpha: float,
    r_min: float,
    r_max: float = 30.0,
    L_min: float = 0.1,
    discs: Optional[List[ReconstructedDisc]] = None,
    size_model: str = "POWER_LAW",
) -> FractureSetSizeIntensity:
    \"\"\"Estimate P32 and intensity parameters for a fracture set.

    Parameters
    ----------
    traces : list of Trace for this set
    faces : list of Face
    orientation_result : FractureSetOrientation or None
    alpha : float
        Power-law PDF exponent (k_r = alpha - 1) or Exponential rate (lambda).
    r_min : float
        Estimated r_min_mle.
    r_max : float
    L_min : float
    discs : list of ReconstructedDisc or None
    size_model : str
        POWER_LAW or EXPONENTIAL.
    \"\"\"
    sid = orientation_result.set_id if orientation_result else "unknown"

    # Observed P21 = total trace length / total face area
    total_trace_length = sum(t.observed_length for t in traces)
    total_face_area = sum(f.window_area() for f in faces)
    P21_obs = total_trace_length / max(total_face_area, 1e-9)
    P20_obs = len(traces) / max(total_face_area, 1e-9)

    # Orientation factor C_s = ||n x m_face|| (sin theta)
    if orientation_result is not None:
        from dfnrec.geometry.vector import normal_from_trend_plunge
        mean_normal = normal_from_trend_plunge(
            orientation_result.mean_trend_deg,
            orientation_result.mean_plunge_deg,
        )
        C_s = _orientation_factor(mean_normal, faces)
    else:
        C_s = 0.5  # isotropic default

    C_s_safe = max(C_s, 0.01)

    # Compute mean area E[pi r^2] over the target full range
    r_target_min = 0.0 if size_model == "EXPONENTIAL" else 0.5
    if size_model == "POWER_LAW":
        mean_r2_full = _mean_r2_power_law(alpha, r_target_min, r_max)
        mean_r2_eff = _mean_r2_power_law(alpha, r_min, r_max)
    else:
        mean_r2_full = _mean_r2_exponential(alpha, r_target_min)
        mean_r2_eff = _mean_r2_exponential(alpha, r_min)
        
    mean_area_full = math.pi * mean_r2_full
    mean_area_eff = math.pi * mean_r2_eff

    # P32_eff is estimated via exact stereology relation: P32_eff = P21 / Cs
    P32_eff = P21_obs / C_s_safe

    # Correct/Scale up P32_total using the area integral ratio from r_min_mle to the target r_min
    if size_model == "POWER_LAW":
        num_area = _area_integral_power_law(r_max, alpha) - _area_integral_power_law(r_min, alpha)
        den_area = _area_integral_power_law(r_max, alpha) - _area_integral_power_law(r_target_min, alpha)
    else:
        num_area = _area_integral_exponential(r_max, alpha) - _area_integral_exponential(r_min, alpha)
        den_area = _area_integral_exponential(r_max, alpha) - _area_integral_exponential(r_target_min, alpha)
        
    F_area = num_area / max(den_area, 1e-9)
    # P32_total is the scaled total area density in the target range
    P32_total = P32_eff / max(F_area, 0.01)

    # Number density n0 (P30 total) = P32_total / mean_area_full
    n0 = P32_total / max(mean_area_full, 1e-9)

    # Effective number density P30_eff = P32_eff / mean_area_eff
    P30_eff = P32_eff / max(mean_area_eff, 1e-9)

    # Simulated trace length density P21_sim = P32_eff * Cs
    P21_sim = P32_eff * C_s_safe

    # Print comprehensive stereological decomposition debug output
    print(f"\\n[DIAGNOSTIC] P32 Stereological Decomposition for set: {sid}")
    print(f"  - size_model_used: {size_model}")
    print(f"  - k_r_used: {alpha - 1.0 if size_model == 'POWER_LAW' else alpha:.4f}")
    print(f"  - r0_or_rmin_used: {r_min:.4f}")
    print(f"  - radius_range_eff: [{r_min:.4f}, {r_max:.4f}]")
    print(f"  - P21_obs: {P21_obs:.6f} m/m2")
    print(f"  - orientation_factor_Cs: {C_s:.6f}")
    print(f"  - P32_eff: {P32_eff:.6f} m2/m3")
    print(f"  - area_fraction_F_A: {F_area:.6f}")
    print(f"  - P32_total: {P32_total:.6f} m2/m3")
    print(f"  - mean_area (full): {mean_area_full:.6f} m2")
    print(f"  - mean_area (eff): {mean_area_eff:.6f} m2")
    print(f"  - n0 (P30 total): {n0:.6f} m-3")
    print(f"  - P30_eff: {P30_eff:.6f} m-3")

    return FractureSetSizeIntensity(
        set_id=sid,
        size_model=SizeModel.POWER_LAW if size_model == "POWER_LAW" else SizeModel.EXPONENTIAL,
        k_r=alpha - 1.0 if size_model == "POWER_LAW" else alpha,
        r_min=r_min,
        r_max=r_max,
        lambda_exp=alpha if size_model == "EXPONENTIAL" else None,
        P32_total=P32_total,
        P32_eff=P32_eff,
        P30=n0,
        n0=n0,
        P21_observed=P21_obs,
        P21_simulated=P21_sim,
        P20_observed=P20_obs,
        N_traces_observed=len(traces),
        C_s=C_s,
        n_discs_used=len(discs) if discs else 0,
        metadata={
            "alpha": alpha,
            "mean_area_m2": mean_area_full,
            "mean_area_eff_m2": mean_area_eff,
            "F_area": F_area,
            "P30_eff": P30_eff,
            "P21_obs": P21_obs,
            "orientation_factor_Cs": C_s,
            "P32_eff": P32_eff,
            "radius_range_eff": [r_min, r_max],
            "size_model_used": size_model,
            "r0_or_rmin_used": r_min,
        },
    )"""

content_norm = content.replace("\r\n", "\n")
target_p32_norm = target_p32.replace("\r\n", "\n")
replacement_p32_norm = replacement_p32.replace("\r\n", "\n")

if target_p32_norm in content_norm:
    new_content = content_norm.replace(target_p32_norm, replacement_p32_norm)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("SUCCESS: estimate_p32 replaced successfully!")
else:
    print("ERROR: Target function not found exactly in file content!")
