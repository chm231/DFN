import os

file_path = r"dfnrec/size_intensity/p32_estimator.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target_str = """def estimate_size_model(
    traces: List[Trace],
    set_id: str,
    r_min: float = 0.5,
    r_max: float = 30.0,
    L_min: float = 0.1,
) -> SizeEstimateResult:
    \"\"\"Estimate best size model (POWER_LAW or EXPONENTIAL) and its parameters via joint MLE on chord lengths.

    Parameters
    ----------
    traces : list of Trace
    set_id : str
    r_min, r_max : float
    L_min : float

    Returns
    -------
    SizeEstimateResult (unpacks as (alpha_or_lambda, r_min_used))
    \"\"\"
    set_traces = [t for t in traces if t.set_id == set_id]
    if not set_traces:
        return SizeEstimateResult(3.5, r_min, "POWER_LAW")

    chord_lengths = np.array([t.observed_length for t in set_traces])
    is_contained = np.array([t.is_contained for t in set_traces])

    # 1. Fit POWER_LAW
    best_pl_ll = -1e10
    best_pl_alpha = 3.5
    best_pl_rmin = r_min

    # Search r_min in range [0.1, 1.5] and alpha in [1.5, 6.0]
    r_min_grid = np.linspace(0.1, 1.5, 29)
    alpha_grid = np.linspace(1.5, 6.0, 46)

    for r_cand in r_min_grid:
        for alpha_cand in alpha_grid:
            ll = censored_chord_log_likelihood(
                chord_lengths, is_contained, alpha_cand, r_cand, r_max, L_min, size_model="POWER_LAW"
            )
            if ll > best_pl_ll:
                best_pl_ll = ll
                best_pl_alpha = alpha_cand
                best_pl_rmin = r_cand

    # 2. Fit EXPONENTIAL
    best_exp_ll = -1e10
    best_exp_lambda = 0.25
    best_exp_rmin = r_min

    # Search lambda in range [0.05, 1.0]
    lambda_grid = np.linspace(0.05, 1.0, 39)

    for r_cand in r_min_grid:
        for lambda_cand in lambda_grid:
            ll = censored_chord_log_likelihood(
                chord_lengths, is_contained, lambda_cand, r_cand, r_max, L_min, size_model="EXPONENTIAL"
            )
            if ll > best_exp_ll:
                best_exp_ll = ll
                best_exp_lambda = lambda_cand
                best_exp_rmin = r_cand

    if best_pl_ll >= best_exp_ll:
        return SizeEstimateResult(best_pl_alpha, best_pl_rmin, "POWER_LAW")
    else:
        return SizeEstimateResult(best_exp_lambda, best_exp_rmin, "EXPONENTIAL")"""

replacement_str = """def estimate_size_model(
    traces: List[Trace],
    set_id: str,
    r_min: float = 0.5,
    r_max: float = 30.0,
    L_min: float = 0.1,
) -> SizeEstimateResult:
    \"\"\"Estimate best size model (POWER_LAW or EXPONENTIAL) and its parameters via joint MLE on chord lengths.

    Parameters
    ----------
    traces : list of Trace
    set_id : str
    r_min, r_max : float
    L_min : float

    Returns
    -------
    SizeEstimateResult (unpacks as (alpha_or_lambda, r_min_used))
    \"\"\"
    set_traces = [t for t in traces if t.set_id == set_id]
    if not set_traces:
        return SizeEstimateResult(3.5, r_min, "POWER_LAW")

    chord_lengths = np.array([t.observed_length for t in set_traces])
    is_contained = np.array([t.is_contained for t in set_traces])

    n_traces = len(set_traces)
    n_contained = int(np.sum(is_contained))
    n_clipped = n_traces - n_contained
    len_min = float(np.min(chord_lengths)) if n_traces > 0 else 0.0
    len_med = float(np.median(chord_lengths)) if n_traces > 0 else 0.0
    len_max = float(np.max(chord_lengths)) if n_traces > 0 else 0.0

    # 1. Fit POWER_LAW
    best_pl_ll = -1e10
    best_pl_alpha = 3.5
    best_pl_rmin = r_min

    # Search r_min in range [0.1, 1.5] and alpha in [1.5, 6.0]
    r_min_grid = np.linspace(0.1, 1.5, 29)
    alpha_grid = np.linspace(1.5, 6.0, 46)
    pl_curve = []

    for r_cand in r_min_grid:
        for alpha_cand in alpha_grid:
            ll = censored_chord_log_likelihood(
                chord_lengths, is_contained, alpha_cand, r_cand, r_max, L_min, size_model="POWER_LAW"
            )
            if ll > best_pl_ll:
                best_pl_ll = ll
                best_pl_alpha = alpha_cand
                best_pl_rmin = r_cand

    for alpha_cand in alpha_grid:
        ll = censored_chord_log_likelihood(
            chord_lengths, is_contained, alpha_cand, best_pl_rmin, r_max, L_min, size_model="POWER_LAW"
        )
        pl_curve.append((alpha_cand, ll))

    # 2. Fit EXPONENTIAL
    best_exp_ll = -1e10
    best_exp_lambda = 0.25
    best_exp_rmin = r_min

    # Search lambda in range [0.05, 1.0]
    lambda_grid = np.linspace(0.05, 1.0, 39)
    exp_curve = []

    for r_cand in r_min_grid:
        for lambda_cand in lambda_grid:
            ll = censored_chord_log_likelihood(
                chord_lengths, is_contained, lambda_cand, r_cand, r_max, L_min, size_model="EXPONENTIAL"
            )
            if ll > best_exp_ll:
                best_exp_ll = ll
                best_exp_lambda = lambda_cand
                best_exp_rmin = r_cand

    for lambda_cand in lambda_grid:
        ll = censored_chord_log_likelihood(
            chord_lengths, is_contained, lambda_cand, best_exp_rmin, r_max, L_min, size_model="EXPONENTIAL"
        )
        exp_curve.append((lambda_cand, ll))

    # Determine best model
    if best_pl_ll >= best_exp_ll:
        best_model = "POWER_LAW"
        best_val = best_pl_alpha
        best_r_min = best_pl_rmin
        best_ll = best_pl_ll
        curve = pl_curve
        boundary_hit = (abs(best_val - 1.5) < 1e-5 or abs(best_val - 6.0) < 1e-5)
    else:
        best_model = "EXPONENTIAL"
        best_val = best_exp_lambda
        best_r_min = best_exp_rmin
        best_ll = best_exp_ll
        curve = exp_curve
        boundary_hit = (abs(best_val - 0.05) < 1e-5 or abs(best_val - 1.0) < 1e-5)

    # Print comprehensive diagnostic output
    print(f"\\n[DIAGNOSTIC] Size model estimation for set: {set_id}")
    print(f"  - Number of traces: {n_traces}")
    print(f"  - Contained trace count: {n_contained}")
    print(f"  - Clipped trace count: {n_clipped}")
    print(f"  - Length min/median/max: {len_min:.4f} / {len_med:.4f} / {len_max:.4f}")
    if best_model == "POWER_LAW":
        print(f"  - Alpha grid range: [1.5, 6.0]")
    else:
        print(f"  - Lambda grid range: [0.05, 1.0]")
    print(f"  - Best size model: {best_model}")
    print(f"  - Best parameter (alpha or lambda): {best_val:.4f}")
    print(f"  - Best r_min: {best_r_min:.4f}")
    print(f"  - Best log likelihood: {best_ll:.4f}")
    print(f"  - Boundary hit flag: {boundary_hit}")
    # Print curve (only a few points for brevity)
    curve_str = ", ".join(f"({a:.2f}:{l:.1f})" for a, l in curve[::5])
    print(f"  - Log likelihood curve (selected points): {curve_str}")

    if best_model == "POWER_LAW":
        return SizeEstimateResult(best_pl_alpha, best_pl_rmin, "POWER_LAW")
    else:
        return SizeEstimateResult(best_exp_lambda, best_exp_rmin, "EXPONENTIAL")"""

# Normalise newline representation
content_norm = content.replace("\r\n", "\n")
target_norm = target_str.replace("\r\n", "\n")
replacement_norm = replacement_str.replace("\r\n", "\n")

if target_norm in content_norm:
    new_content = content_norm.replace(target_norm, replacement_norm)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("SUCCESS: estimate_size_model replaced successfully!")
else:
    print("ERROR: Target function not found exactly in file content!")
