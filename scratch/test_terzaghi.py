import numpy as np
import math

# Let's mock the estimate_fisher_orientation with 1.0 / sin_theta weights
def estimate_fisher_orientation_corrected(normals, faces, apply_bias_correction=True):
    n = len(normals)
    if n < 2:
        return None
    
    # Flip to same hemisphere
    ref = normals[0]
    for k in range(1, n):
        if np.dot(normals[k], ref) < 0:
            normals[k] = -normals[k]
            
    weights = np.ones(n, dtype=float)
    if apply_bias_correction and len(faces) > 0:
        m_face = np.array([1.0, 0.0, 0.0]) # face normal
        for k in range(n):
            cross_val = np.linalg.norm(np.cross(normals[k], m_face))
            # Inverse of sin_theta for Terzaghi correction
            weights[k] = 1.0 / max(cross_val, 0.05)
            
    w_sum = weights.sum()
    weights /= w_sum
    
    mean_direction = (normals * weights[:, None]).sum(axis=0)
    R = np.linalg.norm(mean_direction)
    mean_direction /= R
    
    # trend/plunge
    n_vec = mean_direction
    if n_vec[2] < 0:
        n_vec = -n_vec
    horiz = math.sqrt(n_vec[0]**2 + n_vec[1]**2)
    plunge = math.degrees(math.atan2(n_vec[2], horiz))
    trend = math.degrees(math.atan2(n_vec[0], n_vec[1])) % 360.0
    return trend, plunge, R

# Let's test on the discs from S2
# We can load the normals of S2 from storage/output/ground_truth_traces_with_normals.csv
import pandas as pd
df = pd.read_csv("storage/output/ground_truth_traces_with_normals.csv")
df_s2 = df[df["set_id"] == 2]
normals_s2 = df_s2[["normal_x", "normal_y", "normal_z"]].values

# Reconstruct all of them (axial flip)
trend, plunge, R = estimate_fisher_orientation_corrected(normals_s2, [1])
print(f"Set 2 Corrected (Inverse Weighting): trend={trend:.2f}, plunge={plunge:.2f}, R={R:.4f}")

# What if we use direct weighting (the current bug)?
def estimate_fisher_orientation_bug(normals, faces):
    n = len(normals)
    ref = normals[0]
    for k in range(1, n):
        if np.dot(normals[k], ref) < 0:
            normals[k] = -normals[k]
    weights = np.ones(n, dtype=float)
    m_face = np.array([1.0, 0.0, 0.0])
    for k in range(n):
        cross_val = np.linalg.norm(np.cross(normals[k], m_face))
        weights[k] = max(cross_val, 0.01)
    weights /= weights.sum()
    mean_direction = (normals * weights[:, None]).sum(axis=0)
    R = np.linalg.norm(mean_direction)
    mean_direction /= R
    n_vec = mean_direction
    if n_vec[2] < 0:
        n_vec = -n_vec
    horiz = math.sqrt(n_vec[0]**2 + n_vec[1]**2)
    plunge = math.degrees(math.atan2(n_vec[2], horiz))
    trend = math.degrees(math.atan2(n_vec[0], n_vec[1])) % 360.0
    return trend, plunge, R

trend_b, plunge_b, R_b = estimate_fisher_orientation_bug(normals_s2, [1])
print(f"Set 2 Buggy (Direct Weighting): trend={trend_b:.2f}, plunge={plunge_b:.2f}, R={R_b:.4f}")
