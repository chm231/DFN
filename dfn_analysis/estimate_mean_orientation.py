from typing import Optional, Tuple
import numpy as np


def normalize(vector_xyz: np.ndarray, eps: float = 1e-12) -> Optional[np.ndarray]:
    vector_xyz = np.asarray(vector_xyz, dtype=np.float64)
    norm = float(np.linalg.norm(vector_xyz))
    if norm < eps:
        return None
    return vector_xyz / norm


def estimate_mean_normal_axial(normals_xyz: np.ndarray, eps: float = 1e-12) -> Optional[np.ndarray]:
    """
    Estimates the mean normal vector from a set of axial vectors (normals)
    by aligning them to the first vector to ensure they lie in the same hemisphere.
    """
    normals_xyz = np.asarray(normals_xyz, dtype=np.float64)
    if normals_xyz.ndim != 2 or normals_xyz.shape[0] == 0 or normals_xyz.shape[1] != 3:
        return None

    aligned = normals_xyz.copy()
    ref_xyz = normalize(aligned[0], eps=eps)
    if ref_xyz is None:
        return None

    for idx in range(len(aligned)):
        if float(np.dot(aligned[idx], ref_xyz)) < 0.0:
            aligned[idx] *= -1.0

    resultant_xyz = np.sum(aligned, axis=0)
    return normalize(resultant_xyz, eps=eps)


def normal_to_trend_plunge_ned(normal_xyz: np.ndarray, eps: float = 1e-12) -> Tuple[Optional[float], Optional[float]]:
    """
    Converts a 3D normal vector in model coordinates [x, y, z] to
    geological Trend and Plunge under the NED (North, East, Down) coordinate system:
      North = -y
      East = -x
      Down = z

    Returns lower-hemisphere pole trend/plunge.
    """
    n = np.asarray(normal_xyz, dtype=np.float64)
    norm = float(np.linalg.norm(n))
    if norm < eps:
        return None, None

    x_model, y_model, z_model = n / norm

    # Apply coordinate transformation
    x_ned = -y_model
    y_ned = -x_model
    z_ned = z_model

    # Lower hemisphere convention: Down component must be positive
    if z_ned < 0.0:
        x_ned, y_ned, z_ned = -x_ned, -y_ned, -z_ned

    trend = float(np.degrees(np.arctan2(y_ned, x_ned)) % 360.0)
    plunge = float(np.degrees(np.arctan2(z_ned, np.sqrt(x_ned * x_ned + y_ned * y_ned))))

    return trend, plunge
