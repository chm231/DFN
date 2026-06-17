import numpy as np
import scipy.integrate as integrate
from scipy.special import beta, betainc

def chord_pdf_ideal_scalar(c: float, alpha: float, r_min: float, r_max: float, L_min: float, size_model: str) -> float:
    if c < L_min or c <= 0:
        return 0.0
    r_lo = max(r_min, c / 2.0 + 1e-9)
    if r_lo >= r_max:
        return 0.0

    if size_model == "POWER_LAW":
        u_lo = c / (2.0 * r_max)
        u_hi = min(c / (2.0 * r_min), 1.0 - 1e-9)
        if u_lo >= u_hi:
            return 0.0
        integral_val = 0.5 * beta(alpha / 2.0, 0.5) * (betainc(alpha / 2.0, 0.5, u_hi**2) - betainc(alpha / 2.0, 0.5, u_lo**2))
        const = (2.0 ** (alpha - 2.0)) * (c ** (1.0 - alpha))
        return max(const * integral_val, 0.0)
    return 0.0

# Vectorized version of chord_pdf_ideal
def chord_pdf_ideal_vec(c: np.ndarray, alpha: float, r_min: float, r_max: float, L_min: float = 0.0, size_model: str = "POWER_LAW") -> np.ndarray:
    c = np.atleast_1d(c)
    out = np.zeros_like(c, dtype=float)
    valid = (c >= L_min) & (c > 0)
    if not np.any(valid):
        return out
        
    c_val = c[valid]
    r_lo = np.maximum(r_min, c_val / 2.0 + 1e-9)
    in_range = r_lo < r_max
    if not np.any(in_range):
        return out
        
    c_active = c_val[in_range]
    
    if size_model == "POWER_LAW":
        u_lo = c_active / (2.0 * r_max)
        u_hi = np.minimum(c_active / (2.0 * r_min), 1.0 - 1e-9)
        
        # betainc is vectorized
        val_hi = betainc(alpha / 2.0, 0.5, u_hi**2)
        val_lo = betainc(alpha / 2.0, 0.5, u_lo**2)
        integral_val = 0.5 * beta(alpha / 2.0, 0.5) * (val_hi - val_lo)
        const = (2.0 ** (alpha - 2.0)) * (c_active ** (1.0 - alpha))
        out[np.where(valid)[0][in_range]] = np.maximum(const * integral_val, 0.0)
        
    elif size_model == "EXPONENTIAL":
        # Exponential needs a loop/vectorized cosh trapezoid
        for idx, val in enumerate(c_active):
            val_lo = max(2.0 * r_min / val, 1.0)
            val_hi = 2.0 * r_max / val
            if val_lo >= val_hi:
                continue
            t_lo = np.arccosh(val_lo)
            t_max = np.arccosh(val_hi)
            t_grid = np.linspace(t_lo, t_max, 30)
            y = np.exp(-alpha * 0.5 * val * np.cosh(t_grid))
            v = float(np.trapezoid(y, t_grid))
            out[np.where(valid)[0][in_range][idx]] = max(0.25 * val * v, 0.0)
            
    return out

# Test vectorization
c_vals = np.linspace(0.1, 5.0, 100)
pdf_scalar = np.array([chord_pdf_ideal_scalar(c, 3.5, 0.5, 15.0, 0.1, "POWER_LAW") for c in c_vals])
pdf_vec = chord_pdf_ideal_vec(c_vals, 3.5, 0.5, 15.0, 0.1, "POWER_LAW")

print("Max difference (POWER_LAW):", np.max(np.abs(pdf_scalar - pdf_vec)))
