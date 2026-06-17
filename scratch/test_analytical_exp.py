import math
import numpy as np
from scipy import integrate

def chord_pdf_given_r(c: float, r: float) -> float:
    if c <= 0 or c >= 2.0 * r:
        return 0.0
    denom = 2.0 * r * math.sqrt(max(4.0 * r**2 - c**2, 1e-12))
    return c / denom

def chord_pdf_ideal_numerical_exp(
    c: float,
    param: float,
    r_min: float,
    r_max: float,
) -> float:
    r_lo = max(r_min, c / 2.0 + 1e-9)
    if r_lo >= r_max:
        return 0.0

    def integrand(r):
        f_obs_r = r * math.exp(-param * r)
        return chord_pdf_given_r(c, r) * f_obs_r

    val, _ = integrate.quad(integrand, r_lo, r_max, limit=50)
    return max(val, 0.0)

def chord_pdf_ideal_substitution_exp(
    c: float,
    param: float,
    r_min: float,
    r_max: float,
) -> float:
    val_lo = max(2.0 * r_min / c, 1.0)
    val_hi = 2.0 * r_max / c
    if val_lo >= val_hi:
        return 0.0
    t_lo = math.acosh(val_lo)
    t_max = math.acosh(val_hi)
    
    def integrand_cosh(t):
        return math.exp(-param * 0.5 * c * math.cosh(t))
        
    val, _ = integrate.quad(integrand_cosh, t_lo, t_max, limit=50)
    return max(0.25 * c * val, 0.0)

# Test comparison
for c in [0.2, 0.5, 1.0, 2.0, 5.0]:
    num = chord_pdf_ideal_numerical_exp(c, 0.25, 0.5, 30.0)
    sub = chord_pdf_ideal_substitution_exp(c, 0.25, 0.5, 30.0)
    print(f"c={c:.2f} | Numerical={num:.6e} | Substitution={sub:.6e} | Diff={abs(num-sub):.6e}")
