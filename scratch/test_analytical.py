import math
import numpy as np
from scipy import integrate
from scipy.special import betainc, beta

def chord_pdf_given_r(c: float, r: float) -> float:
    if c <= 0 or c >= 2.0 * r:
        return 0.0
    denom = 2.0 * r * math.sqrt(max(4.0 * r**2 - c**2, 1e-12))
    return c / denom

def chord_pdf_ideal_numerical(
    c: float,
    param: float,
    r_min: float,
    r_max: float,
) -> float:
    r_lo = max(r_min, c / 2.0 + 1e-9)
    if r_lo >= r_max:
        return 0.0

    def integrand(r):
        f_obs_r = r ** (1.0 - param)
        return chord_pdf_given_r(c, r) * f_obs_r

    val, _ = integrate.quad(integrand, r_lo, r_max, limit=50)
    return max(val, 0.0)

def chord_pdf_ideal_analytical(
    c: float,
    param: float,
    r_min: float,
    r_max: float,
) -> float:
    # param = alpha
    if c <= 0:
        return 0.0
    r_lo = max(r_min, c / 2.0 + 1e-9)
    if r_lo >= r_max:
        return 0.0
    
    # We want to integrate from r_lo to r_max of:
    # c / (2 * r^alpha * sqrt(4*r^2 - c^2)) dr
    # Let u = c / (2r) => r = c / (2u) => dr = -c / (2u^2) du
    # Limits: r_lo => u_hi = c / (2 * r_lo)
    #         r_max => u_lo = c / (2 * r_max)
    # Integral = \int_{u_lo}^{u_hi} (c / (2 * (c/(2u))^alpha * sqrt(c^2/u^2 - c^2))) * (c / (2u^2)) du
    #          = \int_{u_lo}^{u_hi} (c / (2 * c^alpha / (2^alpha * u^alpha) * (c/u) * sqrt(1 - u^2))) * (c / (2u^2)) du
    #          = \int_{u_lo}^{u_hi} (2^alpha * u^{alpha+1} / (2 * c^alpha * sqrt(1 - u^2))) * (c / (2u^2)) du
    #          = \int_{u_lo}^{u_hi} 2^{alpha-2} c^{1-alpha} u^{alpha-1} / sqrt(1 - u^2) du
    #          = (2 * r_0)^(alpha-1) * c^{1-alpha} * 1/(2 * r_0)^(alpha-1) * 2^{alpha-2} c^{1-alpha} ...
    # Let's simplify the constant:
    # 2^{alpha-2} * c^{1-alpha} * \int_{u_lo}^{u_hi} u^{alpha-1} / sqrt(1 - u^2) du
    
    u_lo = c / (2.0 * r_max)
    u_hi = min(c / (2.0 * r_min), 1.0 - 1e-9)
    if u_lo >= u_hi:
        return 0.0
        
    def I_integral(u_val, a):
        # returns \int_0^u_val u^{a-1} / sqrt(1-u^2) du
        # = 0.5 * beta(a/2, 0.5) * betainc(a/2, 0.5, u_val^2)
        return 0.5 * beta(a / 2.0, 0.5) * betainc(a / 2.0, 0.5, u_val**2)
        
    integral_val = I_integral(u_hi, param) - I_integral(u_lo, param)
    const = (2.0 ** (param - 2.0)) * (c ** (1.0 - param))
    return const * integral_val

# Test comparison
for c in [0.2, 0.5, 1.0, 2.0, 5.0]:
    num = chord_pdf_ideal_numerical(c, 3.0, 0.328, 30.0)
    ana = chord_pdf_ideal_analytical(c, 3.0, 0.328, 30.0)
    print(f"c={c:.2f} | Numerical={num:.6e} | Analytical={ana:.6e} | Diff={abs(num-ana):.6e}")
