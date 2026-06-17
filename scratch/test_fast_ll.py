import math
import numpy as np
import scipy.integrate as integrate
from scipy.special import beta, betainc

def chord_pdf_given_r(c: float, r: float) -> float:
    if c <= 0 or c >= 2.0 * r:
        return 0.0
    denom = 2.0 * r * math.sqrt(max(4.0 * r**2 - c**2, 1e-12))
    return c / denom

def chord_pdf_ideal_slow(
    c: float,
    alpha: float,
    r_min: float,
    r_max: float,
    L_min: float = 0.0,
    size_model: str = "POWER_LAW",
) -> float:
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
        from scipy.special import beta, betainc
        def I_integral(u_val, a):
            if u_val <= 0.0:
                return 0.0
            return 0.5 * beta(a / 2.0, 0.5) * betainc(a / 2.0, 0.5, u_val**2)

        integral_val = I_integral(u_hi, alpha) - I_integral(u_lo, alpha)
        const = (2.0 ** (alpha - 2.0)) * (c ** (1.0 - alpha))
        return max(const * integral_val, 0.0)
    return 0.0

def censored_chord_log_likelihood_slow(
    chord_lengths: np.ndarray,
    is_contained: np.ndarray,
    alpha: float,
    r_min: float,
    r_max: float,
    L_min: float,
    size_model: str = "POWER_LAW",
    n_grid: int = 80,
) -> float:
    c_grid = np.linspace(L_min + 1e-6, 2 * r_max - 1e-6, n_grid)
    pdf_grid = np.array([chord_pdf_ideal_slow(c, alpha, r_min, r_max, L_min, size_model) for c in c_grid])
    Z = float(np.trapezoid(pdf_grid, c_grid))
    if Z < 1e-15:
        return -1e10

    log_Z = math.log(Z)
    ll = 0.0
    for c, contained in zip(chord_lengths, is_contained):
        if contained:
            p = chord_pdf_ideal_slow(c, alpha, r_min, r_max, L_min, size_model)
            if p < 1e-300:
                ll -= 20.0
            else:
                ll += math.log(p) - log_Z
        else:
            c_tail = np.linspace(c, 2 * r_max - 1e-6, max(n_grid // 2, 10))
            pdf_tail = np.array([chord_pdf_ideal_slow(ct, alpha, r_min, r_max, L_min, size_model) for ct in c_tail])
            surv = float(np.trapezoid(pdf_tail, c_tail))
            surv_frac = surv / max(Z, 1e-15)
            ll += math.log(max(surv_frac, 1e-15))

    return ll

def chord_pdf_ideal_fast(
    c: np.ndarray,
    alpha: float,
    r_min: float,
    r_max: float,
    L_min: float = 0.0,
    size_model: str = "POWER_LAW",
) -> np.ndarray:
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
        val_hi = betainc(alpha / 2.0, 0.5, u_hi**2)
        val_lo = betainc(alpha / 2.0, 0.5, u_lo**2)
        integral_val = 0.5 * beta(alpha / 2.0, 0.5) * (val_hi - val_lo)
        const = (2.0 ** (alpha - 2.0)) * (c_active ** (1.0 - alpha))
        out[np.where(valid)[0][in_range]] = np.maximum(const * integral_val, 0.0)
    elif size_model == "EXPONENTIAL":
        # Vectorized EXPONENTIAL
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

def censored_chord_log_likelihood_fast(
    chord_lengths: np.ndarray,
    is_contained: np.ndarray,
    alpha: float,
    r_min: float,
    r_max: float,
    L_min: float,
    size_model: str = "POWER_LAW",
    n_grid: int = 150,
) -> float:
    c_grid = np.linspace(L_min + 1e-6, 2 * r_max - 1e-6, n_grid)
    pdf_grid = chord_pdf_ideal_fast(c_grid, alpha, r_min, r_max, L_min, size_model)
    
    dx = c_grid[1] - c_grid[0]
    Z = float(np.trapezoid(pdf_grid, c_grid))
    if Z < 1e-15:
        return -1e10

    pdf_rev = pdf_grid[::-1]
    cum_trap = np.zeros_like(pdf_rev)
    cum_trap[1:] = np.cumsum(0.5 * (pdf_rev[:-1] + pdf_rev[1:]) * dx)
    survival_grid = cum_trap[::-1]

    log_Z = math.log(Z)
    
    contained_chords = chord_lengths[is_contained]
    if len(contained_chords) > 0:
        pdf_contained = chord_pdf_ideal_fast(contained_chords, alpha, r_min, r_max, L_min, size_model)
        pdf_contained = np.maximum(pdf_contained, 1e-300)
        ll = np.sum(np.log(pdf_contained) - log_Z)
    else:
        ll = 0.0
        
    censored_chords = chord_lengths[~is_contained]
    if len(censored_chords) > 0:
        surv_vals = np.interp(censored_chords, c_grid, survival_grid)
        surv_frac = surv_vals / Z
        surv_frac = np.maximum(surv_frac, 1e-15)
        ll += np.sum(np.log(surv_frac))
        
    return float(ll)

# Test comparison on dummy data
rng = np.random.default_rng(0)
chords = rng.uniform(0.1, 5.0, 50)
contained = rng.choice([True, False], 50)

ll_slow = censored_chord_log_likelihood_slow(chords, contained, 3.5, 0.5, 15.0, 0.1)
ll_fast = censored_chord_log_likelihood_fast(chords, contained, 3.5, 0.5, 15.0, 0.1)

print("Slow LL:", ll_slow)
print("Fast LL:", ll_fast)
print("Diff:", abs(ll_slow - ll_fast))
import time
t0 = time.time()
for _ in range(100):
    censored_chord_log_likelihood_slow(chords, contained, 3.5, 0.5, 15.0, 0.1)
t_slow = time.time() - t0

t0 = time.time()
for _ in range(100):
    censored_chord_log_likelihood_fast(chords, contained, 3.5, 0.5, 15.0, 0.1)
t_fast = time.time() - t0

print(f"Time for 100 runs - Slow: {t_slow:.3f}s | Fast: {t_fast:.3f}s (Speedup: {t_slow/t_fast:.1f}x)")
