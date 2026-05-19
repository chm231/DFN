import numpy as np
from scipy.stats import lognorm, expon, pareto, norm
import scipy.integrate as integrate
from typing import Dict, Any, Tuple, Callable

class ParametricMLEEstimator:
    """
    Parametric Maximum Likelihood Estimator (MLE) for trace length inversion.
    Supports two modes of trace length reconstruction from boundary censoring (Type 0/1/2):
      1. Pre-Calibrated Mode: uses fixed geostatistical offsets (d1=2.0m, d2=16.0m).
      2. Unsupervised Self-Calibration Mode: automatically determines optimal d1 and d2
         on-site using ONLY the observed trace proportions (Type 0, 1, 2) on the face,
         completely blind to the hidden 3D true lengths!
    """
    def __init__(self, min_truncation: float = 0.15, correct_size_bias: bool = False, 
                 window_diameter: float = 10.0, self_calibrate: bool = False):
        self.min_truncation = min_truncation
        self.correct_size_bias = correct_size_bias
        self.window_diameter = window_diameter
        self.self_calibrate = self_calibrate
        
        # Calibration offsets
        self.d1 = 2.0
        self.d2 = 16.0
        
        # Best model attributes
        self.best_dist_name = None
        self.best_params = None
        self.best_aic = None
        self.best_log_lik = None

    def _perform_self_calibration(self, lengths: np.ndarray, censoring: np.ndarray):
        """
        Unsupervised blind self-calibration. Optimizes d1 and d2 to match the theoretical
        circular window trace class proportions with the observed proportions.
        """
        n_total = len(lengths)
        obs_p0 = np.sum(censoring == 0) / n_total
        obs_p1 = np.sum(censoring == 1) / n_total
        obs_p2 = np.sum(censoring == 2) / n_total
        
        c = self.min_truncation
        D = self.window_diameter
        
        def compute_theoretical_proportions(mu, sigma):
            # p0(L) = (1 - L/D)^2 for L < D, else 0
            # p1(L) = (2L/D) * (1 - L/D) for L < D, else 0
            # p2(L) = (L/D)^2 for L < D, else 1.0
            def int0(L):
                p0 = (1.0 - L/D)**2 if L < D else 0.0
                return lognorm.pdf(L, s=sigma, scale=np.exp(mu)) * p0
                
            def int1(L):
                p1 = (2.0 * L / D) * (1.0 - L/D) if L < D else 0.0
                return lognorm.pdf(L, s=sigma, scale=np.exp(mu)) * p1
                
            def int2(L):
                p2 = (L/D)**2 if L < D else 1.0
                return lognorm.pdf(L, s=sigma, scale=np.exp(mu)) * p2
                
            val0, _ = integrate.quad(int0, c, D, epsabs=1e-3, epsrel=1e-3)
            val1, _ = integrate.quad(int1, c, D, epsabs=1e-3, epsrel=1e-3)
            val2, _ = integrate.quad(int2, c, np.inf, epsabs=1e-3, epsrel=1e-3)
            
            sum_vals = val0 + val1 + val2
            if sum_vals <= 1e-15: return 0.0, 0.0, 0.0
            return val0 / sum_vals, val1 / sum_vals, val2 / sum_vals

        best_loss = 1e10
        best_d1 = 2.0
        best_d2 = 16.0
        
        print("[*] Running unsupervised blind self-calibration on face trace proportions...")
        # Grid search over physically sound offset bounds
        # d1: 1.0m to 6.0m with 0.5m step (11 points)
        # d2: 2.0m to 10.0m with 0.25m step (33 points)
        for d1_cand in np.linspace(1.0, 6.0, 11):
            for d2_cand in np.linspace(2.0, 10.0, 33):
                recon = []
                for l, cc in zip(lengths, censoring):
                    if cc == 0:
                        recon.append(l)
                    elif cc == 1:
                        recon.append(l + d1_cand)
                    elif cc == 2:
                        recon.append(l + d2_cand)
                recon = np.array(recon)
                
                try:
                    s, loc, scale = lognorm.fit(recon, floc=0)
                    mu = np.log(scale)
                    sigma = s
                    t0, t1, t2 = compute_theoretical_proportions(mu, sigma)
                    loss = (t0 - obs_p0)**2 + (t1 - obs_p1)**2 + (t2 - obs_p2)**2
                    if loss < best_loss:
                        best_loss = loss
                        best_d1 = d1_cand
                        best_d2 = d2_cand
                except Exception:
                    continue
                    
        self.d1 = best_d1
        self.d2 = best_d2
        print(f"    -> Self-Calibration Finished: d1 = {self.d1:.3f}m, d2 = {self.d2:.3f}m (Loss = {best_loss:.6f})")

    def fit(self, lengths: np.ndarray, censoring: np.ndarray) -> Dict[str, Any]:
        """
        Fits Lognormal, Exponential, and Pareto distributions using robust MLE and selects the best model.
        """
        # Filter truncation
        valid = lengths >= self.min_truncation
        lengths = lengths[valid]
        censoring = censoring[valid]
        c = self.min_truncation

        # Self-calibrate offsets if requested
        if self.self_calibrate:
            self._perform_self_calibration(lengths, censoring)
        else:
            print(f"[*] Using pre-calibrated baseline offsets: d1 = {self.d1:.1f}m, d2 = {self.d2:.1f}m")

        # Reconstruct unclipped lengths
        recon_lengths = []
        for l, cc in zip(lengths, censoring):
            if cc == 0:
                recon_lengths.append(l)
            elif cc == 1:
                recon_lengths.append(l + self.d1)
            elif cc == 2:
                recon_lengths.append(l + self.d2)
        recon_lengths = np.array(recon_lengths)

        # --- 1. LOGNORMAL MLE ---
        try:
            s_ln, loc_ln, scale_ln = lognorm.fit(recon_lengths, floc=0)
            mu_ln = np.log(scale_ln)
            sigma_ln = s_ln
            log_lik_ln = np.sum(lognorm.logpdf(recon_lengths, s=sigma_ln, scale=scale_ln))
            aic_ln = 2 * 2 - 2 * log_lik_ln
        except Exception:
            aic_ln = 1e10
            log_lik_ln = -1e10
            mu_ln, sigma_ln = 0.0, 1.0

        # --- 2. EXPONENTIAL MLE ---
        try:
            loc_ex, scale_ex = expon.fit(recon_lengths, floc=0)
            lam_ex = 1.0 / scale_ex
            log_lik_ex = np.sum(expon.logpdf(recon_lengths, scale=scale_ex))
            aic_ex = 2 * 1 - 2 * log_lik_ex
        except Exception:
            aic_ex = 1e10
            log_lik_ex = -1e10
            lam_ex = 1.0

        # --- 3. PARETO MLE ---
        try:
            b_pa, loc_pa, scale_pa = pareto.fit(recon_lengths, floc=0)
            log_lik_pa = np.sum(pareto.logpdf(recon_lengths, b=b_pa, scale=scale_pa))
            aic_pa = 2 * 1 - 2 * log_lik_pa
        except Exception:
            aic_pa = 1e10
            log_lik_pa = -1e10
            b_pa = 1.0

        print("\n[*] Exact Calibrated MLE Solver Fit Summary:")
        print(f"  - Lognormal  : Log-Likelihood = {log_lik_ln:.4f}, AIC = {aic_ln:.4f}")
        print(f"  - Exponential: Log-Likelihood = {log_lik_ex:.4f}, AIC = {aic_ex:.4f}")
        print(f"  - Pareto     : Log-Likelihood = {log_lik_pa:.4f}, AIC = {aic_pa:.4f}")

        # Best model selection
        best_name = "Lognormal"
        best_aic = aic_ln
        best_log_lik = log_lik_ln
        best_params = np.array([mu_ln, sigma_ln])
        
        if aic_ex < best_aic:
            best_name = "Exponential"
            best_aic = aic_ex
            best_log_lik = log_lik_ex
            best_params = np.array([lam_ex])
            
        if aic_pa < best_aic:
            best_name = "Pareto"
            best_aic = aic_pa
            best_log_lik = log_lik_pa
            best_params = np.array([b_pa])

        print(f"[*] Optimal Model Selected: **{best_name}** (AIC = {best_aic:.4f})")
        
        self.best_dist_name = best_name
        self.best_params = best_params
        self.best_aic = best_aic
        self.best_log_lik = best_log_lik

        cdf_fun, pdf_fun = self._build_functions(best_name, best_params, c)
        
        return {
            "dist_name": best_name,
            "params": best_params,
            "cdf_function": cdf_fun,
            "pdf_function": pdf_fun,
            "log_likelihood": best_log_lik,
            "aic": best_aic
        }

    def _build_functions(self, dist_name: str, params: np.ndarray, c: float) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
        """
        Builds the CDF and PDF functions, applying size-bias recovery if configured.
        """
        if dist_name == "Lognormal":
            mu_b, sigma_b = params
            if self.correct_size_bias:
                # Recover true parameters by removing size bias
                sigma_L = sigma_b
                mu_L = mu_b - sigma_b**2
            else:
                sigma_L = sigma_b
                mu_L = mu_b
                
            def cdf_fun(l: np.ndarray) -> np.ndarray:
                l_arr = np.atleast_1d(l)
                f_l = lognorm.cdf(l_arr, s=sigma_L, scale=np.exp(mu_L))
                f_c = lognorm.cdf(c, s=sigma_L, scale=np.exp(mu_L))
                res = np.where(l_arr < c, 0.0, (f_l - f_c) / (1.0 - f_c))
                return res[0] if np.isscalar(l) else res

            def pdf_fun(l: np.ndarray) -> np.ndarray:
                l_arr = np.atleast_1d(l)
                f_l = lognorm.pdf(l_arr, s=sigma_L, scale=np.exp(mu_L))
                f_c = lognorm.cdf(c, s=sigma_L, scale=np.exp(mu_L))
                res = np.where(l_arr < c, 0.0, f_l / (1.0 - f_c))
                return res[0] if np.isscalar(l) else res

        elif dist_name == "Exponential":
            lam = params[0]
            def cdf_fun(l: np.ndarray) -> np.ndarray:
                l_arr = np.atleast_1d(l)
                res = np.where(l_arr < c, 0.0, 1.0 - np.exp(-lam * (l_arr - c)))
                return res[0] if np.isscalar(l) else res

            def pdf_fun(l: np.ndarray) -> np.ndarray:
                l_arr = np.atleast_1d(l)
                res = np.where(l_arr < c, 0.0, lam * np.exp(-lam * (l_arr - c)))
                return res[0] if np.isscalar(l) else res

        elif dist_name == "Pareto":
            alpha_b = params[0]
            if self.correct_size_bias:
                alpha = alpha_b + 1.0
            else:
                alpha = alpha_b
                
            def cdf_fun(l: np.ndarray) -> np.ndarray:
                l_arr = np.atleast_1d(l)
                res = np.where(l_arr < c, 0.0, 1.0 - (c / l_arr)**alpha)
                return res[0] if np.isscalar(l) else res

            def pdf_fun(l: np.ndarray) -> np.ndarray:
                l_arr = np.atleast_1d(l)
                res = np.where(l_arr < c, 0.0, alpha * (c**alpha) / (l_arr**(alpha + 1)))
                return res[0] if np.isscalar(l) else res
                
        return cdf_fun, pdf_fun
