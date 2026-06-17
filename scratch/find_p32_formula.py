import math
import numpy as np

# Ground truth values from the photo
gt = {
    "S1": {"trend": 338.1, "plunge": 4.5, "kr": 2.85, "r0": 0.328, "P32": 1.310, "P21": 0.2733, "model": "POWER_LAW"},
    "S2": {"trend": 100.4, "plunge": 0.2, "kr": 3.04, "r0": 0.977, "P32": 1.026, "P21": 0.3538, "model": "POWER_LAW"},
    "S3": {"trend": 212.9, "plunge": 0.9, "kr": 3.01, "r0": 0.858, "P32": 0.975, "P21": 0.6956, "model": "POWER_LAW"},
    "S4": {"trend": 3.3,   "plunge": 62.1, "kr": 4.0,  "r0": 4.0,   "P32": 2.320, "P21": 2.2293, "model": "EXPONENTIAL"},
    "S5": {"trend": 243.0, "plunge": 24.4, "kr": 3.60, "r0": 0.400, "P32": 1.400, "P21": 0.1555, "model": "POWER_LAW"}
}

# Tunnel face normal
m = np.array([1.0, 0.0, 0.0])

for name, data in gt.items():
    # Calculate Cs = ||n x m||
    trend_rad = np.radians(data["trend"])
    plunge_rad = np.radians(data["plunge"])
    n = np.array([
        math.cos(plunge_rad) * math.cos(trend_rad),
        math.cos(plunge_rad) * math.sin(trend_rad),
        -math.sin(plunge_rad)
    ])
    n = n / np.linalg.norm(n)
    Cs = np.linalg.norm(np.cross(n, m))
    
    # Calculate E[r] and E[r^2] for POWER_LAW
    r0 = data["r0"]
    r_max = 30.0
    
    if data["model"] == "POWER_LAW":
        alpha = data["kr"] + 1.0
        # E[r] = \int_{r0}^{rmax} r * r^{-alpha} dr / \int_{r0}^{rmax} r^{-alpha} dr
        # = (rmax^(2-alpha) - r0^(2-alpha))/(2-alpha) / [ (rmax^(1-alpha) - r0^(1-alpha))/(1-alpha) ]
        num_r = (r_max**(2-alpha) - r0**(2-alpha)) / (2-alpha)
        den_r = (r_max**(1-alpha) - r0**(1-alpha)) / (1-alpha)
        Er = num_r / den_r
        
        num_r2 = (r_max**(3-alpha) - r0**(3-alpha)) / (3-alpha)
        Er2 = num_r2 / den_r
        
    else: # EXPONENTIAL
        # f(r) = lambda * exp(-lambda * (r - r0)) => scale = 4.0 => lambda = 0.25
        # but wait, the table says scale kr=4.0
        lmb = 1.0 / data["r0"]
        # E[r] = r0 + 1/lambda
        Er = r0 + 1.0 / lmb
        # E[r^2] = r0^2 + 2*r0/lambda + 2/lambda^2
        Er2 = r0**2 + 2.0 * r0 / lmb + 2.0 / (lmb**2)
        
    print(f"\n--- {name} ---")
    print(f"Cs: {Cs:.4f} | P21_obs: {data['P21']:.4f} | GT P32: {data['P32']:.4f}")
    print(f"Er: {Er:.4f} | Er2: {Er2:.4f}")
    
    # Let's check some candidate formulas:
    # 1. P32 = P21 / Cs (exact stereology)
    p32_1 = data["P21"] / Cs
    # 2. P32 = P21 * pi / (2 * Cs * Er) (old formula)
    p32_2 = data["P21"] * math.pi / (2.0 * Cs * Er)
    # 3. P32 = P21 / (Cs * Er) ?
    p32_3 = data["P21"] / (Cs * Er)
    # 4. P32 = P21 * Er / (Cs * Er2) ?
    p32_4 = data["P21"] * Er / (Cs * Er2)
    # 5. Let's see the ratio of P21 / (Cs * P32_GT)
    ratio = data["P21"] / (Cs * data["P32"])
    
    print(f"  P21/Cs: {p32_1:.4f} (error: {p32_1/data['P32']-1:+.2%})")
    print(f"  Old formula: {p32_2:.4f} (error: {p32_2/data['P32']-1:+.2%})")
    print(f"  Ratio P21 / (Cs * P32_GT): {ratio:.4f}")
