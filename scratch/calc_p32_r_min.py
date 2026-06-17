import math

def mean_r2(alpha, r_min, r_max):
    num = (r_max**(3-alpha) - r_min**(3-alpha)) / (3-alpha)
    den = (r_max**(1-alpha) - r_min**(1-alpha)) / (1-alpha)
    return num / den

alpha = 3.826043162057008
r_max = 250.0
P30 = 0.03518515694404851

for r_min in [1.0, 0.328]:
    e_r2 = mean_r2(alpha, r_min, r_max)
    e_pi_r2 = math.pi * e_r2
    P32_calc = P30 * e_pi_r2
    print(f"r_min={r_min}: e_r2={e_r2:.4f}, e_pi_r2={e_pi_r2:.4f}, P32={P32_calc:.4f}")
