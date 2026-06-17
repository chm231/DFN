import math

def mean_r2(alpha, r_min, r_max):
    # E[r^2] = \int_{r_min}^{r_max} r^2 * C * r^{-alpha} dr
    # C = (alpha - 1) / (r_min^{1-alpha} - r_max^{1-alpha})
    # \int r^{2-alpha} dr = r^{3-alpha} / (3-alpha)
    num = (r_max**(3-alpha) - r_min**(3-alpha)) / (3-alpha)
    den = (r_max**(1-alpha) - r_min**(1-alpha)) / (1-alpha)
    return num / den

alpha = 3.826043162057008
r_min = 1.0
r_max = 250.0
P30 = 0.03518515694404851

e_r2 = mean_r2(alpha, r_min, r_max)
e_pi_r2 = math.pi * e_r2
P32_calc = P30 * e_pi_r2
print("e_r2:", e_r2)
print("e_pi_r2:", e_pi_r2)
print("P32_calc:", P32_calc)
