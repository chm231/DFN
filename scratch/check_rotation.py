import numpy as np

def mean_pole_vector_from_trend_plunge(trend_deg, plunge_deg):
    tr = np.radians(trend_deg)
    pl = np.radians(plunge_deg)
    n = np.array([np.cos(pl) * np.sin(tr), np.cos(pl) * np.cos(tr), -np.sin(pl)])
    return n / np.linalg.norm(n)

# Let's test the basis vectors
basis = {
    'East (+X)': np.array([1.0, 0.0, 0.0]),
    'North (+Y)': np.array([0.0, 1.0, 0.0]),
    'Up (+Z)': np.array([0.0, 0.0, 1.0]),
    'West (-X)': np.array([-1.0, 0.0, 0.0]),
    'South (-Y)': np.array([0.0, -1.0, 0.0]),
    'Down (-Z)': np.array([0.0, 0.0, -1.0])
}

print("Current Projection (as in code):")
print("n = [nx, ny, nz]")
print("X_proj = -ny / (1 - nz)")
print("Y_proj = nx / (1 - nz)")
print("-" * 50)

for name, n in basis.items():
    # If nz > 0, flip for lower hemisphere
    n_lower = n.copy()
    if n_lower[2] > 0:
        n_lower = -n_lower
    
    # Projection
    denom = 1.0 - n_lower[2]
    # Handle denominator = 0 if nz = 1 (but it's flipped to nz = -1, so denom = 2)
    x_proj = -n_lower[1] / denom
    y_proj = n_lower[0] / denom
    print(f"{name:12} -> n_lower={n_lower} -> X_proj={x_proj:6.2f}, Y_proj={y_proj:6.2f}")

print("\nStandard Projection (North is Up, East is Right):")
print("X_proj = nx / (1 - nz)")
print("Y_proj = ny / (1 - nz)")
print("-" * 50)

for name, n in basis.items():
    n_lower = n.copy()
    if n_lower[2] > 0:
        n_lower = -n_lower
    
    denom = 1.0 - n_lower[2]
    x_proj = n_lower[0] / denom
    y_proj = n_lower[1] / denom
    print(f"{name:12} -> n_lower={n_lower} -> X_proj={x_proj:6.2f}, Y_proj={y_proj:6.2f}")
