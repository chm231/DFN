import os
import sys
import argparse
import time
import re
import numpy as np
import scipy.integrate as integrate
import h5py
import matplotlib.pyplot as plt
from typing import List, Dict, Any


# ==============================================================================
# SECTION 1: CORE GEOMETRY & STATISTICAL SAMPLING FUNCTIONS
# ==============================================================================

def mean_pole_vector_from_trend_plunge(trend_deg, plunge_deg):
    """
    Converts trend and plunge (in degrees) to a mean unit pole vector.
    Assumes coordinate system: x=East, y=North, z=Up. Plunge is positive downward.
    """
    tr = np.radians(trend_deg)
    pl = np.radians(plunge_deg)
    n = np.array([np.cos(pl) * np.sin(tr), np.cos(pl) * np.cos(tr), -np.sin(pl)])
    return n / np.linalg.norm(n)

def axis_angle_to_rotmat(axis, angle):
    """Generates 3D rotation matrix from axis and angle."""
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    R = np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c]
    ])
    return R

def sample_fisher_normals(mean_n, kappa, N, seed=None):
    """Samples unit normal vectors from a 3D Fisher/von Mises-Fisher distribution."""
    mean_n = mean_n / np.linalg.norm(mean_n)
    rng = np.random.default_rng(seed)
    U = rng.uniform(0.0, 1.0, N)
    phi = 2.0 * np.pi * rng.uniform(0.0, 1.0, N)
    
    if kappa < 1e-10:
        cosTheta = 2.0 * U - 1.0
    else:
        # MATLAB-identical Fisher polar angle formula:
        # cosTheta = (1/kappa) * log(exp(-kappa) + U*(exp(kappa) - exp(-kappa)))
        cosTheta = (1.0 / kappa) * np.log(np.exp(-kappa) + U * (np.exp(kappa) - np.exp(-kappa)))
        cosTheta = np.clip(cosTheta, -1.0, 1.0)
        
    sinTheta = np.sqrt(np.maximum(0.0, 1.0 - cosTheta**2))
    v0 = np.column_stack([sinTheta * np.cos(phi), sinTheta * np.sin(phi), cosTheta])
    
    ez = np.array([0.0, 0.0, 1.0])
    if np.linalg.norm(mean_n - ez) < 1e-12:
        R = np.eye(3)
    elif np.linalg.norm(mean_n + ez) < 1e-12:
        R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    else:
        axis_rot = np.cross(ez, mean_n)
        axis_rot = axis_rot / np.linalg.norm(axis_rot)
        angle = np.arccos(np.dot(ez, mean_n))
        R = axis_angle_to_rotmat(axis_rot, angle)
        
    normals = (R @ v0.T).T
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    return normals

def normal_to_strike_dip_basis_vectorized(normals):
    """Builds two orthonormal vectors in the fracture plane for a set of unit normals."""
    nrm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / nrm
    N = len(normals)
    
    ref = np.zeros((N, 3), dtype=normals.dtype)
    ref[:, 2] = 1.0
    
    # Identify near-vertical normals to shift reference vector to prevent cross product collapse
    mask = np.abs(normals[:, 2]) > 0.95
    ref[mask, :] = [1.0, 0.0, 0.0]
    
    strike_u = np.cross(ref, normals, axis=1)
    strike_u = strike_u / np.linalg.norm(strike_u, axis=1, keepdims=True)
    
    dip_u = np.cross(normals, strike_u, axis=1)
    dip_u = dip_u / np.linalg.norm(dip_u, axis=1, keepdims=True)
    
    return strike_u, dip_u

def size_pdf_truncated(r, sizeDist):
    """Computes probability density function (PDF) value for size distribution types."""
    rmin = sizeDist['rmin']
    rmax = sizeDist['rmax']
    val = np.zeros_like(r)
    mask = (r >= rmin) & (r <= rmax)
    
    dtype = sizeDist['type'].lower()
    if dtype == 'powerlaw':
        k = sizeDist['kr']
        alpha = k + 1.0
        if np.abs(alpha - 1.0) < 1e-12:
            C = 1.0 / np.log(rmax / rmin)
        else:
            C = (1.0 - alpha) / (rmax**(1.0 - alpha) - rmin**(1.0 - alpha))
        val[mask] = C * r[mask]**(-alpha)
    elif dtype == 'exponential':
        lbl = 1.0 / sizeDist['r0']
        C = lbl / (np.exp(-lbl * rmin) - np.exp(-lbl * rmax))
        val[mask] = C * np.exp(-lbl * r[mask])
    elif dtype == 'lognormal':
        mu = sizeDist['mu']
        sig = sizeDist['sigma']
        raw = lambda x: (1.0 / (x * sig * np.sqrt(2.0 * np.pi))) * np.exp(-(np.log(x) - mu)**2 / (2.0 * sig**2))
        Z, _ = integrate.quad(raw, rmin, rmax)
        val[mask] = raw(r[mask]) / Z
    elif dtype == 'uniform':
        val[mask] = 1.0 / (rmax - rmin)
    else:
        raise ValueError(f"Unknown sizeDist type: {sizeDist['type']}")
    return val

def sample_radius(sizeDist, N, seed=None):
    """Samples truncated fracture radii according to specified distribution."""
    rmin = sizeDist['rmin']
    rmax = sizeDist['rmax']
    rng = np.random.default_rng(seed)
    U = rng.uniform(0.0, 1.0, N)
    
    dtype = sizeDist['type'].lower()
    if dtype == 'powerlaw':
        # Case A:
        # inverse CDF uses exponent kr
        # -> survival exponent = kr (S_R(r) proportional to r^-kr)
        # -> pdf exponent = kr + 1 (f_R(r) proportional to r^-(kr+1))
        k = sizeDist['kr']
        alpha = k + 1.0
        if np.abs(alpha - 1.0) < 1e-12:
            r = rmin * (rmax / rmin)**U
        else:
            r = (rmin**(1.0 - alpha) + U * (rmax**(1.0 - alpha) - rmin**(1.0 - alpha)))**(1.0 / (1.0 - alpha))
    elif dtype == 'exponential':
        # Exponential survival function integration for inverse transform sampling
        lbl = 1.0 / sizeDist['r0']
        A = np.exp(-lbl * rmin)
        B = np.exp(-lbl * rmax)
        r = -(1.0 / lbl) * np.log(A - U * (A - B))
    elif dtype == 'lognormal':
        mu = sizeDist['mu']
        sig = sizeDist['sigma']
        r = np.zeros(N)
        cnt = 0
        while cnt < N:
            batch = rng.lognormal(mu, sig, N)
            valid = batch[(batch >= rmin) & (batch <= rmax)]
            take = min(len(valid), N - cnt)
            if take > 0:
                r[cnt : cnt + take] = valid[:take]
                cnt += take
    elif dtype == 'uniform':
        r = rmin + (rmax - rmin) * U
    else:
        raise ValueError(f"Unknown sizeDist type: {sizeDist['type']}")
    return r

def compute_num_fractures_from_P32(P32, sizeDist, V):
    """Computes the target number of fractures to generate based on box volume and target realized P32."""
    rmin = sizeDist['rmin']
    rmax = sizeDist['rmax']
    
    # Numerically integrate area * PDF
    integrand = lambda r: np.pi * r**2 * size_pdf_truncated(np.array([r]), sizeDist)[0]
    meanArea, _ = integrate.quad(integrand, rmin, rmax)
    
    N_real = V * P32 / meanArea
    return max(0, int(round(N_real)))

def sample_centers_from_surface_points(box, radius, strike_u, dip_u, centerMode, seed=None):
    """Samples fracture disc center coordinates within the designated box domain."""
    N = len(radius)
    rng = np.random.default_rng(seed)
    
    x0, y0, z0 = box.get('x0', 0.0), box.get('y0', 0.0), box.get('z0', 0.0)
    
    # Uniform point coordinates in the box
    centers = np.zeros((N, 3))
    centers[:, 0] = x0 + box['dx'] * rng.uniform(0.0, 1.0, N)
    centers[:, 1] = y0 + box['dy'] * rng.uniform(0.0, 1.0, N)
    centers[:, 2] = z0 + box['dz'] * rng.uniform(0.0, 1.0, N)
    
    omega = 2.0 * np.pi * rng.uniform(0.0, 1.0, N)
    
    mode = centerMode.lower()
    if mode == 'report':
        rho = radius * rng.uniform(0.0, 1.0, N)
    elif mode == 'area_uniform':
        rho = radius * np.sqrt(rng.uniform(0.0, 1.0, N))
    else:
        raise ValueError(f"Unknown centerMode: {centerMode}")
        
    centers[:, 0] += rho * (strike_u[:, 0] * np.cos(omega) + dip_u[:, 0] * np.sin(omega))
    centers[:, 1] += rho * (strike_u[:, 1] * np.cos(omega) + dip_u[:, 1] * np.sin(omega))
    centers[:, 2] += rho * (strike_u[:, 2] * np.cos(omega) + dip_u[:, 2] * np.sin(omega))
    return centers

# ==============================================================================
# SECTION 2: INPUT PARSING & EXPORTER LOGIC
# ==============================================================================

def read_tunnel_polygon(filepath):
    """Parses tunnel polygon coordinates from .dat format and converts from mm to meters."""
    if not os.path.exists(filepath):
        print(f"WARNING: Tunnel polygon file not found: {filepath}. Skipping tunnel geometry inclusion.")
        return None
        
    x_coords = []
    y_coords = []
    with open(filepath, 'r', encoding='utf-8') as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('X='):
                match = re.search(r'X=\s*([-\d.]+)\s*Y=\s*([-\d.]+)', line)
                if match:
                    x_coords.append(float(match.group(1)))
                    y_coords.append(float(match.group(2)))
                    
    if not x_coords:
        raise ValueError(f"No coordinates parsed from file: {filepath}")
        
    poly = np.column_stack([x_coords, y_coords]) / 1000.0
    return poly

# ==============================================================================
# SECTION 2.5: DISC CLIPPING & DFN VISUALIZATION
# ==============================================================================

def _generate_disc_polygon(center, normal, radius, n_pts=36):
    """N-gon approximation of a disc surface in 3D."""
    normal = normal / np.linalg.norm(normal)
    ref = np.array([0., 0., 1.]) if abs(normal[2]) < 0.95 else np.array([1., 0., 0.])
    u = np.cross(ref, normal);  u /= np.linalg.norm(u)
    v = np.cross(normal, u);    v /= np.linalg.norm(v)
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    return center + radius * (np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v))

def _clip_polygon_halfspace(poly, plane_pt, plane_normal):
    """Sutherland-Hodgman clip against one half-space (plane_normal points inward = valid side)."""
    if not poly:
        return poly
    result = []
    n = len(poly)
    for i in range(n):
        p1, p2 = poly[i], poly[(i + 1) % n]
        d1 = np.dot(p1 - plane_pt, plane_normal)
        d2 = np.dot(p2 - plane_pt, plane_normal)
        if d1 >= 0:
            result.append(p1)
            if d2 < 0:
                t = d1 / (d1 - d2)
                result.append(p1 + t * (p2 - p1))
        elif d2 >= 0:
            t = d1 / (d1 - d2)
            result.append(p1 + t * (p2 - p1))
    return result

def clip_disc_with_cropbox(center, normal, radius, cb):
    """Clip a fracture disc against an AABB crop box.
    cb: dict with xmin/xmax/ymin/ymax/zmin/zmax. Returns Nx3 array or None."""
    poly = list(_generate_disc_polygon(center, normal, radius))
    planes = [
        (np.array([cb['xmin'], 0., 0.]), np.array([ 1.,  0.,  0.])),
        (np.array([cb['xmax'], 0., 0.]), np.array([-1.,  0.,  0.])),
        (np.array([0., cb['ymin'], 0.]), np.array([ 0.,  1.,  0.])),
        (np.array([0., cb['ymax'], 0.]), np.array([ 0., -1.,  0.])),
        (np.array([0., 0., cb['zmin']]), np.array([ 0.,  0.,  1.])),
        (np.array([0., 0., cb['zmax']]), np.array([ 0.,  0., -1.])),
    ]
    for pt, nrm in planes:
        poly = _clip_polygon_halfspace(poly, pt, nrm)
        if len(poly) < 3:
            return None
    return np.array(poly)

def _intersect_poly_plane(poly, dim, val, tol=1e-8):
    """Line segment where a 3D convex polygon intersects an axis-aligned plane.
    dim: 0=x 1=y 2=z. Returns 2x3 array or None."""
    pts = []
    n = len(poly)
    for i in range(n):
        p1, p2 = poly[i], poly[(i + 1) % n]
        v1, v2 = p1[dim], p2[dim]
        if abs(v1 - val) < tol:
            pts.append(p1.copy())
        if (v1 - val) * (v2 - val) < 0:
            t = (val - v1) / (v2 - v1)
            pts.append(p1 + t * (p2 - p1))
        if abs(v1 - val) < tol and abs(v2 - val) < tol:
            pts.append(p2.copy())
    if len(pts) < 2:
        return None
    keep = [pts[0]]
    for p in pts[1:]:
        if all(np.linalg.norm(p - q) > 1e-7 for q in keep):
            keep.append(p)
    if len(keep) < 2:
        return None
    keep = np.array(keep)
    best_d, best_pair = -np.inf, None
    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            d = np.linalg.norm(keep[i] - keep[j])
            if d > best_d:
                best_d, best_pair = d, np.array([keep[i], keep[j]])
    return best_pair

def plot_clipped_dfn_3d(all_centers, all_normals, all_radii, all_set_ids, cb,
                         filter_by_tunnel=False, tunnel_poly_YZ=None,
                         overlay_tunnel=True, output_dir=None):
    """3D plot of DFN discs clipped to crop box (= MATLAB plot_clipped_dfn_crop)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.path import Path as MplPath

    xmin, xmax = cb['xmin'], cb['xmax']
    ymin, ymax = cb['ymin'], cb['ymax']
    zmin, zmax = cb['zmin'], cb['zmax']

    mask = (
        (all_centers[:, 0] - all_radii <= xmax) & (all_centers[:, 0] + all_radii >= xmin) &
        (all_centers[:, 1] - all_radii <= ymax) & (all_centers[:, 1] + all_radii >= ymin) &
        (all_centers[:, 2] - all_radii <= zmax) & (all_centers[:, 2] + all_radii >= zmin)
    )
    cc = all_centers[mask].astype(float)
    nn = all_normals[mask].astype(float)
    rr = all_radii[mask].astype(float)
    ss = all_set_ids[mask]

    tunnel_path = None
    if filter_by_tunnel and tunnel_poly_YZ is not None:
        tunnel_path = MplPath(tunnel_poly_YZ)

    SET_COLORS = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7', '#C4AD66']
    polys3d, facecolors = [], []
    total_area = 0.0

    for i in range(len(cc)):
        poly = clip_disc_with_cropbox(cc[i], nn[i], float(rr[i]), cb)
        if poly is None:
            continue
        if filter_by_tunnel and tunnel_path is not None:
            if not tunnel_path.contains_points(poly[:, 1:3]).any():
                continue
        polys3d.append(poly)
        facecolors.append(SET_COLORS[(int(ss[i]) - 1) % len(SET_COLORS)])
        c = poly.mean(axis=0)
        total_area += sum(
            np.linalg.norm(np.cross(poly[j] - c, poly[(j + 1) % len(poly)] - c)) / 2
            for j in range(len(poly))
        )

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    if polys3d:
        col = Poly3DCollection(polys3d, alpha=0.4, linewidth=0.3, edgecolor=[0.5, 0.5, 0.5])
        col.set_facecolor(facecolors)
        ax.add_collection3d(col)

    # Crop box wireframe
    for y in [ymin, ymax]:
        ax.plot([xmin, xmax], [y, y], [zmin, zmin], 'k-', lw=0.5)
        ax.plot([xmin, xmax], [y, y], [zmax, zmax], 'k-', lw=0.5)
    for x in [xmin, xmax]:
        ax.plot([x, x], [ymin, ymax], [zmin, zmin], 'k-', lw=0.5)
        ax.plot([x, x], [ymin, ymax], [zmax, zmax], 'k-', lw=0.5)
    for x in [xmin, xmax]:
        for y in [ymin, ymax]:
            ax.plot([x, x], [y, y], [zmin, zmax], 'k-', lw=0.5)

    if overlay_tunnel and tunnel_poly_YZ is not None:
        tY, tZ = tunnel_poly_YZ[:, 0], tunnel_poly_YZ[:, 1]
        for xv in [xmin, xmax]:
            ax.plot([xv] * len(tY), tY, tZ, 'r-', lw=1.5)

    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)
    ax.set_title('Clipped DFN in crop box' + (' (Tunnel Intersecting Only)' if filter_by_tunnel else ''))
    ax.view_init(elev=20, azim=-35)

    V_crop = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
    print(f"  Visible clipped fractures: {len(polys3d)}")
    print(f"  Crop box P32 = {total_area / V_crop:.4f} m2/m3")

    if output_dir:
        tag = 'tunnel_intersect' if filter_by_tunnel else 'all_dfn'
        fig.savefig(os.path.join(output_dir, f'plot_3d_{tag}.png'), dpi=150)
        print(f"  -> Saved: plot_3d_{tag}.png")
    return fig

def plot_2d_trace_maps(all_centers, all_normals, all_radii, all_set_ids, cb,
                        tunnel_poly_YZ=None, overlay_tunnel=True, output_dir=None):
    """2D trace maps at X=0, Y=0, Z=0 (= MATLAB plot_2d_slice_cropbox)."""
    xmin, xmax = cb['xmin'], cb['xmax']
    ymin, ymax = cb['ymin'], cb['ymax']
    zmin, zmax = cb['zmin'], cb['zmax']

    # (axis, dim_index, slice_val, xlabel, ylabel, horiz_idx, vert_idx, h_range, v_range)
    configs = [
        ('x', 0, 0.0, 'Y [m]', 'Z [m]', 1, 2, ymin, ymax, zmin, zmax),
        ('y', 1, 0.0, 'X [m]', 'Z [m]', 0, 2, xmin, xmax, zmin, zmax),
        ('z', 2, 0.0, 'X [m]', 'Y [m]', 0, 1, xmin, xmax, ymin, ymax),
    ]
    figs = []
    for axis, dim, val, xl, yl, i1, i2, hlo, hhi, vlo, vhi in configs:
        mask = (
            (all_centers[:, 0] - all_radii <= xmax) & (all_centers[:, 0] + all_radii >= xmin) &
            (all_centers[:, 1] - all_radii <= ymax) & (all_centers[:, 1] + all_radii >= ymin) &
            (all_centers[:, 2] - all_radii <= zmax) & (all_centers[:, 2] + all_radii >= zmin) &
            (all_centers[:, dim] - all_radii <= val) & (all_centers[:, dim] + all_radii >= val)
        )
        cc = all_centers[mask].astype(float)
        nn = all_normals[mask].astype(float)
        rr = all_radii[mask].astype(float)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal'); ax.grid(True, linewidth=0.3)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f'2D Trace Map at {axis.upper()} = {val:.1f} m')
        ax.set_xlim(hlo - 0.5, hhi + 0.5); ax.set_ylim(vlo - 0.5, vhi + 0.5)
        ax.plot([hlo, hhi, hhi, hlo, hlo], [vlo, vlo, vhi, vhi, vlo], 'k-', lw=1.0)

        total_trace_len = 0.0
        for i in range(len(cc)):
            poly = clip_disc_with_cropbox(cc[i], nn[i], float(rr[i]), cb)
            if poly is None:
                continue
            seg = _intersect_poly_plane(poly, dim, val)
            if seg is not None:
                ax.plot(seg[:, i1], seg[:, i2], '-', color='k', lw=0.5)
                total_trace_len += np.linalg.norm(seg[0] - seg[1])

        if overlay_tunnel and tunnel_poly_YZ is not None and axis == 'x':
            ax.plot(tunnel_poly_YZ[:, 0], tunnel_poly_YZ[:, 1], 'r-', lw=2.0, label='Tunnel')
            ax.legend(loc='upper right')

        A_box = (hhi - hlo) * (vhi - vlo)
        p21 = total_trace_len / A_box if A_box > 0 else 0.0
        print(f"  [{axis.upper()}={val:.1f}] trace length={total_trace_len:.2f} m, P21={p21:.4f} m^-1")

        if output_dir:
            fig.savefig(os.path.join(output_dir, f'plot_2d_trace_{axis}.png'), dpi=150)
            print(f"  -> Saved: plot_2d_trace_{axis}.png")
        figs.append(fig)
    return figs

# ==============================================================================
# SECTION 3: CORE PIPELINE DRIVER
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Python-based 3D Rock Mass DFN Generator (Forsmark/Laxemar statistics)")
    parser.add_argument("--site", type=str, default="laxemar", choices=["forsmark", "laxemar"], help="Stratigraphic site parameter set")
    parser.add_argument("--output-dir", type=str, default="dfn_output_cube250m", help="Output folder directory")
    parser.add_argument("--rmin", type=float, default=0.5, help="Explicit DFN lower size cutoff resolution")
    parser.add_argument("--rmax", type=float, default=250.0, help="Explicit DFN upper size limit")
    parser.add_argument("--p32-label", default="P32_r_ge_0p5m", help="P32 population label for this generated DFN, e.g. P32_r_ge_0p5m")
    parser.add_argument("--size", type=float, default=250.0, help="Dimension of the model generation box (meters)")
    parser.add_argument("--validate",              action="store_true",  help="Run validation suite: size distribution + stereonet plots (= MATLAB run_validation_suite)")
    parser.add_argument("--no-export",              action="store_false", dest="export",          help="Skip HDF5 export (= MATLAB export_for_python=false)")
    parser.add_argument("--no-overlay-tunnel",      action="store_false", dest="overlay_tunnel",  help="Skip tunnel overlay in plots (= MATLAB overlay_tunnel=false)")
    parser.add_argument("--plot-all-dfn",           action="store_true",  help="Plot all DFN discs in crop box (= MATLAB plot_all_dfn) [not yet implemented]")
    parser.add_argument("--plot-tunnel-intersect",  action="store_true",  help="Plot tunnel-intersecting fractures (= MATLAB plot_tunnel_intersect_dfn) [not yet implemented]")
    parser.add_argument("--plot-2d-trace-map",      action="store_true",  help="Plot 2D trace maps on X/Y/Z slices (= MATLAB plot_2d_trace_map) [not yet implemented]")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (None = random, e.g. 42 for validation)")
    parser.set_defaults(export=True, overlay_tunnel=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Define Model Domain (Generation Box)
    box = {
        'dx': args.size,
        'dy': args.size,
        'dz': args.size,
        'x0': -args.size / 2.0,
        'y0': -args.size / 2.0,
        'z0': -args.size / 2.0
    }
    V = box['dx'] * box['dy'] * box['dz']
    
    # 2. Site configurations
    site = args.site.lower()
    p32_label = args.p32_label or f"P32_r_ge_{str(args.rmin).replace('.', 'p')}m"
    sets: List[Dict[str, Any]] = []
    
    if site == "forsmark":
        sets = [
            {'name': 'Set_1', 'P32': 0.602, 'sizeDist': {'type': 'powerlaw', 'kr': 2.88, 'r0': 0.28, 'rmin': max(args.rmin, 0.28), 'rmax': args.rmax}, 'trend': 87.2, 'plunge': 1.7, 'kappa': 21.66},
            {'name': 'Set_2', 'P32': 2.069, 'sizeDist': {'type': 'powerlaw', 'kr': 3.02, 'r0': 0.25, 'rmin': max(args.rmin, 0.25), 'rmax': args.rmax}, 'trend': 135.2, 'plunge': 2.7, 'kappa': 21.54},
            {'name': 'Set_3', 'P32': 0.448, 'sizeDist': {'type': 'powerlaw', 'kr': 2.81, 'r0': 0.14, 'rmin': max(args.rmin, 0.14), 'rmax': args.rmax}, 'trend': 40.6, 'plunge': 2.2, 'kappa': 23.90},
            {'name': 'Set_4', 'P32': 0.226, 'sizeDist': {'type': 'powerlaw', 'kr': 2.95, 'r0': 0.15, 'rmin': max(args.rmin, 0.15), 'rmax': args.rmax}, 'trend': 190.4, 'plunge': 0.7, 'kappa': 30.63},
            {'name': 'Set_5', 'P32': 0.605, 'sizeDist': {'type': 'powerlaw', 'kr': 2.92, 'r0': 0.25, 'rmin': max(args.rmin, 0.25), 'rmax': args.rmax}, 'trend': 342.9, 'plunge': 80.3, 'kappa': 8.18}
        ]
    elif site == "laxemar":
        sets = [
            {'name': 'Set_1', 'P32': 1.310, 'sizeDist': {'type': 'powerlaw', 'kr': 2.85, 'r0': 0.328, 'rmin': max(args.rmin, 0.328), 'rmax': args.rmax}, 'trend': 338.1, 'plunge': 4.5, 'kappa': 13.06},
            {'name': 'Set_2', 'P32': 1.026, 'sizeDist': {'type': 'powerlaw', 'kr': 3.04, 'r0': 0.977, 'rmin': max(args.rmin, 0.977), 'rmax': args.rmax}, 'trend': 100.4, 'plunge': 0.2, 'kappa': 19.62},
            {'name': 'Set_3', 'P32': 0.975, 'sizeDist': {'type': 'powerlaw', 'kr': 3.01, 'r0': 0.858, 'rmin': max(args.rmin, 0.858), 'rmax': args.rmax}, 'trend': 212.9, 'plunge': 0.9, 'kappa': 10.46},
            {'name': 'Set_4', 'P32': 2.320, 'sizeDist': {'type': 'exponential', 'r0': 4.0, 'rmin': args.rmin, 'rmax': args.rmax}, 'trend': 3.3, 'plunge': 62.1, 'kappa': 10.13},
            {'name': 'Set_5', 'P32': 1.400, 'sizeDist': {'type': 'powerlaw', 'kr': 3.60, 'r0': 0.400, 'rmin': max(args.rmin, 0.400), 'rmax': args.rmax}, 'trend': 243.0, 'plunge': 24.4, 'kappa': 23.52}
        ]

    set_meta_ids = np.array([idx + 1 for idx in range(len(sets))], dtype=np.int32)
    set_table_r0 = np.array([float(seti["sizeDist"].get("r0", np.nan)) for seti in sets], dtype=np.float32)
    set_generation_rmin = np.array([float(seti["sizeDist"]["rmin"]) for seti in sets], dtype=np.float32)
    set_effective_rmin = np.array([float(seti["sizeDist"]["rmin"]) for seti in sets], dtype=np.float32)
        
    print(f"=========================================================")
    print(f" Rock Mass DFN Generator - Python Port ({args.site.upper()})")
    print(f" Box: {args.size}x{args.size}x{args.size} m, Vol: {V:.2e} m3")
    print(f"=========================================================")
    
    # 3. Generate Set-by-Set
    all_centers = []
    all_normals = []
    all_radii = []
    all_set_ids = []
    
    for s_idx, seti in enumerate(sets):
        assert isinstance(seti, dict)
        s_num = s_idx + 1
        dist = seti['sizeDist']
        assert isinstance(dist, dict)
        
        # P32 Scaling Filter to adjust for truncation over generation bounds
        if dist['type'] == 'powerlaw':
            kr = dist['kr']
            rmax = dist['rmax']
            r0 = dist['r0']
            rmin = dist['rmin']
            pow_val = 2.0 - kr
            if np.abs(pow_val) < 1e-12:
                int_r0 = np.log(rmax) - np.log(r0)
                int_rmin = np.log(rmax) - np.log(rmin)
            else:
                int_r0 = (rmax**pow_val - r0**pow_val) / pow_val
                int_rmin = (rmax**pow_val - rmin**pow_val) / pow_val
            target_P32 = seti['P32'] * (int_rmin / int_r0)
        elif dist['type'] == 'exponential':
            lbl = 1.0 / dist['r0']
            rmax = dist['rmax']
            rmin = dist['rmin']
            int_func = lambda r: -np.exp(-lbl * r) * (r**2 + 2 * r / lbl + 2 / (lbl**2))
            int_r0 = int_func(rmax) - int_func(0.0)
            int_rmin = int_func(rmax) - int_func(rmin)
            target_P32 = seti['P32'] * (int_rmin / int_r0)
        else:
            target_P32 = seti['P32']
            
        Ntarget = compute_num_fractures_from_P32(target_P32, dist, V)
        print(f"[*] Set {s_num} ({seti['name']}): Target N = {Ntarget:,} (Scaled P32 = {target_P32:.4f})")
        
        if Ntarget <= 0:
            continue
            
        mean_n = mean_pole_vector_from_trend_plunge(seti['trend'], seti['plunge'])
        
        # Chunked sampling to handle memory limits cleanly
        chunk_size = 2000000
        num_chunks = int(np.ceil(Ntarget / chunk_size))
        
        set_centers = []
        set_normals = []
        set_radii = []
        
        for ch in range(num_chunks):
            n_chunk = min(chunk_size, Ntarget - ch * chunk_size)
            base = None if args.seed is None else args.seed + s_idx * 1000 + ch * 3
            r_ch = sample_radius(dist, n_chunk, seed=None if base is None else base)
            n_ch = sample_fisher_normals(mean_n, seti['kappa'], n_chunk, seed=None if base is None else base + 1)
            strike_u, dip_u = normal_to_strike_dip_basis_vectorized(n_ch)
            c_ch = sample_centers_from_surface_points(box, r_ch, strike_u, dip_u, 'area_uniform', seed=None if base is None else base + 2)
            
            set_centers.append(c_ch)
            set_normals.append(n_ch)
            set_radii.append(r_ch)
            
        set_centers = np.vstack(set_centers)
        set_normals = np.vstack(set_normals)
        set_radii = np.concatenate(set_radii)
        
        realized_P32 = np.sum(np.pi * set_radii**2) / V
        print(f"  -> Realized P32: {realized_P32:.4f} m2/m3")
        
        all_centers.append(set_centers)
        all_normals.append(set_normals)
        all_radii.append(set_radii)
        all_set_ids.append(np.full(len(set_radii), s_num, dtype=np.uint16))
        
    all_centers = np.vstack(all_centers).astype(np.float32)
    all_normals = np.vstack(all_normals).astype(np.float32)
    all_radii = np.concatenate(all_radii).astype(np.float32)
    all_set_ids = np.concatenate(all_set_ids).astype(np.uint16)
    
    total_N = len(all_radii)
    print(f"\n=========================================")
    print(f" DFN Generation Finished. Total Fractures: {total_N:,}")
    print(f"=========================================")
    
    # 4. Load Tunnel Polygon YZ
    tunnel_poly_YZ = None
    tunnel_Y = None
    tunnel_Z = None
    
    # Search for the tunnel polygon DAT from the repo root first, then fall back to nearby locations.
    tunnel_candidates = [
        os.path.join(_project_root, "storage", "data", "단면_폴리곤.dat"),
        os.path.join(_parent, "storage", "data", "단면_폴리곤.dat"),
        os.path.join(_parent, "단면_폴리곤.dat"),
        os.path.join(os.getcwd(), "storage", "data", "단면_폴리곤.dat"),
        os.path.join(os.getcwd(), "단면_폴리곤.dat"),
    ]
    tunnel_file = next((path for path in tunnel_candidates if os.path.exists(path)), None)

    if tunnel_file is not None:
        print(f"[*] Parsing tunnel file: {tunnel_file}")
        poly = read_tunnel_polygon(tunnel_file)
        if poly is not None:
            # Shift Z coordinates for sync exactly as in MATLAB
            poly_shifted = poly.copy()
            poly_shifted[:, 1] = poly_shifted[:, 1] - 4.0
            
            # Close the polygon loop
            poly_shifted = np.vstack([poly_shifted, poly_shifted[0]])
            tunnel_Y = poly_shifted[:, 0].astype(np.float32)
            tunnel_Z = poly_shifted[:, 1].astype(np.float32)
            tunnel_poly_YZ = poly_shifted.astype(np.float32)
            
    # 5. Define crop box and domain box
    # Domain boundary box calculation with 5m margins
    xmin_d = float(np.min(all_centers[:, 0] - all_radii) - 5.0)
    xmax_d = float(np.max(all_centers[:, 0] + all_radii) + 5.0)
    
    if tunnel_poly_YZ is not None:
        ymin_d = float(np.min(tunnel_poly_YZ[:, 0]) - 15.0)
        ymax_d = float(np.max(tunnel_poly_YZ[:, 0]) + 15.0)
        zmin_d = float(np.min(tunnel_poly_YZ[:, 1]) - 15.0)
        zmax_d = float(np.max(tunnel_poly_YZ[:, 1]) + 15.0)
    else:
        ymin_d = float(np.min(all_centers[:, 1] - all_radii) - 5.0)
        ymax_d = float(np.max(all_centers[:, 1] + all_radii) + 5.0)
        zmin_d = float(np.min(all_centers[:, 2] - all_radii) - 5.0)
        zmax_d = float(np.max(all_centers[:, 2] + all_radii) + 5.0)
        
    domain_box = np.array([xmin_d, xmax_d, ymin_d, ymax_d, zmin_d, zmax_d], dtype=np.float32)
    
    # 50x50x50m Central crop box for downstream block detection
    cx = 50.0
    crop_box = np.array([-cx/2, cx/2, -cx/2, cx/2, -cx/2, cx/2], dtype=np.float32)
    # Crop box as dict for geometry clipping functions
    cb = {'xmin': float(crop_box[0]), 'xmax': float(crop_box[1]),
          'ymin': float(crop_box[2]), 'ymax': float(crop_box[3]),
          'zmin': float(crop_box[4]), 'zmax': float(crop_box[5])}

    # 6. Save HDF5  (= MATLAB export_for_python)
    if args.export:
        out_hdf5_file = os.path.join(args.output_dir, "dfn_export_for_python.h5")
        print(f"[*] Exporting Python HDF5 to: {out_hdf5_file}")
        
        # Save to temp path first to prevent file locks (same as MATLAB lock prevention)
        import tempfile
        temp_fd, temp_path = tempfile.mkstemp(suffix=".h5")
        os.close(temp_fd)
        
        with h5py.File(temp_path, 'w') as f:
            # Fractures group
            f.create_dataset('/fractures/centers', data=all_centers, dtype=np.float32)
            f.create_dataset('/fractures/normals', data=all_normals, dtype=np.float32)
            f.create_dataset('/fractures/radii', data=all_radii.reshape(-1, 1), dtype=np.float32)
            f.create_dataset('/fractures/set_id', data=all_set_ids.reshape(-1, 1), dtype=np.uint16)
            
            # Tunnel group
            if tunnel_poly_YZ is not None:
                f.create_dataset('/tunnel/poly_YZ', data=tunnel_poly_YZ, dtype=np.float32)
                f.create_dataset('/tunnel/profile_Y', data=tunnel_Y, dtype=np.float32)
                f.create_dataset('/tunnel/profile_Z', data=tunnel_Z, dtype=np.float32)
                
            # Meta group
            f.create_dataset('/meta/domain_box', data=domain_box.reshape(1, 6), dtype=np.float32)
            f.create_dataset('/meta/crop_box', data=crop_box.reshape(1, 6), dtype=np.float32)
            f.create_dataset('/meta/generation_rmin', data=np.array([args.rmin], dtype=np.float32))
            f.create_dataset('/meta/generation_rmax', data=np.array([args.rmax], dtype=np.float32))
            f.create_dataset('/meta/set_ids', data=set_meta_ids, dtype=np.int32)
            f.create_dataset('/meta/set_table_r0', data=set_table_r0, dtype=np.float32)
            f.create_dataset('/meta/set_generation_rmin', data=set_generation_rmin, dtype=np.float32)
            f.create_dataset('/meta/set_effective_rmin', data=set_effective_rmin, dtype=np.float32)
            f.create_dataset('/meta/p32_label', data=np.bytes_(p32_label))
            f.create_dataset('/meta/site', data=np.bytes_(site))
            f.create_dataset('/meta/powerlaw_pdf_convention', data=np.bytes_("f_R(r) proportional to r^-(kr+1)"))
            f.create_dataset('/meta/powerlaw_survival_convention', data=np.bytes_("S_R(r) proportional to r^-kr"))
            f.create_dataset('/meta/kr_parameter_interpretation', data=np.bytes_("survival exponent"))
            
            # File attributes
            f.attrs['created_by'] = 'generate_dfn.py'
            f.attrs['python_version'] = sys.version
            f.attrs['num_fractures'] = total_N
            
        if os.path.exists(out_hdf5_file):
            os.remove(out_hdf5_file)
        os.replace(temp_path, out_hdf5_file)
        print("  [OK] Export complete.")
    else:
        print("[*] Skipping HDF5 export (--no-export).")
    
    # 7. Visualization (= MATLAB plot_* opts)
    if args.plot_all_dfn:
        print("[*] Generating 3D DFN plot (all fractures in crop box)...")
        plot_clipped_dfn_3d(all_centers, all_normals, all_radii, all_set_ids, cb,
                             filter_by_tunnel=False,
                             tunnel_poly_YZ=tunnel_poly_YZ,
                             overlay_tunnel=args.overlay_tunnel,
                             output_dir=args.output_dir)

    if args.plot_tunnel_intersect:
        if tunnel_poly_YZ is None:
            print("[WARNING] --plot-tunnel-intersect: no tunnel polygon loaded.")
        else:
            print("[*] Generating 3D DFN plot (tunnel-intersecting fractures)...")
            plot_clipped_dfn_3d(all_centers, all_normals, all_radii, all_set_ids, cb,
                                 filter_by_tunnel=True,
                                 tunnel_poly_YZ=tunnel_poly_YZ,
                                 overlay_tunnel=args.overlay_tunnel,
                                 output_dir=args.output_dir)

    if args.plot_2d_trace_map:
        print("[*] Generating 2D trace maps (X=0, Y=0, Z=0)...")
        plot_2d_trace_maps(all_centers, all_normals, all_radii, all_set_ids, cb,
                            tunnel_poly_YZ=tunnel_poly_YZ,
                            overlay_tunnel=args.overlay_tunnel,
                            output_dir=args.output_dir)

    # 8. V&V validation suite  (= MATLAB run_validation_suite)
    if args.validate:
        print("[*] Running synthetic basis vector test:")
        test_basis = {
            'East (+X)': np.array([1.0, 0.0, 0.0]),
            'North (+Y)': np.array([0.0, 1.0, 0.0]),
            'Up (+Z)': np.array([0.0, 0.0, 1.0]),
            'West (-X)': np.array([-1.0, 0.0, 0.0]),
            'South (-Y)': np.array([0.0, -1.0, 0.0])
        }
        for name, vec in test_basis.items():
            vl = vec.copy()
            if vl[2] > 0:
                vl = -vl
            denom = 1.0 - vl[2]
            # Standard Wulff Net Equal-Angle Projection
            tx = vl[0] / denom
            ty = vl[1] / denom
            print(f"  {name:12} -> n_lower=[{vl[0]:.1f}, {vl[1]:.1f}, {vl[2]:.1f}] -> X_proj={tx:.2f}, Y_proj={ty:.2f}")

        print("[*] Generating verification plots...")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for s_idx, seti in enumerate(sets):
            assert isinstance(seti, dict)
            s_num = s_idx + 1
            mask = all_set_ids == s_num
            r = all_radii[mask]
            
            # Subplot size distribution Log-Log
            ax = axes[s_idx]
            valid_r = (r > 1.0) & (r < 250.0)
            r_filtered = r[valid_r]
            if len(r_filtered) > 0:
                counts, edges = np.histogram(np.log10(r_filtered), bins=50)
                bin_centers = 0.5 * (edges[:-1] + edges[1:])
                valid = counts > 0
                ax.scatter(bin_centers[valid], np.log10(counts[valid]), label="Generated", s=10)
                
                # Theoretical slope
                dist = seti['sizeDist']
                assert isinstance(dist, dict)
                if dist['type'] == 'powerlaw':
                    kr = float(dist['kr'])
                    C = np.log10(counts[valid][0]) + kr * bin_centers[valid][0]
                    y_theory = -kr * bin_centers[valid] + C
                    ax.plot(bin_centers[valid], y_theory, 'r-', label=f"Theoretical (kr={kr:.2f})")
                ax.legend()
                ax.grid(True)
                ax.set_title(f"{seti['name']} Size Distribution")
                ax.set_xlabel("log10(Radius) [m]")
                ax.set_ylabel("log10(Count)")
                
        # Stereonet densities
        fig_st, ax_st = plt.subplots(figsize=(8, 8))
        ax_st.plot(np.cos(np.linspace(0, 2*np.pi, 100)), np.sin(np.linspace(0, 2*np.pi, 100)), 'k-', lw=1.5)
        
        # Convert normals to lower hemisphere and project to equal-angle Wulff net
        n = all_normals.copy()
        flip = n[:, 2] > 0
        n[flip] = -n[flip]
        
        # Equal-angle projection (Standard Wulff Net: X_proj=East/nx, Y_proj=North/ny)
        X_proj = n[:, 0] / (1.0 - n[:, 2])
        Y_proj = n[:, 1] / (1.0 - n[:, 2])
        
        # 2D Histogram contour
        counts, xedges, yedges = np.histogram2d(X_proj, Y_proj, bins=80, range=[[-1.0, 1.0], [-1.0, 1.0]])
        # Smooth using simple box filter or gaussian filter (from scipy)
        from scipy.ndimage import gaussian_filter
        density = gaussian_filter(counts, sigma=2.0)
        
        # Mask outside unit circle
        GX, GY = np.meshgrid(0.5*(xedges[:-1]+xedges[1:]), 0.5*(yedges[:-1]+yedges[1:]))
        mask = GX**2 + GY**2 <= 1.0
        density[~mask.T] = np.nan
        
        cp = ax_st.contourf(GX, GY, density.T, levels=12, cmap='jet')
        fig_st.colorbar(cp, label='Pole Density Count')
        
        # Plot mean expected pole locations
        for s_idx, seti in enumerate(sets):
            mean_n = mean_pole_vector_from_trend_plunge(seti['trend'], seti['plunge'])
            if mean_n[2] > 0:
                mean_n = -mean_n
            mX = mean_n[0] / (1.0 - mean_n[2])
            mY = mean_n[1] / (1.0 - mean_n[2])
            ax_st.plot(mX, mY, 'rp', ms=12, mec='black')
            ax_st.text(mX + 0.05, mY + 0.05, f"Set {s_idx+1}", color='red', weight='bold')
            
        ax_st.set_xlim((-1.1, 1.1))
        ax_st.set_ylim((-1.1, 1.1))
        ax_st.text(0, 1.05, 'N', ha='center', va='bottom', weight='bold', fontsize=12)
        ax_st.text(1.05, 0, 'E', ha='left', va='center', weight='bold', fontsize=12)
        ax_st.text(0, -1.05, 'S', ha='center', va='top', weight='bold', fontsize=12)
        ax_st.text(-1.05, 0, 'W', ha='right', va='center', weight='bold', fontsize=12)
        ax_st.axis('off')
        ax_st.set_title("Combined DFN Stereonet Equal-Angle Density Projection")
        
        # Save plots
        size_plot_path = os.path.join(args.output_dir, "vv_size_distribution.png")
        stereo_plot_path = os.path.join(args.output_dir, "vv_stereonet_density.png")
        fig.savefig(size_plot_path)
        fig_st.savefig(stereo_plot_path)
        print(f"  -> Saved verification plots to: {args.output_dir}")

    if args.plot_all_dfn or args.plot_tunnel_intersect or args.plot_2d_trace_map or args.validate:
        plt.show()

if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    _parent = os.path.dirname(_here)
    _project_root = os.path.dirname(_parent)
    main()
