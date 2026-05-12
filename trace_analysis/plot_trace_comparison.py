"""
Side-by-side 2D trace comparison utility (Observed vs Simulated).
Enables direct visual comparison of trace distribution, counts, and orientations per excavation face.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from trace_reconstruction.trace_types import FaceTrace, ExcavationFace


def plot_side_by_side_trace_comparison(
    obs_traces: List[FaceTrace],
    sim_traces: List[FaceTrace],
    faces: List[ExcavationFace],
    save_path: str
):
    """
    Generates and saves a high-fidelity, premium side-by-side trace comparison plot.
    Left column: Observed traces on each face.
    Right column: Simulated/Reconstructed traces on each face.
    """
    n_faces = len(faces)
    if n_faces == 0:
        return
        
    fig, axes = plt.subplots(n_faces, 2, figsize=(14, 6 * n_faces), squeeze=False)
    
    # Elegant color palette for fracture sets (vibrant, modern colors)
    set_colors = {
        1: '#FF5733',  # Vibrant Red/Orange
        2: '#33FF57',  # Vibrant Green
        3: '#3357FF',  # Vibrant Blue
        4: '#F3FF33',  # Yellow
        5: '#FF33F3',  # Magenta
    }
    fallback_color = '#8E44AD'  # Purple
    
    for idx, face in enumerate(faces):
        f_id = face.face_id
        poly = face.tunnel_polygon_yz
        
        face_obs = [t for t in obs_traces if t.face_id == f_id]
        face_sim = [t for t in sim_traces if t.face_id == f_id]
        
        # --- LEFT PANEL: OBSERVED ---
        ax_obs = axes[idx, 0]
        # Plot closed tunnel polygon boundary
        poly_closed = np.vstack([poly, poly[0]])
        ax_obs.plot(poly_closed[:, 0], poly_closed[:, 1], color='#2C3E50', lw=2.5, ls='--', label='Tunnel Boundary')
        
        p21_obs = sum(t.length for t in face_obs)
        for t in face_obs:
            c = set_colors.get(t.set_id or 1, fallback_color)
            ax_obs.plot([t.p0_y, t.p1_y], [t.p0_z, t.p1_z], color=c, lw=2.0)
            
        ax_obs.set_title(f"Face {f_id} (x = {face.x_face:.1f}m) - Observed\n(Count: {len(face_obs)} | P21: {p21_obs:.2f}m)", fontsize=12, fontweight='bold', color='#2C3E50')
        ax_obs.set_xlabel("Y (m)", fontsize=10)
        ax_obs.set_ylabel("Z (m)", fontsize=10)
        ax_obs.grid(True, linestyle=':', alpha=0.6)
        ax_obs.set_aspect('equal', 'box')
        
        # --- RIGHT PANEL: SIMULATED / RECONSTRUCTED ---
        ax_sim = axes[idx, 1]
        ax_sim.plot(poly_closed[:, 0], poly_closed[:, 1], color='#2C3E50', lw=2.5, ls='--', label='Tunnel Boundary')
        
        p21_sim = sum(t.length for t in face_sim)
        for t in face_sim:
            c = set_colors.get(t.set_id or 1, fallback_color)
            ax_sim.plot([t.p0_y, t.p1_y], [t.p0_z, t.p1_z], color=c, lw=2.0)
            
        ax_sim.set_title(f"Face {f_id} (x = {face.x_face:.1f}m) - Simulated\n(Count: {len(face_sim)} | P21: {p21_sim:.2f}m)", fontsize=12, fontweight='bold', color='#2C3E50')
        ax_sim.set_xlabel("Y (m)", fontsize=10)
        ax_sim.set_ylabel("Z (m)", fontsize=10)
        ax_sim.grid(True, linestyle=':', alpha=0.6)
        ax_sim.set_aspect('equal', 'box')
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [Visualizer Complete] Saved side-by-side comparison to: {save_path}")


def plot_overlay_trace_comparison(
    obs_traces: List[FaceTrace],
    sim_traces: List[FaceTrace],
    faces: List[ExcavationFace],
    save_path: str
):
    """
    Generates and saves a high-fidelity overlay comparison plot.
    Observed traces: Red
    Simulated/Reconstructed traces: Blue
    Superimposed on a single plot per face.
    """
    n_faces = len(faces)
    if n_faces == 0:
        return
        
    fig, axes = plt.subplots(n_faces, 1, figsize=(8, 6 * n_faces), squeeze=False)
    
    for idx, face in enumerate(faces):
        f_id = face.face_id
        poly = face.tunnel_polygon_yz
        
        face_obs = [t for t in obs_traces if t.face_id == f_id]
        face_sim = [t for t in sim_traces if t.face_id == f_id]
        
        ax = axes[idx, 0]
        # Plot closed tunnel polygon boundary
        poly_closed = np.vstack([poly, poly[0]])
        ax.plot(poly_closed[:, 0], poly_closed[:, 1], color='#2C3E50', lw=2.5, ls='--', label='Tunnel Boundary')
        
        p21_obs = sum(t.length for t in face_obs)
        p21_sim = sum(t.length for t in face_sim)
        
        # Plot Observed traces (Red)
        for i, t in enumerate(face_obs):
            label = 'Observed Traces' if i == 0 else ""
            ax.plot([t.p0_y, t.p1_y], [t.p0_z, t.p1_z], color='#FF3333', lw=2.0, alpha=0.8, label=label)
            
        # Plot Simulated traces (Blue)
        for i, t in enumerate(face_sim):
            label = 'Simulated Traces' if i == 0 else ""
            ax.plot([t.p0_y, t.p1_y], [t.p0_z, t.p1_z], color='#3333FF', lw=2.0, alpha=0.7, label=label)
            
        ax.set_title(f"Face {f_id} (x = {face.x_face:.1f}m) - Overlay Comparison\n"
                     f"Obs: {len(face_obs)} Traces (P21={p21_obs:.2f}m) | "
                     f"Sim: {len(face_sim)} Traces (P21={p21_sim:.2f}m)", 
                     fontsize=12, fontweight='bold', color='#2C3E50')
        ax.set_xlabel("Y (m)", fontsize=10)
        ax.set_ylabel("Z (m)", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_aspect('equal', 'box')
        ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#BDC3C7', fontsize=9)
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [Visualizer Complete] Saved overlay comparison to: {save_path}")
