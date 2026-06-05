import os
import sys
import numpy as np
import pandas as pd
import pyvista as pv

# Set local imports
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _here not in sys.path:
    sys.path.insert(0, _here)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from load_tunnel_dat import load_tunnel_polygon_from_dat
from rough_face.generator import RoughFace

def get_rough_trace_points(p0, p1, rough_face, num_pts=30):
    """Line segment interpolation followed by projection to rough face to follow topography."""
    ys = np.linspace(p0[1], p1[1], num_pts)
    zs = np.linspace(p0[2], p1[2], num_pts)
    pts_yz = np.column_stack((ys, zs))
    pts_3d = rough_face.project_to_face(pts_yz)
    return pts_3d

def main():
    tunnel_dat = "storage/data/단면_폴리곤.dat"
    csv_path = "storage/output/ground_truth_traces.csv"
    output_dir = "storage/output/rough_face_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load tunnel boundary
    print(f"[*] Loading tunnel polygon: {tunnel_dat}")
    poly_y, poly_z = load_tunnel_polygon_from_dat(tunnel_dat)
    poly_yz = np.column_stack([poly_y, poly_z])
    
    # Determine bounds for RoughFace grid
    y_min, y_max = np.min(poly_yz[:, 0]), np.max(poly_yz[:, 0])
    z_min, z_max = np.min(poly_yz[:, 1]), np.max(poly_yz[:, 1])
    pad = 1.0
    y_range = (y_min - pad, y_max + pad)
    z_range = (z_min - pad, z_max + pad)
    
    # 2. Load traces
    print(f"[*] Loading traces: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Parameters used for generation
    x_positions = [0.0, 3.0, 6.0, 9.0]
    res = 0.05
    dx = 0.2
    lc = 1.0
    
    # Color map for set_ids
    color_map = {
        1: "#FF3B30",  # Apple Red
        2: "#34C759",  # Apple Green
        3: "#007AFF",  # Apple Blue
        4: "#FF9500",  # Apple Orange
        5: "#AF52DE",  # Apple Purple
    }
    default_color = "#8E8E93"
    
    # Process each face
    for idx, x_pos in enumerate(x_positions):
        face_id = idx + 1
        print(f"[*] Visualizing Face {face_id} at x={x_pos}m...")
        
        # Recreate the exact same RoughFace object
        rough_face = RoughFace(
            base_x=x_pos,
            y_range=y_range,
            z_range=z_range,
            resolution=res,
            amplitude=dx,
            correlation_length=lc,
            seed=42 + idx
        )
        
        # Filter traces for this face
        face_df = df[df['face_id'] == face_id]
        
        # Initialize PyVista Plotter
        p = pv.Plotter(off_screen=True, window_size=[1200, 900])
        p.set_background("#1C1C1E")  # Premium dark mode background
        
        # A. Add Rough Face Grid
        grid = pv.StructuredGrid(rough_face.X, rough_face.Y, rough_face.Z)
        p.add_mesh(
            grid, 
            color="#3A3A3C", 
            opacity=0.75, 
            show_edges=False, 
            lighting=True,
            smooth_shading=True,
            label="Rough Excavation Face"
        )
        
        # B. Add Tunnel Boundary Polyline (extruding slightly or just displaying at base_x)
        # Project tunnel polygon onto the rough face to make it look embedded
        tunnel_pts_3d = rough_face.project_to_face(poly_yz)
        tunnel_poly = pv.PolyData(tunnel_pts_3d)
        tunnel_lines = np.arange(len(poly_yz))
        tunnel_poly.lines = np.hstack(([len(poly_yz)], tunnel_lines))
        p.add_mesh(
            tunnel_poly, 
            color="#FFD60A",  # Yellow
            line_width=3, 
            opacity=0.9, 
            label="Tunnel Boundary"
        )
        
        # C. Add Traces colored by set_id
        added_labels = set()
        for _, row in face_df.iterrows():
            p0 = np.array([row['p0_x'], row['p0_y'], row['p0_z']])
            p1 = np.array([row['p1_x'], row['p1_y'], row['p1_z']])
            set_id = int(row['set_id'])
            
            # Get smooth polyline tracing the rough surface topography
            pts_3d = get_rough_trace_points(p0, p1, rough_face, num_pts=20)
            
            poly = pv.PolyData(pts_3d)
            lines = np.arange(len(pts_3d))
            poly.lines = np.hstack(([len(pts_3d)], lines))
            
            color = color_map.get(set_id, default_color)
            label = f"Fracture Set {set_id}" if set_id not in added_labels else None
            if label:
                added_labels.add(set_id)
                
            p.add_mesh(
                poly, 
                color=color, 
                line_width=6, 
                render_lines_as_tubes=True,
                label=label
            )
            
        p.add_legend(bcolor=None)
        p.add_axes(color="white")
        
        # Set camera position for a nice 3D perspective
        # Camera focuses on the center of the tunnel face
        center_y = (y_min + y_max) / 2.0
        center_z = (z_min + z_max) / 2.0
        p.camera_position = [
            (x_pos - 15.0, center_y - 2.0, center_z + 8.0), # Camera position
            (x_pos, center_y, center_z),                    # Focal point
            (0, 0, 1)                                       # View up
        ]
        
        save_path = os.path.join(output_dir, f"face_{face_id}_x_{int(x_pos)}m.png")
        p.screenshot(save_path)
        p.close()
        print(f"  -> Saved visualization to {save_path}")
        
    print("[*] All face visualizations generated successfully!")

if __name__ == "__main__":
    main()
