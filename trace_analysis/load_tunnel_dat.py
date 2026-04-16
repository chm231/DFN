import re
import numpy as np

def load_tunnel_polygon_from_dat(dat_path, scale=0.001):
    """
    Loads tunnel polygon from .dat file.
    Assumes mm unit in .dat and converts to meters by default (scale=0.001).
    Maps Dat X -> Python Y, Dat Y -> Python Z.
    """
    poly_y = []
    poly_z = []
    
    with open(dat_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    start_parsing = False
    for line in lines:
        line = line.strip()
        if '# 2차원 좌표' in line:
            start_parsing = True
            continue
        
        if start_parsing:
            # Match ( 5038.262036,     0.000000)
            match = re.search(r'\(\s*([\d\.-]+),\s*([\d\.-]+)\)', line)
            if match:
                y_val = float(match.group(1)) * scale
                z_val = float(match.group(2)) * scale
                poly_y.append(y_val)
                poly_z.append(z_val)
                
    return np.array(poly_y), np.array(poly_z)

if __name__ == "__main__":
    import os
    # Relative path test
    dat_file = r"c:\Users\user\OneDrive\2026-1\3D DFN modeling\storage\data\단면_폴리곤.dat"
    if os.path.exists(dat_file):
        py, pz = load_tunnel_polygon_from_dat(dat_file)
        print(f"Loaded {len(py)} points.")
        print(f"Sample Y: {py[:3]}")
        print(f"Sample Z: {pz[:3]}")
    else:
        print(f"File not found: {dat_file}")
