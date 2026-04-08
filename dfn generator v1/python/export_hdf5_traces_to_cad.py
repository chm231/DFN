"""
export_hdf5_traces_to_cad.py
HDF5 데이터에서 직접 3D Trace 및 터널 형상을 추출하여 CAD 포맷(SCR, DXF, CSV)으로 저장합니다.

사용자 지정 조건:
- 간격(dx): 3.0m
- 클리핑: 터널 단면(poly_YZ) 내부만 추출
- 터널 형상: 각 단면 위치의 프로파일(Closed Loop) 포함
"""

import os
import argparse
import numpy as np
import h5py
import pandas as pd
from shapely.geometry import LineString, Polygon

def load_hdf5_data(h5_path):
    print(f"[Info] HDF5 loading: {h5_path}")
    data = {}
    with h5py.File(h5_path, 'r') as f:
        raw_c = f['/fractures/centers'][:]
        raw_n = f['/fractures/normals'][:]
        data['centers'] = raw_c.T if raw_c.shape[0] == 3 and raw_c.shape[0] < raw_c.shape[1] else raw_c
        data['normals'] = raw_n.T if raw_n.shape[0] == 3 and raw_n.shape[0] < raw_n.shape[1] else raw_n
        data['radii']   = f['/fractures/radii'][:].ravel()
        
        if '/tunnel/poly_YZ' in f:
            raw_p = f['/tunnel/poly_YZ'][:]
            data['poly_YZ'] = raw_p.T if raw_p.shape[0] == 2 and raw_p.shape[0] < raw_p.shape[1] else raw_p
        else:
            data['poly_YZ'] = None
            
        if '/meta/crop_box' in f:
            data['crop_box'] = f['/meta/crop_box'][:].ravel()
        elif '/meta/domain_box' in f:
            data['crop_box'] = f['/meta/domain_box'][:].ravel()
        else:
            data['crop_box'] = None
    return data

def extract_raw_traces(centers, normals, radii, x_slice):
    """
    해석학적 방정식으로 3D 균열(원판)과 x = x_slice 평면의 교차선(Trace)을 계산합니다.
    (plot_2d_trace_map.py 의 logic과 동일)
    """
    dx = centers[:, 0] - x_slice
    valid = np.abs(dx) <= radii
    
    if not np.any(valid):
        return np.zeros((0, 2, 2))
        
    cy = centers[valid, 1]
    cz = centers[valid, 2]
    nx = normals[valid, 0]
    ny = normals[valid, 1]
    nz = normals[valid, 2]
    r  = radii[valid]
    dx = dx[valid]
    
    s2 = ny**2 + nz**2
    s  = np.sqrt(s2)
    
    non_zero = s > 1e-6
    if not np.any(non_zero): return np.zeros((0, 2, 2))
    
    cy = cy[non_zero]; cz = cz[non_zero]; nx = nx[non_zero]
    ny = ny[non_zero]; nz = nz[non_zero]; r  = r[non_zero]
    dx = dx[non_zero]; s  = s[non_zero];  s2 = s2[non_zero]
    
    d = np.abs(dx) / s
    intersect = d <= r
    
    if not np.any(intersect): return np.zeros((0, 2, 2))
    
    cy = cy[intersect]; cz = cz[intersect]; nx = nx[intersect]
    ny = ny[intersect]; nz = nz[intersect]; r  = r[intersect]
    dx = dx[intersect]; s  = s[intersect];  d  = d[intersect]
    s2 = s2[intersect]
    
    L = np.sqrt(r**2 - d**2)
    My = cy + dx * (nx * ny) / s2
    Mz = cz + dx * (nx * nz) / s2
    uy = nz / s
    uz = -ny / s
    
    segments = np.zeros((len(My), 2, 2))
    segments[:, 0, 0] = My + L * uy
    segments[:, 0, 1] = Mz + L * uz
    segments[:, 1, 0] = My - L * uy
    segments[:, 1, 1] = Mz - L * uz
    
    return segments

def clip_to_tunnel(segments_yz, tunnel_poly_yz):
    """터널 폴리곤 내부만 살리고 나머지는 잘라냄"""
    if tunnel_poly_yz is None:
        return segments_yz
        
    poly = Polygon(tunnel_poly_yz)
    clipped = []
    
    for seg in segments_yz:
        line = LineString(seg)
        if poly.intersects(line):
            inter = poly.intersection(line)
            if inter.geom_type == 'LineString':
                clipped.append(list(inter.coords))
            elif inter.geom_type == 'MultiLineString':
                for geom in inter.geoms:
                    clipped.append(list(geom.coords))
    return clipped

def main():
    parser = argparse.ArgumentParser(description="HDF5 to 3D CAD Trace Exporter (Clipped to Tunnel)")
    parser.add_argument('--input', required=True, help="HDF5 파일 경로")
    parser.add_argument('--dx', type=float, default=3.0, help="추출 간격 (m)")
    parser.add_argument('--outdir', default="cad_export_final", help="저장 디렉토리")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    data = load_hdf5_data(args.input)
    
    centers, normals, radii = data['centers'], data['normals'], data['radii']
    tunnel_poly = data['poly_YZ']
    crop = data['crop_box']
    
    xmin, xmax = (crop[0], crop[1]) if crop is not None else (np.min(centers[:,0]), np.max(centers[:,0]))
    
    # 3m 간격 X 좌표 리스트
    x_positions = np.arange(xmin, xmax + 1e-5, args.dx)
    print(f"[Info] Extracting traces for {len(x_positions)} slices. (X: {xmin:.1f} ~ {xmax:.1f})")

    all_trace_lines = [] # List of ((x1,y1,z1), (x2,y2,z2))
    all_tunnel_lines = []

    for x_pos in x_positions:
        # 1. Trace 추출 및 클리핑
        raw_segs = extract_raw_traces(centers, normals, radii, x_pos)
        clipped_segs = clip_to_tunnel(raw_segs, tunnel_poly)
        
        for seg in clipped_segs:
            # seg is a list of coords [(y1,z1), (y2,z2)]
            all_trace_lines.append(((x_pos, seg[0][0], seg[0][1]), (x_pos, seg[1][0], seg[1][1])))
            
        # 2. 터널 단면 생성 (Profile)
        if tunnel_poly is not None:
            for i in range(len(tunnel_poly)):
                p1 = tunnel_poly[i]
                p2 = tunnel_poly[(i+1) % len(tunnel_poly)] # Close the loop
                all_tunnel_lines.append(((x_pos, p1[0], p1[1]), (x_pos, p2[0], p2[1])))

    # --- 💾 파일 저장 ---
    # 1. CSV 저장
    csv_data = []
    for (p1, p2) in all_trace_lines:
        csv_data.append({'type': 'Trace', 'x1': p1[0], 'y1': p1[1], 'z1': p1[2], 'x2': p2[0], 'y2': p2[1], 'z2': p2[2]})
    for (p1, p2) in all_tunnel_lines:
        csv_data.append({'type': 'Tunnel', 'x1': p1[0], 'y1': p1[1], 'z1': p1[2], 'x2': p2[0], 'y2': p2[1], 'z2': p2[2]})
    
    csv_path = os.path.join(args.outdir, "extracted_3d_data.csv")
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"[Result] CSV Exported: {csv_path}")

    # 2. AutoCAD Script (SCR) 저장
    scr_path = os.path.join(args.outdir, "draw_3d_cad.scr")
    with open(scr_path, 'w') as f:
        # Layer 설정
        f.write("-LAYER\nMAKE\nTunnel_Profile\nCOLOR\n8\nTunnel_Profile\n\n") # Color 8: Gray
        f.write("-LAYER\nMAKE\nFracture_Traces\nCOLOR\n1\nFracture_Traces\n\n") # Color 1: Red
        
        # 터널 그리기
        f.write("-LAYER\nSET\nTunnel_Profile\n\n")
        for (p1, p2) in all_tunnel_lines:
            f.write(f"LINE\n{p1[0]},{p1[1]},{p1[2]}\n{p2[0]},{p2[1]},{p2[2]}\n\n")
            
        # Trace 그리기
        f.write("-LAYER\nSET\nFracture_Traces\n\n")
        for (p1, p2) in all_trace_lines:
            f.write(f"LINE\n{p1[0]},{p1[1]},{p1[2]}\n{p2[0]},{p2[1]},{p2[2]}\n\n")
            
        f.write("ZOOM\nEXTENTS\n")
    print(f"[Result] AutoCAD SCR Exported: {scr_path}")

    # 3. DXF 저장 (Optional)
    try:
        import ezdxf
        dxf_path = os.path.join(args.outdir, "extracted_3d_data.dxf")
        doc = ezdxf.new('R2010')
        doc.layers.new('Tunnel_Profile', dxfattribs={'color': 8})
        doc.layers.new('Fracture_Traces', dxfattribs={'color': 1})
        msp = doc.modelspace()
        
        for (p1, p2) in all_tunnel_lines:
            msp.add_line(p1, p2, dxfattribs={'layer': 'Tunnel_Profile'})
        for (p1, p2) in all_trace_lines:
            msp.add_line(p1, p2, dxfattribs={'layer': 'Fracture_Traces'})
            
        doc.saveas(dxf_path)
        print(f"[Result] DXF Exported: {dxf_path}")
    except ImportError:
        print("[Info] 'ezdxf' package not found. Skipping DXF export.")

    print(f"\n[Done] All tasks completed! Files located at: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
