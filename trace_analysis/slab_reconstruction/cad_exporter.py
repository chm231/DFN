import os
import numpy as np
from typing import List, Tuple
from .slab_types import Slab

def export_slabs_and_traces_to_cad(
    slabs: List[Slab], 
    slab_all_segments: List[List[np.ndarray]], 
    tunnel_poly_yz: np.ndarray,
    scr_path: str
):
    """
    Slab 경계 박스와 내부 Trace들을 AutoCAD SCR 파일로 저장
    """
    os.makedirs(os.path.dirname(scr_path), exist_ok=True)
    
    with open(scr_path, 'w') as f:
        # Layer 설정
        f.write("-LAYER\nMAKE\nSlab_Boundaries\nCOLOR\n5\nSlab_Boundaries\n\n") # Blue
        f.write("-LAYER\nMAKE\nSlab_Traces\nCOLOR\n1\nSlab_Traces\n\n") # Red
        
        # 1. Slab Boundaries (Bounding Boxes based on Tunnel Polygon)
        if tunnel_poly_yz is not None:
            f.write("-LAYER\nSET\nSlab_Boundaries\n\n")
            for slab in slabs:
                # Slab의 8개 꼭짓점 계산 (X_min, X_max에 대해 Polygon 복제)
                x_coords = [slab.x_min, slab.x_max]
                
                # 가이드 선분: X_min 단면 폴리곤
                for i in range(len(tunnel_poly_yz)):
                    p_start = tunnel_poly_yz[i]
                    p_end = tunnel_poly_yz[(i + 1) % len(tunnel_poly_yz)]
                    
                    # X_min 단면
                    f.write("LINE\n")
                    f.write(f"{slab.x_min},{p_start[0]},{p_start[1]}\n")
                    f.write(f"{slab.x_min},{p_end[0]},{p_end[1]}\n\n")
                    
                    # X_max 단면
                    f.write("LINE\n")
                    f.write(f"{slab.x_max},{p_start[0]},{p_start[1]}\n")
                    f.write(f"{slab.x_max},{p_end[0]},{p_end[1]}\n\n")
                    
                    # 두 단면 연결 (X축 방향 선분)
                    f.write("LINE\n")
                    f.write(f"{slab.x_min},{p_start[0]},{p_start[1]}\n")
                    f.write(f"{slab.x_max},{p_start[0]},{p_start[1]}\n\n")

        # 2. Slab Traces (Extracted 3D segments)
        f.write("-LAYER\nSET\nSlab_Traces\n\n")
        total_segs = 0
        for segments in slab_all_segments:
            for seg in segments:
                # seg is (2, 3) -> [ [x0, y0, z0], [x1, y1, z1] ]
                f.write("LINE\n")
                f.write(f"{seg[0,0]},{seg[0,1]},{seg[0,2]}\n")
                f.write(f"{seg[1,0]},{seg[1,1]},{seg[1,2]}\n\n")
                total_segs += 1
                
        f.write("ZOOM\nEXTENTS\n")
        
    print(f" -> [CAD] AutoCAD 스크립트 생성 완료: {scr_path}")
    print(f" -> [CAD] 총 {len(slabs)}개 Slab 및 {total_segs}개 Trace 선분이 포함되었습니다.")
