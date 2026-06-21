import os
import numpy as np
from typing import List, Dict, Any

def export_rough_traces_to_cad(
    all_rough_traces: List[List[Dict[str, Any]]],
    x_positions: np.ndarray,
    tunnel_poly_yz: np.ndarray,
    scr_path: str
):
    """
    비평면 상의 Trace들과 터널 단면 형상을 AutoCAD SCR 파일로 저장.
    """
    os.makedirs(os.path.dirname(scr_path), exist_ok=True)
    
    with open(scr_path, 'w') as f:
        # Layer 설정
        f.write("-LAYER\nMAKE\nRough_Traces\nCOLOR\n1\nRough_Traces\n\n") # Red
        f.write("-LAYER\nMAKE\nTunnel_Slabs\nCOLOR\n8\nTunnel_Slabs\n\n") # Gray
        
        # 1. 터널 Slab 형상 (Wireframe Domains)
        if tunnel_poly_yz is not None and len(x_positions) > 1:
            f.write("-LAYER\nSET\nTunnel_Slabs\n\n")
            for i in range(len(x_positions) - 1):
                x_curr = x_positions[i]
                x_next = x_positions[i+1] # 다음 굴착면까지 연결
                
                # 가로 루프 (시작면)
                f.write("3DPOLY\n")
                for p in tunnel_poly_yz:
                    f.write(f"{x_curr},{p[0]},{p[1]}\n")
                f.write(f"{x_curr},{tunnel_poly_yz[0,0]},{tunnel_poly_yz[0,1]}\n\n")
                
                # 종방향 연결선 (각 꼭짓점 연결)
                for p in tunnel_poly_yz:
                    f.write("LINE\n")
                    f.write(f"{x_curr},{p[0]},{p[1]}\n")
                    f.write(f"{x_next},{p[0]},{p[1]}\n\n")
            
            # 마지막 면의 루프 추가
            x_last = x_positions[-1]
            f.write("3DPOLY\n")
            for p in tunnel_poly_yz:
                f.write(f"{x_last},{p[0]},{p[1]}\n")
            f.write(f"{x_last},{tunnel_poly_yz[0,0]},{tunnel_poly_yz[0,1]}\n\n")

        # 2. Rough Traces (3DPOLY)
        f.write("-LAYER\nSET\nRough_Traces\n\n")
        rough_count = 0
        for face_traces in all_rough_traces:
            for t in face_traces:
                pts = t['points']
                if len(pts) < 2: continue
                
                f.write("3DPOLY\n")
                for p in pts:
                    f.write(f"{p[0]},{p[1]},{p[2]}\n")
                f.write("\n")
                rough_count += 1
                
        f.write("ZOOM\nEXTENTS\n")
        
    print(f" -> [CAD] AutoCAD 스크립트 생성 완료: {scr_path}")
    print(f" -> [CAD] 총 {len(x_positions)}개 단면 및 {rough_count}개 Rough Trace가 내보내졌습니다.")
