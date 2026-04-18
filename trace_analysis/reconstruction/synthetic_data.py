import numpy as np
from typing import List, Dict
from .data_models import Face, Trace

class SyntheticGenerator:
    """검증용 가상 Trace 데이터 생성기"""
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
    
    def generate_faces(self, x_offsets: List[float]) -> List[Face]:
        """지정된 X 위치에 막장면 생성"""
        faces = []
        for i, x in enumerate(x_offsets):
            face = Face(
                face_id=i,
                plane_point=np.array([x, 0.0, 0.0]),
                plane_normal=np.array([1.0, 0.0, 0.0]),
                excavation_step=i
            )
            faces.append(face)
        return faces

    def generate_synthetic_traces(self, faces: List[Face], num_fractures: int = 5) -> Dict:
        """가상 균열(Disc)을 생성하고 각 막장과의 교선을 추출"""
        fractures = []
        for i in range(num_fractures):
            # 무작위 법선 (전방을 향하도록 유도)
            normal = np.random.uniform(-1, 1, 3)
            normal[0] = np.abs(normal[0])  # 주로 X축 방향
            normal /= np.linalg.norm(normal)
            
            # 무작위 중심 및 반경 (막장들을 충분히 가로지르도록 설정)
            center = np.random.uniform(-3, 3, 3)
            center[0] = np.mean([f.plane_point[0] for f in faces]) 
            radius = np.random.uniform(3, 7)
            
            fractures.append({'n': normal, 'c': center, 'r': radius})

        for face in faces:
            x_f = face.plane_point[0]
            for i, frac in enumerate(fractures):
                n = frac['n']
                c = frac['c']
                r = frac['r']
                
                # 평면-원판 교선 계산 (Simplified)
                # n * (p - c) = 0 and p_x = x_f
                # n_x*(x_f - c_x) + n_y*(p_y - c_y) + n_z*(p_z - c_z) = 0
                
                # 평면과 원판 중심 사이의 거리 확인
                dist_to_plane = np.abs(n[0] * (x_f - c[0])) / np.sqrt(n[1]**2 + n[2]**2) if (n[1]**2 + n[2]**2) > 1e-6 else 1e9
                
                if dist_to_plane < r:
                    # 교선의 방향 벡터는 n과 face_normal(1,0,0)의 외적
                    dir_vec = np.cross(n, [1.0, 0.0, 0.0])
                    if np.linalg.norm(dir_vec) < 1e-6: continue
                    dir_vec /= np.linalg.norm(dir_vec)
                    
                    # 교선의 중심점 (평면상에서 원판 중심 투영 지점)
                    # p_x = x_f
                    # n_y*(p_y - c_y) + n_z*(p_z - c_z) = -n_x*(x_f - c_x)
                    # p = c + t*n 에서 p_x = x_f 인 t 찾기: c_x + t*n_x = x_f -> t = (x_f - c_x)/n_x
                    if np.abs(n[0]) < 1e-6: continue
                    t = (x_f - c[0]) / n[0]
                    proj_c = c + t * n
                    
                    # 교선 반각 길이 d' = sqrt(r^2 - dist^2)
                    # 여기서 dist는 proj_c와 c 사이의 거리
                    dist_sq = np.sum((proj_c - c)**2)
                    if r**2 < dist_sq: continue
                    half_len = np.sqrt(r**2 - dist_sq)
                    
                    p1 = proj_c + dir_vec * half_len
                    p2 = proj_c - dir_vec * half_len
                    
                    trace = Trace(
                        trace_id=f"frac_{i}_face_{face.face_id}",
                        face_id=face.face_id,
                        endpoints_3d=np.array([p1, p2])
                    )
                    face.traces.append(trace)
        
        return fractures # GT 반환
