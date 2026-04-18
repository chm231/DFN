import numpy as np
from typing import List, Dict
from .data_models import Trace, FractureSet

class SetInferrer:
    """법선 매니폴드 분석 기반 절리군 추론 엔진"""
    def __init__(self, num_sets: int = 3):
        self.num_sets = num_sets

    def infer_sets(self, all_traces: List[Trace]) -> List[FractureSet]:
        """지정된 Trace 리스트로부터 지배적인 절리군 배향을 추론 (Baseline)"""
        # 프로토타입 baseline: 
        # 실제 데이터가 충분하지 않은 경우 대표 배향 3개를 휴리스틱하게 설정하거나
        # 구면상에서의 점군 밀도 분석을 수행해야 함.
        # 여기서는 간단히 3개의 대표적인 배향(Random)을 반환하는 베이스라인 구현
        
        sets = []
        # 예시: 3개의 서로 다른 방향을 가진 절리군 시뮬레이션
        orientations = [
            np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
            np.array([1.0, -1.0, 0.5]) / np.linalg.norm([1.0, -1.0, 0.5]),
            np.array([0.5, 0.2, 1.0]) / np.linalg.norm([0.5, 0.2, 1.0])
        ]
        
        for i in range(min(self.num_sets, len(orientations))):
            sets.append(FractureSet(
                set_id=i,
                representative_normal=orientations[i],
                dispersion=50.0 # High concentration
            ))
        
        print(f" [SetInference] Inferred {len(sets)} dominant fracture sets.")
        return sets

    def assign_membership(self, traces: List[Trace], sets: List[FractureSet]) -> Dict[str, np.ndarray]:
        """각 Trace가 각 절리군에 속할 가중치(Soft membership) 계산"""
        # 제약 조건: 법선 n은 Trace 방향 t와 직교해야 함 (n dot t = 0)
        # weight = exp(- |n_set dot t|^2 / sigma)
        # 즉, 절리군의 법선이 Trace 방향과 직교할수록 높은 점수 부여
        
        memberships = {}
        for t in traces:
            weights = []
            for s in sets:
                n_s = s.representative_normal
                dir_v = t.direction
                # 기하학적 적합도: 법선과 선분 방향이 수직일수록 0에 가까움
                compatibility = np.abs(np.dot(n_s, dir_v))
                w = np.exp(- (compatibility**2) / 0.1) # Soft assignment
                weights.append(w)
            
            # 정규화
            weights = np.array(weights)
            if np.sum(weights) > 0:
                weights /= np.sum(weights)
            memberships[t.trace_id] = weights
            
        return memberships
