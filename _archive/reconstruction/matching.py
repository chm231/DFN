import numpy as np
from typing import List, Dict, Tuple
from .data_models import Trace, Face

class TraceMatcher:
    """막장 간 Trace 매칭 엔진 (확률적 스코어링)"""
    def __init__(self, weights: Dict[str, float] = None):
        # 가중치 설정 (사용자 가이드 [5] 반영)
        self.weights = weights or {
            'set': 0.4,
            'geometry': 0.3,
            'continuity': 0.1,
            'size': 0.1,
            'positional': 0.1
        }

    def compute_match_score(self, t1: Trace, t2: Trace, m1: np.ndarray, m2: np.ndarray) -> float:
        """두 Trace 사이의 매칭 점수 계산 (높을수록 좋음)"""
        
        # 1. Set Consistency (절리군 멤버십 유사도)
        set_score = np.dot(m1, m2)
        
        # 2. Size similarity (길이 비율)
        size_ratio = min(t1.length, t2.length) / max(t1.length, t2.length)
        size_score = size_ratio
        
        # 3. Geometric compatibility (방향성 일치도)
        geom_score = np.abs(np.dot(t1.direction, t2.direction))
        
        # 4. Positional continuity (YZ 평면상에서의 중점 이동 거리)
        dist_yz = np.linalg.norm(t1.midpoint_3d[1:] - t2.midpoint_3d[1:])
        pos_score = np.exp(-dist_yz / 5.0)
        
        # 최종 가중합
        total_score = (
            self.weights['set'] * set_score +
            self.weights['size'] * size_score +
            self.weights['geometry'] * geom_score +
            self.weights['positional'] * pos_score
        )
        
        return total_score

    def find_matches(self, face1: Face, face2: Face, 
                     memberships1: Dict[str, np.ndarray], 
                     memberships2: Dict[str, np.ndarray], 
                     threshold: float = 0.5) -> List[Tuple[str, str, float]]:
        """두 막장 간의 최적 매칭쌍 탐색 (Greedy baseline)"""
        
        matches = []
        candidates = []
        
        for t1 in face1.traces:
            for t2 in face2.traces:
                score = self.compute_match_score(
                    t1, t2, memberships1[t1.trace_id], memberships2[t2.trace_id]
                )
                if score > threshold:
                    candidates.append((t1.trace_id, t2.trace_id, score))
        
        # 점수순 정렬 후 Greedy 매칭
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        used1 = set()
        used2 = set()
        for id1, id2, score in candidates:
            if id1 not in used1 and id2 not in used2:
                matches.append((id1, id2, score))
                used1.add(id1)
                used2.add(id2)
        
        print(f" [Matching] Found {len(matches)} matches between Face {face1.face_id} and {face2.face_id}")
        return matches
