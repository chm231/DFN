import numpy as np
from typing import List
from .data_models import FractureHypothesis

class QualityEvaluator:
    """복원된 균열 가설의 품질 및 신뢰도 평가"""
    
    def evaluate(self, hypothesis: FractureHypothesis, num_traces: int) -> FractureHypothesis:
        """가설의 여러 지표를 종합하여 최종 신뢰도(Confidence) 산출"""
        
        # 1. 지원 근거 점수 (막장 수/Trace 수)
        # 2개 막장 이상이면 가점
        support_score = min(1.0, num_traces / 4.0) 
        
        # 2. 피팅 정밀도 점수 (Fit Error)
        # 에러가 적을수록 1에 가까움
        fit_score = np.exp(-hypothesis.fit_error * 2.0)
        
        # 3. 절리군 일치 점수 (Prior)
        # 이미 최적화 과정에서 반영되었으나, 최종적으로도 체크
        prior_score = hypothesis.prior_score
        
        # 가중 평균 (Baseline)
        confidence = (0.5 * support_score + 0.4 * fit_score + 0.1 * prior_score)
        
        hypothesis.confidence = float(confidence)
        return hypothesis

def filter_hypotheses(hypotheses: List[FractureHypothesis], min_confidence: float = 0.4) -> List[FractureHypothesis]:
    """저신뢰 가설 필터링"""
    filtered = [h for h in hypotheses if h.confidence >= min_confidence]
    print(f" [Scoring] Filtered {len(hypotheses)} -> {len(filtered)} confident fractures (min={min_confidence})")
    return filtered
