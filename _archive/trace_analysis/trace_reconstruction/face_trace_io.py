"""
[Direction B: Inverse Reconstruction]
Excavation Face에서 획득한 Trace 정보를 CSV 포맷으로 읽고 쓰는 모듈입니다.
전체 DFN 단면(Direction A)이 아닌, 막장면 내부로 클리핑된 순수(Direction B) 데이터만을 다룹니다.
"""
import os
import pandas as pd
from typing import List, Dict
from .trace_types import FaceTrace

def load_face_traces(csv_path: str) -> List[FaceTrace]:
    """저장된 face trace CSV를 파싱하여 FaceTrace 객체 리스트로 반환"""
    df = pd.read_csv(csv_path)
    traces = []
    for _, row in df.iterrows():
        trace = FaceTrace(
            face_id=int(row['face_id']),
            trace_id=int(row['trace_id']),
            x_face=float(row['x_face']),
            p0_y=float(row['y0']),
            p0_z=float(row['z0']),
            p1_y=float(row['y1']),
            p1_z=float(row['z1']),
            confidence=float(row.get('confidence', 1.0))
        )
        traces.append(trace)
    return traces

def save_face_traces(traces: List[FaceTrace], csv_path: str):
    """FaceTrace 객체 리스트를 표준 스키마 CSV로 내보내기 (NaN 허용)"""
    data = []
    for t in traces:
        data.append({
            'face_id': t.face_id,
            'trace_id': t.trace_id,
            'x_face': t.x_face,
            'y0': t.p0_y,
            'z0': t.p0_z,
            'y1': t.p1_y,
            'z1': t.p1_z,
            'length': t.length,
            'local_orientation_2d': t.local_orientation_2d,
            'confidence': t.confidence
        })
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    df.to_csv(csv_path, index=False, na_rep='NaN')

def group_traces_by_face(traces: List[FaceTrace]) -> Dict[int, List[FaceTrace]]:
    """입력된 Trace 리스트를 face_id 별로 분류"""
    grouped = {}
    for t in traces:
        grouped.setdefault(t.face_id, []).append(t)
    return grouped
