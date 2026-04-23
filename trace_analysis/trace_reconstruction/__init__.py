"""
[Direction B: Inverse Reconstruction]
터널 막장면에서 측정한 2D Trace 데이터를 기반으로, 동일 절리면을 추적(Trace Matching)하고
3차원 평면(Plane Reconstruction) 및 안정적인 블록(Block Polyhedron)을 재구성하는 패키지입니다.
이 모듈 내에서의 Trace는 "x=const 평면과 절리면의 교차선 중 터널 폴리곤 내에 위치한 선분"만 의미합니다.
"""

from .trace_types import FaceTrace, ExcavationFace, CensoringType
from .face_trace_io import load_face_traces, save_face_traces
from .excavation_face_traces import extract_excavation_face_traces_from_truth, clip_trace_to_tunnel_polygon

__all__ = [
    "FaceTrace",
    "ExcavationFace",
    "CensoringType",
    "load_face_traces",
    "save_face_traces",
    "extract_excavation_face_traces_from_truth",
    "clip_trace_to_tunnel_polygon",
]
