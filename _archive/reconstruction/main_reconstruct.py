import json
import argparse
import os
import sys
import numpy as np
import h5py
from typing import List

# 로컬 모듈 로드 설정 (상위 디렉토리 참조)
_here = os.path.dirname(os.path.abspath(__file__))
_trace_analysis = os.path.dirname(_here)
_project_root = os.path.dirname(_trace_analysis)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from trace_analysis.load_tunnel_dat import load_tunnel_polygon_from_dat
from trace_analysis.trace_reconstruction.trace_types import ExcavationFace
from trace_analysis.trace_reconstruction.excavation_face_traces import extract_excavation_face_traces_from_truth

from .data_models import Face, Trace, FractureSet
from .synthetic_data import SyntheticGenerator
from .preprocess import preprocess_traces
from .set_inference import SetInferrer
from .matching import TraceMatcher
from .plane_fit import PlaneReconstructor
from .scoring import QualityEvaluator, filter_hypotheses
from .visualize import plot_reconstruction_results

def run_sc_pmfr_pipeline(faces: List[Face], visualize: bool = True):
    """SC-PMFR 전체 파이프라인 통합 실행"""
    
    # 1. 전처리 (Preprocessing)
    all_traces = []
    for face in faces:
        face.traces = preprocess_traces(face.traces)
        all_traces.extend(face.traces)

    if not all_traces:
        print(" [Error] No active traces found after preprocessing.")
        return

    # 2. 절리군 추론 (Set Inference)
    inferrer = SetInferrer(num_sets=3)
    fracture_sets = inferrer.infer_sets(all_traces)
    
    memberships = {}
    for face in faces:
        m = inferrer.assign_membership(face.traces, fracture_sets)
        memberships[face.face_id] = m

    # 3. 막장 간 매칭 (Inter-face Matching)
    matcher = TraceMatcher()
    
    # 막장 간 연속 매칭 (0-1, 1-2, ...)
    raw_match_pairs = []
    for i in range(len(faces) - 1):
        m_pair = matcher.find_matches(faces[i], faces[i+1], memberships[faces[i].face_id], memberships[faces[i+1].face_id])
        raw_match_pairs.append(m_pair)

    # 단순 체인 빌딩 (2개 막장 쌍들을 독립적인 복원 후보로 취급)
    final_matches = []
    for pair_idx, m_list in enumerate(raw_match_pairs):
        for id1, id2, score in m_list:
            final_matches.append({'ids': [id1, id2], 'score': score})

    # 4. 3차원 평면 복원 (Reconstruction)
    reconstructor = PlaneReconstructor(set_lambda=0.5)
    evaluator = QualityEvaluator()
    
    hypotheses = []
    trace_lookup = {t.trace_id: t for f in faces for t in f.traces}
    
    for i, match in enumerate(final_matches):
        match_traces = [trace_lookup[tid] for tid in match['ids']]
        
        # 주배향 정보 (가장 가중치가 높은 세트 선택)
        m_avg = np.mean([memberships[t.face_id][t.trace_id] for t in match_traces], axis=0)
        best_set_idx = np.argmax(m_avg)
        
        # 복원 수행
        hypo = reconstructor.fit_plane_constrained(match_traces, fracture_sets[best_set_idx])
        hypo.hypothesis_id = i
        hypo.prior_score = float(m_avg[best_set_idx])
        
        # 신뢰도 평가
        hypo = evaluator.evaluate(hypo, len(match_traces))
        hypotheses.append(hypo)

    # 5. 필터링 및 시각화
    confident_fractures = filter_hypotheses(hypotheses, min_confidence=0.3)
    
    print(f"\n [Success] Pipeline completed. Reconstructed {len(confident_fractures)} confident fractures.")
    
    # 6. JSON 출력
    output_data = []
    for h in confident_fractures:
        output_data.append({
            "fracture_id": h.hypothesis_id,
            "normal": h.normal.tolist(),
            "center": h.center.tolist(),
            "radius": h.radius,
            "confidence": h.confidence
        })
    
    out_path = "reconstructed_fractures.json"
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
        print(f" [Output] Reconstructed data saved to {out_path}")

    # 7. 시각화
    if visualize:
        plot_reconstruction_results(faces, confident_fractures)

def main():
    parser = argparse.ArgumentParser(description="SC-PMFR Real Data Reconstruction")
    parser.add_argument('--input', default='storage/data/dfn_export_for_python.h5', help="원본 H5 데이터 (Truth)")
    parser.add_argument('--tunnel_dat', default='storage/data/단면_폴리곤.dat', help="터널 단면 polygon 데이터")
    parser.add_argument('--x_curr', type=float, default=20.0, help="기준 막장 위치 (X)")
    parser.add_argument('--num_faces', type=int, default=3, help="분석에 사용할 막장 개수")
    parser.add_argument('--interval', type=float, default=3.0, help="막장 사이의 간격 (m)")
    parser.add_argument('--visualize', action='store_true', help="결과 시각화 여부")
    parser.add_argument('--use_synthetic', action='store_true', help="가상 데이터 모드 강제 활성화")
    args = parser.parse_args()

    print("="*60)
    print(f" [SC-PMFR] Set-Constrained Probabilistic Multi-Face Reconstruction")
    print(f" -> Mode: {'Synthetic' if args.use_synthetic else 'Real Data'}")
    print(f" -> Current X: {args.x_curr}m, Faces: {args.num_faces}")
    print("="*60)

    faces_to_process = []

    if args.use_synthetic:
        gen = SyntheticGenerator(seed=42)
        x_poses = [args.x_curr - (i * args.interval) for i in range(args.num_faces)][::-1]
        faces_to_process = gen.generate_faces(x_offsets=x_poses)
        _ = gen.generate_synthetic_traces(faces_to_process, num_fractures=10)
    else:
        # 1. 터널 및 DFN 데이터 로드
        poly_y, poly_z = load_tunnel_polygon_from_dat(args.tunnel_dat)
        poly_yz = np.column_stack([poly_y, poly_z])
        
        with h5py.File(args.input, 'r') as f:
            centers = f['/fractures/centers'][:]
            normals = f['/fractures/normals'][:]
            radii = f['/fractures/radii'][:].ravel()
            if centers.shape[0] == 3 and centers.shape[0] < centers.shape[1]: centers = centers.T
            if normals.shape[0] == 3 and normals.shape[0] < normals.shape[1]: normals = normals.T

        # 2. 지정된 막장들에서 Trace 추출 및 모델 변환
        face_x_poses = [args.x_curr - (i * args.interval) for i in range(args.num_faces)][::-1]
        
        for i, x in enumerate(face_x_poses):
            # 기존 패키지 Face 객체 생성 및 추출
            exc_face = ExcavationFace(face_id=i, x_face=x, tunnel_polygon_yz=poly_yz, advance_step=args.interval)
            raw_traces = extract_excavation_face_traces_from_truth(centers, normals, radii, exc_face)
            
            # 신규 모델 Face 객체 생성
            recon_face = Face(face_id=i, plane_point=np.array([x, 0, 0]), plane_normal=np.array([1, 0, 0]))
            
            # FaceTrace -> reconstruction.data_models.Trace 변환
            for rt in raw_traces:
                p1_3d = np.array([rt.x_face, rt.p0_y, rt.p0_z])
                p2_3d = np.array([rt.x_face, rt.p1_y, rt.p1_z])
                t = Trace(
                    trace_id=f"T_{i}_{rt.trace_id}",
                    face_id=i,
                    endpoints_3d=np.array([p1_3d, p2_3d])
                )
                recon_face.traces.append(t)
            
            faces_to_process.append(recon_face)
            print(f"  - Face {i} (X={x:.1f}): {len(recon_face.traces)} traces extracted from real data.")

    # 3. 파이프라인 실행
    run_sc_pmfr_pipeline(faces_to_process, visualize=args.visualize)

if __name__ == '__main__':
    main()
