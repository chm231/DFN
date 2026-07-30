# =============================================================================
# 파일 역할:
#   막장면에서 추출한 3D trace(단열 흔적선) 데이터를 단열 세트(set_id)별로
#   집계하여 통계를 낸다. trace 개수, 절단(censoring) 분류, 길이 통계,
#   관측 면적 대비 P21 선밀도 등을 세트별로 요약해 CSV/콘솔로 출력한다.
#
# 주요 입력:
#   - trace 데이터: --trace-h5(HDF5) 또는 --trace-csv(CSV) 중 하나 (택1)
#     각 trace는 set_id, observed_length_m, censoring_class, trace_normal_valid 보유
#   - (옵션) --rough-mesh-h5: rough face collection HDF5.
#     mesh 삼각형 면적 합으로 관측 면적을 구해 P21 계산에 사용
#
# 주요 출력:
#   - 세트별 통계 CSV (--out-csv)
#   - 콘솔 요약 테이블
#
# 핵심 처리 흐름:
#   1) trace 행들을 HDF5 또는 CSV에서 읽는다
#   2) (옵션) rough mesh에서 총 관측 면적(m^2)을 계산한다
#   3) set_id별로 개수/절단/길이/P21/normal 유효비율 등을 집계한다
#   4) CSV로 저장하고 콘솔에 요약을 출력한다
#
# 용어:
#   - censoring_class: 0=양끝 관측(uncensored), 1=한쪽 끝 절단, 2=양쪽 끝 절단
#   - P21: 단위 관측 면적당 trace 총 길이 (선밀도, 1/m)
# =============================================================================

import argparse
import csv
import os
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np


# HDF5 그룹에서 스칼라 값 하나를 안전하게 읽는 도우미.
#   key가 없으면 default 반환, bytes면 문자열로 디코드, 배열이면 첫 원소를 반환한다.
#   인자: group(h5py 그룹), key(데이터셋 이름), default(없을 때 값)  /  반환: 스칼라 값
def _read_scalar(group: h5py.Group, key: str, default):
    if key not in group:
        return default
    value = group[key][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if np.ndim(value) > 0:
        return value.ravel()[0]
    return value


# trace HDF5(/traces 그룹)에서 통계에 필요한 컬럼만 뽑아 dict 리스트로 읽는다.
#   인자: h5_path(입력 HDF5 경로)
#   반환: trace별 dict 목록(set_id, observed_length_m, censoring_class, trace_normal_valid)
def load_trace_rows_from_h5(h5_path: str) -> List[dict]:
    rows: List[dict] = []
    with h5py.File(h5_path, "r") as f:
        # /traces 그룹이 없으면 잘못된 입력이므로 예외 발생.
        if "traces" not in f:
            raise ValueError(f"Could not find /traces in: {h5_path}")
        grp = f["traces"]
        # trace_id 개수만큼 순회하며 각 trace 행을 dict로 만든다.
        n_rows = len(grp["trace_id"])
        for idx in range(n_rows):
            rows.append(
                {
                    "set_id": int(grp["set_id"][idx]),
                    "observed_length_m": float(grp["observed_length_m"][idx]),
                    "censoring_class": int(grp["censoring_class"][idx]),
                    "trace_normal_valid": int(grp["trace_normal_valid"][idx]),
                }
            )
    return rows


# trace CSV에서 동일한 컬럼을 읽어 dict 리스트로 만든다(HDF5 버전과 형식 동일).
#   인자: csv_path(입력 CSV 경로)  /  반환: trace별 dict 목록
def load_trace_rows_from_csv(csv_path: str) -> List[dict]:
    rows: List[dict] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        # DictReader로 헤더 기반으로 각 행을 읽고 형 변환한다.
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "set_id": int(row["set_id"]),
                    "observed_length_m": float(row["observed_length_m"]),
                    "censoring_class": int(row["censoring_class"]),
                    "trace_normal_valid": int(row["trace_normal_valid"]),
                }
            )
    return rows


# rough face mesh를 HDF5에서 읽어 face별 정점/삼각형을 반환한다(P21 면적 계산용).
#   여러 저장 형식을 지원: /rough_faces(collection) > /rough_face(단일) > /mesh(구형).
#   인자: h5_path(rough mesh HDF5)  /  반환: face별 dict(face_id, vertices_xyz, triangles) 목록
def load_rough_face_collection_from_h5(h5_path: str) -> List[dict]:
    with h5py.File(h5_path, "r") as f:
        # 1) 최신 형식: /rough_faces 아래 face별 그룹을 이름순으로 모두 읽는다.
        if "rough_faces" in f:
            rough_faces = []
            faces_grp = f["rough_faces"]
            for face_name in sorted(faces_grp.keys()):
                grp = faces_grp[face_name]
                meta = grp["meta"] if "meta" in grp else None
                face_id = int(_read_scalar(meta, "face_id", len(rough_faces) + 1)) if meta else len(rough_faces) + 1
                rough_faces.append(
                    {
                        "face_id": face_id,
                        "vertices_xyz": grp["mesh/vertices_xyz"][:].astype(np.float64),
                        "triangles": grp["mesh/triangles"][:].astype(np.int32),
                    }
                )
            return rough_faces

        # 2) 하위호환: 단일 face 형식 /rough_face.
        if "rough_face" in f:
            grp = f["rough_face"]
            return [
                {
                    "face_id": 1,
                    "vertices_xyz": grp["mesh/vertices_xyz"][:].astype(np.float64),
                    "triangles": grp["mesh/triangles"][:].astype(np.int32),
                }
            ]

        # 3) 가장 구형: 루트에 바로 /mesh 가 있는 형식.
        if "mesh" in f:
            return [
                {
                    "face_id": 1,
                    "vertices_xyz": f["mesh/vertices_xyz"][:].astype(np.float64),
                    "triangles": f["mesh/triangles"][:].astype(np.int32),
                }
            ]

    # 위 형식 어느 것도 없으면 rough face를 찾을 수 없으므로 예외 발생.
    raise ValueError(f"Could not find rough face collection in: {h5_path}")


# 삼각형 mesh 하나의 전체 표면적을 합산한다(외적 크기의 절반 = 삼각형 면적).
#   인자: vertices_xyz((M,3) 정점), triangles((K,3) 정점 인덱스)
#   반환: 면적 합(float, m^2)
def triangle_area_sum(vertices_xyz: np.ndarray, triangles: np.ndarray) -> float:
    # 각 삼각형의 세 정점을 모아 두 변의 외적으로 면적을 벡터화 계산한다.
    v0 = vertices_xyz[triangles[:, 0]]
    v1 = vertices_xyz[triangles[:, 1]]
    v2 = vertices_xyz[triangles[:, 2]]
    return 0.5 * float(np.sum(np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)))


# 모든 face의 면적을 합해 총 관측 면적을 구한다(P21 분모).
#   인자: rough_faces(face dict 목록)  /  반환: 총 면적(float, m^2)
def compute_total_observation_area(rough_faces: Sequence[dict]) -> float:
    return float(sum(triangle_area_sum(face["vertices_xyz"], face["triangles"]) for face in rough_faces))


# 빈 배열이면 nan, 아니면 평균을 반환하는 안전 평균 도우미.
def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


# 빈 배열이면 nan, 아니면 중앙값을 반환하는 안전 중앙값 도우미.
def _safe_median(values: np.ndarray) -> float:
    return float(np.median(values)) if len(values) else float("nan")


# trace 행들을 set_id별로 묶어 통계 요약 행을 만든다.
#   집계 항목: 개수/절단 분류/길이 평균·중앙값/총길이/P21/normal 유효비율.
#   인자: rows(trace dict 목록), observation_area_m2(관측 면적, 없으면 P21=nan)
#   반환: set_id별 요약 dict 목록
def build_summary_rows(rows: Sequence[dict], observation_area_m2: Optional[float]) -> List[dict]:
    # 등장하는 set_id를 정렬해 순회 대상으로 삼는다.
    set_ids = sorted({int(row["set_id"]) for row in rows})
    summary_rows: List[dict] = []
    for set_id in set_ids:
        # 현재 set에 속하는 trace만 추린다.
        set_rows = [row for row in rows if int(row["set_id"]) == set_id]
        # 절단되지 않은(class 0) trace 길이 배열.
        uncensored = np.asarray(
            [float(row["observed_length_m"]) for row in set_rows if int(row["censoring_class"]) == 0],
            dtype=np.float64,
        )
        # 절단된(class>0) trace 길이 배열.
        censored = np.asarray(
            [float(row["observed_length_m"]) for row in set_rows if int(row["censoring_class"]) > 0],
            dtype=np.float64,
        )
        # 개수/절단 분류/normal 유효 개수 등을 집계한다.
        total_length = float(sum(float(row["observed_length_m"]) for row in set_rows))
        n_total = len(set_rows)
        n_uncensored = int(sum(int(row["censoring_class"]) == 0 for row in set_rows))
        n_one_end_censored = int(sum(int(row["censoring_class"]) == 1 for row in set_rows))
        n_two_end_censored = int(sum(int(row["censoring_class"]) == 2 for row in set_rows))
        n_censored = n_one_end_censored + n_two_end_censored
        n_valid_normal = int(sum(int(row["trace_normal_valid"]) == 1 for row in set_rows))

        # set별 요약 한 줄을 구성해 결과에 추가한다(P21 = 총길이 / 관측면적).
        summary_rows.append(
            {
                "set_id": set_id,
                "n_total": n_total,
                "n_uncensored": n_uncensored,
                "n_one_end_censored": n_one_end_censored,
                "n_two_end_censored": n_two_end_censored,
                "censoring_ratio_total": (n_censored / n_total) if n_total else float("nan"),
                "mean_length_uncensored": _safe_mean(uncensored),
                "mean_length_censored": _safe_mean(censored),
                "median_length_uncensored": _safe_median(uncensored),
                "median_length_censored": _safe_median(censored),
                "total_trace_length": total_length,
                "P21_observed": (total_length / observation_area_m2) if observation_area_m2 and observation_area_m2 > 0.0 else float("nan"),
                "trace_normal_valid_ratio": (n_valid_normal / n_total) if n_total else float("nan"),
            }
        )
    return summary_rows


# set별 요약 행들을 고정된 컬럼 순서로 CSV 파일에 기록한다.
#   인자: summary_rows(요약 dict 목록), csv_path(출력 경로)  /  반환: 없음(파일 저장)
def write_summary_csv(summary_rows: Sequence[dict], csv_path: str) -> None:
    # 출력 CSV의 컬럼 순서를 고정한다.
    fieldnames = [
        "set_id",
        "n_total",
        "n_uncensored",
        "n_one_end_censored",
        "n_two_end_censored",
        "censoring_ratio_total",
        "mean_length_uncensored",
        "mean_length_censored",
        "median_length_uncensored",
        "median_length_censored",
        "total_trace_length",
        "P21_observed",
        "trace_normal_valid_ratio",
    ]
    # 헤더를 쓴 뒤 각 요약 행을 기록한다.
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


# set별 통계 요약을 사람이 읽기 좋은 형태로 콘솔에 출력한다.
#   인자: summary_rows(요약 dict 목록), observation_area_m2(관측 면적, 있으면 표시)
#   반환: 없음(콘솔 출력)
def print_summary_table(summary_rows: Sequence[dict], observation_area_m2: Optional[float]) -> None:
    print("[*] Set-wise trace statistics")
    if observation_area_m2 is not None:
        print(f"    observation_area_m2 = {observation_area_m2:.6f}")
    for row in summary_rows:
        print(
            f"    - Set {row['set_id']}: "
            f"n_total={row['n_total']}, n_uncensored={row['n_uncensored']}, "
            f"n_one_end_censored={row['n_one_end_censored']}, n_two_end_censored={row['n_two_end_censored']}, "
            f"censoring_ratio_total={row['censoring_ratio_total']:.4f}, "
            f"mean_length_uncensored={row['mean_length_uncensored']:.4f}, "
            f"mean_length_censored={row['mean_length_censored']:.4f}, "
            f"median_length_uncensored={row['median_length_uncensored']:.4f}, "
            f"median_length_censored={row['median_length_censored']:.4f}, "
            f"total_trace_length={row['total_trace_length']:.4f}, "
            f"P21_observed={row['P21_observed']:.6f}, "
            f"trace_normal_valid_ratio={row['trace_normal_valid_ratio']:.4f}"
        )


# CLI 진입점: trace 로드 → (옵션)관측면적 계산 → set별 집계 → CSV 저장/콘솔 출력.
#   인자: 없음(argparse로 CLI 인자 읽음)  /  반환: 없음
def main() -> None:
    # CLI 인자 정의 및 파싱.
    parser = argparse.ArgumentParser(description="Summarize set-wise 3D trace statistics.")
    parser.add_argument("--trace-h5", help="Input trace HDF5 created by export_setwise_3d_traces.py")
    parser.add_argument("--trace-csv", help="Input trace CSV created by export_setwise_3d_traces.py")
    parser.add_argument("--rough-mesh-h5", help="Rough face collection HDF5 used to compute observed P21 area")
    parser.add_argument(
        "--out-csv",
        default="storage/output/trace_dataset_collection/setwise_trace_statistics.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    # trace 입력은 HDF5/CSV 중 정확히 하나만 지정해야 한다.
    if bool(args.trace_h5) == bool(args.trace_csv):
        raise ValueError("Provide exactly one of --trace-h5 or --trace-csv.")

    # 지정된 형식으로 trace 행을 읽는다.
    rows = load_trace_rows_from_h5(args.trace_h5) if args.trace_h5 else load_trace_rows_from_csv(args.trace_csv)
    # rough mesh가 주어지면 총 관측 면적을 계산해 P21 계산에 사용한다.
    observation_area_m2 = None
    if args.rough_mesh_h5:
        rough_faces = load_rough_face_collection_from_h5(args.rough_mesh_h5)
        observation_area_m2 = compute_total_observation_area(rough_faces)

    # set별 집계 → 출력 폴더 생성 → CSV 저장 → 콘솔 요약 출력.
    summary_rows = build_summary_rows(rows, observation_area_m2)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    write_summary_csv(summary_rows, args.out_csv)
    print_summary_table(summary_rows, observation_area_m2)
    print(f"[*] CSV written to: {args.out_csv}")


if __name__ == "__main__":
    main()
