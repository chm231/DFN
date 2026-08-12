r"""
실행 예시:

```
$env:PYTHONPATH='.'
python dfn_analysis\generate_synthetic_rough_face_mesh.py `
  --tunnel-dat 'storage\data\단면_폴리곤.dat' `
  --outdir 'storage\output\rough_face_mesh_collection' `
  --face-x-csv "0,1,2,3" `
  --grid-step 0.2 `
  --amplitude 0.05 `
  --corr-length 1.0 `
  --seed-base 42 `
  --merge-collection-into-hdf5 'storage\data\dfn_export_for_python.h5'

# 주요 parser 옵션 의미:
# --tunnel-dat:
#   터널 단면 polygon 원본 파일이다. [y, z] 단면 형상을 여기서 읽는다.
# --outdir:
#   생성 결과를 저장할 폴더다. HDF5와 preview PNG가 이 안에 생긴다.
# --grid-step:
#   Y-Z 방향 격자 해상도다. 값이 작을수록 정점/삼각형 수가 늘고 계산이 무거워진다.
# --amplitude:
#   x 방향 roughness 크기다. 값이 크면 막장면 요철이 커진다.
# --corr-length:
#   roughness의 공간 상관 길이다. 값이 크면 더 부드럽고 완만한 형상이 만들어진다.
# --base-x:
#   --num-faces 모드에서 첫 번째 face의 기준 x 위치다.
# --face-step:
#   --num-faces 모드에서 face 간 x 간격이다.
# --num-faces:
#   --face-x-csv를 쓰지 않을 때 생성할 face 개수다.
# --face-x-csv:
#   생성할 face들의 x 위치를 직접 지정한다. 있으면 --num-faces, --face-step보다 우선한다.
# --seed-base:
#   난수 seed 시작값이다. face별로 서로 다른 roughness를 만들기 위해 face index를 더해 사용한다.
# --out-h5:
#   outdir 아래에 저장할 HDF5 파일 이름이다.
# --merge-collection-into-hdf5:
#   생성한 multi-face collection을 기존 DFN HDF5 안의 /rough_faces 구조로 병합한다.
# --merge-into-hdf5:
#   이전 단일-face 워크플로우 호환용 alias다. 내부적으로 collection 병합으로 처리된다.
```
"""

# =============================================================================
# 파일 역할:
#   터널 단면 polygon(y, z) 안쪽에, x 방향으로 미세한 요철(roughness)을 준
#   합성(synthetic) 막장면 mesh 여러 장(multi-face collection)을 생성한다.
#   생성한 요철 막장면은 DFN(단열망) trace 검출/검증의 관측면으로 사용된다.
#
# 주요 입력:
#   - 터널 단면 polygon DAT 파일 (--tunnel-dat): [y, z] 단면 형상
#   - 격자/roughness 파라미터 (--grid-step, --amplitude, --corr-length 등)
#   - face x 위치 지정 (--face-x-csv 또는 --base-x/--face-step/--num-faces)
#
# 주요 출력:
#   - multi-face rough mesh collection HDF5 (/rough_faces 하위에 face별 mesh)
#   - 첫 face 미리보기 PNG (mask / roughness field / 3D mesh)
#   - (옵션) 기존 DFN HDF5에 /rough_faces 로 병합
#
# 핵심 처리 흐름:
#   1) DAT에서 단면 polygon 읽기 → 반시계(CCW) 방향으로 정규화
#   2) 단면 bounding box에 규칙 격자 생성 → polygon 내부 격자점 mask 계산
#   3) face마다 seed를 바꿔 x 방향 roughness field 생성
#   4) 내부 격자를 3D 정점/삼각형 mesh로 삼각분할
#   5) collection HDF5 저장 + 미리보기 PNG 저장 + (옵션) 기존 HDF5 병합
# =============================================================================

import argparse
import os
import re
from typing import List, Tuple

import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter


# 터널 단면 DAT 파일을 읽어 [y, z] polygon 좌표 배열을 만든다.
#   인자: dat_path(입력 DAT 경로), scale(원본 단위→m 변환 계수, 기본 mm→m)
#   반환: (N, 2) float64 배열 (열0=y, 열1=z)
def load_tunnel_polygon_from_dat(dat_path: str, scale: float = 0.001) -> np.ndarray:
    """DAT 파일에서 터널 단면 [y, z] 좌표를 읽는다."""
    # 각 줄에서 "(y, z)" 패턴을 정규식으로 찾아 scale을 곱해 누적한다.
    poly_y = []
    poly_z = []
    with open(dat_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(r"\(\s*([\d\.-]+),\s*([\d\.-]+)\)", line)
            if not match:
                continue
            poly_y.append(float(match.group(1)) * scale)
            poly_z.append(float(match.group(2)) * scale)

    # 좌표를 하나도 못 읽으면 잘못된 파일이므로 예외 발생.
    if not poly_y:
        raise ValueError(f"터널 단면 polygon을 읽지 못했습니다: {dat_path}")

    # (N, 2) 형태로 합쳐 반환.
    return np.column_stack([poly_y, poly_z]).astype(np.float64)


# polygon의 부호 있는 면적(신발끈 공식)을 계산한다. 부호로 회전 방향을 판별한다.
#   인자: poly_yz((N,2) 좌표)  /  반환: 부호 있는 면적(float, +면 CCW)
def signed_polygon_area(poly_yz: np.ndarray) -> float:
    """폴리곤 방향 판별용 부호 있는 면적."""
    y = poly_yz[:, 0]
    z = poly_yz[:, 1]
    return 0.5 * float(np.dot(y, np.roll(z, -1)) - np.dot(z, np.roll(y, -1)))


# polygon 정점 순서를 반시계(CCW)로 통일한다. 내부 판정을 안정적으로 만든다.
#   인자: poly_yz((N,2))  /  반환: CCW로 정렬된 (N,2) 복사본
def ensure_ccw_polygon(poly_yz: np.ndarray) -> np.ndarray:
    """기하 판정의 일관성을 위해 polygon을 반시계 방향으로 맞춘다."""
    # 면적 부호가 음수이면 시계 방향이므로 역순으로 뒤집는다.
    if signed_polygon_area(poly_yz) < 0.0:
        return poly_yz[::-1].copy()
    return poly_yz.copy()


# 단면 bounding box를 grid_step 간격으로 나눈 규칙 격자를 생성한다.
#   인자: poly_yz(단면 좌표), grid_step(격자 간격 m)
#   반환: (grid_y, grid_z) meshgrid 2D 배열 튜플
def build_regular_grid(poly_yz: np.ndarray, grid_step: float) -> Tuple[np.ndarray, np.ndarray]:
    """터널 단면 bounding box 안에 규칙 격자를 만든다."""
    # 단면의 y, z 최소/최대로 bounding box 범위를 정한다.
    y_min = float(np.min(poly_yz[:, 0]))
    y_max = float(np.max(poly_yz[:, 0]))
    z_min = float(np.min(poly_yz[:, 1]))
    z_max = float(np.max(poly_yz[:, 1]))

    # y, z 각 축 좌표를 만든 뒤 2D meshgrid로 확장한다.
    y_coords = np.arange(y_min, y_max + 0.5 * grid_step, grid_step, dtype=np.float64)
    z_coords = np.arange(z_min, z_max + 0.5 * grid_step, grid_step, dtype=np.float64)
    grid_y, grid_z = np.meshgrid(y_coords, z_coords)
    return grid_y, grid_z


# 격자점이 polygon 내부인지 아닌지 판별하는 boolean mask를 만든다.
#   인자: poly_yz(단면), grid_y/grid_z(격자 좌표)
#   반환: grid와 같은 shape의 bool 배열 (True=내부)
def build_inside_mask(poly_yz: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray) -> np.ndarray:
    """격자점 중 터널 polygon 내부에 있는 점만 남긴다."""
    # matplotlib Path의 점 포함 판정으로 내부 여부를 벡터화하여 계산한다.
    path = Path(poly_yz)
    points_yz = np.column_stack([grid_y.ravel(), grid_z.ravel()])
    mask = path.contains_points(points_yz, radius=1e-10)
    return mask.reshape(grid_y.shape)


# face 표면의 x 방향 요철(roughness) field를 생성한다.
#   원리: white noise를 gaussian smoothing해 공간 상관을 준 뒤 진폭을 맞춘다.
#   인자: shape(격자 크기), amplitude(RMS 진폭 m), corr_length(상관 길이 m),
#         grid_step(격자 간격 m), seed(난수 시드)
#   반환: shape 크기의 x offset 배열(float64)
def generate_rough_field(
    shape: Tuple[int, int],
    amplitude: float,
    corr_length: float,
    grid_step: float,
    seed: int,
) -> np.ndarray:
    """face 법선 방향 x 오프셋 roughness field를 만든다."""
    # 1) seed로 재현 가능한 white noise를 생성한다.
    rng = np.random.default_rng(seed)
    white_noise = rng.standard_normal(shape)
    # 2) 상관 길이(m)를 격자 셀 단위 sigma로 바꿔 gaussian smoothing한다.
    sigma_cells = max(corr_length / max(grid_step, 1e-9), 1e-6)
    smooth = gaussian_filter(white_noise, sigma=sigma_cells, mode="nearest")
    # 3) 평균을 0으로 맞춘다(면 전체 이동 성분 제거).
    smooth = smooth - np.mean(smooth)
    std = float(np.std(smooth))
    # 4) 분산이 거의 0이면 평평한 면(모두 0)으로 처리한다.
    if std < 1e-12:
        return np.zeros(shape, dtype=np.float64)
    # 5) 표준편차로 정규화 후 목표 진폭(amplitude)을 곱해 RMS를 맞춘다.
    return amplitude * (smooth / std)


# 내부 mask 격자를 3D 삼각형 mesh로 변환한다.
#   정점 x = face_x + rough_x(요철), y/z = 격자 좌표.
#   인자: grid_y/grid_z(격자), mask(내부 여부), rough_x(x offset), face_x(기준 x)
#   반환: (vertices_xyz (M,3), triangles (K,3) 인덱스) 튜플
def triangulate_masked_grid(
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    mask: np.ndarray,
    rough_x: np.ndarray,
    face_x: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """polygon 내부 격자를 3D mesh 정점/삼각형으로 바꾼다."""
    # 격자 각 셀의 정점 인덱스를 담을 표(-1=미사용)를 준비한다.
    n_rows, n_cols = grid_y.shape
    vertex_index = -np.ones((n_rows, n_cols), dtype=np.int32)
    vertices = []

    # 1) 내부 격자점마다 3D 정점을 만들고, 셀 위치에 정점 인덱스를 기록한다.
    next_idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            if not mask[r, c]:
                continue
            vertex_index[r, c] = next_idx
            vertices.append(
                [
                    face_x + rough_x[r, c],
                    float(grid_y[r, c]),
                    float(grid_z[r, c]),
                ]
            )
            next_idx += 1

    # 2) 네 꼭짓점이 모두 내부인 셀만 두 개의 삼각형으로 분할한다.
    triangles = []
    for r in range(n_rows - 1):
        for c in range(n_cols - 1):
            if not all([mask[r, c], mask[r, c + 1], mask[r + 1, c], mask[r + 1, c + 1]]):
                continue
            v00 = int(vertex_index[r, c])
            v01 = int(vertex_index[r, c + 1])
            v10 = int(vertex_index[r + 1, c])
            v11 = int(vertex_index[r + 1, c + 1])
            triangles.append([v00, v01, v11])
            triangles.append([v00, v11, v10])

    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int32)


# 생성할 face들의 x 위치 목록을 결정한다.
#   우선순위: --face-x-csv(명시적 목록) > --base-x/--face-step/--num-faces(등간격)
#   인자: args(파싱된 CLI 인자)  /  반환: face x 위치 배열(float64)
def resolve_face_positions(args: argparse.Namespace) -> np.ndarray:
    """생성할 face x 위치들을 결정한다."""
    # CSV로 x 목록이 주어지면 그대로 파싱해 사용한다.
    if args.face_x_csv:
        values = [float(x.strip()) for x in args.face_x_csv.split(",") if x.strip()]
        return np.asarray(values, dtype=np.float64)

    if args.num_faces < 1:
        raise ValueError("--num-faces는 1 이상이어야 합니다.")

    # base_x에서 face_step 간격으로 num_faces개 위치를 등간격 생성한다.
    return args.base_x + np.arange(args.num_faces, dtype=np.float64) * args.face_step


# multi-face rough mesh collection을 독립 HDF5 파일로 저장한다.
#   구조: /rough_faces/face_XXXXXX/{mesh, field, meta}, /tunnel, /grid
#   인자: out_path(저장 경로), poly_yz, grid_y/grid_z, face_results(face별 dict 목록)
#   반환: 없음(파일 저장)
def save_collection_hdf5(
    out_path: str,
    poly_yz: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    face_results: List[dict],
) -> None:
    """multi-face rough mesh collection을 독립 HDF5로 저장한다."""
    with h5py.File(out_path, "w") as f:
        faces_grp = f.create_group("rough_faces")
        # face마다 mesh(정점/삼각형), field(mask/rough_x), meta(식별정보)를 저장한다.
        for result in face_results:
            face_key = f"face_{result['face_id']:06d}"
            grp = faces_grp.create_group(face_key)

            mesh = grp.create_group("mesh")
            mesh.create_dataset("vertices_xyz", data=result["vertices_xyz"].astype(np.float32))
            mesh.create_dataset("triangles", data=result["triangles"].astype(np.int32))

            field = grp.create_group("field")
            field.create_dataset("inside_mask", data=result["inside_mask"].astype(np.uint8))
            field.create_dataset("rough_x", data=result["rough_x"].astype(np.float32))

            meta = grp.create_group("meta")
            meta.create_dataset("face_id", data=np.array([result["face_id"]], dtype=np.int32))
            meta.create_dataset("face_x", data=np.array([result["face_x"]], dtype=np.float32))
            meta.create_dataset("seed", data=np.array([result["seed"]], dtype=np.int32))
            meta.create_dataset("source_name", data=np.bytes_(face_key))

        # 공통 정보: 터널 단면 polygon과 격자 좌표를 함께 저장한다.
        tunnel = f.create_group("tunnel")
        tunnel.create_dataset("poly_yz", data=poly_yz.astype(np.float32))

        grid = f.create_group("grid")
        grid.create_dataset("grid_y", data=grid_y.astype(np.float32))
        grid.create_dataset("grid_z", data=grid_z.astype(np.float32))


# 생성한 collection을 기존 DFN HDF5 파일에 /rough_faces 로 덧붙여(병합) 저장한다.
#   기존 rough 관련 그룹은 지우고 새로 쓰며, 단일 face면 하위호환 구조도 남긴다.
#   인자: target_h5_path(병합 대상 기존 HDF5), poly_yz, grid_y/grid_z, face_results
#   반환: 없음(파일 갱신)
def merge_collection_into_existing_hdf5(
    target_h5_path: str,
    poly_yz: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    face_results: List[dict],
) -> None:
    """생성한 multi-face collection을 기존 DFN HDF5에 /rough_faces 로 병합한다."""
    # 'a'(append) 모드로 열어 기존 DFN 데이터를 보존하며 rough_faces만 갱신한다.
    with h5py.File(target_h5_path, "a") as f:
        # 기존 /rough_faces가 있으면 지우고 새로 생성한다.
        if "rough_faces" in f:
            del f["rough_faces"]
        faces_grp = f.create_group("rough_faces")

        # face별 mesh/field/meta를 collection과 동일한 구조로 기록한다.
        for result in face_results:
            face_key = f"face_{result['face_id']:06d}"
            grp = faces_grp.create_group(face_key)

            mesh = grp.create_group("mesh")
            mesh.create_dataset("vertices_xyz", data=result["vertices_xyz"].astype(np.float32))
            mesh.create_dataset("triangles", data=result["triangles"].astype(np.int32))

            field = grp.create_group("field")
            field.create_dataset("inside_mask", data=result["inside_mask"].astype(np.uint8))
            field.create_dataset("rough_x", data=result["rough_x"].astype(np.float32))

            meta = grp.create_group("meta")
            meta.create_dataset("face_id", data=np.array([result["face_id"]], dtype=np.int32))
            meta.create_dataset("face_x", data=np.array([result["face_x"]], dtype=np.float32))
            meta.create_dataset("seed", data=np.array([result["seed"]], dtype=np.int32))
            meta.create_dataset("source_name", data=np.bytes_(face_key))

        # 예전 단일-face 그룹(/rough_face)이 있으면 정리한다.
        if "rough_face" in f:
            del f["rough_face"]

        # 단일 face backward compatibility
        # face가 하나뿐이면 예전 워크플로우가 읽는 /rough_face 구조도 함께 남긴다.
        if len(face_results) == 1:
            result = face_results[0]
            grp = f.create_group("rough_face")
            mesh = grp.create_group("mesh")
            mesh.create_dataset("vertices_xyz", data=result["vertices_xyz"].astype(np.float32))
            mesh.create_dataset("triangles", data=result["triangles"].astype(np.int32))
            meta = grp.create_group("meta")
            meta.create_dataset("base_x", data=np.array([result["face_x"]], dtype=np.float32))

        # 기존 파일에 터널 단면 정보가 없을 때만 새로 추가한다(덮어쓰기 방지).
        if "/tunnel/poly_YZ" not in f and "tunnel" not in f:
            tunnel = f.create_group("tunnel")
            tunnel.create_dataset("poly_yz", data=poly_yz.astype(np.float32))

        # 격자 좌표는 rough_face_meta에 갱신 저장하고, 병합 출처를 attribute로 남긴다.
        meta = f.require_group("rough_face_meta")
        if "grid_y" in meta:
            del meta["grid_y"]
        if "grid_z" in meta:
            del meta["grid_z"]
        meta.create_dataset("grid_y", data=grid_y.astype(np.float32))
        meta.create_dataset("grid_z", data=grid_z.astype(np.float32))
        meta.attrs["merged_by"] = "generate_synthetic_rough_face_mesh.py"


# 첫 face의 중간 결과를 3-패널(내부 mask / roughness field / 3D mesh) PNG로 저장한다.
#   생성 결과를 눈으로 검증하기 위한 시각화 함수(계산 로직 아님).
#   인자: out_png(저장 경로) 및 시각화에 필요한 격자/mask/mesh 데이터들
#   반환: 없음(이미지 저장)
def plot_intermediate_visualizations(
    out_png: str,
    poly_yz: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    mask: np.ndarray,
    rough_x: np.ndarray,
    vertices_xyz: np.ndarray,
    triangles: np.ndarray,
    face_x: float,
) -> None:
    """첫 번째 face를 빠르게 확인할 수 있는 미리보기 이미지를 저장한다."""
    fig = plt.figure(figsize=(16, 5))

    # 패널 1: 내부 mask와 단면 polygon 외곽선.
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("Inside Mask")
    ax1.pcolormesh(grid_y, grid_z, mask.astype(float), shading="nearest", cmap="Greys")
    poly_closed = np.vstack([poly_yz, poly_yz[0]])
    ax1.plot(poly_closed[:, 0], poly_closed[:, 1], color="crimson", linewidth=2.0)
    ax1.set_xlabel("Y (m)")
    ax1.set_ylabel("Z (m)")
    ax1.set_aspect("equal")

    # 패널 2: 내부 영역의 roughness field(x offset) 컬러맵.
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("Roughness Field (X Offset)")
    rough_plot = np.where(mask, rough_x, np.nan)
    im = ax2.pcolormesh(grid_y, grid_z, rough_plot, shading="nearest", cmap="viridis")
    ax2.plot(poly_closed[:, 0], poly_closed[:, 1], color="white", linewidth=1.5)
    ax2.set_xlabel("Y (m)")
    ax2.set_ylabel("Z (m)")
    ax2.set_aspect("equal")
    fig.colorbar(im, ax=ax2, shrink=0.85, label="X offset (m)")

    # 패널 3: 생성된 3D rough face mesh(trisurf) 표시.
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.set_title(f"Synthetic Rough Face Mesh @ x={face_x:.2f} m")
    if len(vertices_xyz) > 0 and len(triangles) > 0:
        tri = mtri.Triangulation(vertices_xyz[:, 1], vertices_xyz[:, 2], triangles)
        ax3.plot_trisurf(
            vertices_xyz[:, 0],
            vertices_xyz[:, 1],
            vertices_xyz[:, 2],
            triangles=tri.triangles,
            cmap="viridis",
            linewidth=0.15,
            edgecolor="k",
            alpha=0.95,
        )
    ax3.set_xlim(-5.0, 5.0)
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_zlabel("Z (m)")

    # 레이아웃 정리 후 파일로 저장하고 figure를 닫아 메모리를 해제한다.
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


# 생성된 face별 정점/삼각형 수와 roughness 표준편차를 콘솔에 요약 출력한다.
#   인자: face_results(face별 결과 dict 목록)  /  반환: 없음(출력)
def print_summary(face_results: List[dict]) -> None:
    """face별 생성 결과를 간단히 요약한다."""
    print(f"[*] Generated {len(face_results):,} rough faces.")
    for result in face_results:
        inside_values = result["rough_x"][result["inside_mask"]]
        print(
            f"    - Face {result['face_id']:03d} @ x={result['face_x']:.2f} m: "
            f"{len(result['vertices_xyz']):,} vertices, {len(result['triangles']):,} triangles, "
            f"roughness std={inside_values.std():.4f} m"
        )


# CLI 진입점: 인자 파싱 → 단면/격자/roughness 계산 → face mesh 생성 → 저장/시각화/병합.
#   인자: 없음(argparse로 CLI 인자 읽음)  /  반환: 없음
def main() -> None:
    # CLI 인자 정의 및 파싱.
    parser = argparse.ArgumentParser(
        description="터널 단면 polygon 내부에 multi-face synthetic rough face mesh collection을 생성한다."
    )
    parser.add_argument("--tunnel-dat", required=True, help="터널 단면 polygon DAT 파일")
    parser.add_argument("--outdir", default="storage/output/rough_face_mesh_collection", help="출력 폴더")
    parser.add_argument("--grid-step", type=float, default=0.20, help="Y-Z 격자 간격 (m)")
    parser.add_argument("--amplitude", type=float, default=0.05, help="roughness RMS 진폭 (m)")
    parser.add_argument("--corr-length", type=float, default=1.00, help="roughness 상관 길이 (m)")
    parser.add_argument("--base-x", type=float, default=0.0, help="기준 face x 위치 (m)")
    parser.add_argument("--face-step", type=float, default=3.0, help="face 간격 (m)")
    parser.add_argument("--num-faces", type=int, default=1, help="생성할 face 개수")
    parser.add_argument("--face-x-csv", help='명시적 face x 목록, 예: "0,3,6,9"')
    parser.add_argument("--seed-base", type=int, default=42, help="base 난수 시드")
    parser.add_argument("--out-h5", default="synthetic_rough_face_collection.h5", help="출력 HDF5 파일명")
    parser.add_argument("--merge-collection-into-hdf5", help="생성 결과를 기존 DFN HDF5의 /rough_faces 로 병합")
    parser.add_argument("--merge-into-hdf5", help="이전 옵션 호환용 alias. 내부적으로 collection 병합을 수행")
    args = parser.parse_args()

    # 출력 폴더를 준비한다.
    os.makedirs(args.outdir, exist_ok=True)

    # deprecated 옵션(--merge-into-hdf5)을 현재 collection 병합 옵션으로 흡수한다.
    if args.merge_into_hdf5:
        print("[*] Warning: --merge-into-hdf5 는 deprecated 입니다. collection 방식으로 병합합니다.")
        if args.merge_collection_into_hdf5 and args.merge_collection_into_hdf5 != args.merge_into_hdf5:
            raise ValueError("--merge-into-hdf5 와 --merge-collection-into-hdf5 값이 다릅니다.")
        args.merge_collection_into_hdf5 = args.merge_into_hdf5

    # 단면 polygon 읽어 CCW 정규화 → 격자 생성 → 내부 mask → face x 위치 목록 계산.
    poly_yz = ensure_ccw_polygon(load_tunnel_polygon_from_dat(args.tunnel_dat))
    grid_y, grid_z = build_regular_grid(poly_yz, args.grid_step)
    inside_mask = build_inside_mask(poly_yz, grid_y, grid_z)
    face_x_values = resolve_face_positions(args)

    # face마다: seed 결정 → roughness field 생성 → 내부만 남김 → mesh 삼각분할.
    face_results = []
    for face_idx, face_x in enumerate(face_x_values, start=1):
        # face index마다 seed를 바꿔 서로 다른 요철 패턴을 만든다.
        seed = args.seed_base + face_idx - 1
        rough_x = generate_rough_field(
            shape=grid_y.shape,
            amplitude=args.amplitude,
            corr_length=args.corr_length,
            grid_step=args.grid_step,
            seed=seed,
        )
        # polygon 바깥 격자점의 요철은 0으로 만들어 내부에만 요철을 남긴다.
        rough_x = np.where(inside_mask, rough_x, 0.0)
        vertices_xyz, triangles = triangulate_masked_grid(
            grid_y=grid_y,
            grid_z=grid_z,
            mask=inside_mask,
            rough_x=rough_x,
            face_x=float(face_x),
        )
        face_results.append(
            {
                "face_id": face_idx,
                "face_x": float(face_x),
                "seed": seed,
                "inside_mask": inside_mask,
                "rough_x": rough_x,
                "vertices_xyz": vertices_xyz,
                "triangles": triangles,
            }
        )

    # 출력 HDF5/PNG 경로를 구성한다.
    h5_path = os.path.join(args.outdir, args.out_h5)
    png_path = os.path.join(args.outdir, "synthetic_rough_face_mesh_preview.png")

    # collection HDF5 저장.
    save_collection_hdf5(
        out_path=h5_path,
        poly_yz=poly_yz,
        grid_y=grid_y,
        grid_z=grid_z,
        face_results=face_results,
    )
    # 첫 face 미리보기 PNG 저장.
    plot_intermediate_visualizations(
        out_png=png_path,
        poly_yz=poly_yz,
        grid_y=grid_y,
        grid_z=grid_z,
        mask=face_results[0]["inside_mask"],
        rough_x=face_results[0]["rough_x"],
        vertices_xyz=face_results[0]["vertices_xyz"],
        triangles=face_results[0]["triangles"],
        face_x=face_results[0]["face_x"],
    )

    # 병합 옵션이 있으면 기존 DFN HDF5에 rough_faces를 병합한다.
    if args.merge_collection_into_hdf5:
        merge_collection_into_existing_hdf5(
            target_h5_path=args.merge_collection_into_hdf5,
            poly_yz=poly_yz,
            grid_y=grid_y,
            grid_z=grid_z,
            face_results=face_results,
        )

    # 생성 결과 요약과 저장 경로를 출력한다.
    print_summary(face_results)
    print(f"[*] HDF5 saved to: {h5_path}")
    print(f"[*] Preview saved to: {png_path}")
    if args.merge_collection_into_hdf5:
        print(f"[*] Merged into existing DFN HDF5: {args.merge_collection_into_hdf5}")


if __name__ == "__main__":
    main()
