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


def load_tunnel_polygon_from_dat(dat_path: str, scale: float = 0.001) -> np.ndarray:
    """DAT 파일에서 터널 단면 [y, z] 좌표를 읽는다."""
    poly_y = []
    poly_z = []
    with open(dat_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(r"\(\s*([\d\.-]+),\s*([\d\.-]+)\)", line)
            if not match:
                continue
            poly_y.append(float(match.group(1)) * scale)
            poly_z.append(float(match.group(2)) * scale)

    if not poly_y:
        raise ValueError(f"터널 단면 polygon을 읽지 못했습니다: {dat_path}")

    return np.column_stack([poly_y, poly_z]).astype(np.float64)


def signed_polygon_area(poly_yz: np.ndarray) -> float:
    """폴리곤 방향 판별용 부호 있는 면적."""
    y = poly_yz[:, 0]
    z = poly_yz[:, 1]
    return 0.5 * float(np.dot(y, np.roll(z, -1)) - np.dot(z, np.roll(y, -1)))


def ensure_ccw_polygon(poly_yz: np.ndarray) -> np.ndarray:
    """기하 판정의 일관성을 위해 polygon을 반시계 방향으로 맞춘다."""
    if signed_polygon_area(poly_yz) < 0.0:
        return poly_yz[::-1].copy()
    return poly_yz.copy()


def build_regular_grid(poly_yz: np.ndarray, grid_step: float) -> Tuple[np.ndarray, np.ndarray]:
    """터널 단면 bounding box 안에 규칙 격자를 만든다."""
    y_min = float(np.min(poly_yz[:, 0]))
    y_max = float(np.max(poly_yz[:, 0]))
    z_min = float(np.min(poly_yz[:, 1]))
    z_max = float(np.max(poly_yz[:, 1]))

    y_coords = np.arange(y_min, y_max + 0.5 * grid_step, grid_step, dtype=np.float64)
    z_coords = np.arange(z_min, z_max + 0.5 * grid_step, grid_step, dtype=np.float64)
    grid_y, grid_z = np.meshgrid(y_coords, z_coords)
    return grid_y, grid_z


def build_inside_mask(poly_yz: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray) -> np.ndarray:
    """격자점 중 터널 polygon 내부에 있는 점만 남긴다."""
    path = Path(poly_yz)
    points_yz = np.column_stack([grid_y.ravel(), grid_z.ravel()])
    mask = path.contains_points(points_yz, radius=1e-10)
    return mask.reshape(grid_y.shape)


def generate_rough_field(
    shape: Tuple[int, int],
    amplitude: float,
    corr_length: float,
    grid_step: float,
    seed: int,
) -> np.ndarray:
    """face 법선 방향 x 오프셋 roughness field를 만든다."""
    rng = np.random.default_rng(seed)
    white_noise = rng.standard_normal(shape)
    sigma_cells = max(corr_length / max(grid_step, 1e-9), 1e-6)
    smooth = gaussian_filter(white_noise, sigma=sigma_cells, mode="nearest")
    smooth = smooth - np.mean(smooth)
    std = float(np.std(smooth))
    if std < 1e-12:
        return np.zeros(shape, dtype=np.float64)
    return amplitude * (smooth / std)


def triangulate_masked_grid(
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    mask: np.ndarray,
    rough_x: np.ndarray,
    face_x: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """polygon 내부 격자를 3D mesh 정점/삼각형으로 바꾼다."""
    n_rows, n_cols = grid_y.shape
    vertex_index = -np.ones((n_rows, n_cols), dtype=np.int32)
    vertices = []

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


def resolve_face_positions(args: argparse.Namespace) -> np.ndarray:
    """생성할 face x 위치들을 결정한다."""
    if args.face_x_csv:
        values = [float(x.strip()) for x in args.face_x_csv.split(",") if x.strip()]
        return np.asarray(values, dtype=np.float64)

    if args.num_faces < 1:
        raise ValueError("--num-faces는 1 이상이어야 합니다.")

    return args.base_x + np.arange(args.num_faces, dtype=np.float64) * args.face_step


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

        tunnel = f.create_group("tunnel")
        tunnel.create_dataset("poly_yz", data=poly_yz.astype(np.float32))

        grid = f.create_group("grid")
        grid.create_dataset("grid_y", data=grid_y.astype(np.float32))
        grid.create_dataset("grid_z", data=grid_z.astype(np.float32))


def merge_collection_into_existing_hdf5(
    target_h5_path: str,
    poly_yz: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    face_results: List[dict],
) -> None:
    """생성한 multi-face collection을 기존 DFN HDF5에 /rough_faces 로 병합한다."""
    with h5py.File(target_h5_path, "a") as f:
        if "rough_faces" in f:
            del f["rough_faces"]
        faces_grp = f.create_group("rough_faces")

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

        if "rough_face" in f:
            del f["rough_face"]

        # 단일 face backward compatibility
        if len(face_results) == 1:
            result = face_results[0]
            grp = f.create_group("rough_face")
            mesh = grp.create_group("mesh")
            mesh.create_dataset("vertices_xyz", data=result["vertices_xyz"].astype(np.float32))
            mesh.create_dataset("triangles", data=result["triangles"].astype(np.int32))
            meta = grp.create_group("meta")
            meta.create_dataset("base_x", data=np.array([result["face_x"]], dtype=np.float32))

        if "/tunnel/poly_YZ" not in f and "tunnel" not in f:
            tunnel = f.create_group("tunnel")
            tunnel.create_dataset("poly_yz", data=poly_yz.astype(np.float32))

        meta = f.require_group("rough_face_meta")
        if "grid_y" in meta:
            del meta["grid_y"]
        if "grid_z" in meta:
            del meta["grid_z"]
        meta.create_dataset("grid_y", data=grid_y.astype(np.float32))
        meta.create_dataset("grid_z", data=grid_z.astype(np.float32))
        meta.attrs["merged_by"] = "generate_synthetic_rough_face_mesh.py"


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

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("Inside Mask")
    ax1.pcolormesh(grid_y, grid_z, mask.astype(float), shading="nearest", cmap="Greys")
    poly_closed = np.vstack([poly_yz, poly_yz[0]])
    ax1.plot(poly_closed[:, 0], poly_closed[:, 1], color="crimson", linewidth=2.0)
    ax1.set_xlabel("Y (m)")
    ax1.set_ylabel("Z (m)")
    ax1.set_aspect("equal")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("Roughness Field (X Offset)")
    rough_plot = np.where(mask, rough_x, np.nan)
    im = ax2.pcolormesh(grid_y, grid_z, rough_plot, shading="nearest", cmap="viridis")
    ax2.plot(poly_closed[:, 0], poly_closed[:, 1], color="white", linewidth=1.5)
    ax2.set_xlabel("Y (m)")
    ax2.set_ylabel("Z (m)")
    ax2.set_aspect("equal")
    fig.colorbar(im, ax=ax2, shrink=0.85, label="X offset (m)")

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

    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


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


def main() -> None:
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

    os.makedirs(args.outdir, exist_ok=True)

    if args.merge_into_hdf5:
        print("[*] Warning: --merge-into-hdf5 는 deprecated 입니다. collection 방식으로 병합합니다.")
        if args.merge_collection_into_hdf5 and args.merge_collection_into_hdf5 != args.merge_into_hdf5:
            raise ValueError("--merge-into-hdf5 와 --merge-collection-into-hdf5 값이 다릅니다.")
        args.merge_collection_into_hdf5 = args.merge_into_hdf5

    poly_yz = ensure_ccw_polygon(load_tunnel_polygon_from_dat(args.tunnel_dat))
    grid_y, grid_z = build_regular_grid(poly_yz, args.grid_step)
    inside_mask = build_inside_mask(poly_yz, grid_y, grid_z)
    face_x_values = resolve_face_positions(args)

    face_results = []
    for face_idx, face_x in enumerate(face_x_values, start=1):
        seed = args.seed_base + face_idx - 1
        rough_x = generate_rough_field(
            shape=grid_y.shape,
            amplitude=args.amplitude,
            corr_length=args.corr_length,
            grid_step=args.grid_step,
            seed=seed,
        )
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

    h5_path = os.path.join(args.outdir, args.out_h5)
    png_path = os.path.join(args.outdir, "synthetic_rough_face_mesh_preview.png")

    save_collection_hdf5(
        out_path=h5_path,
        poly_yz=poly_yz,
        grid_y=grid_y,
        grid_z=grid_z,
        face_results=face_results,
    )
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

    if args.merge_collection_into_hdf5:
        merge_collection_into_existing_hdf5(
            target_h5_path=args.merge_collection_into_hdf5,
            poly_yz=poly_yz,
            grid_y=grid_y,
            grid_z=grid_z,
            face_results=face_results,
        )

    print_summary(face_results)
    print(f"[*] HDF5 saved to: {h5_path}")
    print(f"[*] Preview saved to: {png_path}")
    if args.merge_collection_into_hdf5:
        print(f"[*] Merged into existing DFN HDF5: {args.merge_collection_into_hdf5}")


if __name__ == "__main__":
    main()
