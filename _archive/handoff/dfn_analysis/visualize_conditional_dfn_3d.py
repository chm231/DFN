"""Interactive 3D visualisation of the conditional DFN with tunnel faces.

Renders the discs in ``conditional_dfn.csv`` (visible + hidden) as oriented
circular polygons in global 3D space, together with the observation faces
(tunnel cross-section windows at each face x). Coloured by fracture set.

Visualisation only — no geometry is computed or changed here (CLAUDE.md §12).

Coordinate convention: x = East = tunnel advance, y = North, z = Up.
Discs are clipped to a local box around the tunnel so the picture stays
readable (the full DFN spans hundreds of metres).

Usage
-----
    python dfn_analysis/visualize_conditional_dfn_3d.py            # interactive window + screenshot
    python dfn_analysis/visualize_conditional_dfn_3d.py --no-window  # screenshot only (headless)
    python dfn_analysis/visualize_conditional_dfn_3d.py --no-window --html  # + interactive HTML (vtk.js)

HTML export needs trame:  pip install "pyvista[jupyter]"  (trame, trame-vtk, trame-vuetify).
"""
# ======================================================================
# [파일 역할]
#   조건부 DFN(conditional_dfn.csv, 가시+은닉 디스크)을 PyVista로 3D 대화형
#   시각화한다. 각 디스크를 방향을 가진 원형 폴리곤으로 그리고, 터널 관측 면과
#   면 위의 트레이스(관측=파랑, 조건화=빨강)를 함께 표시한다.
#   기하 계산은 하지 않으며 오직 시각화 목적(CLAUDE.md §12).
#
# [주요 입력] (--pipeline-dir 아래)
#   - conditional_hidden/conditional_dfn.csv : 시각화할 디스크 목록
#   - dfn_export_for_python.h5               : 터널 단면 폴리곤(YZ)
#   - trace_dataset/trace_dataset_3d.csv     : 관측 트레이스 및 면 x위치
#
# [주요 출력]
#   - 대화형 PyVista 창(기본) 또는 --no-window 시 스크린샷 PNG
#     (기본 경로: <pipeline>/conditional_hidden/conditional_dfn_3d.png)
#   - (옵션) --html : 브라우저에서 회전·확대 가능한 대화형 HTML(vtk.js)
#     (기본 경로: <pipeline>/conditional_hidden/conditional_dfn_3d.html)
#     ※ trame 패키지 필요: pip install "pyvista[jupyter]" (미설치 시 안내 후 건너뜀)
#
# [핵심 처리 흐름]
#   1) 조건부 DFN CSV 및 터널 폴리곤/면 x위치 로드
#   2) 국소 뷰 박스로 디스크 필터링(은닉은 반경 임계로 정리)
#   3) 디스크를 와이어프레임 원으로, 관측 면을 반투명 폴리곤으로 렌더링
#   4) 관측/조건화 트레이스를 튜브로 오버레이
#   5) 카메라/범례 설정 후 표시 또는 스크린샷 저장
#
# [좌표 규약] x = East = 터널 굴진 방향, y = North, z = Up.
# ======================================================================
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import h5py
import pyvista as pv

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
# Reuse the trace geometry from the conditioning module (single source of truth).
from dfn_analysis.generate_conditional_hidden_dfn import (  # noqa: E402
    load_observed_traces, visible_trace_on_face, _ccw_polygon,
)

# Per-set colours (sets 1,2,3,5 are the reconstructed/conditioned powerlaw sets)
SET_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 5: "#d62728"}
FACE_COLOR = "#888888"
OBSERVED_COLOR = "#1f77b4"    # blue
CONDITIONED_COLOR = "#d62728"  # red


# ----------------------------------------------------------------------
# conditional_dfn.csv를 읽어 병렬 numpy 배열로 반환한다.
#   인자: path CSV 경로
#   반환: (centers, normals, radii, set_ids, sources) 각 배열
def load_conditional_dfn(path: Path):
    """Load discs from conditional_dfn.csv into parallel numpy arrays."""
    # 각 행에서 중심/법선/반경/세트/소스(visible|hidden)를 리스트로 수집
    centers, normals, radii, set_ids, sources = [], [], [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            centers.append([float(row["cx"]), float(row["cy"]), float(row["cz"])])
            normals.append([float(row["nx"]), float(row["ny"]), float(row["nz"])])
            radii.append(float(row["radius"]))
            set_ids.append(int(row["set_id"]))
            sources.append(row["source"])
    return (np.array(centers), np.array(normals), np.array(radii),
            np.array(set_ids), np.array(sources))


# 디스크 중심이 국소 뷰 박스 안에 있는지 여부의 불리언 마스크를 반환한다.
#   인자: centers (N,3) 중심 배열, box {xmin..zmax} 박스 경계
#   반환: (N,) 불리언 마스크
def in_box(centers: np.ndarray, box: dict) -> np.ndarray:
    """Boolean mask of disc centres inside the local viewing box."""
    return (
        (centers[:, 0] >= box["xmin"]) & (centers[:, 0] <= box["xmax"])
        & (centers[:, 1] >= box["ymin"]) & (centers[:, 1] <= box["ymax"])
        & (centers[:, 2] >= box["zmin"]) & (centers[:, 2] <= box["zmax"])
    )


# 각 단위 법선에 대해 디스크 평면 내 직교 기저 벡터 두 개(u, v)를 벡터화 계산.
#   인자: normals (N,3) 법선 배열
#   반환: (u, v) 각각 (N,3) 평면 내 직교 단위벡터
def _disc_basis(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two orthonormal in-plane vectors for each unit normal (vectorised)."""
    # 법선 정규화 후 참조축 선택(법선이 z에 가까우면 x축을 참조로 사용)
    n = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    ref = np.zeros_like(n)
    ref[:, 2] = 1.0
    ref[np.abs(n[:, 2]) > 0.95] = [1.0, 0.0, 0.0]
    # 외적으로 평면 내 직교 기저 u, v 생성 및 정규화
    u = np.cross(ref, n)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(n, u)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return u, v


# 방향을 가진 원형 디스크들을 하나의 폴리곤 PolyData 메시로 합친다.
#   인자: centers 중심, normals 법선, radii 반경, n_pts 원 둘레 분할 수
#   반환: 모든 디스크를 담은 pv.PolyData(각 디스크는 n_pts각형 폴리곤)
def build_disc_mesh(
    centers: np.ndarray, normals: np.ndarray, radii: np.ndarray, n_pts: int = 24
) -> pv.PolyData:
    """Combine oriented circular discs into a single PolyData of polygon faces."""
    if len(centers) == 0:
        return pv.PolyData()
    # 평면 기저(u,v)와 원 둘레 각도 샘플 준비
    u, v = _disc_basis(normals)
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    ct, st = np.cos(theta), np.sin(theta)
    # rim[i,k] = c[i] + r[i]*(cos θk * u[i] + sin θk * v[i])
    # 각 디스크의 둘레 점(rim) 좌표를 벡터화로 계산
    rim = (centers[:, None, :]
           + radii[:, None, None] * (ct[None, :, None] * u[:, None, :]
                                     + st[None, :, None] * v[:, None, :]))
    # 점 배열과 VTK 면 연결 정보(각 면: [정점수, 인덱스...]) 구성
    N = len(centers)
    pts = rim.reshape(N * n_pts, 3)
    idx = np.arange(N)[:, None] * n_pts + np.arange(n_pts)[None, :]
    faces = np.hstack([np.full((N, 1), n_pts), idx]).astype(np.int64).ravel()
    return pv.PolyData(pts, faces)


# 터널 관측 창(yz 폴리곤)을 평면 x=xf 위에 놓은 폴리곤 메시를 만든다.
#   인자: poly_yz (N,2) 창 정점, xf 면 x위치 / 반환: pv.PolyData 단일 폴리곤
def build_face_polygon(poly_yz: np.ndarray, xf: float) -> pv.PolyData:
    """Tunnel observation window (y-z polygon) placed at plane x = xf."""
    # yz 정점에 x=xf를 붙여 3D 점으로 만들고 단일 면으로 구성
    pts = np.column_stack([np.full(len(poly_yz), xf), poly_yz[:, 0], poly_yz[:, 1]])
    face = np.hstack([[len(poly_yz)], np.arange(len(poly_yz))]).astype(np.int64)
    return pv.PolyData(pts, face)


# 3D 선분 목록 [(p0,p1),...]을 하나의 라인 셀 PolyData로 합친다.
#   인자: segments 선분 리스트 / 반환: pv.PolyData(라인 셀들)
def build_lines(segments: List[Tuple[np.ndarray, np.ndarray]]) -> pv.PolyData:
    """Combine [(p0, p1), ...] 3D segments into one PolyData of line cells."""
    if not segments:
        return pv.PolyData()
    # 모든 끝점을 펼쳐 점 배열로 만들고, 각 선분을 [2, i0, i1] 라인 셀로 연결
    pts = np.array([p for seg in segments for p in seg], dtype=float)
    n = len(segments)
    lines = np.hstack([np.full((n, 1), 2),
                       np.arange(2 * n).reshape(n, 2)]).astype(np.int64).ravel()
    return pv.PolyData(pts, lines=lines)


# 가시 디스크들이 각 면에 남기는 트레이스(조건화 트레이스, 빨강)를 수집한다.
#   인자: centers/normals/radii 가시 디스크, face_xs 면 목록, poly_ccw 관측 창
#   반환: 트레이스 3D 선분 리스트 [(p0,p1),...]
def conditioned_face_segments(centers, normals, radii, face_xs, poly_ccw):
    """Traces the visible discs leave on each face (the conditioned traces)."""
    # 각 디스크를 모든 면에 재투영하여 유효한 트레이스만 모은다
    segs = []
    for c, nrm, r in zip(centers, normals, radii):
        for xf in face_xs:
            seg = visible_trace_on_face(c, nrm, r, xf, poly_ccw)
            if seg is not None:
                segs.append(seg)
    return segs


# ----------------------------------------------------------------------
# 엔트리 포인트: 조건부 DFN을 로드해 디스크/면/트레이스를 PyVista로 렌더링하고,
#                대화형 창 표시 또는 --no-window 시 스크린샷을 저장한다.
def main() -> None:
    # CLI 인자 정의(파이프라인 폴더, 뷰 박스 범위, 최소 반경, 트레이스/창 옵션 등)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pipeline-dir", type=Path,
                    default=REPO / "storage/output/pipeline_test_laxemar")
    ap.add_argument("--box-x", type=float, nargs=2, default=[-8.0, 11.0],
                    help="Local box x-range [m].")
    ap.add_argument("--box-half-yz", type=float, default=3.0,
                    help="Local box half-extent in y and z about the window [m].")
    ap.add_argument("--min-radius", type=float, default=1.5,
                    help="Only show HIDDEN discs with radius >= this [m] "
                         "(declutter; visible discs always shown). Set 0 for all.")
    ap.add_argument("--traces", action=argparse.BooleanOptionalAction, default=True,
                    help="Overlay observed (blue) and conditioned (red) face traces; "
                         "discs are drawn neutral grey so the traces read clearly.")
    ap.add_argument("--no-window", action="store_true",
                    help="Render off-screen and only save the screenshot.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Screenshot path (default: <pipeline>/conditional_hidden/conditional_dfn_3d.png).")
    ap.add_argument("--html", nargs="?", const="<default>", default=None,
                    help="대화형 HTML(vtk.js)로 export. 경로 생략 시 기본 "
                         "<pipeline>/conditional_hidden/conditional_dfn_3d.html 에 저장. "
                         "trame 패키지 필요: pip install \"pyvista[jupyter]\".")
    args = ap.parse_args()

    # 입력 CSV 경로와 출력 PNG 경로 결정 후 조건부 DFN 로드
    pdir = args.pipeline_dir
    dfn_csv = pdir / "conditional_hidden/conditional_dfn.csv"
    out_png = args.out or (pdir / "conditional_hidden/conditional_dfn_3d.png")

    centers, normals, radii, set_ids, sources = load_conditional_dfn(dfn_csv)

    # Tunnel window + observed face x-locations
    # 터널 창 폴리곤과 관측 면 x위치 목록 로드
    with h5py.File(pdir / "dfn_export_for_python.h5", "r") as f:
        poly_yz = np.array(f["tunnel/poly_YZ"])
    face_xs = sorted({round(float(r["face_x_m"]), 3)
                      for r in csv.DictReader(open(pdir / "trace_dataset/trace_dataset_3d.csv"))})

    # 창 y/z 범위에 half-extent를 더해 국소 뷰 박스 정의(x는 인자 범위)
    ymin, ymax = poly_yz[:, 0].min(), poly_yz[:, 0].max()
    zmin, zmax = poly_yz[:, 1].min(), poly_yz[:, 1].max()
    box = dict(
        xmin=args.box_x[0], xmax=args.box_x[1],
        ymin=ymin - args.box_half_yz, ymax=ymax + args.box_half_yz,
        zmin=zmin - args.box_half_yz, zmax=zmax + args.box_half_yz,
    )
    # 박스 내 디스크 마스크 계산. 가시는 모두 표시, 은닉은 작은 반경 제외로 정리
    in_local = in_box(centers, box)
    # Show all visible discs in the box; declutter hidden by a radius threshold.
    small_hidden = (sources == "hidden") & (radii < args.min_radius)
    mask = in_local & ~small_hidden
    n_dropped = int((in_local & small_hidden).sum())
    print(f"discs total={len(centers):,}  in local box={in_local.sum():,}  shown={mask.sum():,}  "
          f"(hidden dropped by r<{args.min_radius} m: {n_dropped:,})")
    print(f"  box x[{box['xmin']},{box['xmax']}] y[{box['ymin']:.1f},{box['ymax']:.1f}] "
          f"z[{box['zmin']:.1f},{box['zmax']:.1f}]")

    # PyVista 플로터 생성(--no-window면 오프스크린) 및 배경 흰색 설정
    plotter = pv.Plotter(off_screen=args.no_window, window_size=(1400, 900))
    plotter.set_background("white")

    # Discs as circle OUTLINES (wireframe). In trace mode discs are neutral grey
    # so the blue/red traces read unambiguously; otherwise coloured by set.
    # 표시 대상 은닉/가시 마스크 분리
    hid_m = mask & (sources == "hidden")
    vis_m = mask & (sources == "visible")
    # 트레이스 모드: 디스크는 중립 회색 와이어프레임(트레이스 색 강조)
    if args.traces:
        if hid_m.any():
            plotter.add_mesh(build_disc_mesh(centers[hid_m], normals[hid_m], radii[hid_m]),
                             color="#c0c0c0", style="wireframe", line_width=1, opacity=0.30)
        if vis_m.any():
            plotter.add_mesh(build_disc_mesh(centers[vis_m], normals[vis_m], radii[vis_m]),
                             color="#606060", style="wireframe", line_width=2, opacity=0.7,
                             label="Visible discs")
    # 비트레이스 모드: 세트별 색상으로 은닉/가시 디스크를 그림
    else:
        for s, color in SET_COLORS.items():
            sel_h = hid_m & (set_ids == s)
            if sel_h.any():
                plotter.add_mesh(build_disc_mesh(centers[sel_h], normals[sel_h], radii[sel_h]),
                                 color=color, style="wireframe", line_width=1, opacity=0.35)
            sel_v = vis_m & (set_ids == s)
            if sel_v.any():
                plotter.add_mesh(build_disc_mesh(centers[sel_v], normals[sel_v], radii[sel_v]),
                                 color=color, style="wireframe", line_width=3, opacity=1.0,
                                 label=f"Set {s}")
            else:
                plotter.add_mesh(pv.PolyData(np.zeros((1, 3))), color=color,
                                 point_size=1, label=f"Set {s}")

    # Tunnel observation faces (the only filled surfaces — spatial reference)
    # 관측 면(반투명 폴리곤)을 공간 기준으로 각 x위치에 그림
    for xf in face_xs:
        face = build_face_polygon(poly_yz, xf)
        plotter.add_mesh(face, color=FACE_COLOR, opacity=0.35,
                         show_edges=True, edge_color="black", line_width=3)
    plotter.add_mesh(build_face_polygon(poly_yz, face_xs[0]), color=FACE_COLOR,
                     opacity=0.0, label="Tunnel face")

    # Face traces: observed (blue) and conditioned = visible discs re-projected (red)
    # 면 트레이스 오버레이: 관측(파랑) 및 가시 디스크 재투영(빨강)을 튜브로 표시
    n_obs = n_cond = 0
    if args.traces:
        # 관측 트레이스 로드 및 가시 디스크로부터 조건화 트레이스 계산
        poly_ccw = _ccw_polygon(poly_yz)
        obs_by_face, _ = load_observed_traces(pdir / "trace_dataset/trace_dataset_3d.csv")
        obs_segs = [seg for segs in obs_by_face.values() for seg in segs]
        vis_all = sources == "visible"
        cond_segs = conditioned_face_segments(
            centers[vis_all], normals[vis_all], radii[vis_all], face_xs, poly_ccw)
        # 관측/조건화 선분을 라인 메시로 만들어 튜브 렌더링
        n_obs, n_cond = len(obs_segs), len(cond_segs)
        obs_lines = build_lines(obs_segs)
        if obs_lines.n_points:
            plotter.add_mesh(obs_lines.tube(radius=0.035), color=OBSERVED_COLOR,
                             label="Observed traces")
        cond_lines = build_lines(cond_segs)
        if cond_lines.n_points:
            plotter.add_mesh(cond_lines.tube(radius=0.035), color=CONDITIONED_COLOR,
                             label="Conditioned traces")
        print(f"traces: observed={n_obs}  conditioned={n_cond}")

    # 화면 좌상단 정보 텍스트/범례/축/카메라 위치 구성
    n_vis = int(vis_m.sum())
    n_hid = int(hid_m.sum())
    info = f"Conditional DFN (local box)\nvisible={n_vis}  hidden={n_hid}  faces={len(face_xs)}"
    if args.traces:
        info += f"\nobserved traces={n_obs} (blue)  conditioned={n_cond} (red)"
    plotter.add_text(info, position="upper_left", font_size=11, color="black")
    plotter.add_legend(bcolor="white", border=True)
    plotter.add_axes(xlabel="x (advance)", ylabel="y (North)", zlabel="z (Up)")
    mid_x = (box["xmin"] + box["xmax"]) / 2
    plotter.camera_position = [
        (mid_x + 14, box["ymin"] - 28, box["zmax"] + 12),  # camera: front-side-above
        (mid_x, 0.0, 0.0),                                 # focal: tunnel centre
        (0, 0, 1),                                          # up = z
    ]

    # (옵션) 대화형 HTML export: 브라우저에서 회전·확대 가능한 3D 뷰(vtk.js/trame).
    #   geometry(디스크·면·트레이스)는 그대로 직렬화된다. 2D 텍스트/범례는 vtk.js
    #   변환 특성상 생략될 수 있다. plotter.show()가 렌더 컨텍스트를 정리하므로,
    #   export 는 반드시 show()/screenshot 이전에 호출한다.
    if args.html is not None:
        html_path = (out_png.with_suffix(".html") if args.html == "<default>"
                     else Path(args.html))
        html_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            plotter.export_html(str(html_path))
            print(f"Interactive HTML written to {html_path}")
        except ImportError as e:
            print(f"[html] export 건너뜀 — 추가 패키지 필요: "
                  f"pip install \"pyvista[jupyter]\" (trame, trame-vtk, trame-vuetify). 상세: {e}")

    # 헤드리스 모드면 스크린샷 저장, 아니면 대화형 창 표시
    if args.no_window:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(out_png))
        print(f"Screenshot written to {out_png}")
    else:
        # Interactive: showing then screenshotting on close is unreliable
        # (render context is gone once the window closes). Use --no-window to
        # save an image.
        plotter.show()


if __name__ == "__main__":
    main()
