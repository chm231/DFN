"""
plot_2d_trace_map.py
터널 방향(X축) 2D Trace Map 인터랙티브 시각화

지정된 X 단면(Slice)을 통과하는 3D 원판 균열들의 2D 교차선(Trace)을
수학적으로 계산하여, 터널 폴리곤과 함께 Matplotlib에 시각화합니다.
하단의 슬라이더(Slider)를 조작하여 실시간으로 다른 X 좌표 위치의 단면을 볼 수 있습니다.

사용법:
    & "C:\Users\user\miniconda3\python.exe" "c:\Users\user\OneDrive\2026-1\3D DFN modeling\dfn generator v1\python\plot_2d_trace_map.py" --input "c:\Users\user\OneDrive\2026-1\3D DFN modeling\dfn generator v1\src\main\dfn_output_cube250m\dfn_export_for_python.h5"
"""

import argparse
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection

def load_data(h5_path):
    print(f"📂 HDF5 로드 중: {h5_path}")
    data = {}
    with h5py.File(h5_path, 'r') as f:
        # MATLAB에서 넘어온 [N x 3] 형태에 대한 전치 보정
        raw_c = f['/fractures/centers'][:]
        raw_n = f['/fractures/normals'][:]
        data['centers'] = raw_c.T if raw_c.shape[0] == 3 and raw_c.shape[0] < raw_c.shape[1] else raw_c
        data['normals'] = raw_n.T if raw_n.shape[0] == 3 and raw_n.shape[0] < raw_n.shape[1] else raw_n
        data['radii']   = f['/fractures/radii'][:].ravel()
        
        if '/tunnel/poly_YZ' in f:
            raw_p = f['/tunnel/poly_YZ'][:]
            data['poly_YZ'] = raw_p.T if raw_p.shape[0] == 2 and raw_p.shape[0] < raw_p.shape[1] else raw_p
        else:
            data['poly_YZ'] = None
            
        if '/meta/crop_box' in f:
            data['crop_box'] = f['/meta/crop_box'][:].ravel()
        elif '/meta/domain_box' in f:
            data['crop_box'] = f['/meta/domain_box'][:].ravel()
        else:
            data['crop_box'] = None
    return data

def extract_traces(centers, normals, radii, x_slice):
    """
    해석학적 방정식으로 3D 균열(원판)과 x = x_slice 평면의 교차선(Trace)을 계산합니다.
    (cupy 등 GPU 모듈 없이 numpy만으로 벡터화된 초고속 처리가 가능합니다.)
    
    Y-Z 평면상의 (P1, P2) 좌표 집합인 (N, 2, 2) 배열을 반환합니다.
    """
    dx = centers[:, 0] - x_slice
    
    # 1차 필터링 (Bounding Box - X축 기준 반경 안에 없는 것은 탈락)
    valid = np.abs(dx) <= radii
    
    cy = centers[valid, 1]
    cz = centers[valid, 2]
    nx = normals[valid, 0]
    ny = normals[valid, 1]
    nz = normals[valid, 2]
    r  = radii[valid]
    dx = dx[valid]
    
    s2 = ny**2 + nz**2
    s  = np.sqrt(s2)
    
    # 균열이 YZ 평면과 완벽히 평행한 경우(s=0) 교차선(Trace)이 아니므로 제외
    non_zero = s > 1e-6
    cy = cy[non_zero]; cz = cz[non_zero]; nx = nx[non_zero]
    ny = ny[non_zero]; nz = nz[non_zero]; r  = r[non_zero]
    dx = dx[non_zero]; s  = s[non_zero];  s2 = s2[non_zero]
    
    # 2차 판별 (원판 중심에서 교차점까지의 3D 공간 거리)
    d = np.abs(dx) / s
    intersect = d <= r  # 거리가 반지름 이내면 원 안에서 만남
    
    cy = cy[intersect]; cz = cz[intersect]; nx = nx[intersect]
    ny = ny[intersect]; nz = nz[intersect]; r  = r[intersect]
    dx = dx[intersect]; s  = s[intersect];  d  = d[intersect]
    s2 = s2[intersect]
    
    # 교차선 절반 길이
    L = np.sqrt(r**2 - d**2)
    
    # 평면 위의 중심점 (교차선의 중점)
    My = cy + dx * (nx * ny) / s2
    Mz = cz + dx * (nx * nz) / s2
    
    # 교차선의 방향 벡터
    uy = nz / s
    uz = -ny / s
    
    # 양 끝점 (선분) 계산
    P1y = My + L * uy
    P1z = Mz + L * uz
    P2y = My - L * uy
    P2z = Mz - L * uz
    
    segments = np.zeros((len(P1y), 2, 2))
    segments[:, 0, 0] = P1y
    segments[:, 0, 1] = P1z
    segments[:, 1, 0] = P2y
    segments[:, 1, 1] = P2z
    
    return segments


def main():
    parser = argparse.ArgumentParser(description="인터랙티브 2D Trace Map 시각화 (X 단면 슬라이스)")
    parser.add_argument('--input', required=True, help="dfn_export_for_python.h5 경로")
    args = parser.parse_args()

    data = load_data(args.input)
    centers = data['centers']
    normals = data['normals']
    radii = data['radii']
    poly_YZ = data['poly_YZ']
    crop = data['crop_box']  # [xmin, xmax, ymin, ymax, zmin, zmax]

    if crop is not None:
        xmin, xmax = crop[0], crop[1]
        ymin, ymax = crop[2], crop[3]
        zmin, zmax = crop[4], crop[5]
        
        # crop 영역에 완전히 벗어나는 전체 균열 사전 필터링 (메모리 로드 줄임)
        # centers 반경이 crop 박스 내에 전혀 들어오지 않는 경우 배제
        in_bbox = ~((centers[:, 0] + radii < xmin) | (centers[:, 0] - radii > xmax) |
                    (centers[:, 1] + radii < ymin) | (centers[:, 1] - radii > ymax) |
                    (centers[:, 2] + radii < zmin) | (centers[:, 2] - radii > zmax))
        centers = centers[in_bbox]
        normals = normals[in_bbox]
        radii = radii[in_bbox]
    else:
        # crop 박스 정보가 없을 경우 전체 데이터에서 xmin, xmax 산출
        xmin, xmax = np.min(centers[:, 0]), np.max(centers[:, 0])
        ymin, ymax = np.min(centers[:, 1]), np.max(centers[:, 1])
        zmin, zmax = np.min(centers[:, 2]), np.max(centers[:, 2])

    print(f"전체 필터링 후 균열 수: {len(radii):,} (X 도메인: {xmin:.1f} ~ {xmax:.1f})")

    # 초기 슬라이스 (중앙)
    init_x = 0.0 if (xmin <= 0 <= xmax) else (xmin + xmax) / 2

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    plt.subplots_adjust(bottom=0.2)
    
    ax.set_facecolor('white')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(ymin, ymax)
    ax.set_ylim(zmin, zmax)
    ax.grid(True, linestyle='--', alpha=0.5)

    # 터널 폴리곤 (항상 표출)
    if poly_YZ is not None:
        # 폴리곤 선 그리드 (Closed 구조)
        poly_Y = np.append(poly_YZ[:, 0], poly_YZ[0, 0])
        poly_Z = np.append(poly_YZ[:, 1], poly_YZ[0, 1])
        ax.plot(poly_Y, poly_Z, color='red', linewidth=2.5, label='Tunnel Cross-section')

    # Trace LineCollection (초기 설정)
    segs = extract_traces(centers, normals, radii, x_slice=init_x)
    line_col = LineCollection(segs, colors='black', linewidths=0.6, alpha=0.7)
    ax.add_collection(line_col)

    trace_info_txt = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                             verticalalignment='top', fontsize=11,
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax.set_xlabel('Y (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)
    ax.legend(loc='lower right')

    def draw_update(val=None):
        x_current = slider.val
        ax.set_title(f'2D DFN Trace Map on Y-Z Plane (X = {x_current:.2f} m)', fontsize=14, fontweight='bold')
        
        # 교차선 재계산
        new_segs = extract_traces(centers, normals, radii, x_slice=x_current)
        line_col.set_segments(new_segs)
        
        # 부가 정보 (교차 균열 수, P21 강도 등 계산 가능)
        trace_count = len(new_segs)
        
        # Trace 들의 전체 길이 계산 (P21 = Trace 총 길이 / 면적)
        if trace_count > 0:
            total_trace_length = np.sum(np.linalg.norm(new_segs[:, 0, :] - new_segs[:, 1, :], axis=1))
        else:
            total_trace_length = 0.0
            
        area = (ymax - ymin) * (zmax - zmin)
        p21 = total_trace_length / area if area > 0 else 0
        
        info_str = f"Trace Lines: {trace_count:,} ea\nTrace Max L: {total_trace_length:,.1f} m\nArea: {area:,.0f} m²\nP21 Intensity: {p21:.3f} m⁻¹"
        trace_info_txt.set_text(info_str)
        fig.canvas.draw_idle()

    # 슬라이더 생성
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.04])
    slider = Slider(
        ax=ax_slider,
        label='X Slice [m]',
        valmin=xmin,
        valmax=xmax,
        valinit=init_x,
        valstep=0.1,
        color='gray'
    )

    # 슬라이더가 움직일 때 마다 업데이트
    slider.on_changed(draw_update)

    # 초기 화면 렌더링
    draw_update(init_x)
    
    print("▶ 2D Trace Map 인터랙티브 창을 실행합니다. (X-축 슬라이더 조작 가능)")
    plt.show()

if __name__ == '__main__':
    main()
