import os
import sys
import numpy as np
import h5py
import scipy.ndimage as ndimage
from skimage import measure
from scipy.interpolate import interp1d
from typing import List, Tuple, Dict, Any

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

class RoughTunnel:
    """
    터널 설계 단면 폴리곤(Horse-shoe, Circular 등 자유 형상)을 따라
    X축 방향으로 압출하고 3차원 Gaussian 요철(Roughness)을 입히는 클래스.
    """
    def __init__(
        self,
        tunnel_poly_yz: np.ndarray,
        length_range: Tuple[float, float] = (-25.0, 25.0),
        x_res: float = 0.2,
        n_theta_pts: int = 180,          # 둘레 격자 해상도
        amplitude: float = 0.2,          # 요철 진폭 (m)
        correlation_length: float = 2.0, # 상관 거리 (m)
        seed: int = 42
    ):
        self.poly_yz = np.array(tunnel_poly_yz, dtype=np.float64)
        self.x_min, self.x_max = length_range
        self.amplitude = amplitude
        self.correlation_length = correlation_length
        self.seed = seed
        
        # 1. 터널 중심점(Centroid) 계산
        self.centroid = np.mean(self.poly_yz, axis=0)
        
        # 2. 둘레 해상도 향상을 위한 폴리곤 Y-Z 점들의 선형 보간 수행
        n_poly = len(self.poly_yz)
        poly_closed = np.vstack([self.poly_yz, self.poly_yz[0]])
        diffs = np.diff(poly_closed, axis=0)
        segment_dists = np.linalg.norm(diffs, axis=1)
        cum_dists = np.hstack([0, np.cumsum(segment_dists)])
        total_perimeter = cum_dists[-1]
        
        # 둘레 전체 구간을 n_theta_pts로 등분
        self.theta_vec = np.linspace(0, total_perimeter, n_theta_pts)
        y_interp = interp1d(cum_dists, poly_closed[:, 0], kind='linear')
        z_interp = interp1d(cum_dists, poly_closed[:, 1], kind='linear')
        
        self.y_nodes = y_interp(self.theta_vec)
        self.z_nodes = z_interp(self.theta_vec)
        
        # 3. 중심점 기준 각 노드의 오리지널 반경 벡터 및 방향 벡터 계산
        dy = self.y_nodes - self.centroid[0]
        dz = self.z_nodes - self.centroid[1]
        self.r_poly = np.sqrt(dy**2 + dz**2)
        r_poly_safe = np.where(self.r_poly > 1e-12, self.r_poly, 1.0)
        self.dir_y = dy / r_poly_safe
        self.dir_z = dz / r_poly_safe
        
        # 4. Grid 축 생성
        self.x_vec = np.arange(self.x_min, self.x_max + x_res, x_res)
        self.X_grid, self.Theta_grid = np.meshgrid(self.x_vec, np.arange(len(self.theta_vec)))
        
        # 5. 요철 Delta (r 방향 편차) 계산
        self.Delta = self._generate_roughness()
        
        # 6. 최종 요철이 부여된 3D 터널 벽면 좌표계 구축
        # shape: (n_theta, n_x)
        self.R_rough = self.r_poly[:, None] + self.Delta
        self.Y_grid = self.centroid[0] + self.R_rough * self.dir_y[:, None]
        self.Z_grid = self.centroid[1] + self.R_rough * self.dir_z[:, None]
        
    def _generate_roughness(self) -> np.ndarray:
        """Gaussian Smoothing 필터를 이용해 공간 연속성을 가진 요철 Delta 값 도출"""
        np.random.seed(self.seed)
        noise = np.random.normal(0, 1, self.X_grid.shape)
        
        avg_pixel_size = 0.15 
        sigma = self.correlation_length / avg_pixel_size
        
        smoothed = ndimage.gaussian_filter(noise, sigma=sigma)
        s_min, s_max = smoothed.min(), smoothed.max()
        
        if s_max - s_min > 1e-9:
            delta = (smoothed - s_min) / (s_max - s_min) * 2.0 - 1.0 # [-1, 1] 범위
            return delta * self.amplitude
        return smoothed * 0.0

    def evaluate_3d_point(self, x_idx: float, theta_idx: float) -> np.ndarray:
        """격자 인덱스로부터 선형 보간을 적용해 정확한 [x, y, z] 좌표 반환"""
        n_theta, n_x = self.X_grid.shape
        tx = np.clip(x_idx, 0, n_x - 1)
        tt = np.clip(theta_idx, 0, n_theta - 1)
        
        x_val = self.x_min + tx * (self.x_max - self.x_min) / (n_x - 1)
        
        # 인덱스 위치 보간 추출
        x_int = int(np.floor(tx))
        t_int = int(np.floor(tt))
        
        y_val = self.Y_grid[t_int % n_theta, x_int % n_x]
        z_val = self.Z_grid[t_int % n_theta, x_int % n_x]
        
        return np.array([x_val, y_val, z_val])

def extract_rough_tunnel_traces(
    centers: np.ndarray,
    normals: np.ndarray,
    radii: np.ndarray,
    tunnel: RoughTunnel
) -> List[Dict[str, Any]]:
    """
    3차원 요철 설계 터널 폴리곤 벽면과 DFN 불연속면(Discs)의 3D 교선 집합을 추출합니다.
    """
    all_traces = []
    
    # 3D 터널 격자 포인트를 (N_theta, N_x, 3) 텐서로 변환
    tunnel_pts = np.stack((tunnel.X_grid, tunnel.Y_grid, tunnel.Z_grid), axis=-1)
    n_theta, n_x, _ = tunnel_pts.shape
    
    # Bounding Box 사전 필터링
    t_xmin, t_xmax = tunnel.x_min, tunnel.x_max
    
    dy = tunnel.Y_grid - tunnel.centroid[0]
    dz = tunnel.Z_grid - tunnel.centroid[1]
    t_box_max_radius = np.max(np.sqrt(dy**2 + dz**2))
    
    for i in range(len(radii)):
        Cx, Cy, Cz = centers[i]
        R_f = radii[i]
        
        # 터널 X범위 및 단면 반경 영역에 속하지 않는 것은 스킵
        if Cx - R_f > t_xmax or Cx + R_f < t_xmin:
            continue
        dist_to_tunnel_centroid = np.sqrt((Cy - tunnel.centroid[0])**2 + (Cz - tunnel.centroid[1])**2)
        if dist_to_tunnel_centroid - R_f > t_box_max_radius:
            continue
            
        N = normals[i]
        C = np.array([Cx, Cy, Cz])
        
        # 1. 터널 벽면의 각 점에서 균열 평면까지의 법선거리(Signed Distance) 필드 계산
        # D(x, theta) = N . (P(x, theta) - C)
        diff = tunnel_pts - C
        dist_field = np.sum(diff * N, axis=-1)
        
        # 2. 등치선(Zero-crossing) 추적하여 교선 윤곽선 획득
        contours = measure.find_contours(dist_field, 0.0)
        
        for contour in contours:
            pts_3d = []
            for pt in contour:
                # 격자 좌표로부터 실제 3D [x, y, z] 좌표 보간 및 추출
                p3d = tunnel.evaluate_3d_point(pt[1], pt[0])
                pts_3d.append(p3d)
                
            pts_3d = np.array(pts_3d)
            if len(pts_3d) < 2:
                continue
                
            # 3. 균열 원판의 유효 반경(R_f) 이내에 속하는지 체크하여 클리핑
            d_to_center = np.linalg.norm(pts_3d - C, axis=1)
            valid_mask = d_to_center <= R_f
            
            # 마스크 분할을 통한 3D 폴리라인 조각 생성
            from .intersection import split_mask_to_polylines
            sub_traces = split_mask_to_polylines(pts_3d, valid_mask)
            
            for poly in sub_traces:
                if len(poly) < 2:
                    continue
                all_traces.append({
                    'fracture_id': i,
                    'points': poly,
                    'center': C,
                    'normal': N,
                    'radius': R_f,
                    'length': np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1))
                })
                
    return all_traces

def plot_rough_tunnel_3d(
    tunnel: RoughTunnel,
    traces: List[Dict[str, Any]],
    save_path: str | None = None
):
    """
    3차원 요철 터널 벽면 메쉬와 그 위에 생성된 3D 굴곡 교선들을 PyVista로 시각화 및 내보내기합니다.
    """
    if not HAS_PYVISTA:
        print("[Error] PyVista가 설치되어 있지 않습니다.")
        return
        
    p = pv.Plotter(off_screen=True)
    p.set_background("white")
    
    grid = pv.StructuredGrid(tunnel.X_grid, tunnel.Y_grid, tunnel.Z_grid)
    grid.point_data["요철 변차 (Radial Deviation)"] = tunnel.Delta.ravel()
    
    p.add_mesh(
        grid, 
        cmap="coolwarm", 
        scalars="요철 변차 (Radial Deviation)",
        opacity=0.6, 
        show_edges=False, 
        label="Rough Tunnel Polygon Wall"
    )
    
    for i, t in enumerate(traces):
        pts = t['points']
        poly = pv.PolyData(pts)
        lines = np.arange(len(pts))
        poly.lines = np.hstack(([len(pts)], lines))
        
        p.add_mesh(
            poly, 
            color="#228b22" if i % 2 == 0 else "#d95f02", 
            line_width=4.5, 
            label="3D Rough Joint Trace" if i == 0 else None
        )
        
    p.add_legend()
    p.add_axes()
    p.add_scalar_bar("Delta (m)", label_font_size=12)
    
    if save_path:
        p.show(screenshot=save_path)
        print(f"[*] 3D Rough Tunnel Visualization saved to: {save_path}")
    else:
        p.show()

def main():
    print("=" * 80)
    print(" 3D Rough Tunnel Polygon Wall Trace Intersecting Simulator")
    print("=" * 80)
    
    _here = os.path.dirname(os.path.abspath(__file__))
    _parent = os.path.dirname(os.path.dirname(_here))
    hdf5_path = os.path.join(_parent, "storage", "data", "dfn_export_for_python.h5")
    outdir = os.path.join(_parent, "storage", "output", "rough_tunnel_simulation")
    os.makedirs(outdir, exist_ok=True)
    
    # 1. 3D DFN 로드 및 터널 설계 단면 폴리곤 획득
    print(f"[*] Loading 3D DFN & Tunnel Polygon Y-Z from: {hdf5_path}")
    
    # storage/data/단면_폴리곤.dat 에서 실제 마제형 터널 단면 데이터 로드
    dat_path = os.path.join(_parent, "storage", "data", "단면_폴리곤.dat")
    from trace_analysis.load_tunnel_dat import load_tunnel_polygon_from_dat
    
    if os.path.exists(dat_path):
        print(f"[*] Loading actual tunnel polygon from .dat file: {dat_path}")
        poly_y, poly_z = load_tunnel_polygon_from_dat(dat_path)
        poly_yz = np.column_stack([poly_y, poly_z])
    else:
        print("[Warning] .dat 설계 파일이 없습니다. HDF5 및 원형 폴리곤 폴백을 시도합니다.")
        poly_yz = None
        
    with h5py.File(hdf5_path, 'r') as f:
        raw_c = f['/fractures/centers'][:]
        raw_n = f['/fractures/normals'][:]
        radii = f['/fractures/radii'][:].ravel()
        centers = raw_c.T if raw_c.shape[0] == 3 and raw_c.shape[0] < raw_c.shape[1] else raw_c
        normals = raw_n.T if raw_n.shape[0] == 3 and raw_n.shape[0] < raw_n.shape[1] else raw_n
        
        # HDF5 폴백 체크
        if poly_yz is None and 'poly_YZ' in f:
            poly_yz = f['poly_YZ'][:]
            
    if poly_yz is None:
        print("[Error] 터널 폴리곤 데이터를 찾을 수 없습니다. 기본 원형 폴리곤으로 대체합니다.")
        angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
        poly_yz = np.column_stack([5.0 * np.cos(angles), 5.0 * np.sin(angles)])
        
    print(f"    -> Parsed Tunnel Polygon Y-Z with {len(poly_yz)} vertices.")
    print(f"    -> Real 3D DFN contains {len(radii):,} fractures.")
    
    # 2. 터널 폴리곤 기반 3D 요철 터널 벽면 생성
    print("[*] Building 3D Rough Tunnel Polygon Wall (Length 50m)...")
    tunnel = RoughTunnel(
        tunnel_poly_yz=poly_yz,
        length_range=(-25.0, 25.0),
        amplitude=0.3,          # 30cm 진폭 요철
        correlation_length=1.5,  # 1.5m 상관 거리
        seed=100
    )
    
    # 3. 요철 터널 벽면과의 3D 교선 추출
    print("\n[*] Intersecting 3D fractures with the wavy polygon tunnel wall...")
    traces = extract_rough_tunnel_traces(centers, normals, radii, tunnel)
    print(f"    -> Successfully extracted {len(traces)} 3D joint traces running along the rough tunnel wall!")
    
    # 4. 고해상도 3D 시각화 이미지 캡처
    save_img = os.path.join(outdir, "rough_tunnel_3d_view.png")
    print(f"\n[*] Rendering publication-grade 3D visualization to: {save_img}")
    plot_rough_tunnel_3d(tunnel, traces, save_path=save_img)
    
    print("\n" + "=" * 80)
    print(" ROUGH TUNNEL SIMULATION SUCCESS")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
