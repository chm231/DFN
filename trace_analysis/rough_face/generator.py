import numpy as np
import scipy.ndimage as ndimage
from typing import Tuple, Optional

class RoughFace:
    """
    터널 X축을 따라 비평면(Rough) 굴착면을 생성하고 관리하는 클래스.
    Y-Z 그리드 상에 거칠기가 가미된 X 좌표(Height Map)를 유지합니다.
    """
    def __init__(
        self, 
        base_x: float, 
        y_range: Tuple[float, float], 
        z_range: Tuple[float, float],
        resolution: float = 0.1,
        amplitude: float = 0.3,
        correlation_length: float = 1.0,
        seed: Optional[int] = None
    ):
        self.base_x = base_x
        self.y_range = y_range
        self.z_range = z_range
        self.resolution = resolution
        self.amplitude = amplitude
        self.correlation_length = correlation_length
        self.seed = seed
        
        # 그리드 생성
        self.y_vec = np.arange(y_range[0], y_range[1] + resolution, resolution)
        self.z_vec = np.arange(z_range[0], z_range[1] + resolution, resolution)
        self.Y, self.Z = np.meshgrid(self.y_vec, self.z_vec)
        
        # 거칠기 생성
        self.X = self._generate_rough_heights()

    def _generate_rough_heights(self) -> np.ndarray:
        """
        Gaussian Random Field를 사용하여 공간적으로 상관관계가 있는 거칠기를 생성합니다.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
            
        # 1. 화이트 노이즈 생성
        noise = np.random.normal(0, 1, self.Y.shape)
        
        # 2. Gaussian Smoothing (거칠기의 scale 조절)
        # sigma = correlation_length / resolution
        sigma = self.correlation_length / self.resolution
        smoothed = ndimage.gaussian_filter(noise, sigma=sigma)
        
        # 3. 진폭(Amplitude) 조절 및 정규화 [-amplitude, amplitude]
        s_min, s_max = smoothed.min(), smoothed.max()
        if s_max - s_min > 1e-9:
            normalized = ((smoothed - s_min) / (s_max - s_min) - 0.5) * 2.0 * self.amplitude
        else:
            normalized = smoothed * 0.0
            
        return self.base_x + normalized

    def project_to_face(self, points_yz: np.ndarray) -> np.ndarray:
        """
        2D (Y, Z) 좌표를 굴착면 상의 3D (X, Y, Z) 좌표로 투영합니다.
        가까운 그리드 값을 참조합니다.
        """
        if len(points_yz) == 0:
            return np.empty((0, 3))
            
        y = points_yz[:, 0]
        z = points_yz[:, 1]
        
        # 인덱스 계산
        iy = ((y - self.y_range[0]) / self.resolution).astype(int)
        iz = ((z - self.z_range[0]) / self.resolution).astype(int)
        
        # 경계 체크 및 클리핑
        iy = np.clip(iy, 0, self.Y.shape[1] - 1)
        iz = np.clip(iz, 0, self.Y.shape[0] - 1)
        
        x = self.X[iz, iy]
        
        return np.column_stack((x, y, z))
