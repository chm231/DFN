# =============================================================================
# 파일 역할: 절리(fracture) 법선 벡터 집합으로부터 평균 방향(mean orientation)을
#           추정하고, 이를 지질학적 방위각(trend)/경사각(plunge)으로 변환한다.
#   - 법선은 방향이 없는 축(axial) 벡터이므로, 반구 정렬 후 평균을 구한다.
# 주요 입력: 모델 좌표계 [x, y, z]로 표현된 단위 법선 벡터 배열 (N x 3)
# 주요 출력: 평균 법선 벡터(단위 벡터), NED 좌표계 기준 trend/plunge(도 단위)
# 핵심 처리 흐름:
#   1) normalize: 벡터 정규화(단위화)
#   2) estimate_mean_normal_axial: 법선들을 같은 반구로 정렬 후 합벡터 평균
#   3) normal_to_trend_plunge_ned: 모델 좌표 → NED 변환 → trend/plunge 산출
# =============================================================================
from typing import Optional, Tuple
import numpy as np


# 벡터를 단위 벡터로 정규화한다.
#   인자: vector_xyz(3D 벡터), eps(0 판정 임계값)
#   반환: 단위 벡터(np.ndarray) 또는 크기가 eps 미만이면 None
def normalize(vector_xyz: np.ndarray, eps: float = 1e-12) -> Optional[np.ndarray]:
    # 입력을 float64 배열로 변환하고 벡터 크기(노름)를 계산
    vector_xyz = np.asarray(vector_xyz, dtype=np.float64)
    norm = float(np.linalg.norm(vector_xyz))
    # 크기가 0에 가까우면 방향을 정의할 수 없으므로 None 반환
    if norm < eps:
        return None
    # 크기로 나누어 단위 벡터 반환
    return vector_xyz / norm


# 축(axial) 벡터인 법선 집합의 평균 법선 벡터를 추정한다.
#   법선은 +/- 방향 구분이 없으므로, 첫 벡터를 기준으로 같은 반구에 맞춘 뒤 평균한다.
#   인자: normals_xyz(N x 3 법선 배열), eps(0 판정 임계값)
#   반환: 평균 법선(단위 벡터) 또는 입력이 부적절하면 None
def estimate_mean_normal_axial(normals_xyz: np.ndarray, eps: float = 1e-12) -> Optional[np.ndarray]:
    """
    Estimates the mean normal vector from a set of axial vectors (normals)
    by aligning them to the first vector to ensure they lie in the same hemisphere.
    """
    # 입력을 float64 배열로 변환하고 (N x 3) 형태인지 검증
    normals_xyz = np.asarray(normals_xyz, dtype=np.float64)
    if normals_xyz.ndim != 2 or normals_xyz.shape[0] == 0 or normals_xyz.shape[1] != 3:
        return None

    # 원본을 보존하기 위해 복사하고, 첫 번째 법선을 기준 방향(ref)으로 설정
    aligned = normals_xyz.copy()
    ref_xyz = normalize(aligned[0], eps=eps)
    if ref_xyz is None:
        return None

    # 기준과 내적이 음수인 법선은 반대 반구에 있으므로 부호를 뒤집어 같은 반구로 정렬
    for idx in range(len(aligned)):
        if float(np.dot(aligned[idx], ref_xyz)) < 0.0:
            aligned[idx] *= -1.0

    # 정렬된 법선들의 합벡터(resultant)를 구하고 정규화하여 평균 방향 반환
    resultant_xyz = np.sum(aligned, axis=0)
    return normalize(resultant_xyz, eps=eps)


# 모델 좌표계 [x, y, z]의 법선 벡터를 NED(North-East-Down) 좌표계 기준
# 지질학적 방위각(trend)/경사각(plunge)으로 변환한다.
#   좌표 대응: North = -y, East = -x, Down = z
#   인자: normal_xyz(3D 법선), eps(0 판정 임계값)
#   반환: (trend, plunge) 도 단위. 입력 크기가 0이면 (None, None)
def normal_to_trend_plunge_ned(normal_xyz: np.ndarray, eps: float = 1e-12) -> Tuple[Optional[float], Optional[float]]:
    """
    Converts a 3D normal vector in model coordinates [x, y, z] to
    geological Trend and Plunge under the NED (North, East, Down) coordinate system:
      North = -y
      East = -x
      Down = z

    Returns lower-hemisphere pole trend/plunge.
    """
    # 법선을 단위화. 크기가 0에 가까우면 방향 불가로 (None, None) 반환
    n = np.asarray(normal_xyz, dtype=np.float64)
    norm = float(np.linalg.norm(n))
    if norm < eps:
        return None, None

    x_model, y_model, z_model = n / norm

    # 모델 좌표 → NED 좌표 변환 (North=-y, East=-x, Down=z)
    # Apply coordinate transformation
    x_ned = -y_model
    y_ned = -x_model
    z_ned = z_model

    # 하반구(lower hemisphere) 규약: Down 성분이 양수가 되도록 벡터 전체 부호 반전
    # Lower hemisphere convention: Down component must be positive
    if z_ned < 0.0:
        x_ned, y_ned, z_ned = -x_ned, -y_ned, -z_ned

    # trend: 수평면(N-E)상 방위각(0~360도), plunge: 수평면 대비 하향 경사각
    trend = float(np.degrees(np.arctan2(y_ned, x_ned)) % 360.0)
    plunge = float(np.degrees(np.arctan2(z_ned, np.sqrt(x_ned * x_ned + y_ned * y_ned))))

    return trend, plunge
