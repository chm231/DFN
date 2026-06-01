"""
set_classifier.py
==================
복원된 평면들의 법선벡터를 분석하여 균열 세트를 자동 분류하고
Fisher 집중 계수(kappa)를 추정하는 모듈.

방법:
1. Spherical K-means: 단위 법선벡터를 구면 위에서 클러스터링
2. BIC 기반 최적 K 선정
3. vMF Fisher kappa 추정: Orientation Tensor 기반
4. 평균 법선벡터 → Dip/DipDirection 변환
"""

import numpy as np
from typing import List, Dict, Tuple
from .slab_types import ReconstructedPlane


def _hemisphere_align(normals: np.ndarray) -> np.ndarray:
    """
    법선벡터들을 단일 반구면으로 정렬합니다.
    Orientation Tensor의 주축(principal axis) 방향으로 일관성 있게 뒤집습니다.
    """
    N = normals.shape[0]
    if N <= 1:
        return normals.copy()
    
    # 단위 벡터 정규화
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms_safe = np.where(norms > 1e-12, norms, 1.0)
    n = normals / norms_safe
    
    # Orientation Tensor 계산 및 주축 추출
    tensor = np.einsum('ij,ik->jk', n, n) / N
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
    
    # 주축 방향으로 반구면 정렬
    dots = np.dot(n, principal_axis)
    n_aligned = n.copy()
    n_aligned[dots < 0] *= -1
    
    return n_aligned


def calculate_fisher_kappa(normals: np.ndarray) -> Tuple[float, np.ndarray, float]:
    """
    Fisher 집중 계수(kappa)를 Orientation Tensor 기반으로 추정합니다.
    
    Returns:
        kappa: Fisher concentration parameter
        mean_normal: 평균 법선벡터 (단위벡터)
        R_magnitude: Resultant vector의 크기
    """
    N = normals.shape[0]
    if N <= 1:
        mean_n = normals[0] if N == 1 else np.array([0, 0, 1.0])
        return 0.0, mean_n, 0.0
    
    # 반구면 정렬
    n_aligned = _hemisphere_align(normals)
    
    # Resultant vector 계산
    R_vector = np.sum(n_aligned, axis=0)
    R_mag = np.linalg.norm(R_vector)
    mean_normal = R_vector / (R_mag + 1e-12)
    
    # Fisher kappa 추정 (maximum likelihood)
    denominator = N - R_mag
    if denominator < 1e-6:
        kappa = 1e6  # 극도로 집중된 분포
    else:
        kappa = (N - 1) / denominator
    
    return kappa, mean_normal, R_mag


def normal_to_dip_dipdirection(normal: np.ndarray) -> Tuple[float, float]:
    """
    3D 법선벡터를 지질공학적 Dip / Dip Direction으로 변환합니다.
    
    좌표계 규약: X = 터널 진행 방향(East), Y = North, Z = Up
    
    Args:
        normal: (3,) 단위 법선벡터 [nx, ny, nz]
    
    Returns:
        dip: 경사각 (0~90도)
        dip_direction: 경사 방향 (0~360도, 북 기준 시계방향)
    """
    nx, ny, nz = normal
    
    # 법선이 위를 향하도록 보정 (nz > 0)
    if nz < 0:
        nx, ny, nz = -nx, -ny, -nz
    
    # Dip: 수평면과 균열면의 교각
    horizontal_mag = np.sqrt(nx**2 + ny**2)
    dip = np.degrees(np.arctan2(horizontal_mag, abs(nz)))
    
    # Dip Direction: 법선의 수평 투영 방향 (북 기준 시계방향)
    if horizontal_mag < 1e-10:
        dip_direction = 0.0  # 수평 균열
    else:
        dip_direction = np.degrees(np.arctan2(nx, ny))  # East = +X, North = +Y
        if dip_direction < 0:
            dip_direction += 360.0
    
    return float(dip), float(dip_direction)


def spherical_kmeans(normals: np.ndarray, k: int, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    구면 K-means 클러스터링.
    
    단위 법선벡터들을 구면 거리(arccosine of dot product)를 사용하여 클러스터링합니다.
    
    Args:
        normals: (N, 3) 단위 법선벡터
        k: 클러스터 수
        max_iter: 최대 반복 횟수
    
    Returns:
        labels: (N,) 클러스터 레이블
        centers: (k, 3) 클러스터 중심 (단위벡터)
    """
    N = normals.shape[0]
    
    # 반구면 정렬
    n_aligned = _hemisphere_align(normals)
    
    # 초기 중심점 선택 (K-means++ 유사)
    rng = np.random.RandomState(42)
    centers = np.zeros((k, 3))
    centers[0] = n_aligned[rng.randint(N)]
    
    for c_idx in range(1, k):
        # 기존 중심들과의 최소 각거리 계산
        cos_dists = np.abs(np.dot(n_aligned, centers[:c_idx].T))
        min_cos = np.max(cos_dists, axis=1)  # 가장 가까운 기존 중심과의 유사도
        probs = 1.0 - min_cos  # 먼 것일수록 높은 확률
        probs = np.maximum(probs, 0.0)
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs /= probs_sum
        else:
            probs = np.ones(N) / N
        centers[c_idx] = n_aligned[rng.choice(N, p=probs)]
    
    # 반복 수렴
    labels = np.zeros(N, dtype=int)
    for _ in range(max_iter):
        # Assignment: 가장 가까운 중심에 할당 (cosine similarity 최대)
        cos_sim = np.abs(np.dot(n_aligned, centers.T))  # (N, k)
        new_labels = np.argmax(cos_sim, axis=1)
        
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        
        # Update: 각 클러스터의 새 중심 계산
        for c_idx in range(k):
            mask = labels == c_idx
            if mask.sum() == 0:
                continue
            cluster_normals = n_aligned[mask]
            # 반구면 정렬 후 resultant vector
            aligned = _hemisphere_align(cluster_normals)
            resultant = np.sum(aligned, axis=0)
            norm_val = np.linalg.norm(resultant)
            if norm_val > 1e-12:
                centers[c_idx] = resultant / norm_val
            
    return labels, centers


def compute_bic_spherical(normals: np.ndarray, labels: np.ndarray, k: int) -> float:
    """
    구면 클러스터링에 대한 BIC (Bayesian Information Criterion) 계산.
    
    BIC = -2 * log_likelihood + k_params * ln(N)
    """
    N = normals.shape[0]
    if N <= k:
        return 1e10
    
    log_likelihood = 0.0
    for c_idx in range(k):
        mask = labels == c_idx
        n_c = mask.sum()
        if n_c < 2:
            continue
        
        cluster_normals = normals[mask]
        kappa_c, mean_n, _ = calculate_fisher_kappa(cluster_normals)
        
        # vMF log-likelihood 근사: kappa * sum(n_i . mu) - N_c * log(C(kappa))
        # C(kappa) 정규화 상수 근사 → log(C) ≈ log(kappa/(2π)) - kappa (for large kappa)
        dots = np.dot(cluster_normals, mean_n)
        dots = np.clip(np.abs(dots), 0, 1)
        log_likelihood += np.sum(np.log(np.maximum(dots, 1e-12))) * min(kappa_c, 100)
    
    # BIC 계산 (k_params = k * 4: 3 for mean_normal + 1 for kappa per cluster)
    k_params = k * 4
    bic = -2.0 * log_likelihood + k_params * np.log(N)
    
    return bic


def classify_sets(
    planes: List[ReconstructedPlane],
    max_k: int = 6
) -> Tuple[int, Dict[int, Tuple[np.ndarray, float]]]:
    """
    복원된 평면들의 법선벡터를 분석하여 균열 세트를 자동 분류합니다.
    
    Args:
        planes: 복원된 평면 리스트
        max_k: 탐색할 최대 세트 수
    
    Returns:
        optimal_k: 최적 세트 수
        set_stats: {set_id: (mean_normal, kappa)} 딕셔너리
    """
    if len(planes) < 2:
        if planes:
            planes[0].set_id = 1
            dip, dip_dir = normal_to_dip_dipdirection(planes[0].normal)
            planes[0].dip = dip
            planes[0].dip_direction = dip_dir
        return 1, {1: (np.array([0, 0, 1.0]), 0.0)}
    
    # 법선벡터 수집
    normals = np.array([p.normal for p in planes])
    
    # 단위 벡터 정규화
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms_safe = np.where(norms > 1e-12, norms, 1.0)
    normals = normals / norms_safe
    
    # BIC 기반 최적 K 탐색
    best_k = 1
    best_bic = float('inf')
    best_labels = np.zeros(len(planes), dtype=int)
    best_centers = normals[:1]
    
    max_k = min(max_k, len(planes))
    
    for k in range(1, max_k + 1):
        labels, centers = spherical_kmeans(normals, k)
        bic = compute_bic_spherical(_hemisphere_align(normals), labels, k)
        
        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_labels = labels
            best_centers = centers
    
    # 세트 통계 계산 및 평면에 할당
    set_stats = {}
    for c_idx in range(best_k):
        mask = best_labels == c_idx
        set_id = c_idx + 1
        
        if mask.sum() == 0:
            continue
        
        cluster_normals = normals[mask]
        kappa, mean_normal, _ = calculate_fisher_kappa(cluster_normals)
        set_stats[set_id] = (mean_normal, kappa)
        
        # 각 평면에 세트 ID 및 Dip/DipDirection 할당
        plane_indices = np.where(mask)[0]
        for pi in plane_indices:
            planes[pi].set_id = set_id
            dip, dip_dir = normal_to_dip_dipdirection(planes[pi].normal)
            planes[pi].dip = dip
            planes[pi].dip_direction = dip_dir
    
    return best_k, set_stats
