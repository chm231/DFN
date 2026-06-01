"""
dfn_parameter_extractor.py
===========================
복원된 평면 + 추정 반경 + 세트 분류 결과를 종합하여
DFN 통계 파라미터를 추출하는 모듈.

추출 파라미터:
- P32: 체적 면적 밀도 (m²/m³)
- P30: 체적 개수 밀도 (1/m³)
- α_R: Pareto shape parameter (Power-law exponent)
- r_min: Pareto scale parameter
- kappa: Fisher concentration parameter
- Terzaghi 방향 편향 보정 P32
"""

import numpy as np
from typing import List, Dict, Tuple
from .slab_types import ReconstructedPlane, DFNSetResult, DFNParameterResult
from .set_classifier import calculate_fisher_kappa, normal_to_dip_dipdirection


def fit_pareto_mle(radii: np.ndarray) -> Tuple[float, float]:
    """
    Pareto (Power-law) 분포의 MLE 피팅.
    
    f(r) = α * r_min^α / r^(α+1),  r >= r_min
    
    MLE:
        r_min = min(radii)
        α = N / Σ ln(r_i / r_min)
    
    Returns:
        alpha: Shape parameter
        r_min: Scale parameter (최소값)
    """
    if len(radii) < 2:
        return 2.5, 1.0  # 기본값
    
    r_min = np.min(radii)
    if r_min <= 0:
        r_min = 0.1
    
    # 최소값보다 큰 값만 사용 (안전 처리)
    valid = radii[radii >= r_min]
    if len(valid) < 2:
        return 2.5, r_min
    
    log_ratios = np.log(valid / r_min)
    sum_log = np.sum(log_ratios)
    
    if sum_log < 1e-12:
        return 2.5, r_min
    
    alpha = len(valid) / sum_log
    
    return float(alpha), float(r_min)


def compute_domain_volume(
    slab_x_range: Tuple[float, float],
    tunnel_poly_yz: np.ndarray = None,
    tunnel_area: float = None
) -> float:
    """
    분석 도메인의 체적을 계산합니다.
    
    V = tunnel_area × x_span
    """
    x_min, x_max = slab_x_range
    x_span = abs(x_max - x_min)
    
    if tunnel_area is not None:
        return tunnel_area * x_span
    
    if tunnel_poly_yz is not None and len(tunnel_poly_yz) >= 3:
        # Shoelace formula for polygon area
        n_pts = len(tunnel_poly_yz)
        area = 0.0
        for i in range(n_pts):
            j = (i + 1) % n_pts
            area += tunnel_poly_yz[i, 0] * tunnel_poly_yz[j, 1]
            area -= tunnel_poly_yz[j, 0] * tunnel_poly_yz[i, 1]
        area = 0.5 * abs(area)
        return area * x_span
    
    # 기본값: 10m × 10m 터널 단면 × x_span
    return 100.0 * x_span


def extract_dfn_parameters(
    planes: List[ReconstructedPlane],
    set_stats: Dict[int, Tuple[np.ndarray, float]],
    slab_x_range: Tuple[float, float],
    tunnel_poly_yz: np.ndarray = None,
    tunnel_area: float = None
) -> DFNParameterResult:
    """
    복원된 평면들로부터 DFN 통계 파라미터를 추출합니다.
    
    Args:
        planes: 반경이 추정되고 세트가 분류된 ReconstructedPlane 리스트
        set_stats: {set_id: (mean_normal, kappa)} 세트별 통계
        slab_x_range: (x_min, x_max) Slab 분석 범위
        tunnel_poly_yz: 터널 단면 폴리곤 좌표
        tunnel_area: 터널 단면적 (직접 지정 시)
    
    Returns:
        DFNParameterResult: 종합 DFN 파라미터 결과
    """
    # 도메인 체적 계산
    domain_volume = compute_domain_volume(slab_x_range, tunnel_poly_yz, tunnel_area)
    
    # 세트별 분리
    set_planes = {}
    for p in planes:
        sid = p.set_id
        if sid < 0:
            sid = 0  # 미분류 세트
        if sid not in set_planes:
            set_planes[sid] = []
        set_planes[sid].append(p)
    
    # 세트별 파라미터 추출
    set_results = {}
    
    for sid, s_planes in set_planes.items():
        n_planes = len(s_planes)
        
        # 반경 수집
        radii = np.array([p.estimated_radius for p in s_planes])
        mean_radius = float(np.mean(radii))
        
        # 법선벡터 수집
        normals = np.array([p.normal for p in s_planes])
        
        # Fisher kappa 재계산
        kappa, mean_normal, R_mag = calculate_fisher_kappa(normals)
        
        # Dip / DipDirection
        dip, dip_direction = normal_to_dip_dipdirection(mean_normal)
        
        # Pareto MLE 피팅
        alpha_R, r_min = fit_pareto_mle(radii)
        
        # P30: 체적 개수 밀도
        P30 = n_planes / domain_volume if domain_volume > 0 else 0.0
        
        # P32: 체적 면적 밀도
        total_area = np.sum(np.pi * radii**2)
        P32 = total_area / domain_volume if domain_volume > 0 else 0.0
        
        # Terzaghi 방향 편향 보정
        # sin(θ) = sqrt(ny² + nz²): 터널 축(X)과 법선 사이의 교차각 정현
        sin_thetas = np.sqrt(normals[:, 1]**2 + normals[:, 2]**2)
        mean_sin_theta = float(np.mean(sin_thetas)) if len(sin_thetas) > 0 else 1.0
        
        # 보정된 P32: P21_proxy / mean_sin_theta
        # P21 근사: 터널 단면적 대비 관측 면적 → 여기서는 P32를 직접 보정
        P32_terzaghi = P32 / mean_sin_theta if mean_sin_theta > 0.01 else P32
        
        set_results[sid] = DFNSetResult(
            set_id=sid,
            n_planes=n_planes,
            mean_normal=mean_normal,
            dip=dip,
            dip_direction=dip_direction,
            kappa=kappa,
            mean_radius=mean_radius,
            radii=radii,
            alpha_R=alpha_R,
            r_min=r_min,
            P30=P30,
            P32=P32,
            P32_terzaghi=P32_terzaghi,
            mean_sin_theta=mean_sin_theta
        )
    
    return DFNParameterResult(
        n_sets=len(set_results),
        domain_volume=domain_volume,
        total_reconstructed=len(planes),
        set_results=set_results
    )


def format_dfn_summary_table(
    result: DFNParameterResult,
    gt_centers: np.ndarray = None,
    gt_normals: np.ndarray = None,
    gt_radii: np.ndarray = None,
    gt_set_ids: np.ndarray = None,
    crop_limit: float = 25.0
) -> str:
    """
    DFN 파라미터 추출 결과를 보기 좋은 테이블 문자열로 포매팅합니다.
    Ground Truth가 제공되면 비교 컬럼을 추가합니다.
    
    Returns:
        포매팅된 테이블 문자열
    """
    lines = []
    lines.append("=" * 110)
    lines.append("              DFN PARAMETER EXTRACTION SUMMARY REPORT")
    lines.append("=" * 110)
    lines.append(f"  Domain Volume: {result.domain_volume:.1f} m³")
    lines.append(f"  Total Reconstructed Planes: {result.total_reconstructed}")
    lines.append(f"  Number of Sets: {result.n_sets}")
    lines.append("")
    
    # 헤더
    has_gt = (gt_centers is not None and gt_radii is not None and gt_set_ids is not None)
    
    if has_gt:
        header = f" {'SET':>4} | {'N':>4} | {'Dip':>6} | {'DipDir':>6} | {'kappa':>8} | {'alphaR':>6} | {'meanR':>6} | {'P32':>8} | {'P32_Trz':>8} | {'P32_GT':>8} | {'Err(%)':>8}"
    else:
        header = f" {'SET':>4} | {'N':>4} | {'Dip':>6} | {'DipDir':>6} | {'kappa':>8} | {'alphaR':>6} | {'meanR':>6} | {'P32':>8} | {'P32_Trz':>8}"
    
    lines.append(header)
    lines.append("-" * 110)
    
    # Ground Truth P32 계산 (제공된 경우)
    gt_p32_by_set = {}
    if has_gt:
        db_volume = (2.0 * crop_limit) ** 3
        unique_sets = sorted(set(gt_set_ids.astype(int)))
        for sid in unique_sets:
            mask_set = (gt_set_ids == sid)
            mask_crop = (
                (np.abs(gt_centers[:, 0]) <= crop_limit) &
                (np.abs(gt_centers[:, 1]) <= crop_limit) &
                (np.abs(gt_centers[:, 2]) <= crop_limit)
            )
            mask_combined = mask_set & mask_crop
            total_area = np.sum(np.pi * (gt_radii[mask_combined] ** 2))
            gt_p32_by_set[int(sid)] = total_area / db_volume
    
    for sid, sr in sorted(result.set_results.items()):
        if has_gt and sid in gt_p32_by_set:
            gt_p32 = gt_p32_by_set[sid]
            err = abs(sr.P32 - gt_p32) / gt_p32 * 100 if gt_p32 > 0 else 0.0
            line = (
                f" Set{sid:>2} | {sr.n_planes:>4} | {sr.dip:>5.1f}° | {sr.dip_direction:>5.1f}° | "
                f"{sr.kappa:>8.2f} | {sr.alpha_R:>6.3f} | {sr.mean_radius:>5.2f}m | "
                f"{sr.P32:>8.4f} | {sr.P32_terzaghi:>8.4f} | {gt_p32:>8.4f} | {err:>7.2f}%"
            )
        else:
            line = (
                f" Set{sid:>2} | {sr.n_planes:>4} | {sr.dip:>5.1f}° | {sr.dip_direction:>5.1f}° | "
                f"{sr.kappa:>8.2f} | {sr.alpha_R:>6.3f} | {sr.mean_radius:>5.2f}m | "
                f"{sr.P32:>8.4f} | {sr.P32_terzaghi:>8.4f}"
            )
        lines.append(line)
    
    lines.append("=" * 110)
    
    return "\n".join(lines)


def export_dfn_parameters_json(result: DFNParameterResult, filepath: str):
    """
    DFN 파라미터 결과를 JSON 파일로 내보냅니다.
    """
    import json
    
    out = {
        "n_sets": int(result.n_sets),
        "domain_volume": float(result.domain_volume),
        "total_reconstructed": int(result.total_reconstructed),
        "sets": {}
    }
    
    for sid, sr in result.set_results.items():
        out["sets"][str(sid)] = {
            "n_planes": int(sr.n_planes),
            "mean_normal": [float(x) for x in sr.mean_normal],
            "dip": float(sr.dip),
            "dip_direction": float(sr.dip_direction),
            "kappa": float(sr.kappa),
            "mean_radius": float(sr.mean_radius),
            "alpha_R": float(sr.alpha_R),
            "r_min": float(sr.r_min),
            "P30": float(sr.P30),
            "P32": float(sr.P32),
            "P32_terzaghi": float(sr.P32_terzaghi),
            "mean_sin_theta": float(sr.mean_sin_theta)
        }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=4, ensure_ascii=False)
