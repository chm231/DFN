# -*- coding: utf-8 -*-
"""무한평면 확장 대상 절리 선별 (관측 우선 할당 규칙).

wedge_top20 / build_wedge_case / make_bounding_planes 가 같은 규칙을 써야
전달 CSV 의 rank 번호와 실제 판정에 쓰인 평면이 일치한다. 그래서 선별은
여기 한 곳에만 둔다.

규칙: 관측(label="observed") 절리 중 반지름 상위 n_obs 개를 먼저 배정하고,
남은 자리를 미관측 절리 반지름 상위로 채운다. 관측이 n_obs 개보다 적으면
부족분은 미관측이 더 채운다. n_obs=0 이면 예전 순수 반지름 순위와 동일하다.

배경: 관측 절리는 터널 단면 관측창에 잘린 절리선에서 복원되어 반지름이
구조적으로 과소추정된다(관측 최대 ~4.8 m vs 미관측 최대 ~19 m). 그래서
순수 반지름 순위로는 관측 절리가 상위 20 에 0~1 개만 들어간다.
"""
import numpy as np

N_OBS_DEFAULT = 10


def select_top_idx(fractures, n_top, n_obs=N_OBS_DEFAULT):
    """관측 우선 할당으로 무한평면 확장 대상 인덱스를 고른다.

    반환: (idx, summary) — idx 는 관측분 먼저, 각 그룹 내 반지름 내림차순.
    """
    radii = np.array([f_["radius_m"] for f_ in fractures], dtype=np.float64)
    labels = np.array([f_.get("label", "unobserved") for f_ in fractures])
    obs_i = np.nonzero(labels == "observed")[0]
    unobs_i = np.nonzero(labels != "observed")[0]

    n_from_obs = min(n_obs, len(obs_i), n_top)
    pick_obs = obs_i[np.argsort(radii[obs_i])[::-1][:n_from_obs]]
    pick_unobs = unobs_i[np.argsort(radii[unobs_i])[::-1][:n_top - n_from_obs]]
    idx = np.concatenate([pick_obs, pick_unobs]).astype(int)

    summary = (f"select n_top={n_top} n_obs_req={n_obs} "
               f"observed={len(pick_obs)} unobserved={len(pick_unobs)} "
               f"r_range={radii[idx].min():.2f}~{radii[idx].max():.2f}m")
    return idx, summary
