# =============================================================================
# 파일 역할: 임의(arbitrary) trace dataset에 대해 파이프라인을 돌릴 수 있도록,
#           set별 크기분포/방향 파라미터를 외부 JSON config에서 런타임에 주입한다.
#   - 기존 코드는 forsmark/laxemar preset(SITE_* dict)에 하드코딩돼 있었다.
#   - 이 모듈은 JSON을 읽어 그 preset dict들에 새 dataset을 "site"처럼 등록한다.
#     → 기존 site 기반 코드 경로를 수정 없이 임의 데이터셋에 재사용할 수 있다.
#
# 주요 입력: dataset config JSON 경로(아래 스키마). 다른 인자 없음(라이브러리 모듈).
# JSON 스키마:
# {
#   "dataset_name": "my_dataset",
#   "sets": {
#     "1": {"p32_base": 0.602, "dist_type": "powerlaw", "r0": 0.28,
#           "trend": 182.8, "plunge": -1.7, "kappa": 22.1},
#     ...
#   }
# }
#   - p32_base / dist_type / r0 : 크기분포 (모집단 반지름 샘플링 + support 스케일링)
#   - trend / plunge / kappa    : Fisher 방향 (교차확률 MC)
#
# 주요 출력: 등록된 dataset_name (이 값을 --site 로 넘겨 사용). 부수효과로 세 preset
#           dict(SITE_SET_CONFIG / SITE_FISHER_PARAMS / SITE_SET_SUPPORT_INFO)에 항목 추가.
#
# 핵심 처리 흐름:
#   1) load_dataset_config: JSON 로드 + 필수 키(dataset_name/sets) 유효성 검사
#   2) register_dataset: set별로 크기분포/Fisher방향/지지구간 항목을 구성
#   3) 세 preset dict에 dataset_name 키로 주입(덮어쓰기 아닌 새 site 추가)
#   4) load_and_register: 위 두 단계를 한 번에 수행하고 dataset_name 반환
# =============================================================================
from __future__ import annotations

import json
from typing import Any, Dict


# JSON config 파일을 읽어 유효성(필수 키)만 확인한 뒤 dict로 반환한다.
def load_dataset_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "dataset_name" not in cfg or "sets" not in cfg:
        raise ValueError("dataset config must contain 'dataset_name' and 'sets'")
    if not cfg["sets"]:
        raise ValueError("dataset config 'sets' is empty")
    return cfg


# config의 set별 파라미터를 기존 preset dict들(SITE_SET_CONFIG / SITE_FISHER_PARAMS /
# SITE_SET_SUPPORT_INFO)에 dataset_name 키로 주입한다.
#   - 이렇게 하면 --site <dataset_name> 으로 기존 site 코드 경로를 그대로 탄다.
#   - 반환: 등록된 dataset_name (문자열)
def register_dataset(cfg: Dict[str, Any]) -> str:
    # preset dict를 보유한 모듈을 늦게 import (순환 import 방지)
    from dfn_analysis import build_p32_pilot_summary as bp
    from dfn_analysis import estimate_radius_powerlaw_window_mc as wm

    name = str(cfg["dataset_name"])
    set_cfg: Dict[int, Dict[str, Any]] = {}
    fisher: Dict[int, tuple] = {}
    support: Dict[int, Dict[str, Any]] = {}

    # set별로 세 preset dict 항목을 구성
    for sid_str, s in cfg["sets"].items():
        sid = int(sid_str)
        # 크기분포 (build_p32_pilot_summary.SITE_SET_CONFIG 형식)
        #   powerlaw 추정은 dist_type만 사용하고 r0/p32_base는 쓰지 않으므로 선택적이다
        #   (r0는 exponential 전용, p32_base는 검증용 P32_reference 전용).
        set_cfg[sid] = {
            "p32_base": float(s.get("p32_base", float("nan"))),
            "dist_type": str(s.get("dist_type", "powerlaw")),
            "r0": float(s.get("r0", 0.0)),
        }
        # Fisher 방향 (estimate_radius_powerlaw_window_mc.SITE_FISHER_PARAMS 형식: (trend, plunge, kappa))
        fisher[sid] = (float(s["trend"]), float(s["plunge"]), float(s["kappa"]))
        # 지지구간/크기 타입 (SITE_SET_SUPPORT_INFO 형식); r0는 선택적(powerlaw 미사용)
        support[sid] = {"type": str(s.get("dist_type", "powerlaw")), "table_r0": float(s.get("r0", 0.0))}

    # 기존 preset dict에 등록 (덮어쓰기가 아니라 새 site 키 추가)
    bp.SITE_SET_CONFIG[name] = set_cfg
    wm.SITE_FISHER_PARAMS[name] = fisher
    wm.SITE_SET_SUPPORT_INFO[name] = support
    return name


# 편의 함수: JSON 경로 하나로 로드 + 등록을 수행하고 dataset_name을 돌려준다.
def load_and_register(path: str) -> str:
    return register_dataset(load_dataset_config(path))
