# -*- coding: utf-8 -*-
# [구명칭 호환 셔틀] v2에서 estimate_p32.py 로 개명되었다.
#   기본 보정계수 모드가 해석식(analytic_esinphi, 표집 없는 결정론적 구적)인데
#   파일명의 "mc_calibrated"(순방향 MC 보정)는 구모드 명칭이라 혼동을 주기 때문.
#   기존 명령 `python -m dfn_analysis.estimate_p32_mc_calibrated ...` 도 그대로 동작한다.
from dfn_analysis.estimate_p32 import *  # noqa: F401,F403
from dfn_analysis.estimate_p32 import main  # noqa: F401

if __name__ == "__main__":
    main()
