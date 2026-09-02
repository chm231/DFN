# -*- coding: utf-8 -*-
# [구명칭 호환 셔틀] v2에서 radius_powerlaw_likelihood.py 로 개명되었다.
#   기본 우도 방식이 hybrid(참 현길이 분포는 해석식, 창·절단 커널만 kr불변 MC 1회)인데
#   파일명의 "window_mc"는 legacy 전량-MC 모드 명칭이라 혼동을 주기 때문.
#   기존 import/명령 모두 그대로 동작한다.
from dfn_analysis.radius_powerlaw_likelihood import *  # noqa: F401,F403
from dfn_analysis.radius_powerlaw_likelihood import main  # noqa: F401

if __name__ == "__main__":
    main()
