# -*- coding: utf-8 -*-
"""[7] export_domain_dfn_json 실현 1개의 단계별 소요 시간 실측 (monkeypatch 타이머)."""
import json as _json
import sys
import time

from dfn_analysis import export_domain_dfn_json as E

T = {}


def timed(name, fn):
    def wrap(*a, **k):
        t0 = time.perf_counter()
        r = fn(*a, **k)
        T[name] = T.get(name, 0.0) + (time.perf_counter() - t0)
        return r
    return wrap


E.G.generate_hidden_discs = timed("생성 generate_hidden_discs", E.G.generate_hidden_discs)
E.G.remove_face_intersecting = timed("조건화 remove_face_intersecting", E.G.remove_face_intersecting)
E.select_intersecting = timed("도메인 절단 select_intersecting", E.select_intersecting)
E.json.dump = timed("JSON 쓰기", _json.dump)

sp = sys.argv[1]
sys.argv = ["x", "--pipeline-dir", "demo_output/dfm_demo", "--sets", "1", "2", "3", "4",
            "--rmax-local", "25", "--lmin-det", "0.5", "--seed", "7",
            "--halo", "5", "--halo-z", "7.48", "--out", sp + "/timing_export_profiled.json"]
t0 = time.perf_counter()
E.main()
total = time.perf_counter() - t0
print("\n=== 단계별 시간 ===")
acc = 0.0
for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
    print(f"{k:45s} {v:7.1f} s  ({v/total*100:4.1f}%)")
    acc += v
print(f"{'기타(입력 로드·벽면 절리선·문서 조립 등)':43s} {total-acc:7.1f} s  ({(total-acc)/total*100:4.1f}%)")
print(f"{'총':45s} {total:7.1f} s")
