# DFN Domain Viewer (prototype)

`export_domain_dfn_json.py` 가 생성한 도메인 DFN JSON을 읽어 표시하는
독립 실행형 3D 뷰어 프로토타입입니다. (PySide6 + pyvistaqt + PyVista/VTK)

## 실행

```bash
pip install -r requirements.txt
python viewer.py                          # 기본: handoffv1/example_io/dfn_domain_*.json
python viewer.py path/to/dfn_domain.json  # 파일 지정
python viewer.py --selftest               # 2초 후 자동 종료 + selftest_screenshot.png 저장
```

## 기능

- 절리 디스크를 (set, observed/unobserved) 그룹별 단일 메시로 렌더링
  — 약 2만 개 디스크도 인터랙티브하게 동작
- 세트별 표시 토글 (색상: Set 1 파랑 / 2 초록 / 3 주황 / 4 보라)
- observed(관측 복원, 불투명) / unobserved(확률 생성, 반투명) 토글
- X 클리핑 슬라이더: x ≤ 값 인 부분만 표시 (굴진 방향 단면 검사)
- 터널 벽면 절리선(빨강), 굴착 구간 터널 표면, 도메인 박스 표시 토글
- 스크린샷 저장

## 좌표계

JSON meta 기준: x = 터널 굴진 방향(East), y = North, z = Up, 단위 m.
막장면은 x ≈ const 평면.

## 배포 (참고)

독립 exe 패키징: `pip install pyinstaller` 후
`pyinstaller --onefile --windowed viewer.py`
