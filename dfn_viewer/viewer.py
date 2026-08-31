"""DFN 도메인 뷰어 프로토타입 (독립 프로그램).

export_domain_dfn_json.py 가 생성한 도메인 DFN JSON을 읽어
PySide6 + pyvistaqt 창에서 인터랙티브하게 표시한다.

좌표계: x = 터널 굴진 방향, 막장면 = x ≈ const 평면, 단위 m (JSON meta 기준).

실행:
    python viewer.py [dfn_domain_*.json]
    python viewer.py --selftest   # 2초 후 자동 종료 + 스크린샷 저장 (검증용)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QPushButton, QSlider, QVBoxLayout, QWidget,
)
import vtk

DEFAULT_JSON = (
    Path(__file__).resolve().parent.parent
    / "handoffv1" / "example_io" / "dfn_domain_x22.204-32.204_halo5.json"
)

SET_COLORS = {
    1: "#1f77b4",
    2: "#2ca02c",
    3: "#ff7f0e",
    4: "#9467bd",
}
FALLBACK_COLOR = "#7f7f7f"
LABEL_OPACITY = {"observed": 1.0, "unobserved": 0.35}
DISC_SIDES = 24


def build_disc_mesh(centers_xyz, normals_xyz, radii_m, n_sides=DISC_SIDES):
    """디스크(원판) 묶음을 하나의 PolyData 로 벡터화 생성한다."""
    centers = np.asarray(centers_xyz, dtype=float)
    normals = np.asarray(normals_xyz, dtype=float)
    radii = np.asarray(radii_m, dtype=float)
    n = len(centers)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # 법선과 평행하지 않은 기준축을 골라 디스크 평면의 정규직교기저 (u, v) 구성
    ref = np.where(np.abs(normals[:, 2:3]) < 0.9, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    u = np.cross(normals, ref)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(normals, u)

    theta = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    ring = (
        np.cos(theta)[None, :, None] * u[:, None, :]
        + np.sin(theta)[None, :, None] * v[:, None, :]
    )
    pts = centers[:, None, :] + radii[:, None, None] * ring

    faces = np.empty((n, n_sides + 1), dtype=np.int64)
    faces[:, 0] = n_sides
    faces[:, 1:] = np.arange(n * n_sides).reshape(n, n_sides)
    return pv.PolyData(pts.reshape(-1, 3), faces=faces.ravel())


def build_trace_mesh(traces):
    """tunnel_wall_traces 세그먼트 리스트를 라인 PolyData 로 만든다."""
    p0_xyz = np.array([t["p0_xyz_m"] for t in traces], dtype=float)
    p1_xyz = np.array([t["p1_xyz_m"] for t in traces], dtype=float)
    n = len(traces)
    pts = np.empty((2 * n, 3))
    pts[0::2] = p0_xyz
    pts[1::2] = p1_xyz
    lines = np.empty((n, 3), dtype=np.int64)
    lines[:, 0] = 2
    lines[:, 1] = np.arange(0, 2 * n, 2)
    lines[:, 2] = np.arange(1, 2 * n, 2)
    return pv.PolyData(pts, lines=lines.ravel())


def build_tunnel_mesh(polygon_yz, x_range):
    """YZ 단면 폴리곤을 x 방향으로 압출한 터널 표면을 만든다."""
    poly_yz = np.asarray(polygon_yz, dtype=float)
    n = len(poly_yz)
    pts_xyz = np.column_stack([np.full(n, x_range[0]), poly_yz[:, 0], poly_yz[:, 1]])
    loop = pv.PolyData(
        pts_xyz, lines=np.concatenate([[n + 1], np.arange(n), [0]])
    )
    return loop.extrude([x_range[1] - x_range[0], 0.0, 0.0], capping=False)


class DFNViewerWindow(QMainWindow):
    def __init__(self, json_path):
        super().__init__()
        self.setWindowTitle("DFN Domain Viewer (prototype)")
        self.resize(1400, 900)

        central = QWidget()
        layout = QHBoxLayout(central)
        self.setCentralWidget(central)

        self.panel = QVBoxLayout()
        panel_widget = QWidget()
        panel_widget.setLayout(self.panel)
        panel_widget.setFixedWidth(280)
        layout.addWidget(panel_widget)

        self.plotter = QtInteractor(central)
        layout.addWidget(self.plotter.interactor, stretch=1)
        self.plotter.set_background("white")
        self.plotter.add_axes()

        # (set_id, label) -> {actor, count}
        self.groups = {}
        self.set_checks = {}
        self.label_checks = {}
        self.clip_plane = vtk.vtkPlane()
        self.clip_plane.SetNormal(-1.0, 0.0, 0.0)
        self.meta = {}

        self._build_static_panel()
        if json_path is not None:
            self.load_json(json_path)

    # ---------- UI ----------

    def _build_static_panel(self):
        open_btn = QPushButton("JSON 열기...")
        open_btn.clicked.connect(self._open_dialog)
        self.panel.addWidget(open_btn)

        self.file_label = QLabel("(파일 없음)")
        self.file_label.setWordWrap(True)
        self.panel.addWidget(self.file_label)

        self.sets_box = QGroupBox("절리 세트")
        self.sets_layout = QVBoxLayout(self.sets_box)
        self.panel.addWidget(self.sets_box)

        labels_box = QGroupBox("균열 종류")
        labels_layout = QVBoxLayout(labels_box)
        for label, text in [
            ("observed", "observed (관측 복원)"),
            ("unobserved", "unobserved (확률 생성)"),
        ]:
            cb = QCheckBox(text)
            cb.setChecked(True)
            cb.stateChanged.connect(self._update_visibility)
            self.label_checks[label] = cb
            labels_layout.addWidget(cb)
        self.panel.addWidget(labels_box)

        clip_box = QGroupBox("X 클리핑 (x ≤ 슬라이더 값만 표시)")
        clip_layout = QVBoxLayout(clip_box)
        self.clip_check = QCheckBox("클리핑 적용")
        self.clip_check.stateChanged.connect(self._update_clipping)
        clip_layout.addWidget(self.clip_check)
        self.clip_slider = QSlider(Qt.Horizontal)
        self.clip_slider.setRange(0, 100)
        self.clip_slider.setValue(100)
        self.clip_slider.valueChanged.connect(self._update_clipping)
        clip_layout.addWidget(self.clip_slider)
        self.clip_value_label = QLabel("x = -")
        clip_layout.addWidget(self.clip_value_label)
        self.panel.addWidget(clip_box)

        overlay_box = QGroupBox("보조 표시")
        overlay_layout = QVBoxLayout(overlay_box)
        self.trace_check = QCheckBox("터널 벽면 절리선")
        self.trace_check.setChecked(True)
        self.trace_check.stateChanged.connect(self._update_visibility)
        overlay_layout.addWidget(self.trace_check)
        self.tunnel_check = QCheckBox("터널 표면 (굴착 구간)")
        self.tunnel_check.setChecked(True)
        self.tunnel_check.stateChanged.connect(self._update_visibility)
        overlay_layout.addWidget(self.tunnel_check)
        self.domain_check = QCheckBox("도메인 박스")
        self.domain_check.setChecked(True)
        self.domain_check.stateChanged.connect(self._update_visibility)
        overlay_layout.addWidget(self.domain_check)
        self.panel.addWidget(overlay_box)

        shot_btn = QPushButton("스크린샷 저장...")
        shot_btn.clicked.connect(self._save_screenshot)
        self.panel.addWidget(shot_btn)

        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        self.panel.addWidget(self.stats_label)
        self.panel.addStretch(1)

    def _open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "도메인 DFN JSON 선택", "", "JSON (*.json)"
        )
        if path:
            self.load_json(Path(path))

    def _save_screenshot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "스크린샷 저장", "dfn_view.png", "PNG (*.png)"
        )
        if path:
            self.plotter.screenshot(path)

    # ---------- 데이터 로딩 ----------

    def load_json(self, json_path):
        json_path = Path(json_path)
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        self.meta = data.get("meta", {})
        fractures = data["fractures"]

        self.plotter.clear()
        self.plotter.add_axes()
        self.groups.clear()
        for cb in self.set_checks.values():
            cb.deleteLater()
        self.set_checks.clear()

        # (set_id, label) 별로 묶어 디스크 메시 생성 — 액터 수를 최소화
        grouped = {}
        for fr in fractures:
            grouped.setdefault((fr["set_id"], fr["label"]), []).append(fr)

        for (set_id, label), rows in sorted(grouped.items()):
            mesh = build_disc_mesh(
                [r["center_xyz_m"] for r in rows],
                [r["normal_xyz"] for r in rows],
                [r["radius_m"] for r in rows],
            )
            color = SET_COLORS.get(set_id, FALLBACK_COLOR)
            actor = self.plotter.add_mesh(
                mesh,
                color=color,
                opacity=LABEL_OPACITY.get(label, 1.0),
                name=f"set{set_id}_{label}",
            )
            self.groups[(set_id, label)] = {"actor": actor, "count": len(rows)}

        for set_id in sorted({sid for sid, _ in self.groups}):
            total = sum(
                g["count"] for (sid, _), g in self.groups.items() if sid == set_id
            )
            cb = QCheckBox(f"Set {set_id} ({total}개)")
            cb.setChecked(True)
            cb.setStyleSheet(
                f"QCheckBox {{ color: {SET_COLORS.get(set_id, FALLBACK_COLOR)};"
                f" font-weight: bold; }}"
            )
            cb.stateChanged.connect(self._update_visibility)
            self.set_checks[set_id] = cb
            self.sets_layout.addWidget(cb)

        traces = data.get("tunnel_wall_traces", [])
        self.trace_actor = None
        if traces:
            self.trace_actor = self.plotter.add_mesh(
                build_trace_mesh(traces), color="red", line_width=4,
                name="wall_traces",
            )

        domain = self.meta.get("domain", {})
        self.tunnel_actor = None
        polygon_yz = domain.get("tunnel_polygon_yz_m")
        tunnel_x = domain.get("excavated_tunnel_x_range_m")
        if polygon_yz and tunnel_x:
            self.tunnel_actor = self.plotter.add_mesh(
                build_tunnel_mesh(polygon_yz, tunnel_x),
                color="lightgray", opacity=0.3, name="tunnel",
            )

        self.domain_actor = None
        x_range = domain.get("x_range_m")
        yz = domain.get("yz_bounds_m")
        if x_range and yz:
            box = pv.Box(
                (x_range[0], x_range[1], yz["y_min"], yz["y_max"],
                 yz["z_min"], yz["z_max"])
            )
            self.domain_actor = self.plotter.add_mesh(
                box, style="wireframe", color="black", line_width=1,
                name="domain_box",
            )
            self._clip_x_range = (float(x_range[0]), float(x_range[1]))
        else:
            b = self.plotter.bounds
            self._clip_x_range = (b[0], b[1])

        self.file_label.setText(json_path.name)
        self.plotter.reset_camera()
        self.plotter.view_isometric()
        self._update_visibility()
        self._update_clipping()

    # ---------- 상태 갱신 ----------

    def _update_visibility(self):
        shown = 0
        for (set_id, label), g in self.groups.items():
            visible = (
                self.set_checks[set_id].isChecked()
                and self.label_checks[label].isChecked()
            )
            g["actor"].SetVisibility(visible)
            if visible:
                shown += g["count"]
        if self.trace_actor is not None:
            self.trace_actor.SetVisibility(self.trace_check.isChecked())
        if self.tunnel_actor is not None:
            self.tunnel_actor.SetVisibility(self.tunnel_check.isChecked())
        if self.domain_actor is not None:
            self.domain_actor.SetVisibility(self.domain_check.isChecked())
        total = sum(g["count"] for g in self.groups.values())
        self.stats_label.setText(
            f"표시 균열: {shown} / {total}\n(클리핑은 개수에 미반영)"
        )
        self.plotter.render()

    def _update_clipping(self):
        x0, x1 = getattr(self, "_clip_x_range", (0.0, 1.0))
        x = x0 + (x1 - x0) * self.clip_slider.value() / 100.0
        self.clip_value_label.setText(f"x = {x:.2f} m")
        self.clip_plane.SetOrigin(x, 0.0, 0.0)
        enabled = self.clip_check.isChecked()
        for g in self.groups.values():
            mapper = g["actor"].GetMapper()
            mapper.RemoveAllClippingPlanes()
            if enabled:
                mapper.AddClippingPlane(self.clip_plane)
        self.plotter.render()


def main():
    argv = [a for a in sys.argv[1:] if a != "--selftest"]
    selftest = "--selftest" in sys.argv[1:]
    json_path = Path(argv[0]) if argv else DEFAULT_JSON
    if not json_path.exists():
        print(f"입력 JSON을 찾을 수 없습니다: {json_path}")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = DFNViewerWindow(json_path)
    window.show()

    if selftest:
        out_png = Path(__file__).parent / "selftest_screenshot.png"

        def _finish():
            window.plotter.screenshot(str(out_png))
            print(f"selftest screenshot: {out_png}")
            app.quit()

        QTimer.singleShot(2000, _finish)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
