"""보고서(슬라이드) 키워드 블록 ①~④에 대응하는 figure 생성.

실측 데이터(storage/output/pipeline_test_laxemar/)에 근거해 그린다.
  ① 절리 전처리         : 평면 막장면 위 절리선 / 길이·중심 / censoring / 짧은 trace 제거
  ② 절리 통계 역산      : pole stereonet / kr profile likelihood / bootstrap CI / Laxemar 검증
  ③ 절리망 생성         : (a) 관측 trace(실측)  (b) 역산 파라미터 기반 확률적 생성(개념 예시)
  ④ 안정성 해석 연계    : 통합 disk dataset → 안정성 모듈 흐름도(개념도)

③(b)와 ④는 설계단계이므로 "개념/예시(preview)"로 명시한다.

실행:
    python scripts/make_report_figures.py --base storage/output/pipeline_test_laxemar
출력:
    docs/figures/fig_block1_preprocessing.png ... fig_block4_stability_link.png
"""
import argparse
import csv
import os

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

try:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

SETS = [1, 2, 3, 5]
SET_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 5: "#9467bd"}
CLASS_COLORS = {0: "#2ca02c", 1: "#ff7f0e", 2: "#d62728"}
CLASS_LABELS = {0: "Type 0 (내부)", 1: "Type 1 (한 끝 절단)", 2: "Type 2 (양 끝 절단)"}


# ----------------------------- 데이터 로딩 ---------------------------------
def load_traces(base):
    path = os.path.join(base, "trace_dataset", "trace_dataset_3d.h5")
    with h5py.File(path, "r") as f:
        g = f["traces"]
        d = {
            "set_id": g["set_id"][:].astype(int),
            "face_id": g["face_id"][:].astype(int),
            "face_x": g["face_x_m"][:].astype(float),
            "p0": g["p0_xyz"][:].astype(float),
            "p1": g["p1_xyz"][:].astype(float),
            "length": g["observed_length_m"][:].astype(float),
            "censoring": g["censoring_class"][:].astype(int),
            "normal": g["trace_normal_xyz"][:].astype(float),
            "normal_valid": g["trace_normal_valid"][:].astype(int),
            "pl_start": g["polyline_vertex_start"][:].astype(int),
            "pl_count": g["polyline_vertex_count"][:].astype(int),
            "pl_xyz": g["polyline_vertices_xyz"][:].astype(float),
            "radius": g["radius_m"][:].astype(float),
        }
        d["tunnel_poly_yz"] = f["meta/tunnel_poly_yz"][:].astype(float)
        d["face_x_positions"] = f["meta/face_x_positions_m"][:].astype(float)
    return d


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def polygon_area(poly_yz):
    y, z = poly_yz[:, 0], poly_yz[:, 1]
    return 0.5 * abs(float(np.dot(y, np.roll(z, -1)) - np.dot(z, np.roll(y, -1))))


# --------------------------- 블록 ① 전처리 ---------------------------------
def block1(tr, out):
    poly = tr["tunnel_poly_yz"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))

    # (a) 평면 막장면 위 절리선 (제5장 데이터셋) — v2: 요철 컬러맵 제거
    axa = ax[0, 0]
    axa.fill(poly[:, 0], poly[:, 1], color="#f5f5f5", ec="none", zorder=0)
    axa.plot(poly[:, 0], poly[:, 1], "k-", lw=1.6, zorder=3)
    sel = np.where(tr["face_id"] == 1)[0]
    seen = set()
    for i in sel:
        s = int(tr["set_id"][i])
        lab = f"Set {s}" if s not in seen else None
        seen.add(s)
        axa.plot([tr["p0"][i, 1], tr["p1"][i, 1]], [tr["p0"][i, 2], tr["p1"][i, 2]],
                 "-", color=SET_COLORS.get(s, "0.3"), lw=1.3, label=lab, zorder=2)
    a_face = polygon_area(poly)
    axa.text(0.02, 0.02,
             f"관측창 = 터널 단면 다각형\n면적 $A_{{face}}$ = {a_face:.2f} m²  ·  절리선 {len(sel)}개",
             transform=axa.transAxes, fontsize=8.5, va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    axa.set_aspect("equal"); axa.set_xlabel("y (m)"); axa.set_ylabel("z (m)")
    axa.set_title(f"(a) 평면 막장면 위 절리선 (막장면 1, x = {tr['face_x'][sel][0]:.1f} m, y-z)")
    axa.legend(loc="upper right", fontsize=8, ncol=2)

    # (b) 길이 · 중심점 (실측 trace 1개 주석) — 2D 방향각(θ)은 v1에서 미사용이라 표기 제거
    axb = ax[0, 1]
    cand = [i for i in sel if tr["censoring"][i] == 0 and tr["length"][i] > 2.0]
    i = max(cand, key=lambda k: tr["length"][k]) if cand else sel[int(np.argmax(tr["length"][sel]))]
    p0 = tr["p0"][i, 1:3]; p1 = tr["p1"][i, 1:3]
    c = 0.5 * (p0 + p1)
    L = float(np.hypot(*(p1 - p0)))
    axb.plot(poly[:, 0], poly[:, 1], color="0.7", lw=1.2)
    axb.plot([p0[0], p1[0]], [p0[1], p1[1]], "-", color="#1f4e79", lw=2.5)
    axb.plot(*p0, "o", color="#1f4e79", ms=7); axb.plot(*p1, "o", color="#1f4e79", ms=7)
    axb.plot(*c, "s", color="#c00000", ms=8)
    axb.annotate("P1=(y1,z1)", p0, textcoords="offset points", xytext=(6, 8), fontsize=9)
    axb.annotate("P2=(y2,z2)", p1, textcoords="offset points", xytext=(6, 8), fontsize=9)
    axb.annotate(f"중심 C\n({c[0]:.2f}, {c[1]:.2f})", c, textcoords="offset points",
                 xytext=(8, -22), fontsize=9, color="#c00000")
    axb.text(c[0], c[1] + 0.9, f"L = {L:.2f} m", fontsize=11, ha="center",
             color="#1f4e79", fontweight="bold")
    axb.set_aspect("equal"); axb.set_xlabel("y (m)"); axb.set_ylabel("z (m)")
    axb.set_title("(b) 길이 L · 중심점 C 정의 (실측 trace 예)")

    # (c) censoring class 분류
    axc = ax[1, 0]
    axc.plot(poly[:, 0], poly[:, 1], "k-", lw=1.5)
    seen = set()
    for i in sel:
        cl = int(tr["censoring"][i])
        lab = CLASS_LABELS[cl] if cl not in seen else None
        seen.add(cl)
        axc.plot([tr["p0"][i, 1], tr["p1"][i, 1]], [tr["p0"][i, 2], tr["p1"][i, 2]],
                 "-", color=CLASS_COLORS.get(cl, "0.5"), lw=1.4, label=lab)
    axc.set_aspect("equal"); axc.set_xlabel("y (m)"); axc.set_ylabel("z (m)")
    axc.set_title("(c) 터널 경계 접촉 유형(censoring class) — 막장면 1")
    axc.legend(loc="upper right", fontsize=8)

    # (d) 짧은 trace 제거 (품질관리 임계값)
    axd = ax[1, 1]
    lengths = tr["length"]
    thr = 0.5  # 품질관리(QC) 예시 임계값
    bins = np.linspace(0, np.percentile(lengths, 99), 40)
    axd.hist(lengths[lengths >= thr], bins=bins, color="#4c78a8", alpha=0.9, label="채택 (≥ 임계값)")
    axd.hist(lengths[lengths < thr], bins=bins, color="#d62728", alpha=0.8, label="제거 (< 임계값)")
    axd.axvline(thr, color="k", ls="--", lw=1.3)
    axd.text(thr, axd.get_ylim()[1] * 0.9, f" QC 임계값 예시 = {thr} m", fontsize=9)
    axd.set_xlabel("관측 trace 길이 L (m)"); axd.set_ylabel("trace 수")
    n_removed = int(np.sum(lengths < thr))
    axd.set_title(f"(d) 임계값 이하 짧은 trace 제거 (제거 {n_removed} / 전체 {len(lengths)})")
    axd.legend(fontsize=9)

    fig.suptitle("① 절리 전처리 (Joint Preprocessing)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(out, "fig_block1_preprocessing.png")
    fig.savefig(p, dpi=180); plt.close(fig)
    return p


# ------------------------ 스테레오넷 보조 함수 -----------------------------
def pole_vec_enu(trend_deg, plunge_deg):
    tr = np.radians(trend_deg); pl = np.radians(plunge_deg)
    n = np.array([np.cos(pl) * np.sin(tr), np.cos(pl) * np.cos(tr), -np.sin(pl)])
    return n / np.linalg.norm(n)


def to_lower_hemisphere(n):
    n = n.copy()
    flip = n[:, 2] > 0.0
    n[flip] *= -1.0
    return n


def schmidt_xy(n_xyz):
    """등면적(Schmidt) 하반구 투영: pole (E,N,U) → (x,y) 평면좌표."""
    n = to_lower_hemisphere(np.atleast_2d(n_xyz))
    trend = np.arctan2(n[:, 0], n[:, 1])           # from North(+y), CW
    plunge = np.arcsin(np.clip(-n[:, 2], -1, 1))   # >=0 downward
    colat = np.pi / 2.0 - plunge                   # nadir 기준 각
    r = np.sqrt(2.0) * np.sin(colat / 2.0)
    return r * np.sin(trend), r * np.cos(trend)


# --------------------------- 블록 ② 통계 역산 ------------------------------
def block2(base, tr, out):
    comp = {int(r["set_id"]): r for r in read_csv_rows(os.path.join(base, "comparison_vs_laxemar.csv"))}
    krs = {int(r["set_id"]): r for r in read_csv_rows(os.path.join(base, "kr", "kr_summary_by_set.csv"))}
    p32 = {int(r["set_id"]): r for r in read_csv_rows(os.path.join(base, "p32", "p32_summary.csv"))}
    prof = read_csv_rows(os.path.join(base, "kr", "kr_profile_likelihood.csv"))

    fig, ax = plt.subplots(2, 2, figsize=(13, 11))

    # (a) pole 스테레오넷 (set별 유효 normal + 평균 pole)
    axa = ax[0, 0]
    circ = np.linspace(0, 2 * np.pi, 200)
    axa.plot(np.cos(circ), np.sin(circ), "k-", lw=1.2)
    axa.plot([0], [0], "k+", ms=8)
    for s in SETS:
        m = (tr["set_id"] == s) & (tr["normal_valid"] == 1)
        if not np.any(m):
            continue
        x, y = schmidt_xy(tr["normal"][m])
        axa.scatter(x, y, s=10, color=SET_COLORS[s], alpha=0.5, label=f"Set {s}")
        gx, gy = schmidt_xy(pole_vec_enu(float(comp[s]["gt_trend"]), float(comp[s]["gt_plunge"])))
        ex, ey = schmidt_xy(pole_vec_enu(float(comp[s]["est_trend"]), float(comp[s]["est_plunge"])))
        axa.scatter(gx, gy, s=140, marker="*", color=SET_COLORS[s], edgecolor="k", zorder=5)
        axa.scatter(ex, ey, s=90, marker="X", color=SET_COLORS[s], edgecolor="k", zorder=5)
    axa.text(0, 1.06, "N", ha="center", fontsize=10)
    axa.set_aspect("equal"); axa.axis("off")
    axa.set_title("(a) 절리군별 pole 등면적 투영 (하반구)\n★ 실제,  × 역산 평균 pole")
    axa.legend(loc="lower right", fontsize=8)

    # (b) kr window-MC profile likelihood
    axb = ax[1, 0]  # place after; keep index but assign below
    axb = ax[0, 1]
    for s in SETS:
        best_lmin = krs[s]["best_lmin_fit"]
        rows = [r for r in prof if int(r["set_id"]) == s and r["lmin_fit"] == best_lmin]
        if not rows:
            rows = [r for r in prof if int(r["set_id"]) == s]
        rows.sort(key=lambda r: float(r["kr_window_mc"]))
        kr = np.array([float(r["kr_window_mc"]) for r in rows])
        dl = -np.array([float(r["delta_loglik"]) for r in rows])  # 봉우리(0=최적)로 표시
        axb.plot(kr, dl, "-", color=SET_COLORS[s], lw=1.6, label=f"Set {s}")
        khat = float(krs[s]["kr_hat"])
        axb.axvline(khat, color=SET_COLORS[s], ls=":", lw=1.0)
    axb.axhline(0, color="k", lw=0.8)
    axb.set_xlabel("반지름 지수 $k_r$"); axb.set_ylabel("상대 log-likelihood (0 = 최적)")
    axb.set_title("(b) 반지름 지수 $k_r$ — window-aware forward-MC profile likelihood")
    axb.legend(fontsize=8); axb.set_ylim(-8, 1)

    # (c) 불확실성: kr, P32 bootstrap CI
    axc = ax[1, 0]
    xs = np.arange(len(SETS))
    kr_hat = np.array([float(krs[s]["kr_hat"]) for s in SETS])
    kr_lo = np.array([float(krs[s]["kr_ci_low"]) for s in SETS])
    kr_hi = np.array([float(krs[s]["kr_ci_high"]) for s in SETS])
    kr_true = np.array([float(krs[s]["kr_true"]) for s in SETS])
    axc.vlines(xs - 0.1, kr_lo, kr_hi, color="#1f77b4", lw=6, alpha=0.3)
    axc.scatter(xs - 0.1, kr_hat, marker="o", s=45, color="#1f77b4", zorder=5, label="$k_r$ 역산 (● 점, 막대=CI)")
    axc.scatter(xs - 0.1, kr_true, marker="_", s=220, color="k", zorder=6, label="$k_r$ 실제")
    axc.set_ylabel("$k_r$", color="#1f77b4"); axc.tick_params(axis="y", labelcolor="#1f77b4")
    axc2 = axc.twinx()
    p_hat = np.array([float(p32[s]["P32_hat"]) for s in SETS])
    p_lo = np.array([float(p32[s]["P32_ci_low"]) for s in SETS])
    p_hi = np.array([float(p32[s]["P32_ci_high"]) for s in SETS])
    p_ref = np.array([float(p32[s]["P32_reference"]) for s in SETS])
    axc2.vlines(xs + 0.1, p_lo, p_hi, color="#d62728", lw=6, alpha=0.3)
    axc2.scatter(xs + 0.1, p_hat, marker="s", s=40, color="#d62728", zorder=5, label="$P_{32}$ 역산 (■ 점, 막대=CI)")
    axc2.scatter(xs + 0.1, p_ref, marker="_", s=220, color="dimgray", zorder=6, label="$P_{32}$ 참조")
    axc2.set_ylabel("$P_{32}$ (m²/m³)", color="#d62728"); axc2.tick_params(axis="y", labelcolor="#d62728")
    axc.set_xticks(xs); axc.set_xticklabels([f"Set {s}" for s in SETS])
    axc.set_title("(c) 불확실성: $k_r$·$P_{32}$ bootstrap 신뢰구간")
    h1, l1 = axc.get_legend_handles_labels(); h2, l2 = axc2.get_legend_handles_labels()
    axc.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    # (d) Laxemar 검증: 역산/실제 비율
    axd = ax[1, 1]
    metrics = ["κ", "$k_r$", "$P_{32}$"]
    ratios = {s: [float(comp[s]["est_kappa"]) / float(comp[s]["gt_kappa"]),
                  float(comp[s]["kr_hat"]) / float(comp[s]["kr_true"]),
                  float(comp[s]["p32_hat"]) / float(comp[s]["p32_ref"])] for s in SETS}
    w = 0.2
    base_x = np.arange(len(metrics))
    for j, s in enumerate(SETS):
        axd.bar(base_x + (j - 1.5) * w, ratios[s], w, color=SET_COLORS[s], label=f"Set {s}")
    axd.axhline(1.0, color="k", ls="--", lw=1.2)
    axd.set_xticks(base_x); axd.set_xticklabels(metrics)
    axd.set_ylabel("역산 / 실제 비율")
    axd.set_title("(d) Laxemar 벤치마크 검증 (SKB, 2006) — 1.0에 가까울수록 정확")
    axd.legend(fontsize=8, ncol=2)

    fig.suptitle("② 절리 통계 정보 역산 (Statistical Inversion)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(out, "fig_block2_inversion.png")
    fig.savefig(p, dpi=180); plt.close(fig)
    return p


# ------------------- 블록 ③ 보조: 디스크/샘플러 ----------------------------
def disc_polygon(center, normal, radius, n=48):
    normal = normal / np.linalg.norm(normal)
    a = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, a); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    t = np.linspace(0, 2 * np.pi, n)
    return center + radius * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))


def sample_powerlaw_radius(kr, rmin, rmax, n, rng):
    u = rng.random(n)
    ratio = (rmin / rmax) ** kr
    return rmin * (1.0 - u * (1.0 - ratio)) ** (-1.0 / kr)


def sample_vmf(mean, kappa, n, rng):
    mean = mean / np.linalg.norm(mean)
    u = rng.random(n)
    w = 1.0 + np.log(u + (1.0 - u) * np.exp(-2.0 * kappa)) / kappa
    phi = 2.0 * np.pi * rng.random(n)
    s = np.sqrt(np.clip(1.0 - w * w, 0.0, 1.0))
    x = np.column_stack([s * np.cos(phi), s * np.sin(phi), w])
    a = np.array([0.0, 0.0, 1.0])
    axis = np.cross(a, mean); an = np.linalg.norm(axis)
    if an < 1e-9:
        return x if mean[2] > 0 else -x
    axis /= an
    ang = np.arccos(np.clip(mean[2], -1, 1))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    return x @ R.T


def _tunnel_prism(ax, poly, x0, x1, color="0.5"):
    for xx in (x0, x1):
        ax.plot(poly[:, 0] * 0 + xx, poly[:, 0], poly[:, 1], color=color, lw=1.0)
    for k in range(0, len(poly), 3):
        ax.plot([x0, x1], [poly[k, 0]] * 2, [poly[k, 1]] * 2, color=color, lw=0.5, alpha=0.5)


# --------------------------- 블록 ③ 생성 ----------------------------------
def block3(base, tr, out):
    comp = {int(r["set_id"]): r for r in read_csv_rows(os.path.join(base, "comparison_vs_laxemar.csv"))}
    krs = {int(r["set_id"]): r for r in read_csv_rows(os.path.join(base, "kr", "kr_summary_by_set.csv"))}
    poly = tr["tunnel_poly_yz"]
    xpos = tr["face_x_positions"]
    x0, x1 = float(xpos.min()) - 0.5, float(xpos.max()) + 0.5

    fig = plt.figure(figsize=(14, 6.5))

    # (a) 관측 균열 정보(실측 trace, 4개 막장면)
    axa = fig.add_subplot(1, 2, 1, projection="3d")
    _tunnel_prism(axa, poly, x0, x1)
    seen = set()
    for i in range(len(tr["set_id"])):
        s = int(tr["set_id"][i])
        v = tr["pl_xyz"][tr["pl_start"][i]:tr["pl_start"][i] + tr["pl_count"][i]]
        lab = f"Set {s}" if s not in seen else None
        seen.add(s)
        axa.plot(v[:, 0], v[:, 1], v[:, 2], color=SET_COLORS.get(s, "0.3"), lw=0.8, label=lab)
    axa.set_title("(a) 관측 균열 복원 입력\n4개 막장면 trace (실측)")
    axa.set_xlabel("x (m, 굴진)"); axa.set_ylabel("y (m)"); axa.set_zlabel("z (m)")
    axa.legend(fontsize=8, loc="upper left")
    axa.view_init(elev=18, azim=-60)

    # (b) 미관측 영역 확률적 생성 (역산 파라미터 기반 개념 예시)
    axb = fig.add_subplot(1, 2, 2, projection="3d")
    _tunnel_prism(axb, poly, x0, x1)
    ymin, ymax = poly[:, 0].min() - 1, poly[:, 0].max() + 1
    zmin, zmax = poly[:, 1].min() - 1, poly[:, 1].max() + 1
    rng = np.random.default_rng(7)
    per_set = 22
    for s in SETS:
        kr = float(krs[s]["kr_hat"]); rmin = float(krs[s]["set_likelihood_rmin"])
        mean = pole_vec_enu(float(comp[s]["est_trend"]), float(comp[s]["est_plunge"]))
        kappa = float(comp[s]["est_kappa"])
        radii = sample_powerlaw_radius(kr, rmin, 3.0, per_set, rng)
        normals = sample_vmf(mean, kappa, per_set, rng)
        cx = rng.uniform(x0, x1, per_set)
        cy = rng.uniform(ymin, ymax, per_set)
        cz = rng.uniform(zmin, zmax, per_set)
        for k in range(per_set):
            poly3 = disc_polygon(np.array([cx[k], cy[k], cz[k]]), normals[k], radii[k])
            axb.plot_trisurf(poly3[:, 0], poly3[:, 1], poly3[:, 2],
                             color=SET_COLORS[s], alpha=0.35, linewidth=0)
    axb.set_title("(b) 미관측 영역 확률적 생성\n(역산 파라미터 기반 예시 · 개념도)")
    axb.set_xlabel("x (m, 굴진)"); axb.set_ylabel("y (m)"); axb.set_zlabel("z (m)")
    axb.view_init(elev=18, azim=-60)

    fig.suptitle("③ 절리망 생성 (Fracture Network Generation)  —  (b)는 설계단계 개념 예시",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(out, "fig_block3_generation.png")
    fig.savefig(p, dpi=180); plt.close(fig)
    return p


# --------------------------- 블록 ④ 안정성 연계 ----------------------------
def block4(out):
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    fc=fc, ec="#333", lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=18, lw=1.6, color="#333"))

    box(0.4, 4.4, 3.0, 1.6, "관측 균열 복원\n(관측 trace → disc,\n관측 형상 보존)", "#cfe3f7")
    box(0.4, 1.0, 3.0, 1.6, "미관측 균열 생성\n(역산 파라미터 기반\n확률적 DFN)", "#d9ead3")
    box(4.6, 2.7, 3.0, 1.6, "통합 disk-fracture\ndataset\n(center · radius · normal)", "#fff2cc")
    box(8.8, 2.7, 3.0, 1.6, "안정성 해석 모듈\n(블록 판별 · 안정성)", "#f4cccc")

    arrow(3.4, 5.2, 4.6, 3.9)
    arrow(3.4, 1.8, 4.6, 3.3)
    arrow(7.6, 3.5, 8.8, 3.5)

    ax.text(6.0, 6.6, "④ 안정성 해석 연계 (Link to Stability Analysis)",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(6.1, 1.6, "관측(조건부) + 생성 균열을 단일 원판 데이터셋으로 통합하여 입력",
            ha="center", fontsize=9, color="0.3")
    fig.tight_layout()
    p = os.path.join(out, "fig_block4_stability_link.png")
    fig.savefig(p, dpi=180); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="storage/output/pipeline_test_laxemar")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tr = load_traces(args.base)
    for p in (block1(tr, args.out),
              block2(args.base, tr, args.out),
              block3(args.base, tr, args.out),
              block4(args.out)):
        print("[*] written:", p)


if __name__ == "__main__":
    main()
