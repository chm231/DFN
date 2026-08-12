"""제6장 보충 원고용 그림 3장 — 프로파일 우도 / 격자 이산화 / Kaplan-Meier 진단.

보고서 제3절·제6절의 압축된 서술을 되살리면서 함께 실을 그림들이다.
모두 v2 파이프라인(`pipeline_v2_laxemar`) 출력에 근거한다.

실행:
    python scripts/make_fig_supplement.py --base storage/output/pipeline_v2_laxemar
출력:
    docs/figures/fig_kr_profile_likelihood_v2.png
    docs/figures/fig_grid_discretization_v2.png
    docs/figures/fig_km_diagnostic_v2.png
"""
import argparse
import csv
import os

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

SETS = [1, 2, 3, 5]
SET_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 5: "#9467bd"}

# H-4 ① 실측: 다각형 정확면적 73.9266 m² 대비 mesh 면적 오차
GRID_STEPS = [0.2, 0.1, 0.05]
FLAT_AREAS = [71.0800, 72.4700, 73.2325]
POLY_AREA = 73.9266
ROUGH_02_AREA = 71.4156          # 요철 mesh, grid_step 0.2 m


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# --------------------------- 프로파일 우도 -----------------------------------
def fig_profile_likelihood(base, out):
    krs = {int(r["set_id"]): r for r in read_csv_rows(os.path.join(base, "kr", "kr_summary_by_set.csv"))}
    prof = read_csv_rows(os.path.join(base, "kr", "kr_profile_likelihood.csv"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    for ax, s in zip(axes.ravel(), SETS):
        best = krs[s]["best_lmin_fit"]
        rows = [r for r in prof if int(r["set_id"]) == s and r["lmin_fit"] == best]
        rows.sort(key=lambda r: float(r["kr_window_mc"]))
        kr = np.array([float(r["kr_window_mc"]) for r in rows])
        dl = np.array([float(r["delta_loglik"]) for r in rows])   # 0 = 최적, 클수록 나쁨

        ax.plot(kr, dl, "-", color=SET_COLORS[s], lw=1.8)
        ax.axhline(2.0, color="0.4", ls="--", lw=1.1)
        ax.text(kr[-1], 2.0, " $\\Delta\\log L=2$", va="center", fontsize=8, color="0.35")

        inside = kr[dl <= 2.0]
        if len(inside):
            ax.axvspan(inside.min(), inside.max(), color=SET_COLORS[s], alpha=0.12)
            ax.text(0.03, 0.94, "$\\Delta\\log L\\leq2$ 구간  [%.2f, %.2f]" % (inside.min(), inside.max()),
                    transform=ax.transAxes, fontsize=8.5, va="top")

        khat, ktrue = float(krs[s]["kr_hat"]), float(krs[s]["kr_true"])
        ax.axvline(khat, color=SET_COLORS[s], ls="-", lw=1.2)
        ax.axvline(ktrue, color="k", ls=":", lw=1.4)
        ax.plot([], [], color=SET_COLORS[s], lw=1.2, label="역산 $\\hat{k_r}$ = %.2f" % khat)
        ax.plot([], [], color="k", ls=":", lw=1.4, label="참값 = %.2f" % ktrue)

        ax.set_xlim(kr.min(), kr.max())
        ax.set_ylim(-0.4, 12)
        ax.set_xlabel("반지름 멱법칙 지수 $k_r$")
        ax.set_ylabel("$\\Delta\\log L$  (0 = 최대우도)")
        ax.set_title("Set %d  (절리선 %s개)" % (s, krs[s]["n_used"]), fontsize=11)
        ax.legend(fontsize=8.5, loc="upper right")

    fig.suptitle("$k_r$ 프로파일 우도 — 하이브리드 우도, $\\ell_{min,fit}$ = 0.5 m",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(out, "fig_kr_profile_likelihood_v2.png")
    fig.savefig(p, dpi=180); plt.close(fig)
    return p


# --------------------------- 격자 이산화 -------------------------------------
def fig_grid_discretization(out):
    h = np.array(GRID_STEPS)
    err = 100.0 * (np.array(FLAT_AREAS) - POLY_AREA) / POLY_AREA
    rough_err = 100.0 * (ROUGH_02_AREA - POLY_AREA) / POLY_AREA

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.plot(h, err, "o-", color="#1f4e79", lw=1.8, ms=8, label="평면 mesh (격자 이산화)")
    ref = err[0] * (h / h[0])
    ax.plot(h, ref, "--", color="0.55", lw=1.3, label="$O(h)$ 기준선")
    ax.plot([0.2], [rough_err], "s", color="#b06a1f", ms=9, label="요철 mesh (격자 0.2 m)")
    ax.axhline(0.0, color="#3a7d44", lw=1.8)
    ax.text(0.005, 0.25, "터널 단면 다각형 (정확, 73.93 m²) — v2가 채택한 기준",
            fontsize=9, color="#3a7d44")

    for x, y, off in zip(h, err, [(-46, -16), (8, -14), (8, -14)]):
        ax.annotate("%.2f%%" % y, (x, y), textcoords="offset points", xytext=off, fontsize=9)
    ax.annotate("%.2f%%  (요철 기여 +0.47%%p)" % rough_err, (0.2, rough_err),
                textcoords="offset points", xytext=(-150, 10), fontsize=9, color="#b06a1f")

    ax.set_xlim(0, 0.23); ax.set_ylim(-4.8, 0.9)
    ax.set_xlabel("mesh 격자 간격 $h$ (m)")
    ax.set_ylabel("1면 관측면적 오차 (%, 다각형 대비)")
    ax.set_title("관측면적 오차의 원인 — 요철이 아니라 격자 이산화", fontsize=12.5, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    p = os.path.join(out, "fig_grid_discretization_v2.png")
    fig.savefig(p, dpi=180); plt.close(fig)
    return p


# --------------------------- Kaplan-Meier ------------------------------------
def km_curve(lengths, censored):
    """우편향 자료의 Kaplan-Meier 생존함수. censored=True 는 절단(Type 1·2)."""
    order = np.argsort(lengths)
    t, c = lengths[order], censored[order]
    times, surv, s = [0.0], [1.0], 1.0
    for x in np.unique(t[~c]):
        n_risk = int(np.sum(t >= x))
        d = int(np.sum((t == x) & (~c)))
        if n_risk > 0:
            s *= 1.0 - d / n_risk
        times.append(float(x)); surv.append(s)
    return np.array(times), np.array(surv)


def fig_km_diagnostic(base, out):
    path = os.path.join(base, "trace_dataset", "trace_dataset_3d.h5")
    with h5py.File(path, "r") as f:
        g = f["traces"]
        set_id = g["set_id"][:].astype(int)
        length = g["observed_length_m"][:].astype(float)
        cls = g["censoring_class"][:].astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    for ax, s in zip(axes.ravel(), SETS):
        m = set_id == s
        L, C = length[m], cls[m] > 0
        t_km, s_km = km_curve(L, C)
        ax.step(t_km, s_km, where="post", color=SET_COLORS[s], lw=1.9,
                label="Kaplan–Meier (절단 반영)")

        # 절단을 무시한 경험적 생존함수 — 과소평가되는 꼬리를 대비로 보여준다
        xs = np.sort(L)
        ax.step(xs, 1.0 - np.arange(len(xs)) / len(xs), where="post",
                color="0.45", ls="--", lw=1.4, label="절단 무시 (경험적)")

        ax.set_xscale("log")
        lo, hi = max(L.min(), 0.1), L.max() * 1.15
        ax.set_xlim(lo, hi)
        # 로그축 기본 눈금은 mathtext 음지수(U+2212)를 쓰는데 한글 글꼴에 해당 글리프가 없다
        ticks = [t for t in (0.2, 0.5, 1, 2, 5, 10) if lo <= t <= hi]
        ax.set_xticks(ticks, minor=False)
        ax.set_xticklabels(["%g" % t for t in ticks])
        ax.set_xticks([], minor=True)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("관측 절리선 길이 $\\ell$ (m)")
        ax.set_ylabel("$\\hat{S}(\\ell)$")
        n_cen = int(np.sum(C))
        ax.set_title("Set %d  (절리선 %d개, 절단 %d개 = %.0f%%)"
                     % (s, len(L), n_cen, 100 * n_cen / len(L)), fontsize=11)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8.5, loc="lower left")

    fig.suptitle("Kaplan–Meier 생존곡선 — 경계 절단의 영향 진단 (보조 진단, $k_r$ 추정에는 미사용)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(out, "fig_km_diagnostic_v2.png")
    fig.savefig(p, dpi=180); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="storage/output/pipeline_v2_laxemar")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for p in (fig_profile_likelihood(args.base, args.out),
              fig_grid_discretization(args.out),
              fig_km_diagnostic(args.base, args.out)):
        print("[*] written:", p)


if __name__ == "__main__":
    main()
