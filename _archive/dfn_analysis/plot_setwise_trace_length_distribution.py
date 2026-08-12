import argparse
import csv
import math
import os
from typing import Dict, List, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


'''
python dfn_analysis/plot_setwise_trace_length_distribution.py --trace-h5 storage/output/trace_dataset_collection/trace_dataset_3d.h5 --out-png storage/output/trace_dataset_collection/setwise_trace_length_distribution.png
'''
def load_rows_from_h5(h5_path: str) -> List[dict]:
    with h5py.File(h5_path, "r") as f:
        if "traces" not in f:
            raise ValueError(f"Could not find /traces in: {h5_path}")
        grp = f["traces"]
        set_ids = grp["set_id"][:].astype(np.int32)
        lengths = grp["observed_length_m"][:].astype(np.float64)
    return [{"set_id": int(set_id), "observed_length_m": float(length)} for set_id, length in zip(set_ids, lengths)]


def load_rows_from_csv(csv_path: str) -> List[dict]:
    rows: List[dict] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "set_id": int(row["set_id"]),
                    "observed_length_m": float(row["observed_length_m"]),
                }
            )
    return rows


def group_lengths_by_set(rows: Sequence[dict]) -> Dict[int, np.ndarray]:
    grouped: Dict[int, List[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["set_id"]), []).append(float(row["observed_length_m"]))
    return {set_id: np.asarray(lengths, dtype=np.float64) for set_id, lengths in sorted(grouped.items())}


def _hist_bin_count(lengths: np.ndarray) -> int:
    return max(16, min(60, int(math.sqrt(len(lengths)) * 2) if len(lengths) > 0 else 16))


def plot_setwise_length_distribution(grouped_lengths: Dict[int, np.ndarray], out_png: str) -> None:
    set_ids = list(grouped_lengths.keys())
    if not set_ids:
        raise ValueError("No trace rows found to plot.")

    n_sets = len(set_ids)
    fig, axes = plt.subplots(
        n_sets,
        1,
        figsize=(12, max(3.2 * n_sets, 4.5)),
        sharex=True,
    )
    if n_sets == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10")
    global_max_length = max(float(np.max(lengths)) for lengths in grouped_lengths.values())
    x_ticks = np.linspace(0.0, global_max_length, 9)

    for idx, (ax_hist, set_id) in enumerate(zip(axes, set_ids)):
        lengths = grouped_lengths[set_id]
        color = cmap(idx % 10)
        ax_hist.hist(
            lengths,
            bins=_hist_bin_count(lengths),
            density=False,
            alpha=0.55,
            color=color,
            edgecolor="black",
            linewidth=0.8,
        )
        ax_hist.axvline(float(np.median(lengths)), color=color, linestyle="--", linewidth=1.3)
        ax_hist.set_title(
            f"Set {set_id} Trace Length Distribution (n={len(lengths)}, median={np.median(lengths):.3f} m)"
        )
        ax_hist.set_ylabel("Count")
        ax_hist.set_xlim(0.0, global_max_length)
        ax_hist.set_xticks(x_ticks)
        ax_hist.xaxis.set_major_locator(MaxNLocator(nbins=9))
        ax_hist.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Observed trace length (m)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def print_setwise_summary(grouped_lengths: Dict[int, np.ndarray]) -> None:
    print("[*] Set-wise trace length summary")
    for set_id, lengths in grouped_lengths.items():
        print(
            f"    - Set {set_id}: n={len(lengths):,}, "
            f"mean={np.mean(lengths):.4f} m, median={np.median(lengths):.4f} m, "
            f"p10={np.percentile(lengths, 10):.4f} m, p90={np.percentile(lengths, 90):.4f} m"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot set-wise observed 3D trace length distributions.")
    parser.add_argument("--trace-h5", help="Input trace HDF5 created by export_setwise_3d_traces.py")
    parser.add_argument("--trace-csv", help="Input trace CSV created by export_setwise_3d_traces.py")
    parser.add_argument(
        "--out-png",
        default="storage/output/trace_dataset_collection/setwise_trace_length_distribution.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    if bool(args.trace_h5) == bool(args.trace_csv):
        raise ValueError("Provide exactly one of --trace-h5 or --trace-csv.")

    rows = load_rows_from_h5(args.trace_h5) if args.trace_h5 else load_rows_from_csv(args.trace_csv)
    grouped_lengths = group_lengths_by_set(rows)
    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    plot_setwise_length_distribution(grouped_lengths, args.out_png)
    print_setwise_summary(grouped_lengths)
    print(f"[*] PNG written to: {args.out_png}")


if __name__ == "__main__":
    main()
