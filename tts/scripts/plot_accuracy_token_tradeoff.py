#!/usr/bin/env python3
"""
Accuracy vs Tokens: full sweep (lines) or N=32-only scatter (default).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# Full sweep: (tokens_k, acc_percent), sorted by tokens within each series.
SERIES_FULL = {
    "CoT (greedy)": [(2.2, 36.7)],
    "e2c_self_consistency": [(11.1, 40.0), (21.8, 40.0), (43.7, 43.3), (86.2, 50.0)],
    "e2c_select_lm_judge": [(3.4, 43.3), (4.6, 26.7), (6.5, 40.0), (9.7, 40.0)],
    "e2c_ToT": [(9.9, 46.7), (18.3, 46.7), (32.0, 43.3), (54.8, 53.3)],
    "e2c_react_loop": [(5.7, 43.3), (7.0, 43.3), (7.8, 46.7), (12.4, 53.3)],
}

# N=32 budget: last point of each multi-budget method.
# CoT greedy: table only has N=1 — include as baseline for comparison (labeled in legend).
SERIES_N32 = {
    "CoT (greedy, N=1)": (2.2, 36.7),
    "e2c_self_consistency": (86.2, 50.0),
    "e2c_select_lm_judge": (9.7, 40.0),
    "e2c_ToT": (54.8, 53.3),
    "e2c_react_loop": (12.4, 53.3),
}

MARKER_N32 = {
    "CoT (greedy, N=1)": "D",
    "e2c_self_consistency": "o",
    "e2c_select_lm_judge": "s",
    "e2c_ToT": "^",
    "e2c_react_loop": "v",
}


def _style_linear_tokens_x(ax, xmax: float = 100.0) -> None:
    """Linear x in tokens (k) with uniform tick spacing (standard paper-style readout)."""
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))


def plot_n32_scatter(out_path: Path, dpi: int = 200) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = plt.cm.tab10(np.linspace(0, 0.75, len(SERIES_N32)))

    for (name, (tx, acc)), color in zip(SERIES_N32.items(), colors):
        ax.scatter(
            [tx],
            [acc],
            color=color,
            marker=MARKER_N32.get(name, "o"),
            s=120,
            zorder=4,
            edgecolors="white",
            linewidths=1.0,
            label=name,
        )

    ax.set_xlabel("Tokens (k)", fontsize=11)
    ax.set_ylabel("Acc. (%)", fontsize=11)
    ax.set_title(
        "Accuracy–Token Trade-off (N=32; CoT baseline = N=1)",
        fontsize=12,
        fontweight="semibold",
    )
    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    ax.set_ylim(32, 56)
    _style_linear_tokens_x(ax, xmax=100.0)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.92)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_full_tradeoff(out_path: Path, dpi: int = 200) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = plt.cm.tab10(np.linspace(0, 0.85, len(SERIES_FULL)))

    for (name, pts), color in zip(SERIES_FULL.items(), colors):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if len(pts) == 1:
            ax.scatter(xs, ys, color=color, label=name, marker="D", s=81, zorder=5)
        else:
            ax.plot(xs, ys, color=color, label=name, marker="o", linewidth=2)
            ax.scatter(
                xs,
                ys,
                color=color,
                s=36,
                zorder=4,
                edgecolors="white",
                linewidths=0.8,
            )

    ax.set_xlabel("Tokens (k)", fontsize=11)
    ax.set_ylabel("Acc. (%)", fontsize=11)
    ax.set_title("Accuracy–Token Trade-off", fontsize=12, fontweight="semibold")
    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    ax.set_ylim(22, 58)
    _style_linear_tokens_x(ax, xmax=100.0)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.92)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: N32 -> outputs/accuracy_token_tradeoff.png)",
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument(
        "--full",
        action="store_true",
        help="Plot all N budgets as lines instead of N=32 scatter only.",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1] / "outputs"
    if args.full:
        out = args.out or (root / "accuracy_token_tradeoff_full.png")
        plot_full_tradeoff(out, dpi=args.dpi)
    else:
        out = args.out or (root / "accuracy_token_tradeoff.png")
        plot_n32_scatter(out, dpi=args.dpi)
    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
