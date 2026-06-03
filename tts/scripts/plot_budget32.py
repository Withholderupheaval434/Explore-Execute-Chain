#!/usr/bin/env python3
"""
Figure: Accuracy-Token Trade-off (K/N = 4 → 16 → 32).
Each method shows three waypoints connected by sequential arrows.
Single-pass references (Greedy CoT, ReAct) are lone markers.
E2C-ReAct Loop at K=32 highlighted with a gold star.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator

# ── Table 3: (tokens_k, acc_percent) at K/N = 4, 16, 32 ─────────────────
# Full-chain baselines (dashed arrows)
FC_SERIES = {
    "Self-Consistency": [(11.1, 40.0), (43.7, 43.3), (86.2, 50.0)],
    "Tree-of-Thoughts": [(13.6, 43.3), (43.4, 46.7), (71.3, 50.0)],
}
# E2C methods (solid arrows)
E2C_SERIES = {
    r"E$^2$C-Select":    [(3.4, 43.3), (6.5, 40.0),  (9.7,  40.0)],
    r"E$^2$C-ToT":       [(9.9, 46.7), (32.0, 43.3), (54.8, 53.3)],
    r"E$^2$C-ReAct Loop":[(7.0, 43.3), (7.8,  46.7), (12.4, 53.3)],  # star at K=32
}
# Single-pass references (no arrow)
FC_REF = {
    "Greedy CoT ($N{=}1$)": (2.2,  36.7),
    "ReAct ($N{=}1$)":       (7.6,  43.3),
}

FC_COLORS     = ["#3a6fc4", "#44a860"]
E2C_COLORS    = ["#8b5fcf", "#1a9e8f", "#f0b800"]   # gold = ReAct Loop
FC_REF_COLORS = ["#9e9e9e", "#d4622a"]

FC_MARKERS    = ["o", "^"]
E2C_MARKERS   = ["P", "v", "o"]
FC_REF_MARKERS= ["D", "s"]

BUDGET_LABELS_FC  = ["$N{=}4$",  "$N{=}16$",  "$N{=}32$"]
BUDGET_LABELS_E2C = ["$K{=}4$",  "$K{=}16$",  "$K{=}32$"]
# vertical offsets for budget labels (per point index): above or below marker
LABEL_VA = ["top", "top", "bottom"]   # 0=below, 2=above
LABEL_DY = [-1.1, -1.1, +1.0]


def _draw_arrow(ax, x0, y0, x1, y1, color, linestyle, lw=1.8):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            linestyle=linestyle,
            mutation_scale=12,
        ),
        zorder=3,
    )


def _plot_series(ax, pts, color, mk, linestyle, lw,
                 budget_labels, label_dy, is_star_last=False, legend_label=None):
    """Plot 3 waypoints with two sequential arrows."""
    for i in range(len(pts) - 1):
        _draw_arrow(ax, pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1],
                    color, linestyle, lw)
    for i, (x, y) in enumerate(pts):
        is_last = (i == len(pts) - 1)
        kw = dict(label=legend_label) if (is_last and legend_label) else {}
        if is_last and is_star_last:
            ax.scatter(x, y, color=color, marker="*",
                       s=580, zorder=6,
                       edgecolors="#9a7100", linewidths=1.4, **kw)
        else:
            ax.scatter(x, y, color=color, marker=mk,
                       s=90, zorder=5, edgecolors="white", linewidths=0.9, **kw)
        # small budget label
        ax.text(x, y + label_dy[i], budget_labels[i], color=color,
                fontsize=6.5, ha="center",
                va="bottom" if label_dy[i] > 0 else "top", zorder=7)


def make_figure(out_path: Path, dpi: int = 260) -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f9f9f9")

    # ── Full-chain baselines ─────────────────────────────────────────────
    for (name, pts), color, mk in zip(FC_SERIES.items(), FC_COLORS, FC_MARKERS):
        _plot_series(ax, pts, color, mk, "dashed", 1.6,
                     BUDGET_LABELS_FC, LABEL_DY, legend_label=name)

    # ── E2C methods ──────────────────────────────────────────────────────
    for (name, pts), color, mk in zip(E2C_SERIES.items(), E2C_COLORS, E2C_MARKERS):
        is_react = name.endswith("ReAct Loop")
        _plot_series(ax, pts, color, mk, "solid", 2.1,
                     BUDGET_LABELS_E2C, LABEL_DY,
                     is_star_last=is_react, legend_label=name)

    # ── Single-pass references ────────────────────────────────────────────
    for (name, (tx, acc)), color, mk in zip(
            FC_REF.items(), FC_REF_COLORS, FC_REF_MARKERS):
        ax.scatter(tx, acc, color=color, marker=mk,
                   s=110, zorder=4, edgecolors="white", linewidths=0.9,
                   label=name)

    # ── Axes ─────────────────────────────────────────────────────────────
    ax.set_xlabel("Tokens (k)", fontsize=11.5, labelpad=6)
    ax.set_ylabel("Accuracy (%)", fontsize=11.5, labelpad=6)
    ax.set_title(
        r"Accuracy$-$Token Trade-off  (AIME'2024,  $K$/$N$ = 4 $\rightarrow$ 16 $\rightarrow$ 32)",
        fontsize=11.5, fontweight="bold", pad=10,
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(33, 57)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(True, which="major", linestyle="--", linewidth=0.6,
            color="#cccccc", alpha=0.8, zorder=1)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4,
            color="#dddddd", alpha=0.6, zorder=1)
    ax.tick_params(labelsize=10, length=4)

    # ── Legend ───────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    sep = Line2D([], [], color="none")
    n_fc  = len(FC_SERIES)
    n_e2c = len(E2C_SERIES)
    all_h = handles[:n_fc] + [sep] + handles[n_fc:n_fc+n_e2c] + [sep] + handles[n_fc+n_e2c:]
    all_l = labels[:n_fc]  + [""]  + labels[n_fc:n_fc+n_e2c]  + [""]  + labels[n_fc+n_e2c:]
    ax.legend(all_h, all_l,
              loc="lower right", fontsize=8.4,
              framealpha=0.95, edgecolor="#bbbbbb",
              handletextpad=0.6, labelspacing=0.42, borderpad=0.7)

    fig.tight_layout(pad=1.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path.resolve()}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "outputs" / "tradeoff_budget32.png"
    make_figure(out)
