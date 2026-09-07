"""
karma_level5_plots.py
─────────────────────
Publication-quality Level 5 uncertainty plots for KARMA using seaborn.

Produces a single figure with four panels:

    Panel A (top-left)    Aleatoric entropy H^d_h per history,
                          coloured by coverage reliability tier.

    Panel B (top-right)   Epistemic std sqrt(Var) per bin per history,
                          stacked horizontal bars.

    Panel C (bottom-left) Noise floor rho_floor(h) with lambda/2 and
                          lambda/4 threshold lines.

    Panel D (bottom-right) Coverage map: aleatoric vs total epistemic
                           scatter, bubble size = pi*(h),
                           coloured by reliability tier.

Colour palette
--------------
    Green  #1D9E75  reliable    floor < lambda/4
    Amber  #EF9F27  marginal    lambda/4 <= floor < lambda/2
    Red    #E24B4A  unreliable  floor >= lambda/2

Dependencies: numpy, matplotlib, seaborn
"""

from __future__ import annotations
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Data classes  (mirror karma_level5_uncertainty.py)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class HistoryUncertainty:
    h_idx: int
    d: int
    kernel: np.ndarray
    n_queries: int
    aleatoric: float
    epistemic: np.ndarray
    noise_floor: float
    coverage: float
    reliable: bool
    pi_star: float


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty computations  (self-contained, no import from other module)
# ─────────────────────────────────────────────────────────────────────────────


def _aleatoric(kernel: np.ndarray) -> float:
    k = np.clip(kernel, 1e-12, 1.0)
    return float(-np.sum(k * np.log(k)))


def _epistemic(kernel: np.ndarray, n: int) -> np.ndarray:
    if n == 0:
        return np.full_like(kernel, np.inf)
    return kernel * (1.0 - kernel) / n


def _noise_floor_b(pool_size: int, delta_pred: float, M: int) -> float:
    mc = np.sqrt(1.0 / (2.0 * M)) if M > 0 else np.inf
    pool = 1.0 / np.sqrt(pool_size) if pool_size > 0 else np.inf
    return float(mc + delta_pred + pool)


def _coverage_bar(nf: float, lam: float) -> float:
    if np.isinf(nf):
        return 0.0
    return float(max(0.0, 1.0 - nf / (lam / 2.0)))


def build_results(
    kernels: dict,
    counts: dict,
    pool_sizes: dict,
    pi_star: dict,
    lam: float,
    delta_pred: float,
    M: int,
    d: int = 0,
) -> list[HistoryUncertainty]:
    results = []
    for h_idx, pi_h in pi_star.items():
        ker = kernels.get(h_idx, np.full(3, 1 / 3))
        n = counts.get(h_idx, 0)
        ps = pool_sizes.get(h_idx, n)
        nf = _noise_floor_b(ps, delta_pred, M)
        cov = _coverage_bar(nf, lam)
        results.append(
            HistoryUncertainty(
                h_idx=h_idx,
                d=d,
                kernel=ker,
                n_queries=n,
                aleatoric=_aleatoric(ker),
                epistemic=_epistemic(ker, n),
                noise_floor=nf,
                coverage=cov,
                reliable=nf < lam / 2.0,
                pi_star=pi_h,
            )
        )
    results.sort(key=lambda r: r.noise_floor, reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

_GREEN = "#1D9E75"
_AMBER = "#EF9F27"
_RED = "#E24B4A"

_GREEN_LIGHT = "#9FE1CB"
_AMBER_LIGHT = "#FAC775"
_RED_LIGHT = "#F7C1C1"

_BLUE_RAMP = ["#185FA5", "#378ADD", "#85B7EB", "#B5D4F4", "#E6F1FB"]
_GRAY_MID = "#888780"
_GRAY_LIGHT = "#D3D1C7"


def _tier_colour(r: HistoryUncertainty, lam: float) -> str:
    if r.noise_floor >= lam / 2.0:
        return _RED
    elif r.noise_floor >= lam / 4.0:
        return _AMBER
    return _GREEN


def _tier_colour_light(r: HistoryUncertainty, lam: float) -> str:
    if r.noise_floor >= lam / 2.0:
        return _RED_LIGHT
    elif r.noise_floor >= lam / 4.0:
        return _AMBER_LIGHT
    return _GREEN_LIGHT


# ─────────────────────────────────────────────────────────────────────────────
# Global seaborn / matplotlib theme
# ─────────────────────────────────────────────────────────────────────────────


def _set_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="sans-serif",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 0.6,
            "axes.edgecolor": _GRAY_MID,
            "grid.color": _GRAY_LIGHT,
            "grid.linewidth": 0.4,
            "grid.linestyle": "--",
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.color": _GRAY_MID,
            "ytick.color": _GRAY_MID,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "normal",
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "legend.framealpha": 0.7,
            "legend.edgecolor": _GRAY_LIGHT,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Panel A — Aleatoric entropy
# ─────────────────────────────────────────────────────────────────────────────


def _panel_aleatoric(
    ax: plt.Axes,
    results: list[HistoryUncertainty],
    labels: list[str],
    lam: float,
    N: int,
) -> None:
    n = len(results)
    y = np.arange(n)
    values = [r.aleatoric for r in results]
    colours = [_tier_colour(r, lam) for r in results]
    ec = [_tier_colour_light(r, lam) for r in results]

    bars = ax.barh(
        y,
        values,
        color=colours,
        edgecolor=ec,
        linewidth=0.6,
        height=0.55,
        zorder=3,
    )

    # Annotate values on bars
    for bar, val, r in zip(bars, values, results):
        x_pos = bar.get_width() + 0.01
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color=_GRAY_MID,
        )
        # n_queries annotation inside bar if room
        if bar.get_width() > 0.15:
            ax.text(
                bar.get_width() - 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"n={r.n_queries}",
                va="center",
                ha="right",
                fontsize=6.5,
                color="white",
            )

    max_H = np.log(N)
    ax.axvline(
        max_H,
        color=_GRAY_MID,
        linewidth=0.8,
        linestyle="--",
        zorder=2,
        label=f"max H = ln({N}) = {max_H:.2f}",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Aleatoric entropy  Hᵈₕ  (nats)")
    ax.set_title("A  ·  Aleatoric uncertainty per history", loc="left", pad=6)
    ax.set_xlim(0, max_H * 1.25)
    ax.legend(loc="lower right", handlelength=1.5)
    ax.invert_yaxis()


# ─────────────────────────────────────────────────────────────────────────────
# Panel B — Epistemic std per bin
# ─────────────────────────────────────────────────────────────────────────────


def _panel_epistemic(
    ax: plt.Axes,
    results: list[HistoryUncertainty],
    labels: list[str],
    lam: float,
) -> None:
    n = len(results)
    y = np.arange(n)
    N = results[0].kernel.shape[0]

    lefts = np.zeros(n)
    for s in range(N):
        values = np.array(
            [
                np.sqrt(r.epistemic[s]) if not np.isinf(r.epistemic[s]) else 0.0
                for r in results
            ]
        )
        ax.barh(
            y,
            values,
            left=lefts,
            color=_BLUE_RAMP[s % len(_BLUE_RAMP)],
            edgecolor="white",
            linewidth=0.4,
            height=0.55,
            label=f"s′ = {s}",
            zorder=3,
        )
        lefts += values

    # Total epistemic std annotation
    for i, r in enumerate(results):
        total = (
            float(np.sum(np.sqrt(r.epistemic[r.epistemic < np.inf])))
            if r.n_queries > 0
            else 0.0
        )
        ax.text(
            lefts[i] + 0.001,
            i,
            f"{total:.3f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color=_GRAY_MID,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Epistemic std  √Var(T̂(s′|h))")
    ax.set_title("B  ·  Epistemic uncertainty by bin", loc="left", pad=6)
    ax.legend(loc="lower right", title="next bin", title_fontsize=7)
    ax.invert_yaxis()


# ─────────────────────────────────────────────────────────────────────────────
# Panel C — Noise floor with thresholds
# ─────────────────────────────────────────────────────────────────────────────


def _panel_noise_floor(
    ax: plt.Axes,
    results: list[HistoryUncertainty],
    labels: list[str],
    lam: float,
    delta_pred: float,
    M: int,
) -> None:
    n = len(results)
    y = np.arange(n)
    values = [r.noise_floor if not np.isinf(r.noise_floor) else lam for r in results]
    colours = [_tier_colour(r, lam) for r in results]
    ec = [_tier_colour_light(r, lam) for r in results]

    bars = ax.barh(
        y,
        values,
        color=colours,
        edgecolor=ec,
        linewidth=0.6,
        height=0.55,
        zorder=3,
    )

    # Threshold lines
    ax.axvline(
        lam / 2.0,
        color=_RED,
        linewidth=1.0,
        linestyle="--",
        zorder=4,
        label=f"λ/2 = {lam/2:.3f}  (coverage threshold)",
    )
    ax.axvline(
        lam / 4.0,
        color=_AMBER,
        linewidth=0.8,
        linestyle=":",
        zorder=4,
        label=f"λ/4 = {lam/4:.3f}",
    )

    # Irreducible floor marker
    irred = np.sqrt(1.0 / (2.0 * M)) + delta_pred if M > 0 else delta_pred
    ax.axvline(
        irred,
        color=_BLUE_RAMP[2],
        linewidth=0.7,
        linestyle=(0, (4, 3)),
        zorder=4,
        label=f"irreducible floor = {irred:.3f}",
    )

    # Annotate values
    for bar, val, r in zip(bars, values, results):
        flag = "  ✗" if not r.reliable else ""
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}{flag}",
            va="center",
            ha="left",
            fontsize=7.5,
            color=_RED if not r.reliable else _GRAY_MID,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Noise floor  ρ_floor(h)")
    ax.set_title("C  ·  Estimation noise floor", loc="left", pad=6)
    ax.legend(loc="lower right", handlelength=2)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.35)


# ─────────────────────────────────────────────────────────────────────────────
# Panel D — Coverage map scatter
# ─────────────────────────────────────────────────────────────────────────────


def _panel_coverage_map(
    ax: plt.Axes,
    results: list[HistoryUncertainty],
    labels: list[str],
    lam: float,
    N: int,
) -> None:
    aleat = np.array([r.aleatoric for r in results])
    epist = np.array(
        [
            (
                float(np.sum(np.sqrt(r.epistemic[r.epistemic < np.inf])))
                if r.n_queries > 0
                else 0.0
            )
            for r in results
        ]
    )
    sizes = np.array([max(60, r.pi_star * 2800) for r in results])
    colours = [_tier_colour(r, lam) for r in results]
    ec = [_tier_colour_light(r, lam) for r in results]

    sc = ax.scatter(
        aleat,
        epist,
        s=sizes,
        c=colours,
        edgecolors=ec,
        linewidths=1.0,
        alpha=0.88,
        zorder=4,
    )

    # History labels offset from points
    for i, (r, lab) in enumerate(zip(results, labels)):
        ax.annotate(
            lab,
            xy=(aleat[i], epist[i]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=7.5,
            color=_GRAY_MID,
        )

    # Threshold lines
    max_H = np.log(N)
    ax.axvline(
        max_H * 0.75,
        color=_GRAY_MID,
        linewidth=0.6,
        linestyle="--",
        zorder=2,
        alpha=0.6,
    )
    if len(epist) > 2:
        ax.axhline(
            np.percentile(epist, 66),
            color=_GRAY_MID,
            linewidth=0.6,
            linestyle=":",
            zorder=2,
            alpha=0.6,
        )

    # Quadrant annotations
    x_lo, x_hi = ax.get_xlim() if ax.get_xlim()[1] > 0 else (0, max_H)
    y_lo, y_hi = ax.get_ylim() if ax.get_ylim()[1] > 0 else (0, 1)
    ax.text(
        0.04,
        0.04,
        "ideal\n(low both)",
        transform=ax.transAxes,
        fontsize=7.5,
        color=_GREEN,
        va="bottom",
    )
    ax.text(
        0.62,
        0.04,
        "high aleatoric\n(model uncertain)",
        transform=ax.transAxes,
        fontsize=7.5,
        color=_AMBER,
        va="bottom",
    )
    ax.text(
        0.04,
        0.72,
        "high epistemic\n(need more data)",
        transform=ax.transAxes,
        fontsize=7.5,
        color=_RED,
        va="bottom",
    )

    # Bubble size legend
    for pi_val, lab in [(0.10, "π*=0.10"), (0.20, "π*=0.20")]:
        ax.scatter(
            [],
            [],
            s=max(60, pi_val * 2800),
            c=[_GRAY_LIGHT],
            edgecolors=[_GRAY_MID],
            linewidths=0.8,
            label=lab,
            alpha=0.7,
        )

    ax.set_xlabel("Aleatoric entropy  Hᵈₕ  (nats)")
    ax.set_ylabel("Total epistemic std  Σ √Var(T̂(s′|h))")
    ax.set_title("D  ·  Coverage map", loc="left", pad=6)
    ax.legend(loc="upper right", title="bubble size", title_fontsize=7)


# ─────────────────────────────────────────────────────────────────────────────
# Colour legend patch (shared across all panels)
# ─────────────────────────────────────────────────────────────────────────────


def _reliability_legend(lam: float) -> list[mpatches.Patch]:
    return [
        mpatches.Patch(
            facecolor=_GREEN,
            edgecolor=_GREEN_LIGHT,
            linewidth=0.6,
            label=f"reliable  (floor < λ/4 = {lam/4:.3f})",
        ),
        mpatches.Patch(
            facecolor=_AMBER,
            edgecolor=_AMBER_LIGHT,
            linewidth=0.6,
            label=f"marginal  (λ/4 ≤ floor < λ/2 = {lam/2:.3f})",
        ),
        mpatches.Patch(
            facecolor=_RED,
            edgecolor=_RED_LIGHT,
            linewidth=0.6,
            label=f"unreliable  (floor ≥ λ/2 = {lam/2:.3f})",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main figure
# ─────────────────────────────────────────────────────────────────────────────


def plot_level5(
    results: list[HistoryUncertainty],
    lam: float,
    delta_pred: float,
    M: int,
    hist_labels: list[str] | None = None,
    d_name: str = "X¹",
    N: int = 3,
    figsize: tuple = (14, 9),
    max_rows: int = 20,
    save_path: str | None = None,
    dpi: int = 180,
) -> plt.Figure:
    """
    Four-panel Level 5 uncertainty figure using seaborn/matplotlib.

    Parameters
    ----------
    results      : list[HistoryUncertainty]   from build_results()
    lam          : float   lambda trimming threshold
    delta_pred   : float   certified hat_Delta_pred
    M            : int     MC draws
    hist_labels  : list[str] | None
    d_name       : str     target variable name for titles
    N            : int     number of bins
    figsize      : tuple
    max_rows     : int     cap histories shown in bar panels
    save_path    : str | None   if given, saves figure to this path
    dpi          : int

    Returns
    -------
    plt.Figure
    """
    _set_theme()

    rows = results[:max_rows]
    n = len(rows)
    labels = hist_labels[:n] if hist_labels else [f"h{r.h_idx}" for r in rows]

    # Global coverage scalar for subtitle
    total_pi = sum(r.pi_star for r in results)
    cov_scalar = sum(r.pi_star for r in results if r.reliable) / total_pi
    cov_str = f"{cov_scalar:.3f}  {'✓' if cov_scalar >= 0.9 else '✗  Pillar 3 flagged'}"

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    fig.patch.set_facecolor("white")

    # Title block
    fig.text(
        0.5,
        0.985,
        f"Level 5 — uncertainty-aware explanation  ·  target {d_name}",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="normal",
        color="#2C2C2A",
    )
    fig.text(
        0.5,
        0.962,
        f"λ = {lam}   ·   δ_pred = {delta_pred}   ·   M = {M}"
        f"   ·   global Cov(K*) = {cov_str}",
        ha="center",
        va="top",
        fontsize=8.5,
        color=_GRAY_MID,
    )

    # Grid: 2 rows × 2 cols with tight spacing
    gs = fig.add_gridspec(
        2,
        2,
        left=0.12,
        right=0.96,
        top=0.91,
        bottom=0.13,
        hspace=0.38,
        wspace=0.38,
    )

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    _panel_aleatoric(ax_A, rows, labels, lam, N)
    _panel_epistemic(ax_B, rows, labels, lam)
    _panel_noise_floor(ax_C, rows, labels, lam, delta_pred, M)
    _panel_coverage_map(ax_D, rows, labels, lam, N)

    # Shared reliability colour legend at the bottom
    fig.legend(
        handles=_reliability_legend(lam),
        loc="lower center",
        ncol=3,
        fontsize=8.5,
        framealpha=0.85,
        edgecolor=_GRAY_LIGHT,
        bbox_to_anchor=(0.5, 0.01),
        handlelength=1.4,
        handleheight=0.9,
        borderpad=0.6,
        columnspacing=1.6,
    )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  saved → {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Example
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Synthetic data matching the worked example from the paper
    kernels = {
        0: np.array([0.74, 0.18, 0.08]),
        1: np.array([0.22, 0.55, 0.23]),
        2: np.array([0.05, 0.12, 0.83]),
        3: np.array([0.31, 0.38, 0.31]),
        4: np.array([0.19, 0.71, 0.10]),
        5: np.array([0.28, 0.28, 0.44]),
    }
    counts = {0: 210, 1: 160, 2: 190, 3: 130, 4: 180, 5: 8}
    pool_sizes = {0: 210, 1: 160, 2: 190, 3: 130, 4: 180, 5: 8}
    pi_star = {0: 0.20, 1: 0.18, 2: 0.19, 3: 0.15, 4: 0.17, 5: 0.11}

    lam = 0.25
    delta_pred = 0.05
    M = 200
    N = 3

    results = build_results(
        kernels,
        counts,
        pool_sizes,
        pi_star,
        lam=lam,
        delta_pred=delta_pred,
        M=M,
        d=0,
    )

    labels = [
        "h=(0,0,0)",
        "h=(1,0,1)",
        "h=(2,2,1)",
        "h=(0,1,2)",
        "h=(1,1,0)",
        "h=(2,0,0)",
    ]

    fig = plot_level5(
        results,
        lam=lam,
        delta_pred=delta_pred,
        M=M,
        hist_labels=labels,
        d_name="X¹",
        N=N,
        figsize=(14, 9),
        save_path="/mnt/user-data/outputs/karma_level5.png",
        dpi=180,
    )
    plt.show()
