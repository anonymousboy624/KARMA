"""
karma_levels1to4_plots.py
─────────────────────────
Publication-quality plots for KARMA explanation Levels 1–4.

Level 1  Variable importance        Φ̃^d   horizontal bar chart
Level 2  Lag profiles               φ^d_k  line plot per variable
Level 3  Regime heatmap             ΔTV(h,h') regime distance matrix
Level 4  Average causal effects     ACE^d(d',k,x)  heatmap + bar

Each level has its own standalone function returning a plt.Figure.
A combined figure (all four panels) is also provided.

Dependencies: numpy, matplotlib, seaborn, networkx (for DAG layout)
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette  (mirrors karma_level5_plots.py)
# ─────────────────────────────────────────────────────────────────────────────

_BLUE_RAMP = [
    "#042C53",
    "#0C447C",
    "#185FA5",
    "#378ADD",
    "#85B7EB",
    "#B5D4F4",
    "#E6F1FB",
]
_TEAL_RAMP = [
    "#04342C",
    "#085041",
    "#0F6E56",
    "#1D9E75",
    "#5DCAA5",
    "#9FE1CB",
    "#E1F5EE",
]
_AMBER_RAMP = [
    "#412402",
    "#633806",
    "#854F0B",
    "#BA7517",
    "#EF9F27",
    "#FAC775",
    "#FAEEDA",
]
_CORAL_RAMP = [
    "#4A1B0C",
    "#712B13",
    "#993C1D",
    "#D85A30",
    "#F0997B",
    "#F5C4B3",
    "#FAECE7",
]
_GRAY_RAMP = [
    "#2C2C2A",
    "#444441",
    "#5F5E5A",
    "#888780",
    "#B4B2A9",
    "#D3D1C7",
    "#F1EFE8",
]
_RED_RAMP = [
    "#501313",
    "#791F1F",
    "#A32D2D",
    "#E24B4A",
    "#F09595",
    "#F7C1C1",
    "#FCEBEB",
]

_VAR_COLOURS = [
    _BLUE_RAMP[2],
    _TEAL_RAMP[3],
    _CORAL_RAMP[3],
    _AMBER_RAMP[3],
    _GRAY_RAMP[3],
]

_GREEN = _TEAL_RAMP[3]
_AMBER = _AMBER_RAMP[4]
_RED = _RED_RAMP[3]
_GRAY = _GRAY_RAMP[3]


# ─────────────────────────────────────────────────────────────────────────────
# Global theme
# ─────────────────────────────────────────────────────────────────────────────


def _set_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.5,
            "axes.edgecolor": _GRAY_RAMP[3],
            "grid.color": _GRAY_RAMP[5],
            "grid.linewidth": 0.35,
            "grid.linestyle": "--",
            "xtick.major.width": 0.4,
            "ytick.major.width": 0.4,
            "xtick.color": _GRAY_RAMP[3],
            "ytick.color": _GRAY_RAMP[3],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "normal",
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "legend.framealpha": 0.75,
            "legend.edgecolor": _GRAY_RAMP[5],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared annotation helper
# ─────────────────────────────────────────────────────────────────────────────


def _panel_label(ax: plt.Axes, letter: str) -> None:
    """Bold panel letter in top-left corner (A, B, C, D ...)."""
    ax.text(
        -0.08,
        1.04,
        letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=_GRAY_RAMP[1],
    )


# ═══════════════════════════════════════════════════════════════════════════
#  LEVEL 1 — Variable importance
# ═══════════════════════════════════════════════════════════════════════════


def plot_level1(
    phi: np.ndarray,
    var_names: list[str],
    lam: float = 0.08,
    figsize: tuple = (6, 3.5),
    save_path: str | None = None,
    dpi: int = 180,
) -> plt.Figure:
    """
    Level 1 — variable importance bar chart.

    Parameters
    ----------
    phi       : (D,) float   normalised importance Φ̃^d, sums to 1
    var_names : list[str]    length D
    lam       : float        trimming threshold lambda (shown as reference line)
    figsize   : tuple
    save_path : str | None
    dpi       : int

    Returns
    -------
    plt.Figure
    """
    _set_theme()
    D = len(phi)
    assert len(var_names) == D

    order = np.argsort(phi)[::-1]
    phi_ord = phi[order]
    names = [var_names[i] for i in order]
    colours = [_VAR_COLOURS[i % len(_VAR_COLOURS)] for i in order]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")

    bars = ax.barh(
        range(D),
        phi_ord,
        color=colours,
        edgecolor=[_GRAY_RAMP[5]] * D,
        linewidth=0.5,
        height=0.55,
        zorder=3,
    )

    # Value annotations
    for bar, val, name in zip(bars, phi_ord, names):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=8,
            color=_GRAY_RAMP[3],
        )

    # Lambda reference line
    ax.axvline(
        lam,
        color=_RED,
        linewidth=0.8,
        linestyle="--",
        zorder=4,
        label=f"λ = {lam}  (trim threshold)",
    )

    ax.set_yticks(range(D))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Normalised importance  Φ̃ᵈ")
    ax.set_title(
        "Level 1  ·  Variable importance",
        loc="left",
        pad=6,
        fontsize=10,
    )
    ax.set_xlim(0, max(phi_ord) * 1.25)
    ax.legend(loc="lower right")
    _panel_label(ax, "A")

    # Cumulative coverage annotation on right spine
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel("Cumulative  Σ Φ̃", fontsize=8, color=_GRAY_RAMP[3])
    ax2.tick_params(axis="x", labelsize=7.5, colors=_GRAY_RAMP[3])
    cumulative = np.cumsum(phi_ord)
    ax2.set_xticks(cumulative)
    ax2.set_xticklabels([f"{c:.2f}" for c in cumulative], rotation=30)
    ax2.spines["top"].set_visible(True)
    ax2.spines["top"].set_linewidth(0.4)
    ax2.spines["top"].set_color(_GRAY_RAMP[5])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  saved → {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  LEVEL 2 — Lag profiles
# ═══════════════════════════════════════════════════════════════════════════


def plot_level2(
    phi_dk: np.ndarray,
    var_names: list[str],
    K_star: int,
    lam: float = 0.08,
    figsize: tuple = (7, 4),
    save_path: str | None = None,
    dpi: int = 180,
) -> plt.Figure:
    """
    Level 2 — lag profiles φ^d_k vs k.

    Parameters
    ----------
    phi_dk    : (D, K*) float   lag-resolved influence per variable and lag
    var_names : list[str]       length D
    K_star    : int             Markov order
    lam       : float           lambda reference
    figsize   : tuple
    save_path : str | None
    dpi       : int

    Returns
    -------
    plt.Figure
    """
    _set_theme()
    D, K = phi_dk.shape
    assert len(var_names) == D
    lags = np.arange(1, K + 1)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")

    markers = ["o", "s", "^", "D", "v"]

    for d in range(D):
        colour = _VAR_COLOURS[d % len(_VAR_COLOURS)]
        ax.plot(
            lags,
            phi_dk[d],
            color=colour,
            linewidth=1.8,
            marker=markers[d % len(markers)],
            markersize=5.5,
            label=var_names[d],
            zorder=4,
        )
        # Shaded band: ± 1 pseudo-SE (illustrative; replace with bootstrap CIs)
        ax.fill_between(
            lags,
            phi_dk[d] * 0.85,
            phi_dk[d] * 1.15,
            color=colour,
            alpha=0.08,
            zorder=2,
        )

    # Lambda trim threshold
    ax.axhline(
        lam,
        color=_RED,
        linewidth=0.8,
        linestyle="--",
        zorder=3,
        label=f"λ = {lam}",
    )

    # Annotate lag 1 values
    for d in range(D):
        ax.annotate(
            f"  {phi_dk[d, 0]:.2f}",
            xy=(1, phi_dk[d, 0]),
            fontsize=7.5,
            color=_VAR_COLOURS[d % len(_VAR_COLOURS)],
            va="center",
        )

    ax.set_xticks(lags)
    ax.set_xticklabels([f"k = {k}" for k in lags])
    ax.set_xlabel("Lag  k")
    ax.set_ylabel("Lag-resolved influence  φᵈₖ")
    ax.set_title("Level 2  ·  Lag profiles", loc="left", pad=6, fontsize=10)
    ax.legend(loc="upper right", title="variable", title_fontsize=7.5)
    _panel_label(ax, "B")

    # Momentum / mean-reversion annotations
    peak_lags = [np.argmax(phi_dk[d]) + 1 for d in range(D)]
    for d, pk in enumerate(peak_lags):
        if pk == 1:
            label_text = "momentum"
        elif pk == K:
            label_text = "long memory"
        else:
            label_text = f"peak k={pk}"
        ax.annotate(
            label_text,
            xy=(pk, phi_dk[d, pk - 1]),
            xytext=(0, 12),
            textcoords="offset points",
            fontsize=7,
            color=_VAR_COLOURS[d % len(_VAR_COLOURS)],
            ha="center",
            arrowprops=dict(
                arrowstyle="-",
                color=_VAR_COLOURS[d % len(_VAR_COLOURS)],
                lw=0.6,
            ),
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  saved → {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  LEVEL 3 — Regime heatmap
# ═══════════════════════════════════════════════════════════════════════════


def plot_level3(
    kernels: np.ndarray,
    hist_labels: list[str],
    pi_star: np.ndarray,
    coverage: np.ndarray,
    lam: float = 0.08,
    figsize: tuple = (9, 7),
    save_path: str | None = None,
    dpi: int = 180,
    cluster: bool = True,
) -> plt.Figure:
    """
    Level 3 — regime distance matrix Δ_regime(h, h′).

    Shows inter-history TV distances as a clustered heatmap.
    Histories with coverage < λ/2 are flagged with hatching.

    Parameters
    ----------
    kernels      : (H, N) float   factored kernel row per history
                   (use a single target variable d)
    hist_labels  : list[str]      length H
    pi_star      : (H,) float     stationary weights
    coverage     : (H,) float     per-history bar value in [0,1]
    lam          : float
    figsize      : tuple
    save_path    : str | None
    cluster      : bool           whether to apply hierarchical clustering
    dpi          : int

    Returns
    -------
    plt.Figure
    """
    _set_theme()
    H, N = kernels.shape
    assert len(hist_labels) == H

    # ── TV distance matrix ─────────────────────────────────────────────────
    tv_mat = np.zeros((H, H))
    for i in range(H):
        for j in range(H):
            tv_mat[i, j] = 0.5 * np.sum(np.abs(kernels[i] - kernels[j]))

    # ── Hierarchical clustering ────────────────────────────────────────────
    if cluster and H > 2:
        condensed = tv_mat[np.triu_indices(H, k=1)]
        Z = linkage(condensed, method="average")
        order = leaves_list(Z)
    else:
        order = np.arange(H)

    tv_ord = tv_mat[np.ix_(order, order)]
    labs_ord = [hist_labels[i] for i in order]
    pi_ord = pi_star[order]
    cov_ord = coverage[order]

    # ── Figure layout: heatmap + dendrograms ──────────────────────────────
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[0.15, 1],
        height_ratios=[1, 0.15],
        hspace=0.04,
        wspace=0.04,
    )

    ax_dend_left = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_dend_bot = fig.add_subplot(gs[1, 1])
    ax_dend_left.axis("off")
    ax_dend_bot.axis("off")

    # Dendrogram (left)
    if cluster and H > 2:
        dendrogram(
            Z,
            orientation="left",
            ax=ax_dend_left,
            no_labels=True,
            link_color_func=lambda k: _GRAY_RAMP[3],
            above_threshold_color=_GRAY_RAMP[3],
        )
        ax_dend_left.set_xlim(ax_dend_left.get_xlim()[::-1])
        for spine in ax_dend_left.spines.values():
            spine.set_visible(False)
        ax_dend_left.tick_params(left=False, bottom=False)

    # ── Heatmap ────────────────────────────────────────────────────────────
    cmap = sns.color_palette("Blues", as_cmap=True)

    im = ax_heat.imshow(
        tv_ord,
        cmap=cmap,
        vmin=0,
        vmax=0.5,
        aspect="auto",
        zorder=2,
    )

    # Cell annotations
    for i in range(H):
        for j in range(H):
            val = tv_ord[i, j]
            txt_col = "white" if val > 0.28 else _GRAY_RAMP[1]
            ax_heat.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=txt_col,
            )

    # Hatch unreliable histories (coverage < 0.5 -> floor > lambda/4)
    for i, cov in enumerate(cov_ord):
        if cov < 0.5:
            for j in range(H):
                ax_heat.add_patch(
                    mpatches.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        hatch="///",
                        edgecolor=_RED,
                        linewidth=0,
                        alpha=0.35,
                        zorder=3,
                    )
                )
            for j in range(H):
                ax_heat.add_patch(
                    mpatches.Rectangle(
                        (i - 0.5, j - 0.5),
                        1,
                        1,
                        fill=False,
                        hatch="///",
                        edgecolor=_RED,
                        linewidth=0,
                        alpha=0.35,
                        zorder=3,
                    )
                )

    ax_heat.set_xticks(range(H))
    ax_heat.set_yticks(range(H))
    ax_heat.set_xticklabels(labs_ord, rotation=40, ha="right", fontsize=8)
    ax_heat.set_yticklabels(labs_ord, fontsize=8)
    ax_heat.set_title(
        "Level 3  ·  Regime distance matrix  Δ_regime(h, h′)  =  TV(T̂(·|h), T̂(·|h′))",
        loc="left",
        pad=8,
        fontsize=10,
    )
    _panel_label(ax_heat, "C")

    # π* weight bar on right side
    ax_pi = ax_heat.inset_axes([1.02, 0, 0.04, 1])
    for i, (pi_h, cov_h) in enumerate(zip(pi_ord, cov_ord)):
        colour = _GREEN if cov_h >= 0.5 else _RED
        ax_pi.barh(i, pi_h, height=0.8, color=colour, alpha=0.7)
    ax_pi.set_xlim(0, max(pi_ord) * 1.5)
    ax_pi.set_ylim(-0.5, H - 0.5)
    ax_pi.set_yticks([])
    ax_pi.set_xlabel("π*", fontsize=7.5)
    ax_pi.invert_yaxis()
    for spine in ax_pi.spines.values():
        spine.set_visible(False)
    ax_pi.tick_params(labelsize=7)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.12)
    cbar.set_label("TV distance", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)

    # Legend patches
    legend_elements = [
        mpatches.Patch(
            facecolor="white",
            edgecolor=_RED,
            hatch="///",
            linewidth=0.5,
            label="unreliable history  (floor ≥ λ/4)",
        ),
    ]
    ax_heat.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=7.5,
        framealpha=0.8,
    )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  saved → {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  LEVEL 4 — Average causal effects
# ═══════════════════════════════════════════════════════════════════════════


def plot_level4(
    ace: np.ndarray,
    rho: np.ndarray,
    var_names: list[str],
    K_star: int,
    lam: float = 0.08,
    figsize: tuple = (11, 4.5),
    save_path: str | None = None,
    dpi: int = 180,
) -> plt.Figure:
    """
    Level 4 — average causal effect heatmap + bar chart.

    Left panel:  ACE^d(d′, k) heatmap  (source variable d′ × lag k)
                 for a fixed target variable d.
    Right panel: Ranked bar chart of ACE values, coloured by
                 retained (ρ(e) > λ) vs trimmed.

    Parameters
    ----------
    ace       : (D, K*) float   ACE^d(d′, k) averaged over intervention bins x
    rho       : (D, K*) float   TV edge contribution ρ(e)
    var_names : list[str]       length D  (source variables)
    K_star    : int             Markov order
    lam       : float           lambda trim threshold
    figsize   : tuple
    save_path : str | None
    dpi       : int

    Returns
    -------
    plt.Figure
    """
    _set_theme()
    D, K = ace.shape
    assert len(var_names) == D

    fig, (ax_L, ax_R) = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.38},
    )
    fig.patch.set_facecolor("white")

    # ── Left: ACE heatmap ─────────────────────────────────────────────────
    lags = [f"k={k+1}" for k in range(K)]
    cmap_ace = sns.light_palette(_BLUE_RAMP[2], n_colors=256, as_cmap=True)

    im = ax_L.imshow(
        ace,
        cmap=cmap_ace,
        vmin=0,
        vmax=ace.max() * 1.05,
        aspect="auto",
    )

    # Cell annotations: ACE value + retained/trimmed marker
    for d in range(D):
        for k in range(K):
            val = ace[d, k]
            retained = rho[d, k] > lam
            txt_col = "white" if val > ace.max() * 0.55 else _GRAY_RAMP[1]
            ax_L.text(
                k,
                d,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=txt_col,
            )
            # Small marker: filled circle = retained, empty = trimmed
            marker_col = _GREEN if retained else _GRAY_RAMP[4]
            ax_L.plot(
                k + 0.35,
                d - 0.32,
                "o",
                markersize=4,
                color=marker_col,
                zorder=5,
                clip_on=False,
            )

    ax_L.set_xticks(range(K))
    ax_L.set_xticklabels(lags, fontsize=8.5)
    ax_L.set_yticks(range(D))
    ax_L.set_yticklabels(var_names, fontsize=8.5)
    ax_L.set_xlabel("Lag  k")
    ax_L.set_ylabel("Source variable  d′")
    ax_L.set_title(
        "ACE heatmap  ACEᵈ(d′, k)",
        loc="left",
        pad=6,
        fontsize=10,
    )
    _panel_label(ax_L, "D")

    cbar = fig.colorbar(im, ax=ax_L, fraction=0.046, pad=0.04)
    cbar.set_label("Avg causal effect", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)

    # Retained / trimmed legend
    leg_elem = [
        mpatches.Patch(facecolor=_GREEN, label=f"retained  ρ(e) > λ = {lam}"),
        mpatches.Patch(facecolor=_GRAY_RAMP[4], label="trimmed"),
    ]
    ax_L.legend(
        handles=leg_elem,
        loc="lower right",
        fontsize=7.5,
        framealpha=0.85,
        markerscale=0.7,
    )

    # ── Right: ranked bar chart ────────────────────────────────────────────
    # Flatten and sort by ACE descending
    entries = []
    for d in range(D):
        for k in range(K):
            entries.append(
                {
                    "label": f"{var_names[d]}, k={k+1}",
                    "ace": ace[d, k],
                    "rho": rho[d, k],
                    "retained": rho[d, k] > lam,
                    "var": d,
                    "lag": k,
                }
            )
    entries.sort(key=lambda e: e["ace"], reverse=True)
    n_e = len(entries)

    bar_colours = []
    bar_ec = []
    for e in entries:
        if e["retained"]:
            bar_colours.append(_VAR_COLOURS[e["var"] % len(_VAR_COLOURS)])
            bar_ec.append(_GRAY_RAMP[5])
        else:
            bar_colours.append(_GRAY_RAMP[5])
            bar_ec.append(_GRAY_RAMP[4])

    bars = ax_R.barh(
        range(n_e),
        [e["ace"] for e in entries],
        color=bar_colours,
        edgecolor=bar_ec,
        linewidth=0.5,
        height=0.55,
        zorder=3,
    )

    # Lambda reference line
    ax_R.axvline(
        lam,
        color=_RED,
        linewidth=0.9,
        linestyle="--",
        zorder=4,
        label=f"λ = {lam}",
    )

    # Annotate ACE value and rho
    for bar, e in zip(bars, entries):
        ax_R.text(
            bar.get_width() + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{e['ace']:.3f}  (ρ={e['rho']:.3f})",
            va="center",
            ha="left",
            fontsize=7,
            color=_GRAY_RAMP[3] if e["retained"] else _GRAY_RAMP[4],
        )

    ax_R.set_yticks(range(n_e))
    ax_R.set_yticklabels([e["label"] for e in entries], fontsize=8)
    ax_R.invert_yaxis()
    ax_R.set_xlabel("ACEᵈ(d′, k)")
    ax_R.set_title("Ranked causal effects", loc="left", pad=6, fontsize=10)
    ax_R.set_xlim(0, max(e["ace"] for e in entries) * 1.55)
    ax_R.legend(loc="lower right")

    # Variable colour legend (right panel)
    var_patches = [
        mpatches.Patch(
            facecolor=_VAR_COLOURS[d % len(_VAR_COLOURS)],
            label=var_names[d],
        )
        for d in range(D)
    ]
    ax_R.legend(
        handles=var_patches
        + [
            mpatches.Patch(facecolor=_GRAY_RAMP[5], label="trimmed"),
            mpatches.Patch(
                facecolor="white",
                edgecolor=_RED,
                linestyle="--",
                linewidth=1,
                label=f"λ = {lam}",
            ),
        ],
        loc="lower right",
        fontsize=7.5,
        framealpha=0.85,
        title="variable / status",
        title_fontsize=7,
    )

    fig.suptitle(
        "Level 4  ·  Average causal effect  ACEᵈ(d′, k)",
        fontsize=11,
        y=1.01,
        fontweight="normal",
        color=_GRAY_RAMP[1],
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  saved → {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  COMBINED — all four levels in one figure
# ═══════════════════════════════════════════════════════════════════════════


def plot_all_levels(
    phi: np.ndarray,
    phi_dk: np.ndarray,
    kernels: np.ndarray,
    ace: np.ndarray,
    rho: np.ndarray,
    pi_star: np.ndarray,
    coverage: np.ndarray,
    var_names: list[str],
    hist_labels: list[str],
    K_star: int,
    lam: float = 0.08,
    d_name: str = "X¹",
    figsize: tuple = (16, 14),
    save_path: str | None = None,
    dpi: int = 180,
) -> plt.Figure:
    """
    Combined four-panel figure: Levels 1–4 on a single page.

    Layout
    ------
    Row 0  [ Level 1 importance (left)  |  Level 2 lag profiles (right) ]
    Row 1  [ Level 3 regime heatmap (full width)                         ]
    Row 2  [ Level 4 ACE heatmap (left) |  Level 4 ranked bars (right)  ]

    Parameters
    ----------
    phi         : (D,)   normalised importance
    phi_dk      : (D, K) lag-resolved influence
    kernels     : (H, N) transition kernel rows (single target variable d)
    ace         : (D, K) average causal effects
    rho         : (D, K) TV edge contributions
    pi_star     : (H,)   stationary weights
    coverage    : (H,)   per-history coverage bar
    var_names   : (D,)   variable names
    hist_labels : (H,)   history labels
    K_star      : int    Markov order
    lam         : float  lambda
    d_name      : str    target variable name for titles
    figsize     : tuple
    save_path   : str | None
    dpi         : int

    Returns
    -------
    plt.Figure
    """
    _set_theme()
    D = len(var_names)
    H = len(hist_labels)
    K = K_star

    fig = plt.figure(figsize=figsize, facecolor="white")
    fig.text(
        0.5,
        0.995,
        f"KARMA explanation hierarchy  ·  Levels 1–4  ·  target {d_name}  ·  K* = {K_star}  ·  λ = {lam}",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="normal",
        color=_GRAY_RAMP[1],
    )

    gs_outer = gridspec.GridSpec(
        3,
        1,
        top=0.97,
        bottom=0.04,
        hspace=0.44,
        height_ratios=[1, 1.1, 1],
    )

    # ── Row 0: Level 1 + Level 2 ─────────────────────────────────────────
    gs_row0 = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=gs_outer[0],
        wspace=0.38,
    )
    ax1 = fig.add_subplot(gs_row0[0])
    ax2 = fig.add_subplot(gs_row0[1])

    # Level 1
    order = np.argsort(phi)[::-1]
    phi_ord = phi[order]
    names1 = [var_names[i] for i in order]
    cols1 = [_VAR_COLOURS[i % len(_VAR_COLOURS)] for i in order]

    bars1 = ax1.barh(
        range(D),
        phi_ord,
        color=cols1,
        edgecolor=[_GRAY_RAMP[5]] * D,
        linewidth=0.5,
        height=0.55,
        zorder=3,
    )
    for bar, val in zip(bars1, phi_ord):
        ax1.text(
            bar.get_width() + 0.004,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color=_GRAY_RAMP[3],
        )
    ax1.axvline(
        lam, color=_RED, linewidth=0.8, linestyle="--", zorder=4, label=f"λ = {lam}"
    )
    ax1.set_yticks(range(D))
    ax1.set_yticklabels(names1, fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Normalised importance  Φ̃ᵈ")
    ax1.set_title("Level 1  ·  Variable importance", loc="left", pad=5, fontsize=9.5)
    ax1.set_xlim(0, max(phi_ord) * 1.28)
    ax1.legend(loc="lower right", fontsize=7.5)
    _panel_label(ax1, "A")

    # Level 2
    markers = ["o", "s", "^", "D", "v"]
    lags = np.arange(1, K + 1)
    for d in range(D):
        c = _VAR_COLOURS[d % len(_VAR_COLOURS)]
        ax2.plot(
            lags,
            phi_dk[d],
            color=c,
            linewidth=1.8,
            marker=markers[d % len(markers)],
            markersize=5,
            label=var_names[d],
            zorder=4,
        )
        ax2.fill_between(
            lags, phi_dk[d] * 0.85, phi_dk[d] * 1.15, color=c, alpha=0.08, zorder=2
        )
    ax2.axhline(
        lam, color=_RED, linewidth=0.8, linestyle="--", zorder=3, label=f"λ = {lam}"
    )
    ax2.set_xticks(lags)
    ax2.set_xticklabels([f"k={k}" for k in lags])
    ax2.set_xlabel("Lag  k")
    ax2.set_ylabel("Lag-resolved influence  φᵈₖ")
    ax2.set_title("Level 2  ·  Lag profiles", loc="left", pad=5, fontsize=9.5)
    ax2.legend(loc="upper right", title="variable", title_fontsize=7, fontsize=8)
    _panel_label(ax2, "B")

    # ── Row 1: Level 3 (regime heatmap) ──────────────────────────────────
    ax3 = fig.add_subplot(gs_outer[1])

    tv_mat = np.zeros((H, H))
    for i in range(H):
        for j in range(H):
            tv_mat[i, j] = 0.5 * np.sum(np.abs(kernels[i] - kernels[j]))

    if H > 2:
        condensed = tv_mat[np.triu_indices(H, k=1)]
        Z = linkage(condensed, method="average")
        order3 = leaves_list(Z)
    else:
        order3 = np.arange(H)

    tv_ord = tv_mat[np.ix_(order3, order3)]
    labs_ord = [hist_labels[i] for i in order3]
    cov_ord = coverage[order3]

    cmap3 = sns.color_palette("Blues", as_cmap=True)
    im3 = ax3.imshow(tv_ord, cmap=cmap3, vmin=0, vmax=0.5, aspect="auto")

    for i in range(H):
        for j in range(H):
            val = tv_ord[i, j]
            tc = "white" if val > 0.28 else _GRAY_RAMP[1]
            ax3.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=tc)

    for i, cov_h in enumerate(cov_ord):
        if cov_h < 0.5:
            for j in range(H):
                ax3.add_patch(
                    mpatches.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        hatch="///",
                        edgecolor=_RED,
                        linewidth=0,
                        alpha=0.35,
                        zorder=3,
                    )
                )

    ax3.set_xticks(range(H))
    ax3.set_yticks(range(H))
    ax3.set_xticklabels(labs_ord, rotation=35, ha="right", fontsize=8)
    ax3.set_yticklabels(labs_ord, fontsize=8)
    ax3.set_title(
        "Level 3  ·  Regime distance matrix  Δ_regime(h, h′) = TV(T̂(·|h), T̂(·|h′))",
        loc="left",
        pad=5,
        fontsize=9.5,
    )
    _panel_label(ax3, "C")

    cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.02, pad=0.01)
    cbar3.set_label("TV distance", fontsize=8)
    cbar3.ax.tick_params(labelsize=7.5)

    ax3.legend(
        handles=[
            mpatches.Patch(
                facecolor="white",
                edgecolor=_RED,
                hatch="///",
                linewidth=0.5,
                label="unreliable  (floor ≥ λ/4)",
            )
        ],
        loc="lower left",
        fontsize=7.5,
        framealpha=0.85,
    )

    # ── Row 2: Level 4 ────────────────────────────────────────────────────
    gs_row2 = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=gs_outer[2],
        wspace=0.42,
    )
    ax4L = fig.add_subplot(gs_row2[0])
    ax4R = fig.add_subplot(gs_row2[1])

    # ACE heatmap
    cmap4 = sns.light_palette(_BLUE_RAMP[2], n_colors=256, as_cmap=True)
    im4 = ax4L.imshow(ace, cmap=cmap4, vmin=0, vmax=ace.max() * 1.05, aspect="auto")
    for d in range(D):
        for k in range(K):
            val = ace[d, k]
            retained = rho[d, k] > lam
            tc = "white" if val > ace.max() * 0.55 else _GRAY_RAMP[1]
            ax4L.text(
                k, d, f"{val:.3f}", ha="center", va="center", fontsize=7.5, color=tc
            )
            mc = _GREEN if retained else _GRAY_RAMP[4]
            ax4L.plot(
                k + 0.35, d - 0.32, "o", markersize=4, color=mc, zorder=5, clip_on=False
            )
    ax4L.set_xticks(range(K))
    ax4L.set_xticklabels([f"k={k+1}" for k in range(K)], fontsize=8.5)
    ax4L.set_yticks(range(D))
    ax4L.set_yticklabels(var_names, fontsize=8.5)
    ax4L.set_xlabel("Lag  k")
    ax4L.set_ylabel("Source variable  d′")
    ax4L.set_title("Level 4  ·  ACE heatmap", loc="left", pad=5, fontsize=9.5)
    _panel_label(ax4L, "D")
    cbar4 = fig.colorbar(im4, ax=ax4L, fraction=0.046, pad=0.04)
    cbar4.set_label("Avg causal effect", fontsize=8)
    cbar4.ax.tick_params(labelsize=7.5)

    # Ranked bars
    entries = sorted(
        [
            {
                "label": f"{var_names[d]}, k={k+1}",
                "ace": ace[d, k],
                "rho": rho[d, k],
                "retained": rho[d, k] > lam,
                "var": d,
            }
            for d in range(D)
            for k in range(K)
        ],
        key=lambda e: e["ace"],
        reverse=True,
    )
    n_e = len(entries)
    bcs = [
        _VAR_COLOURS[e["var"] % len(_VAR_COLOURS)] if e["retained"] else _GRAY_RAMP[5]
        for e in entries
    ]

    bars4 = ax4R.barh(
        range(n_e),
        [e["ace"] for e in entries],
        color=bcs,
        edgecolor=[_GRAY_RAMP[5]] * n_e,
        linewidth=0.5,
        height=0.55,
        zorder=3,
    )
    ax4R.axvline(lam, color=_RED, linewidth=0.9, linestyle="--", zorder=4)
    for bar, e in zip(bars4, entries):
        ax4R.text(
            bar.get_width() + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{e['ace']:.3f}  ρ={e['rho']:.3f}",
            va="center",
            ha="left",
            fontsize=7,
            color=_GRAY_RAMP[3] if e["retained"] else _GRAY_RAMP[4],
        )
    ax4R.set_yticks(range(n_e))
    ax4R.set_yticklabels([e["label"] for e in entries], fontsize=8)
    ax4R.invert_yaxis()
    ax4R.set_xlabel("ACEᵈ(d′, k)")
    ax4R.set_title("Ranked causal effects", loc="left", pad=5, fontsize=9.5)
    ax4R.set_xlim(0, max(e["ace"] for e in entries) * 1.55)
    var_patches = [
        mpatches.Patch(
            facecolor=_VAR_COLOURS[d % len(_VAR_COLOURS)], label=var_names[d]
        )
        for d in range(D)
    ]
    ax4R.legend(
        handles=var_patches
        + [mpatches.Patch(facecolor=_GRAY_RAMP[5], label="trimmed")],
        loc="lower right",
        fontsize=7.5,
        framealpha=0.85,
        title="variable",
        title_fontsize=7,
    )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  saved → {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Example
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    rng = np.random.default_rng(42)

    D = 3
    K = 3
    N = 3
    H = 6
    lam = 0.08
    K_star = K

    var_names = ["X¹", "X²", "X³"]
    hist_labels = [
        "h=(0,0,0)",
        "h=(1,0,1)",
        "h=(2,2,1)",
        "h=(0,1,2)",
        "h=(1,1,0)",
        "h=(2,0,0)",
    ]

    # Level 1: normalised importance
    phi = np.array([0.58, 0.28, 0.14])

    # Level 2: lag profiles — X1 momentum, X2 flat, X3 near-zero
    phi_dk = np.array(
        [
            [0.41, 0.15, 0.02],  # X1: sharp peak at k=1
            [0.24, 0.20, 0.18],  # X2: flatter
            [0.09, 0.06, 0.04],  # X3: near-zero
        ]
    )

    # Level 3: transition kernels for heatmap (target variable X1)
    kernels = np.array(
        [
            [0.74, 0.18, 0.08],
            [0.22, 0.55, 0.23],
            [0.05, 0.12, 0.83],
            [0.31, 0.38, 0.31],
            [0.19, 0.71, 0.10],
            [0.28, 0.28, 0.44],
        ]
    )
    pi_star = np.array([0.20, 0.18, 0.19, 0.15, 0.17, 0.11])
    coverage = np.array([0.92, 0.78, 0.95, 0.58, 0.88, 0.22])

    # Level 4: ACE and rho
    ace = np.array(
        [
            [0.41, 0.15, 0.03],
            [0.28, 0.11, 0.07],
            [0.04, 0.02, 0.01],
        ]
    )
    rho = np.array(
        [
            [0.38, 0.12, 0.02],
            [0.25, 0.09, 0.05],
            [0.03, 0.01, 0.01],
        ]
    )

    # Individual level plots
    fig1 = plot_level1(
        phi, var_names, lam=lam, save_path="/workspaces/KARMA-/outputs/karma_level1.png"
    )
    fig2 = plot_level2(
        phi_dk,
        var_names,
        K_star=K_star,
        lam=lam,
        save_path="/workspaces/KARMA-/outputs/karma_level2.png",
    )
    fig3 = plot_level3(
        kernels,
        hist_labels,
        pi_star,
        coverage,
        lam=lam,
        save_path="/workspaces/KARMA-/outputs/karma_level3.png",
    )
    fig4 = plot_level4(
        ace,
        rho,
        var_names,
        K_star=K_star,
        lam=lam,
        save_path="/workspaces/KARMA-/outputs/karma_level4.png",
    )

    # Combined figure
    fig_all = plot_all_levels(
        phi,
        phi_dk,
        kernels,
        ace,
        rho,
        pi_star,
        coverage,
        var_names,
        hist_labels,
        K_star=K_star,
        lam=lam,
        d_name="X¹",
        save_path="/workspaces/KARMA-/outputs/karma_levels1to4_combined.png",
        dpi=180,
    )

    plt.show()
