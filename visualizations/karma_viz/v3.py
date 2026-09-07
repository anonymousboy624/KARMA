"""
karma_plots.py
==============
Publication-quality figures for KARMA Levels 1, 2, and 4.
Reads directly from the KARMA JSON output file.

Usage
-----
    python karma_plots.py results.json --outdir ./figs --lam 0.025

Or import and call directly:

    from karma_plots import load_karma_results, plot_level1, plot_level2, plot_level4
    results = load_karma_results("results.json")
    fig1 = plot_level1(results)
    fig1.savefig("level1.pdf")

Dependencies
------------
    pip install numpy matplotlib
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────────────────────


def set_theme() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "normal",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.5,
            "axes.edgecolor": "#888780",
            "grid.color": "#D3D1C7",
            "grid.linewidth": 0.4,
            "grid.linestyle": "--",
            "xtick.major.width": 0.4,
            "ytick.major.width": 0.4,
            "xtick.color": "#888780",
            "ytick.color": "#888780",
            "legend.fontsize": 8,
            "legend.framealpha": 0.75,
            "legend.edgecolor": "#D3D1C7",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

# 10 distinct foreground colours for active variables
_FG = [
    "#185FA5",  # blue
    "#1D9E75",  # teal
    "#D85A30",  # coral
    "#BA7517",  # amber
    "#534AB7",  # purple
    "#444441",  # dark gray
    "#0F6E56",  # dark teal
    "#993C1D",  # dark coral
    "#854F0B",  # dark amber
    "#3C3489",  # dark purple
]
_FG_LIGHT = [
    "#B5D4F4",
    "#9FE1CB",
    "#F5C4B3",
    "#FAC775",
    "#AFA9EC",
    "#B4B2A9",
    "#5DCAA5",
    "#F0997B",
    "#EF9F27",
    "#7F77DD",
]
_ZERO_COL = "#D3D1C7"
_ZERO_ECOL = "#B4B2A9"
_RED = "#E24B4A"
_GRAY = "#888780"
_LGRAY = "#D3D1C7"

ZERO_THRESH = 1e-10


def _var_colour(d: int, active_indices: list[int]) -> str:
    """Map variable index d to a foreground colour."""
    if d in active_indices:
        pos = active_indices.index(d)
        return _FG[pos % len(_FG)]
    return _ZERO_COL


def _var_colour_light(d: int, active_indices: list[int]) -> str:
    if d in active_indices:
        pos = active_indices.index(d)
        return _FG_LIGHT[pos % len(_FG_LIGHT)]
    return _ZERO_ECOL


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────


def load_karma_results(path: str | Path) -> dict:
    """
    Load KARMA JSON results file.

    Expected structure (all fields used by the plots):
    {
      "config":  { "D": int, "N": int, "K_max": int, "lam": float, ... },
      "pillar2": { "K_star": int, "delta_pred": float, ... },
      "pillar3": {
        "variable_importance": {
          "Phi_n": [float, ...],        # length D, normalised importance
          "phi":   [[float, ...], ...]  # shape (D, K_star), lag-resolved
        },
        "retained_edges": [
          {"src": int, "tgt": int, "lag": int, "rho": float}, ...
        ]
      }
    }
    """
    with open(path) as f:
        data = json.load(f)
    return data


def _extract(results: dict, lam: float | None = None) -> dict:
    """Pull arrays and scalars needed by all three plots."""
    cfg = results["config"]
    p2 = results["pillar2"]
    p3 = results["pillar3"]
    vi = p3["variable_importance"]

    D = cfg["D"]
    K_star = p2["K_star"]
    lam = lam if lam is not None else cfg.get("lam", 0.025)
    delta_pred = p2["delta_pred"]

    # Normalised importance  (D,)
    Phi_n = np.array(vi["Phi_n"])

    # Lag-resolved influence  (D, K_star)
    phi_raw = np.array(vi["phi"])  # may be (D, K_star) or (D, K_max)
    phi_dk = phi_raw[:, :K_star]
    phi_dk = np.where(np.abs(phi_dk) < ZERO_THRESH, 0.0, phi_dk)

    # Active variables (Phi_n above zero)
    active = [d for d in range(D) if Phi_n[d] > ZERO_THRESH]

    # Variable names  X^0 … X^{D-1}  (override externally if desired)
    var_names = [f"$X^{{{d}}}$" for d in range(D)]

    # Retained edges
    edges = p3.get("retained_edges", [])

    return dict(
        D=D,
        K_star=K_star,
        lam=lam,
        delta_pred=delta_pred,
        Phi_n=Phi_n,
        phi_dk=phi_dk,
        active=active,
        var_names=var_names,
        edges=edges,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Level 1 — Variable importance
# ─────────────────────────────────────────────────────────────────────────────


def plot_level1(
    results: dict,
    lam: float | None = None,
    var_names: list[str] | None = None,
    figsize: tuple = (6.0, 3.8),
) -> plt.Figure:
    """
    Horizontal bar chart of normalised variable importance Phi_n.

    Parameters
    ----------
    results   : dict from load_karma_results()
    lam       : override lambda from config
    var_names : custom labels, length D
    figsize   : figure size in inches

    Returns
    -------
    plt.Figure
    """
    set_theme()
    ex = _extract(results, lam)
    D, K_star, lam = ex["D"], ex["K_star"], ex["lam"]
    Phi_n, active = ex["Phi_n"], ex["active"]
    names = var_names or ex["var_names"]

    order = np.argsort(Phi_n)[::-1]
    phi_ord = Phi_n[order]
    labs = [names[i] for i in order]
    cols = [_var_colour(i, active) for i in order]
    ecols = [_var_colour_light(i, active) for i in order]
    is_zero = phi_ord < ZERO_THRESH

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(
        range(D),
        phi_ord,
        color=cols,
        edgecolor=ecols,
        linewidth=0.6,
        height=0.58,
        zorder=3,
    )

    for bar, val, zero in zip(bars, phi_ord, is_zero):
        if zero:
            ax.text(
                lam * 0.12,
                bar.get_y() + bar.get_height() / 2,
                "certified zero",
                va="center",
                ha="left",
                fontsize=7.5,
                color=_LGRAY,
                style="italic",
            )
        else:
            ax.text(
                bar.get_width() + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                ha="left",
                fontsize=8,
                color=_GRAY,
            )

    # Lambda reference line
    ax.axvline(
        lam,
        color=_RED,
        linewidth=0.9,
        linestyle="--",
        zorder=4,
        label=f"$\\lambda = {lam}$",
    )

    # Separator between active and zero variables
    first_zero = int(np.searchsorted(-phi_ord, -ZERO_THRESH))
    if 0 < first_zero < D:
        ax.axhline(
            first_zero - 0.5, color=_LGRAY, linewidth=0.6, linestyle=":", zorder=2
        )

    # Cumulative importance on top axis
    active_sorted = np.sort(Phi_n[Phi_n > ZERO_THRESH])[::-1]
    cumul = np.cumsum(active_sorted)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(cumul)
    ax2.set_xticklabels([f"{c:.2f}" for c in cumul], fontsize=7.5, color=_GRAY)
    ax2.set_xlabel("Cumulative $\\sum\\tilde{\\Phi}^{d'}$", fontsize=8, color=_GRAY)
    ax2.spines["top"].set_visible(True)
    ax2.spines["top"].set_linewidth(0.4)
    ax2.spines["top"].set_color(_LGRAY)
    ax2.tick_params(axis="x", colors=_GRAY, width=0.4)

    ax.set_yticks(range(D))
    ax.set_yticklabels(labs, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Normalised importance $\\tilde{\\Phi}^{d'}$")
    ax.set_title(
        f"Level 1  $\\cdot$  Variable importance\n"
        f"$D={D}$,  $K^*={K_star}$,  $\\lambda={lam}$",
        loc="left",
        pad=4,
    )
    ax.set_xlim(0, max(phi_ord) * 1.38)
    ax.legend(loc="lower right", handlelength=1.5)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Level 2 — Lag profiles
# ─────────────────────────────────────────────────────────────────────────────


def plot_level2(
    results: dict,
    lam: float | None = None,
    var_names: list[str] | None = None,
    figsize: tuple = (6.5, 3.6),
    ci_pct: float = 18.0,
) -> plt.Figure:
    """
    Line plot of lag-resolved influence phi^{d'}_k vs lag k.
    Only plots active (non-zero) variables.

    Parameters
    ----------
    results  : dict from load_karma_results()
    lam      : override lambda
    var_names: custom labels
    figsize  : figure size
    ci_pct   : ± percentage band around each profile (pseudo-CI)
               set to 0 to suppress

    Returns
    -------
    plt.Figure
    """
    set_theme()
    ex = _extract(results, lam)
    D, K_star, lam = ex["D"], ex["K_star"], ex["lam"]
    phi_dk, active = ex["phi_dk"], ex["active"]
    names = var_names or ex["var_names"]

    marks = ["o", "s", "^", "D", "v", "P", "X", "h", "8", "*"]
    lags = np.arange(1, K_star + 1)

    fig, ax = plt.subplots(figsize=figsize)

    for i, d in enumerate(active):
        vals = phi_dk[d]
        if vals.max() < ZERO_THRESH:
            continue
        c = _FG[i % len(_FG)]
        ax.plot(
            lags,
            vals,
            color=c,
            linewidth=1.8,
            marker=marks[i % len(marks)],
            markersize=5.5,
            label=names[d],
            zorder=4,
        )
        if ci_pct > 0:
            frac = ci_pct / 100.0
            ax.fill_between(
                lags,
                vals * (1 - frac),
                vals * (1 + frac),
                color=c,
                alpha=0.09,
                zorder=2,
            )

    ax.axhline(
        lam,
        color=_RED,
        linewidth=0.9,
        linestyle="--",
        zorder=3,
        label=f"$\\lambda = {lam}$",
    )

    ax.set_xticks(lags)
    ax.set_xticklabels([f"$k={k}$" for k in lags])
    ax.set_xlabel("Lag $k$")
    ax.set_ylabel("Lag-resolved influence $\\phi^{d'}_k$")
    ax.set_title(
        f"Level 2  $\\cdot$  Lag profiles  $\\phi^{{d'}}_k$\n"
        f"$K^*={K_star}$ lags  (certified-zero variables omitted)",
        loc="left",
        pad=4,
    )
    ax.legend(
        loc="upper right",
        title="source variable",
        title_fontsize=7.5,
        ncol=2,
        fontsize=8,
    )
    ax.set_xlim(0.7, K_star + 0.3)
    ax.set_ylim(-0.02, phi_dk.max() * 1.28)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Level 4 — Ranked causal effects
# ─────────────────────────────────────────────────────────────────────────────


def plot_level4(
    results: dict,
    lam: float | None = None,
    var_names: list[str] | None = None,
    top_n: int = 20,
    figsize: tuple = (6.5, 5.8),
) -> plt.Figure:
    """
    Horizontal bar chart of TV edge contributions rho(e),
    ranked descending, coloured by source variable.

    Parameters
    ----------
    results   : dict from load_karma_results()
    lam       : override lambda
    var_names : custom labels
    top_n     : maximum edges to show
    figsize   : figure size

    Returns
    -------
    plt.Figure
    """
    set_theme()
    ex = _extract(results, lam)
    K_star, lam, active = ex["K_star"], ex["lam"], ex["active"]
    names = var_names or ex["var_names"]
    edges = sorted(ex["edges"], key=lambda e: e["rho"], reverse=True)[:top_n]
    n_e = len(edges)

    if n_e == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No retained edges",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=_GRAY,
        )
        return fig

    labels = [f"{names[e['src']]} → {names[e['tgt']]},  $k={e['lag']}$" for e in edges]
    rho_vals = [e["rho"] for e in edges]
    retained = [e["rho"] > lam for e in edges]
    bar_cols = [
        _var_colour(e["src"], active) if ret else _ZERO_COL
        for e, ret in zip(edges, retained)
    ]
    bar_ecs = [
        _var_colour_light(e["src"], active) if ret else _ZERO_ECOL
        for e, ret in zip(edges, retained)
    ]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(
        range(n_e),
        rho_vals,
        color=bar_cols,
        edgecolor=bar_ecs,
        linewidth=0.6,
        height=0.58,
        zorder=3,
    )

    ax.axvline(
        lam,
        color=_RED,
        linewidth=1.0,
        linestyle="--",
        zorder=4,
        label=f"$\\lambda = {lam}$",
    )

    for bar, e, ret in zip(bars, edges, retained):
        col = _GRAY if ret else "#B4B2A9"
        is_self = e["src"] == e["tgt"]
        suffix = "  (self)" if is_self else ""
        ax.text(
            bar.get_width() + 0.004,
            bar.get_y() + bar.get_height() / 2,
            f"{e['rho']:.4f}{suffix}",
            va="center",
            ha="left",
            fontsize=7.5,
            color=col,
        )

    ax.set_yticks(range(n_e))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel(r"TV edge contribution  $\rho(e)$")
    ax.set_title(
        f"Level 4  $\\cdot$  Ranked causal effects\n"
        f"Top {n_e} edges,  $\\lambda = {lam}$,  $K^* = {K_star}$",
        loc="left",
        pad=4,
    )
    ax.set_xlim(0, max(rho_vals) * 1.48)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)

    # Variable colour legend (active sources only)
    src_vars = sorted({e["src"] for e in edges if e["rho"] > lam})
    var_patches = [
        mpatches.Patch(
            facecolor=_var_colour(d, active),
            edgecolor=_var_colour_light(d, active),
            linewidth=0.6,
            label=names[d],
        )
        for d in src_vars
    ]
    ax.legend(
        handles=var_patches,
        loc="lower right",
        fontsize=8,
        title="source variable",
        title_fontsize=7.5,
        framealpha=0.85,
    )

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Combined figure — all three panels
# ─────────────────────────────────────────────────────────────────────────────


def plot_all(
    results: dict,
    lam: float | None = None,
    var_names: list[str] | None = None,
    top_n: int = 20,
    figsize: tuple = (16, 5),
) -> plt.Figure:
    """
    Three-panel combined figure: Level 1 | Level 2 | Level 4.
    Suitable for a single-page overview in a paper.
    """
    set_theme()
    fig, axes = plt.subplots(
        1, 3, figsize=figsize, gridspec_kw={"width_ratios": [1, 1.1, 1.3]}
    )

    ex = _extract(results, lam)
    D, K_star, lam = ex["D"], ex["K_star"], ex["lam"]
    Phi_n, phi_dk, active = ex["Phi_n"], ex["phi_dk"], ex["active"]
    names = var_names or ex["var_names"]
    edges = sorted(ex["edges"], key=lambda e: e["rho"], reverse=True)[:top_n]

    # ── Panel A: Level 1 ─────────────────────────────────────────────────────
    ax = axes[0]
    order = np.argsort(Phi_n)[::-1]
    phi_ord = Phi_n[order]
    labs = [names[i] for i in order]
    cols = [_var_colour(i, active) for i in order]
    ecols = [_var_colour_light(i, active) for i in order]
    is_zero = phi_ord < ZERO_THRESH

    bars = ax.barh(
        range(D),
        phi_ord,
        color=cols,
        edgecolor=ecols,
        linewidth=0.6,
        height=0.58,
        zorder=3,
    )
    for bar, val, zero in zip(bars, phi_ord, is_zero):
        if zero:
            ax.text(
                lam * 0.12,
                bar.get_y() + bar.get_height() / 2,
                "zero",
                va="center",
                ha="left",
                fontsize=6.5,
                color=_LGRAY,
                style="italic",
            )
        else:
            ax.text(
                bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                ha="left",
                fontsize=7.5,
                color=_GRAY,
            )
    ax.axvline(
        lam,
        color=_RED,
        linewidth=0.8,
        linestyle="--",
        zorder=4,
        label=f"$\\lambda={lam}$",
    )
    first_zero = int(np.searchsorted(-phi_ord, -ZERO_THRESH))
    if 0 < first_zero < D:
        ax.axhline(
            first_zero - 0.5, color=_LGRAY, linewidth=0.5, linestyle=":", zorder=2
        )
    ax.set_yticks(range(D))
    ax.set_yticklabels(labs, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("$\\tilde{\\Phi}^{d'}$", fontsize=9)
    ax.set_title("A  $\\cdot$  Variable importance", loc="left", pad=4, fontsize=9.5)
    ax.set_xlim(0, max(phi_ord) * 1.38)
    ax.legend(loc="lower right", fontsize=7.5, handlelength=1.2)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)

    # ── Panel B: Level 2 ─────────────────────────────────────────────────────
    ax = axes[1]
    marks = ["o", "s", "^", "D", "v", "P", "X", "h"]
    lags = np.arange(1, K_star + 1)
    for i, d in enumerate(active):
        vals = phi_dk[d]
        if vals.max() < ZERO_THRESH:
            continue
        c = _FG[i % len(_FG)]
        ax.plot(
            lags,
            vals,
            color=c,
            linewidth=1.6,
            marker=marks[i % len(marks)],
            markersize=5,
            label=names[d],
            zorder=4,
        )
        ax.fill_between(lags, vals * 0.82, vals * 1.18, color=c, alpha=0.09, zorder=2)
    ax.axhline(
        lam,
        color=_RED,
        linewidth=0.8,
        linestyle="--",
        zorder=3,
        label=f"$\\lambda={lam}$",
    )
    ax.set_xticks(lags)
    ax.set_xticklabels([f"$k={k}$" for k in lags])
    ax.set_xlabel("Lag $k$", fontsize=9)
    ax.set_ylabel("$\\phi^{d'}_k$", fontsize=9)
    ax.set_title("B  $\\cdot$  Lag profiles", loc="left", pad=4, fontsize=9.5)
    ax.legend(loc="upper right", title="source", title_fontsize=7, ncol=2, fontsize=7.5)
    ax.set_xlim(0.7, K_star + 0.3)
    ax.set_ylim(-0.02, phi_dk.max() * 1.28)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)

    # ── Panel C: Level 4 ─────────────────────────────────────────────────────
    ax = axes[2]
    n_e = len(edges)
    elabels = [f"{names[e['src']]}→{names[e['tgt']]}, $k={e['lag']}$" for e in edges]
    rho_vals = [e["rho"] for e in edges]
    retained = [e["rho"] > lam for e in edges]
    bcols = [
        _var_colour(e["src"], active) if r else _ZERO_COL
        for e, r in zip(edges, retained)
    ]
    becols = [
        _var_colour_light(e["src"], active) if r else _ZERO_ECOL
        for e, r in zip(edges, retained)
    ]

    bars = ax.barh(
        range(n_e),
        rho_vals,
        color=bcols,
        edgecolor=becols,
        linewidth=0.5,
        height=0.58,
        zorder=3,
    )
    ax.axvline(
        lam,
        color=_RED,
        linewidth=0.8,
        linestyle="--",
        zorder=4,
        label=f"$\\lambda={lam}$",
    )
    for bar, e, ret in zip(bars, edges, retained):
        ax.text(
            bar.get_width() + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{e['rho']:.3f}",
            va="center",
            ha="left",
            fontsize=7,
            color=_GRAY if ret else "#B4B2A9",
        )
    ax.set_yticks(range(n_e))
    ax.set_yticklabels(elabels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\rho(e)$", fontsize=9)
    ax.set_title("C  $\\cdot$  Ranked causal effects", loc="left", pad=4, fontsize=9.5)
    ax.set_xlim(0, max(rho_vals) * 1.45)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)
    src_vars = sorted({e["src"] for e in edges if e["rho"] > lam})
    ax.legend(
        handles=[
            mpatches.Patch(
                facecolor=_var_colour(d, active),
                edgecolor=_var_colour_light(d, active),
                linewidth=0.5,
                label=names[d],
            )
            for d in src_vars
        ],
        loc="lower right",
        fontsize=7,
        title="source",
        title_fontsize=7,
    )

    fig.suptitle(
        f"KARMA explanation hierarchy  ·  $D={D}$,  $K^*={K_star}$,  "
        f"$\\lambda={lam}$,  $\\hat{{\\Delta}}_{{\\mathrm{{pred}}}}="
        f"{ex['delta_pred']:.4f}$",
        fontsize=10,
        y=1.01,
    )
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate KARMA Level 1, 2, 4 figures from JSON results."
    )
    parser.add_argument(
        "/workspaces/KARMA-/results/beijing_pm25/tcn/karma_results.json",
        help="Path to KARMA JSON results file",
    )
    parser.add_argument("--outdir", default=".", help="Output directory")
    parser.add_argument(
        "--lam", type=float, default=None, help="Override lambda from config"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Max edges to show in Level 4 (default 20)",
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png", "both"],
        help="Output format (default: pdf)",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also produce a single combined three-panel figure",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = load_karma_results(args.results)

    figs = {
        "karma_level1_importance": plot_level1(results, lam=args.lam),
        "karma_level2_lagprofiles": plot_level2(results, lam=args.lam),
        "karma_level4_causal": plot_level4(results, lam=args.lam, top_n=args.top_n),
    }
    if args.combined:
        figs["karma_combined"] = plot_all(results, lam=args.lam, top_n=args.top_n)

    exts = ["pdf", "png"] if args.format == "both" else [args.format]
    for name, fig in figs.items():
        for ext in exts:
            path = outdir / f"{name}.{ext}"
            fig.savefig(path)
            print(f"  saved → {path}")
    print("Done.")


if __name__ == "__main__":
    main()
