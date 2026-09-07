#!/usr/bin/env python
"""
Ablation: KARMA edge-recovery sensitivity to the regularisation parameter λ.

For each VAR configuration the expensive KARMA pipeline is run ONCE
(discretiser fit, K* selection, SuffixPool, TreeKernelEstimator).
Only the final `compute_variable_importance(tree, …, lam=λ)` call is
repeated across the lambda grid — making the sweep cheap.

Each run reports KARMA edge recovery and Kendall's τ against the known VAR
ground truth.

Outputs
-------
  <output_dir>/ablation_lambda_<config>.json   — per-config raw results
  <output_dir>/ablation_lambda_summary.json    — combined
  stdout tables                                — F1, Precision, Recall, τ

Usage
-----
  python -m experiments.ablation_lambda
  python -m experiments.ablation_lambda --configs tiny small medium
  python -m experiments.ablation_lambda --lambdas 0.001 0.01 0.1 0.5 1.0 5.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.comparison_var_multi import (
    ALL_CONFIGS,
    VARConfig,
    generate_stationary_var,
    ground_truth_phi,
    make_numpy_oracle,
    tau_vs_gt,
)

from dataset.var import edge_metrics, simulate_var
from karma.causal_recovery.edge_contribution import compute_variable_importance
from karma.markov_approximation.markov_surrogacy import select_K_and_baseline
from karma.utils.discretiser import Discretiser
from karma.utils.kernel_estimator import TreeKernelEstimator
from karma.utils.sampling import SuffixPool

DEFAULT_LAMBDAS = np.linspace(0, 0.20, 100).tolist()


def karma_setup(
    cfg: VARConfig,
    X_train: np.ndarray,
    X_val: np.ndarray,
    A: np.ndarray,
    seed: int,
    verbose: bool,
) -> dict:
    """Run the lambda-independent part of KARMA on

    Args:
        cfg (VARConfig): Var configuration
        X_train (np.ndarray): training data
        X_val (np.ndarray): validation data
        A (np.ndarray): ground truth adjacency matrix
        seed (int): random seed
        verbose (bool): whether to print verbose output

    Returns:
        dict: a dict with everything needed to cheaply call
    `compute_variable_importance` for any lambda value.
    """    
    f_oracle = make_numpy_oracle(A)
    disc = Discretiser(N=cfg.N)
    disc.fit(X_train)

    T_val = len(X_val)
    X_val_w = np.stack([X_val[t : t + cfg.W] for t in range(T_val - cfg.W)])

    result = select_K_and_baseline(
        f=f_oracle,
        X_train=X_train,
        X_val=X_val_w,
        disc=disc,
        W=cfg.W,
        eps=1e-5,
        K_max=cfg.K_max,
        loss="regression",
        verbose=verbose,
    )
    K_star = result["K_star"]
    b_star = result["b_star"]
    pi_star = result["pi_star"]

    if verbose:
        print(f"    K*={K_star}  Δ_pred={result['delta_pred']:.4f}")

    pool = SuffixPool(disc=disc, K=K_star, W=cfg.W)
    pool.build(X_train)

    tree = TreeKernelEstimator(
        disc,
        K=K_star,
        W=cfg.W,
        M=cfg.M,
        n_pool=cfg.n_pool_min,
        pool=pool,
        b_star=b_star,
    )
    tree.fit(f=f_oracle, pi_star=pi_star, verbose=verbose)

    return {
        "tree": tree,
        "pi_star": pi_star,
        "disc": disc,
        "K_star": K_star,
        "f_oracle": f_oracle,
    }


def run_ablation_config(
    cfg: VARConfig,
    lambdas: list[float],
    args,
) -> dict:
    """Run lambda ablation for given config

    Args:
        cfg (VARConfig): VAR configuration
        lambdas (list[float]): lambda values to sweep
        args (_type_): experiment arguments (seed, verbose, etc.)

    Returns:
        dict: experiment results for this config, including KARMA metrics for each lambda
    """    
    print(f"\n{'═'*72}")
    print(
        f"  {cfg.name.upper()}   D={cfg.D}  K={cfg.K}  edges={cfg.n_edges}"
        f"  T_train={cfg.T_train:,}  W={cfg.W}  N={cfg.N}"
    )
    print(f"{'═'*72}")

    rng = np.random.default_rng(cfg.seed)
    A, true_edges = generate_stationary_var(cfg)

    X = simulate_var(A, cfg.D, cfg.K, cfg.T_train + cfg.T_test, cfg.noise_std, rng)
    X_train, X_test = X[: cfg.T_train], X[cfg.T_train :]

    phi_gt = ground_truth_phi(A, cfg.K_max)

    print("\n  [KARMA] Setting up (disc, K*, pool, tree) …")
    setup = karma_setup(cfg, X_train, X_test, A, seed=args.seed, verbose=args.verbose)
    K_star = setup["K_star"]
    print(f"    K*={K_star}  Sweeping {len(lambdas)} λ values …")

    karma_records = []
    for lam in lambdas:
        vi = compute_variable_importance(
            setup["tree"], setup["pi_star"], setup["disc"], lam=lam
        )
        phi = np.zeros((cfg.D, cfg.K_max), dtype=np.float32)
        phi[:, :K_star] = vi["phi"]
        em = edge_metrics(vi["edges"], true_edges, cfg.D, K_star)
        tau, pval = tau_vs_gt(phi, phi_gt)
        record = {
            "lam": lam,
            "K_star": K_star,
            "n_edges": len(vi["edges"]),
            "precision": em["precision"],
            "recall": em["recall"],
            "f1": em["f1"],
            "tau": tau,
            "tau_p": pval,
        }
        karma_records.append(record)
        print(
            f"    λ={lam:<8g}  P={em['precision']:.3f}  R={em['recall']:.3f}"
            f"  F1={em['f1']:.3f}  τ={tau:+.3f}{'*' if pval < 0.05 else ' '}"
            f"  edges={len(vi['edges'])}"
        )

    return {
        "config": {
            "name": cfg.name,
            "D": cfg.D,
            "K": cfg.K,
            "n_edges": cfg.n_edges,
            "T_train": cfg.T_train,
            "K_star": K_star,
        },
        "true_edges": [list(e) for e in true_edges],
        "lambdas": lambdas,
        "karma": karma_records,
    }


def _lam_header(lambdas: list[float], lw: int = 9) -> str:
    return "".join(f"  λ={lam:<{lw-3}g}" for lam in lambdas)


def print_metric_table(
    all_results: list[dict],
    metric: str,
    label: str,
    lambdas: list[float],
    mark_best: bool = True,
) -> None:
    lw = 9
    hdr = (
        f"\n{'═'*80}\n  {label}\n{'═'*80}\n"
        f"  {'config':>8s}  {'D':>3s}  {'K':>3s}  {'T_train':>8s}"
        + _lam_header(lambdas, lw)
    )
    print(hdr)
    print(f"  {'-'*76}")

    for res in all_results:
        ci = res["config"]
        vals = [r[metric] for r in res["karma"]]
        best_idx = (
            int(np.argmax(vals)) if metric != "complexity" else int(np.argmin(vals))
        )
        row = (
            f"  {ci['name']:>8s}  {ci['D']:>3d}  {ci['K']:>3d}"
            f"  {ci['T_train']:>8,d}"
        )
        for i, v in enumerate(vals):
            mark = "*" if mark_best and i == best_idx else " "
            row += f"  {v:>{lw-2}.3f}{mark}"
        print(row)

    print(f"{'═'*80}")
    print("  * best λ for this config")


def print_tau_comparison(all_results: list[dict], lambdas: list[float]) -> None:
    lw = 9
    print(f"\n{'═'*80}")
    print("  Kendall's τ vs G*  (* p < 0.05 for KARMA)")
    print(f"{'═'*80}")
    print(f"  {'config':>8s}  {'D':>3s}  {'K':>3s}" + _lam_header(lambdas, lw))
    print(f"  {'-'*76}")
    for res in all_results:
        ci = res["config"]
        karma_taus = [r["tau"] for r in res["karma"]]
        karma_ps = [r["tau_p"] for r in res["karma"]]
        row = f"  {ci['name']:>8s}  {ci['D']:>3d}  {ci['K']:>3d}"
        for tau, pval in zip(karma_taus, karma_ps):
            mark = "*" if pval < 0.05 else " "
            row += f"  {tau:>+{lw-2}.3f}{mark}"
        print(row)
    print(f"{'═'*80}")


def print_summary(all_results: list[dict], lambdas: list[float]) -> None:
    print_metric_table(
        all_results, "f1", "KARMA  F1  (higher = better, * best λ)", lambdas
    )
    print_metric_table(
        all_results, "precision", "KARMA  Precision  (* best λ)", lambdas
    )
    print_metric_table(all_results, "recall", "KARMA  Recall  (* best λ)", lambdas)
    print_tau_comparison(all_results, lambdas)


_CURVE_COLORS = {
    "precision": "#3b82f6",  # blue
    "recall": "#10b981",  # green
    "f1": "#f59e0b",  # amber
    "tau": "#8b5cf6",  # violet  (KARMA τ)
}


def plot_lambda_curves(
    all_results: list[dict],
    output_dir: Path,
    lambdas: list[float],
) -> None:
    """One figure per dataset: left — P/R/F1 vs λ; right — τ vs λ + baseline refs."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    lam_arr = np.array(lambdas)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _safe(key, records):
        """Extract a float array and replace any NaN with 0."""
        return np.nan_to_num(np.array([r[key] for r in records], dtype=float), nan=0.0)

    for res in all_results:
        name = res["config"]["name"]
        D, K, n_edges = res["config"]["D"], res["config"]["K"], res["config"]["n_edges"]
        karma = res["karma"]

        precision = _safe("precision", karma)
        recall = _safe("recall", karma)
        f1 = _safe("f1", karma)
        tau = _safe("tau", karma)
        tau_sig = np.array([r["tau_p"] < 0.05 for r in karma])

        fig, (ax_prf, ax_tau) = plt.subplots(
            1, 2, figsize=(12, 4), constrained_layout=True
        )
        fig.suptitle(
            f"λ ablation — {name.upper()}  (D={D}, K={K}, edges={n_edges})",
            fontsize=13,
            fontweight="bold",
        )

        ax_prf.plot(
            lam_arr,
            precision,
            color=_CURVE_COLORS["precision"],
            lw=2,
            label="Precision",
        )
        ax_prf.plot(
            lam_arr, recall, color=_CURVE_COLORS["recall"], lw=2, label="Recall"
        )
        ax_prf.plot(
            lam_arr, f1, color=_CURVE_COLORS["f1"], lw=2.5, label="F1", zorder=5
        )

        best_f1_idx = int(np.argmax(f1))
        ax_prf.axvline(
            lam_arr[best_f1_idx], color=_CURVE_COLORS["f1"], lw=1, ls="--", alpha=0.5
        )
        ax_prf.scatter(
            [lam_arr[best_f1_idx]],
            [f1[best_f1_idx]],
            color=_CURVE_COLORS["f1"],
            s=60,
            zorder=6,
        )

        ax_prf.set_xlabel("λ", fontsize=11)
        ax_prf.set_ylabel("Score", fontsize=11)
        ax_prf.set_ylim(-0.05, 1.05)
        ax_prf.set_title("Edge recovery: Precision / Recall / F1")
        ax_prf.legend(loc="best", fontsize=9)
        ax_prf.grid(True, alpha=0.3)

        ax_tau.plot(lam_arr, tau, color=_CURVE_COLORS["tau"], lw=2.5, label="KARMA τ")
        # shade regions where τ is significant
        ax_tau.fill_between(
            lam_arr,
            tau,
            where=tau_sig,
            color=_CURVE_COLORS["tau"],
            alpha=0.15,
            label="p<0.05",
        )

        ax_tau.axhline(0, color="#94a3b8", lw=0.8, ls="-")
        ax_tau.set_xlabel("λ", fontsize=11)
        ax_tau.set_ylabel("Kendall's τ", fontsize=11)
        ax_tau.set_title("Rank correlation vs ground truth (τ)")
        ax_tau.legend(loc="best", fontsize=9)
        ax_tau.grid(True, alpha=0.3)

        # save
        out = output_dir / f"lambda_curves_{name}.pdf"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  plot → {out}")
    n = len(all_results)
    if n < 2:
        return

    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n), constrained_layout=True)
    fig.suptitle("λ ablation — all configs", fontsize=14, fontweight="bold")

    for row_idx, res in enumerate(all_results):
        name = res["config"]["name"]
        karma = res["karma"]
        precision = _safe("precision", karma)
        recall = _safe("recall", karma)
        f1 = _safe("f1", karma)
        tau = _safe("tau", karma)
        tau_sig = np.array([r["tau_p"] < 0.05 for r in karma])

        ax_prf, ax_tau = axes[row_idx]

        ax_prf.plot(
            lam_arr, precision, color=_CURVE_COLORS["precision"], lw=1.8, label="P"
        )
        ax_prf.plot(lam_arr, recall, color=_CURVE_COLORS["recall"], lw=1.8, label="R")
        ax_prf.plot(lam_arr, f1, color=_CURVE_COLORS["f1"], lw=2.2, label="F1")
        ax_prf.set_ylabel(name, fontsize=10, fontweight="bold")
        ax_prf.set_ylim(-0.05, 1.05)
        ax_prf.grid(True, alpha=0.3)
        if row_idx == 0:
            ax_prf.set_title("Precision / Recall / F1")
            ax_prf.legend(loc="upper right", fontsize=8)

        ax_tau.plot(lam_arr, tau, color=_CURVE_COLORS["tau"], lw=2.0, label="KARMA")
        ax_tau.fill_between(
            lam_arr, tau, where=tau_sig, color=_CURVE_COLORS["tau"], alpha=0.15
        )
        ax_tau.axhline(0, color="#94a3b8", lw=0.8)
        ax_tau.grid(True, alpha=0.3)
        if row_idx == 0:
            ax_tau.set_title("Kendall's τ")
            ax_tau.legend(loc="best", fontsize=8)
        if row_idx == n - 1:
            ax_prf.set_xlabel("λ", fontsize=10)
            ax_tau.set_xlabel("λ", fontsize=10)

    combined_out = output_dir / "lambda_curves_all.pdf"
    fig.savefig(combined_out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  combined plot → {combined_out}")


def plot_tau_curves(
    all_results: list[dict],
    output_dir: Path,
    lambdas: list[float],
) -> None:
    """Dedicated λ vs Kendall's τ figure for KARMA."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping tau plots")
        return

    lam_arr = np.array(lambdas)
    n = len(all_results)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 4 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle("λ vs Kendall's τ  (KARMA)", fontsize=13, fontweight="bold")

    for idx, res in enumerate(all_results):
        ax = axes[idx // ncols][idx % ncols]
        ci = res["config"]
        karma = res["karma"]

        tau = np.nan_to_num(np.array([r["tau"] for r in karma], dtype=float), nan=0.0)
        tau_sig = np.array([r["tau_p"] < 0.05 for r in karma])

        ax.plot(
            lam_arr, tau, color=_CURVE_COLORS["tau"], lw=2.5, label="KARMA", zorder=4
        )
        ax.fill_between(
            lam_arr,
            tau,
            where=tau_sig,
            color=_CURVE_COLORS["tau"],
            alpha=0.18,
            label="p<0.05",
            zorder=3,
        )
        best_idx = int(np.argmax(tau))
        ax.scatter(
            [lam_arr[best_idx]],
            [tau[best_idx]],
            color=_CURVE_COLORS["tau"],
            s=55,
            zorder=5,
        )
        ax.annotate(
            f"λ*={lam_arr[best_idx]:.4g}",
            xy=(lam_arr[best_idx], tau[best_idx]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7.5,
            color=_CURVE_COLORS["tau"],
        )

        ax.axhline(0, color="#94a3b8", lw=0.8, ls="-", zorder=1)
        ax.set_title(
            f"{ci['name'].upper()}  D={ci['D']} K={ci['K']}",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("λ", fontsize=10)
        ax.set_ylabel("Kendall's τ", fontsize=10)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    out = output_dir / "lambda_tau_all.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  tau plot → {out}")

def parse_args():
    p = argparse.ArgumentParser(
        description="Ablation: KARMA edge recovery vs regularisation λ"
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=list(ALL_CONFIGS.keys()),
        choices=list(ALL_CONFIGS.keys()),
    )
    p.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=DEFAULT_LAMBDAS,
        metavar="λ",
        help="Lambda values to sweep (default: 9-point grid from 0.001 to 10)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="results/ablation_lambda")
    p.add_argument("--no_plot", action="store_true", help="Skip matplotlib plots")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    lambdas = sorted(set(args.lambdas))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Configs : {args.configs}")
    print(f"Lambdas : {lambdas}")

    all_results = []
    for name in args.configs:
        cfg = ALL_CONFIGS[name]
        result = run_ablation_config(cfg, lambdas, args)
        all_results.append(result)

        with open(output_dir / f"ablation_lambda_{name}.json", "w") as fh:
            json.dump(result, fh, indent=2, default=float)

    print_summary(all_results, lambdas)

    with open(output_dir / "ablation_lambda_summary.json", "w") as fh:
        json.dump(all_results, fh, indent=2, default=float)

    if not args.no_plot:
        print("\nGenerating plots ...")
        plot_lambda_curves(all_results, output_dir, lambdas)
        plot_tau_curves(all_results, output_dir, lambdas)

    print(f"\nResults → {output_dir}/")


if __name__ == "__main__":
    main()
