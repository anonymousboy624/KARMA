#!/usr/bin/env python
"""
Ablation: KARMA edge-recovery sensitivity to the discretiser bin count N.

For each VAR configuration and N value the KARMA pipeline is run once
(discretiser fit, K* selection, SuffixPool, TreeKernelEstimator). The
regularisation threshold is fixed while N is varied.

Each run reports KARMA edge recovery and Kendall's τ against the known VAR
ground truth.

Outputs
-------
    <output_dir>/ablation_N_<config>.json   — per-config raw results
    <output_dir>/ablation_N_summary.json    — combined
  stdout tables                                — F1, Precision, Recall, τ

Usage
-----
    python -m experiments.ablation_N
    python -m experiments.ablation_N --configs tiny small medium
    python -m experiments.ablation_N --ns 2 3 4 5 --lam 0.025
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

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

DEFAULT_NS = [2, 3, 4, 5]
DEFAULT_LAM = 0.1
DEFAULT_EXCHANGE_NS = [2, 3, 4, 5]



def karma_setup(
    cfg: VARConfig,
    X_train: np.ndarray,
    X_val: np.ndarray,
    A: np.ndarray,
    seed: int,
    verbose: bool,
) -> dict:
    """Run the KARMA setup for one discretiser bin count.

    Args:
        cfg (VARConfig): VAR configuration
        X_train (np.ndarray): training data
        X_val (np.ndarray): validation data
        A (np.ndarray): adjacency matrix
        seed (int): random seed
        verbose (bool): whether to print verbose output

    Returns:
        dict: _description_
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
    ns: list[int],
    args,
) -> dict:
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

    karma_records = []
    for n_bins in ns:
        cfg_n = replace(cfg, N=n_bins)
        cfg_n.N = n_bins
        print(f"\n  [KARMA] N={n_bins}: setting up (disc, K*, pool, tree) …")
        setup = karma_setup(
            cfg_n, X_train, X_test, A, seed=args.seed, verbose=args.verbose
        )
        K_star = setup["K_star"]
        vi = compute_variable_importance(
            setup["tree"], setup["pi_star"], setup["disc"], lam=args.lam
        )
        phi = np.zeros((cfg.D, cfg.K_max), dtype=np.float32)
        phi[:, :K_star] = vi["phi"]
        em = edge_metrics(vi["edges"], true_edges, cfg.D, K_star)
        tau, pval = tau_vs_gt(phi, phi_gt)
        record = {
            "N": n_bins,
            "lam": args.lam,
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
            f"    N={n_bins:<8d}  P={em['precision']:.3f}  R={em['recall']:.3f}"
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
        },
        "true_edges": [list(e) for e in true_edges],
        "ns": ns,
        "lam": args.lam,
        "karma": karma_records,
    }


def run_exchange_rate_tcn(
    ns: list[int],
    args,
    output_dir: Path,
) -> dict:
    """Measure real-data TCN lag_AUC as the discretiser N changes."""
    from experiments.comparison_realdata import (
        DATASETS,
        fit_var_imputer,
        karma_edges_to_time_scores,
        load_dataset,
        load_pretrained_tcn,
        run_karma,
        select_var_order,
        timestep_removal_auc,
    )

    cfg = DATASETS["exchange_rate"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ckpt_dir = Path(args.model_ckpt_dir)
    print(f"\n{'═'*72}")
    print(
        f"  EXCHANGE_RATE / TCN   D={cfg.D}  W={cfg.T}  horizon={cfg.pred_horizon}"
        f"  device={device}"
    )
    print(f"{'═'*72}")

    X_train, X_val, X_test, _, _, _ = load_dataset(cfg)
    X_train_dt = X_train.transpose(0, 2, 1).astype(np.float32)
    X_test_dt = X_test[: args.exchange_n_test].transpose(0, 2, 1).astype(np.float32)
    x_train_mean = X_train_dt.mean(axis=0)
    var_K = select_var_order(X_train_dt, max_K=8)
    var_imputer = fit_var_imputer(X_train_dt, var_K)
    tcn = load_pretrained_tcn(cfg, device, model_ckpt_dir)

    records = []
    for n_bins in ns:
        cfg_n = replace(cfg, N=n_bins)
        print(f"\n  [KARMA / TCN] N={n_bins}: fitting discretiser and kernel …")
        edges, K_star, b_star = run_karma(
            X_train,
            X_val,
            tcn,
            cfg_n,
            device,
            verbose=args.verbose,
            seed=args.seed,
        )
        time_scores = karma_edges_to_time_scores(edges, cfg.T)
        lag_auc, lag_changes = timestep_removal_auc(
            X_test_dt,
            time_scores,
            tcn,
            x_train_mean,
            device,
            var_imputer=var_imputer,
            var_K=var_K,
            karma_b_star=b_star,
        )
        record = {
            "N": n_bins,
            "K_star": K_star,
            "n_edges": len(edges),
            "lag_AUC": lag_auc,
            "lag_changes": lag_changes,
        }
        records.append(record)
        print(
            f"    N={n_bins:<3d}  K*={K_star:<3d}  edges={len(edges):<4d}"
            f"  lag_AUC={lag_auc:.6f}"
        )

    result = {
        "dataset": "exchange_rate",
        "architecture": "tcn",
        "lam": cfg.lam,
        "var_K": var_K,
        "n_test": len(X_test_dt),
        "records": records,
    }
    out_file = output_dir / "ablation_N_exchange_rate_tcn.json"
    with open(out_file, "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    print(f"\nExchange-rate TCN results → {out_file}")
    return result


def print_exchange_rate_summary(result: dict) -> None:
    print(f"\n{'═'*56}")
    print("  Exchange-rate TCN: lag_AUC vs discretiser N")
    print(f"{'═'*56}")
    print("    N   K*   edges   lag_AUC")
    print("  " + "-" * 34)
    for record in result["records"]:
        print(
            f"  {record['N']:>3d}  {record['K_star']:>2d}"
            f"  {record['n_edges']:>5d}  {record['lag_AUC']:>8.6f}"
        )


def _n_header(ns: list[int], lw: int = 9) -> str:
    return "".join(f"  N={n:<{lw-3}d}" for n in ns)


def print_metric_table(
    all_results: list[dict],
    metric: str,
    label: str,
    ns: list[int],
    mark_best: bool = True,
) -> None:
    lw = 9
    hdr = (
        f"\n{'═'*80}\n  {label}\n{'═'*80}\n"
        f"  {'config':>8s}  {'D':>3s}  {'K':>3s}  {'T_train':>8s}" + _n_header(ns, lw)
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
    print("  * best N for this config")


def print_tau_comparison(all_results: list[dict], ns: list[int]) -> None:
    lw = 9
    print(f"\n{'═'*80}")
    print("  Kendall's τ vs G*  (* p < 0.05 for KARMA)")
    print(f"{'═'*80}")
    print(f"  {'config':>8s}  {'D':>3s}  {'K':>3s}" + _n_header(ns, lw))
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


def print_summary(all_results: list[dict], ns: list[int]) -> None:
    print_metric_table(all_results, "f1", "KARMA  F1  (higher = better, * best N)", ns)
    print_metric_table(all_results, "precision", "KARMA  Precision  (* best N)", ns)
    print_metric_table(all_results, "recall", "KARMA  Recall  (* best N)", ns)
    print_tau_comparison(all_results, ns)


_CURVE_COLORS = {
    "precision": "#3b82f6",  # blue
    "recall": "#10b981",  # green
    "f1": "#f59e0b",  # amber
    "tau": "#8b5cf6",  # violet  (KARMA τ)
}


def plot_n_curves(
    all_results: list[dict],
    output_dir: Path,
    ns: list[int],
) -> None:
    """One figure per dataset: P/R/F1 and τ as functions of N."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    n_arr = np.array(ns)
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
            f"N ablation — {name.upper()}  (D={D}, K={K}, edges={n_edges})",
            fontsize=13,
            fontweight="bold",
        )

        ax_prf.plot(
            n_arr,
            precision,
            color=_CURVE_COLORS["precision"],
            lw=2,
            label="Precision",
        )
        ax_prf.plot(n_arr, recall, color=_CURVE_COLORS["recall"], lw=2, label="Recall")
        ax_prf.plot(n_arr, f1, color=_CURVE_COLORS["f1"], lw=2.5, label="F1", zorder=5)

        best_f1_idx = int(np.argmax(f1))
        ax_prf.axvline(
            n_arr[best_f1_idx], color=_CURVE_COLORS["f1"], lw=1, ls="--", alpha=0.5
        )
        ax_prf.scatter(
            [n_arr[best_f1_idx]],
            [f1[best_f1_idx]],
            color=_CURVE_COLORS["f1"],
            s=60,
            zorder=6,
        )

        ax_prf.set_xlabel("N", fontsize=11)
        ax_prf.set_ylabel("Score", fontsize=11)
        ax_prf.set_ylim(-0.05, 1.05)
        ax_prf.set_title("Edge recovery: Precision / Recall / F1")
        ax_prf.legend(loc="best", fontsize=9)
        ax_prf.grid(True, alpha=0.3)

        ax_tau.plot(n_arr, tau, color=_CURVE_COLORS["tau"], lw=2.5, label="KARMA τ")
        ax_tau.fill_between(
            n_arr,
            tau,
            where=tau_sig,
            color=_CURVE_COLORS["tau"],
            alpha=0.15,
            label="p<0.05",
        )

        ax_tau.axhline(0, color="#94a3b8", lw=0.8, ls="-")
        ax_tau.set_xlabel("N", fontsize=11)
        ax_tau.set_ylabel("Kendall's τ", fontsize=11)
        ax_tau.set_title("Rank correlation vs ground truth (τ)")
        ax_tau.legend(loc="best", fontsize=9)
        ax_tau.grid(True, alpha=0.3)

        # save
        out = output_dir / f"N_curves_{name}.pdf"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  plot → {out}")

    n = len(all_results)
    if n < 2:
        return

    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n), constrained_layout=True)
    fig.suptitle("N ablation — all configs", fontsize=14, fontweight="bold")

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
            n_arr, precision, color=_CURVE_COLORS["precision"], lw=1.8, label="P"
        )
        ax_prf.plot(n_arr, recall, color=_CURVE_COLORS["recall"], lw=1.8, label="R")
        ax_prf.plot(n_arr, f1, color=_CURVE_COLORS["f1"], lw=2.2, label="F1")
        ax_prf.set_ylabel(name, fontsize=10, fontweight="bold")
        ax_prf.set_ylim(-0.05, 1.05)
        ax_prf.grid(True, alpha=0.3)
        if row_idx == 0:
            ax_prf.set_title("Precision / Recall / F1")
            ax_prf.legend(loc="upper right", fontsize=8)

        ax_tau.plot(n_arr, tau, color=_CURVE_COLORS["tau"], lw=2.0, label="KARMA")
        ax_tau.fill_between(
            n_arr, tau, where=tau_sig, color=_CURVE_COLORS["tau"], alpha=0.15
        )
        ax_tau.axhline(0, color="#94a3b8", lw=0.8)
        ax_tau.grid(True, alpha=0.3)
        if row_idx == 0:
            ax_tau.set_title("Kendall's τ")
            ax_tau.legend(loc="best", fontsize=8)
        if row_idx == n - 1:
            ax_prf.set_xlabel("N", fontsize=10)
            ax_tau.set_xlabel("N", fontsize=10)

    combined_out = output_dir / "N_curves_all.pdf"
    fig.savefig(combined_out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  combined plot → {combined_out}")


def plot_tau_curves(
    all_results: list[dict],
    output_dir: Path,
    ns: list[int],
) -> None:
    """Dedicated N vs Kendall's τ figure for KARMA."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping tau plots")
        return

    n_arr = np.array(ns)
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
    fig.suptitle("N vs Kendall's τ  (KARMA)", fontsize=13, fontweight="bold")

    for idx, res in enumerate(all_results):
        ax = axes[idx // ncols][idx % ncols]
        ci = res["config"]
        karma = res["karma"]

        tau = np.nan_to_num(np.array([r["tau"] for r in karma], dtype=float), nan=0.0)
        tau_sig = np.array([r["tau_p"] < 0.05 for r in karma])

        ax.plot(n_arr, tau, color=_CURVE_COLORS["tau"], lw=2.5, label="KARMA", zorder=4)
        ax.fill_between(
            n_arr,
            tau,
            where=tau_sig,
            color=_CURVE_COLORS["tau"],
            alpha=0.18,
            label="p<0.05",
            zorder=3,
        )

        best_idx = int(np.argmax(tau))
        ax.scatter(
            [n_arr[best_idx]],
            [tau[best_idx]],
            color=_CURVE_COLORS["tau"],
            s=55,
            zorder=5,
        )
        ax.annotate(
            f"N*={n_arr[best_idx]:.4g}",
            xy=(n_arr[best_idx], tau[best_idx]),
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
        ax.set_xlabel("N", fontsize=10)
        ax.set_ylabel("Kendall's τ", fontsize=10)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    # hide unused axes in the last row
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    out = output_dir / "N_tau_all.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  tau plot → {out}")



def parse_args():
    p = argparse.ArgumentParser(
        description="Ablation: KARMA edge recovery vs discretiser bin count N"
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=list(ALL_CONFIGS.keys()),
        choices=list(ALL_CONFIGS.keys()),
    )
    p.add_argument(
        "--ns",
        nargs="+",
        type=int,
        default=DEFAULT_NS,
        metavar="N",
        help="Discretiser bin counts to evaluate",
    )
    p.add_argument("--lam", type=float, default=DEFAULT_LAM)
    p.add_argument(
        "--exchange_rate",
        action="store_true",
        help="Run the real exchange-rate TCN lag_AUC N sweep",
    )
    p.add_argument(
        "--exchange_ns",
        nargs="+",
        type=int,
        default=DEFAULT_EXCHANGE_NS,
        metavar="N",
        help="Discretiser bin counts for the exchange-rate TCN sweep",
    )
    p.add_argument("--exchange_n_test", type=int, default=200)
    p.add_argument(
        "--model_ckpt_dir",
        default="outputs/checkpoints",
        help="Root directory containing the exchange-rate TCN checkpoint",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="results/ablation_N")
    p.add_argument("--no_plot", action="store_true", help="Skip matplotlib plots")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ns = sorted(set(args.ns))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Configs : {args.configs}")
    print(f"N values: {ns}")
    print(f"Lambda  : {args.lam}")

    all_results = []
    for name in args.configs:
        cfg = ALL_CONFIGS[name]
        result = run_ablation_config(cfg, ns, args)
        all_results.append(result)

        with open(output_dir / f"ablation_N_{name}.json", "w") as fh:
            json.dump(result, fh, indent=2, default=float)

    print_summary(all_results, ns)

    with open(output_dir / "ablation_N_summary.json", "w") as fh:
        json.dump(all_results, fh, indent=2, default=float)

    if not args.no_plot:
        print("\nGenerating plots ...")
        plot_n_curves(all_results, output_dir, ns)
        plot_tau_curves(all_results, output_dir, ns)

    if args.exchange_rate:
        exchange_result = run_exchange_rate_tcn(
            sorted(set(args.exchange_ns)), args, output_dir
        )
        print_exchange_rate_summary(exchange_result)

    print(f"\nResults → {output_dir}/")


if __name__ == "__main__":
    main()
