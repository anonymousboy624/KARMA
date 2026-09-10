#!/usr/bin/env python
"""

n-fold time-series cross-validation variant of comparison_realdata.py.

Each dataset's test set is partitioned into N_FOLDS consecutive
(non-overlapping) chunks.  All attribution methods run independently on
each fold; metrics are then reported as mean ± std across folds.

Models (GRU, LSTM, TCN, Transformer, Random Forest) are NOT
retrained per fold. Methods that learn a mask per batch (DynaMask, WinIT,
ExtremalMask) re-train on each fold's samples.  KARMA runs once on the full
training set and its fixed edge scores are evaluated against every fold.
Random Forest(tree ensembles, not differentiable) skip IG, and TIMING regardless of --skip_* flags.

Usage
-----
  python -m experiments.comparison_realdata_cv --datasets ettm1 ettm2
  python -m experiments.comparison_realdata_cv --n_folds 3 --lag_only
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.comparison_realdata import (
    DATASETS,
    MAX_KARMA_D,
    NON_DIFFERENTIABLE_ARCHS,
    DatasetConfig,
    attribution_complexity,
    fit_var_imputer,
    get_or_train_gru,
    karma_edges_to_attr,
    karma_edges_to_time_scores,
    load_dataset,
    load_pretrained_lstm,
    load_pretrained_rf,
    load_pretrained_tcn,
    load_pretrained_transformer,
    run_ig,
    run_karma,
    run_shaptime,
    run_timeshap,
    run_timing,
    run_tsmule,
    select_var_order,
    set_global_seed,
    timestep_drop_at_frac,
    timestep_removal_auc,
)

N_FOLDS_DEFAULT = 5
N_TEST_MAX = 200
N_TEST_LARGE_D = 50


def _run_fold_for_model(
    arch: str,
    fold: int,
    model,
    x_fold: np.ndarray,  # (B_fold, D, T) — this fold's test windows
    cfg: DatasetConfig,
    args,
    x_train_mean: np.ndarray,
    device: torch.device,
    var_imputer,
    var_K: int,
    karma_attr: np.ndarray | None,  # pre-computed, shape (D, T) or None
    karma_ts: np.ndarray | None,  # pre-computed, shape (T,) or None
    karma_b_star: np.ndarray | None = None,  # pre-computed, shape (T-K*, D) or None
) -> dict:
    """Run all attribution methods on one fold; return a metrics dict.

    Args:
        arch (_type_): model architecture name (gru, lstm, tcn, transformer, rf)
        fold (int): fold index (1-based)
        model (_type_): trained model object (PyTorch or sklearn)
        x_fold (np.ndarray): this fold's test windows, shape (B_fold, D, T)
        cfg (DatasetConfig): dataset configuration
        args (_type_): experiment arguments
        x_train_mean (np.ndarray): mean of training data
        device (torch.device): device to run on (cpu or cuda)
        var_imputer (_type_): var conditional imputer object (fitted on training data)
        var_K (int): VAR K to fit for AUC lag
        karma_attr (np.ndarray | None): Pre-computed global KARMA attribution map, shape (D, T) or None
        karma_ts (np.ndarray | None): Pre-computed global KARMA time scores, shape (T,) or None
        karma_b_star (np.ndarray | None): Pre-computed global KARMA b* values, shape (T-K*, D) or None

    Returns:
        dict: All results for this fold, keyed by method and metric
    """
    r: dict = {}
    p = f"{arch}_"

    def _lag_auc(ts):
        return timestep_removal_auc(
            x_fold,
            ts,
            model,
            x_train_mean,
            device,
            var_imputer=var_imputer,
            var_K=var_K,
            karma_b_star=karma_b_star,
        )

    def _lag_d25(ts):
        return timestep_drop_at_frac(
            x_fold,
            ts,
            model,
            x_train_mean,
            device,
            frac=0.25,
            var_imputer=var_imputer,
            var_K=var_K,
            karma_b_star=karma_b_star,
        )

    def _store_local(m, attr):
        comp = attribution_complexity(attr)
        attr = np.abs(attr).mean(axis=1)
        lag_auc, lag_d25 = 0, 0
        r[f"{p}{m}_complexity"] = comp
        msg = f"cplx={comp:.4f}"
        for _, a in enumerate(attr):
            lag_auc += _lag_auc(a)[0]
            lag_d25 += _lag_d25(a)
        lag_auc /= len(attr)
        lag_d25 /= len(attr)
        r.update(
            {
                f"{p}{m}_lag_auc": lag_auc,
                f"{p}{m}_lag_drop25": lag_d25,
            }
        )
        msg = f"    lag_AUC={lag_auc:.4f}  lag_drop@25%={lag_d25:.4f}"
        print(msg)

    is_differentiable = arch not in NON_DIFFERENTIABLE_ARCHS

    if is_differentiable and not args.skip_ig:
        print("\n  [IG]")
        attr = run_ig(x_fold, model, device, tau=args.tau)
        _store_local("ig", attr)

    if not args.skip_timeshap:
        print("\n  [TimeShap]")
        try:
            attr = run_timeshap(
                x_fold, model, device, nsamples=args.ts_nsamples, tau=args.tau
            )
            _store_local("timeshap", attr)
        except Exception as e:
            print(f"    [TimeShap] skipped: {e}")

    if not args.skip_karma and cfg.D <= MAX_KARMA_D and karma_attr is not None:
        print("\n  [KARMA]")
        attr_k = np.broadcast_to(karma_attr[np.newaxis], x_fold.shape).copy()
        comp = attribution_complexity(attr_k)
        r[f"{p}karma_complexity"] = comp
        msg = f"cplx={comp:.4f}"
        if not args.complexity_only:
            lag_auc, _ = _lag_auc(karma_ts)
            lag_d25 = _lag_d25(karma_ts)
            r[f"{p}karma_lag_auc"] = lag_auc
            r[f"{p}karma_lag_drop25"] = lag_d25
            msg = f"lag_AUC={lag_auc:.4f}  lag_drop@25%={lag_d25:.4f}  " + msg

        print(f"    [fold {fold}] karma: {msg}")
    elif cfg.D > MAX_KARMA_D:
        r[f"{p}karma_lag_auc"] = None
        r[f"{p}karma_auc"] = None

    if is_differentiable and not args.skip_timing:
        print("\n  [TIMING]")
        try:
            attr = run_timing(x_fold, model, device, tau=args.tau)
            _store_local("timing", attr)
        except Exception as e:
            print(f"    [fold {fold}] TIMING failed: {e}")

    if not args.skip_tsmule:
        print("\n  [TS-MuLe]")
        try:
            attr, _ = run_tsmule(
                x_fold, model, device, n_samples=args.tsmule_nsamples, tau=args.tau
            )
            _store_local("tsmule", attr)
        except Exception as e:
            print(f"    [fold {fold}] TS-MuLe failed: {e}")

    if not args.skip_shaptime:
        print("\n  [ShapTime]")
        try:
            attr, _ = run_shaptime(
                x_fold, model, device, Tn=args.shaptime_tn, tau=args.tau
            )
            _store_local("shaptime", attr)
        except Exception as e:
            print(f"    [fold {fold}] ShapTime failed: {e}")

    return r


def _aggregate_folds(fold_results: list[dict]) -> dict:
    """Given a list of per-fold metric dicts, return {key_mean, key_std, key_folds}."""
    from collections import defaultdict

    buckets: dict[str, list] = defaultdict(list)
    for fd in fold_results:
        for k, v in fd.items():
            if v is not None:
                buckets[k].append(float(v))

    agg: dict = {}
    for k, vals in buckets.items():
        arr = np.array(vals)
        agg[f"{k}_mean"] = float(arr.mean())
        agg[f"{k}_std"] = float(arr.std(ddof=0))
        agg[f"{k}_folds"] = vals
    return agg


def _run_cv_for_model(
    arch: str,
    model,
    X_test_dt: np.ndarray,
    X_tr_dt: np.ndarray,
    X_val_dt: np.ndarray,
    X_train: np.ndarray,
    X_val: np.ndarray,
    cfg: DatasetConfig,
    args,
    ckpt_dir: Path,
    x_train_mean: np.ndarray,
    device: torch.device,
    var_imputer,
    var_K: int,
    n_folds: int,
) -> dict:
    """Run N-fold CV for a single arch; return aggregated metrics.

    Args:
        arch (str): model architecture name (gru, lstm, tcn, transformer, rf)
        model (_type_): trained model object (PyTorch or sklearn)
        X_test_dt (np.ndarray): test dataloader, shape (B_test, D, T)
        X_tr_dt (np.ndarray): training dataloader, shape (B_train, D, T)
        X_val_dt (np.ndarray): validation dataloader, shape (B_val, D, T)
        X_train (np.ndarray): training data, shape (B_train, D, T)
        X_val (np.ndarray): validation data, shape (B_val, D, T)
        cfg (DatasetConfig): dataset configuration
        args (_type_): command-line arguments
        ckpt_dir (Path): checkpoint directory for saving/loading models
        x_train_mean (np.ndarray): mean of training data
        device (torch.device): device for training
        var_imputer (_type_): imputer for handling missing values
        var_K (int): number of top features to select
        n_folds (int): number of folds for cross-validation

    Returns:
        dict: _description_
    """
    N = X_test_dt.shape[0]
    fold_size = N // n_folds

    karma_attr, karma_ts, _ = None, None, None
    if not args.skip_karma and cfg.D <= MAX_KARMA_D:
        print("\n  [KARMA] running on training data (once for all folds) …")
        try:
            edges, K_star, _ = run_karma(
                X_train,
                X_val,
                model,
                cfg,
                device,
                verbose=not args.quiet,
                seed=args.seed,
                tau=args.tau,
            )
            karma_attr = karma_edges_to_attr(edges, cfg.D, cfg.T)
            karma_ts = karma_edges_to_time_scores(edges, cfg.T)
            print(karma_ts)
            print(f"    K*={K_star}  edges={len(edges)}")
        except Exception as e:
            print(f"    KARMA failed: {e}")

    fold_results: list[dict] = []
    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else N
        x_fold = X_test_dt[start:end]
        print(f"\n  ── {arch} fold {fold + 1}/{n_folds}  samples [{start}:{end}] ──")

        fd = _run_fold_for_model(
            arch=arch,
            fold=fold + 1,
            model=model,
            x_fold=x_fold,
            cfg=cfg,
            args=args,
            x_train_mean=x_train_mean,
            device=device,
            var_imputer=var_imputer,
            var_K=var_K,
            karma_attr=karma_attr,
            karma_ts=karma_ts,
            karma_b_star=None,
        )
        fold_results.append(fd)

    agg = _aggregate_folds(fold_results)

    if not args.skip_karma and karma_attr is not None:
        cplx = float(attribution_complexity(karma_attr[np.newaxis]))
        key = f"{arch}_karma_complexity"
        agg[f"{key}_mean"] = cplx
        agg[f"{key}_std"] = 0.0
        agg[f"{key}_folds"] = [cplx] * n_folds
        print(
            f"\n  [KARMA complexity] {cplx:.4f} (fold-invariant, training-set attribution)"
        )

    return agg


def run_dataset_cv(
    name: str,
    args,
    device: torch.device,
    ckpt_dir: Path,
    out_dir: Path,
    model_ckpt_dir: Path,
) -> dict:
    """Run Cross Validation Experiment per Dataset

    Args:
        name (str): Dataset name (key in DATASETS)
        args (_type_): Command-line arguments
        device (torch.device): Device for training
        ckpt_dir (Path): Checkpoint directory for saving/loading models
        out_dir (Path): Output directory for saving results
        model_ckpt_dir (Path): Directory for pretrained model checkpoints

    Returns:
        dict: Cross-validation results
    """
    cfg = DATASETS[name]
    n_folds = args.n_folds
    print(f"\n{'='*60}")
    print(f"Dataset: {name}  D={cfg.D}  T={cfg.T}  folds={n_folds}")
    print(f"{'='*60}")

    X_train, X_val, X_test, y_train, y_val, _ = load_dataset(cfg)
    n_test = min(N_TEST_LARGE_D if cfg.D > 20 else N_TEST_MAX, len(X_test))
    # Round down so every fold has the same size
    n_test = (n_test // n_folds) * n_folds
    X_test_s = X_test[:n_test]
    print(
        f"  Train={len(X_train)}  Val={len(X_val)}  Test={n_test} ({n_test//n_folds}/fold)"
    )

    X_tr_dt = X_train.transpose(0, 2, 1)
    X_val_dt = X_val.transpose(0, 2, 1)
    X_test_dt = X_test_s.transpose(0, 2, 1).astype(np.float32)
    x_train_mean = X_tr_dt.mean(axis=0)

    print("  Fitting VAR imputer …")
    var_K = select_var_order(X_tr_dt, max_K=8)
    var_imputer = fit_var_imputer(X_tr_dt, var_K)
    print(f"  VAR order K={var_K}")

    results: dict[str, Any] = {
        "dataset": name,
        "D": cfg.D,
        "T": cfg.T,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": n_test,
        "n_folds": n_folds,
        "fold_size": n_test // n_folds,
        "var_K": var_K,
    }
    if not args.skip_gru:
        print("\n── GRU ──")
        gru = get_or_train_gru(cfg, X_train, y_train, X_val, y_val, device, ckpt_dir)
        results.update(
            _run_cv_for_model(
                "gru",
                gru,
                X_test_dt,
                X_tr_dt,
                X_val_dt,
                X_train,
                X_val,
                cfg,
                args,
                ckpt_dir,
                x_train_mean,
                device,
                var_imputer,
                var_K,
                n_folds,
            )
        )

    if not args.skip_lstm:
        print("\n── LSTM ──")
        try:
            lstm = load_pretrained_lstm(cfg, device, model_ckpt_dir)
            results.update(
                _run_cv_for_model(
                    "lstm",
                    lstm,
                    X_test_dt,
                    X_tr_dt,
                    X_val_dt,
                    X_train,
                    X_val,
                    cfg,
                    args,
                    ckpt_dir,
                    x_train_mean,
                    device,
                    var_imputer,
                    var_K,
                    n_folds,
                )
            )
        except FileNotFoundError as e:
            print(f"  Skipping LSTM: {e}")

    if not args.skip_tcn:
        print("\n── TCN ──")
        try:
            tcn = load_pretrained_tcn(cfg, device, model_ckpt_dir)
            results.update(
                _run_cv_for_model(
                    "tcn",
                    tcn,
                    X_test_dt,
                    X_tr_dt,
                    X_val_dt,
                    X_train,
                    X_val,
                    cfg,
                    args,
                    ckpt_dir,
                    x_train_mean,
                    device,
                    var_imputer,
                    var_K,
                    n_folds,
                )
            )
        except FileNotFoundError as e:
            print(f"  Skipping TCN: {e}")

    if not args.skip_transformer:
        print("\n── Transformer ──")
        try:
            transformer = load_pretrained_transformer(cfg, device, model_ckpt_dir)
            results.update(
                _run_cv_for_model(
                    "transformer",
                    transformer,
                    X_test_dt,
                    X_tr_dt,
                    X_val_dt,
                    X_train,
                    X_val,
                    cfg,
                    args,
                    ckpt_dir,
                    x_train_mean,
                    device,
                    var_imputer,
                    var_K,
                    n_folds,
                )
            )
        except FileNotFoundError as e:
            print(f"  Skipping Transformer: {e}")

    if not args.skip_rf:
        print("\n── Random Forest ──")
        try:
            rf = load_pretrained_rf(cfg, device, model_ckpt_dir)
            results.update(
                _run_cv_for_model(
                    "rf",
                    rf,
                    X_test_dt,
                    X_tr_dt,
                    X_val_dt,
                    X_train,
                    X_val,
                    cfg,
                    args,
                    ckpt_dir,
                    x_train_mean,
                    device,
                    var_imputer,
                    var_K,
                    n_folds,
                )
            )
        except FileNotFoundError as e:
            print(f"  Skipping Random Forest: {e}")

    out_file = out_dir / f"{name}_cv{n_folds}_results.json"
    with open(out_file, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\n  → {out_file}")
    return results


def print_summary_cv(
    all_results: list[dict], complexity_only: bool = False, lag_only: bool = False
) -> None:
    archs = ["gru", "lstm", "tcn", "transformer", "rf"]
    methods = [
        "ig",
        "timeshap",
        "karma",
        "timing",
        "tsmule",
        "shaptime",
    ]
    all_metrics = [
        ("lag_auc", "Lag AUC        (higher = better)"),
        ("lag_drop25", "Lag Drop@25%   (higher = better)"),
        ("complexity", "Complexity     (lower = simpler, Bhatt et al. 2020)"),
    ]
    if complexity_only:
        metrics = [m for m in all_metrics if m[0] == "complexity"]
    elif lag_only:
        metrics = [
            m for m in all_metrics if m[0] in ("lag_auc", "lag_drop25", "complexity")
        ]
    else:
        metrics = all_metrics

    col_w = 16
    ds_w = 16
    arch_w = 6
    width = ds_w + arch_w + col_w * len(methods)

    for metric, label in metrics:
        print(f"\n{'='*width}")
        print(label)
        print("=" * width)
        header = f"{'Dataset':<{ds_w}}{'Arch':<{arch_w}}" + "".join(
            f"{m.upper():>{col_w}}" for m in methods
        )
        print(header)
        print("-" * width)
        for r in all_results:
            for arch in archs:
                row_vals = {
                    m: (
                        r.get(f"{arch}_{m}_{metric}_mean"),
                        r.get(f"{arch}_{m}_{metric}_std"),
                    )
                    for m in methods
                }
                if not any(v[0] is not None for v in row_vals.values()):
                    continue
                row = f"{r['dataset']:<{ds_w}}{arch:<{arch_w}}"
                for m in methods:
                    mean, std = row_vals[m]
                    if mean is None:
                        cell = "—"
                    else:
                        cell = f"{mean:.3f}±{std:.3f}"
                    row += f"{cell:>{col_w}}"
                print(row)
        print("=" * width)


def main():
    p = argparse.ArgumentParser(
        description="5-fold CV attribution comparison on real forecasting datasets"
    )
    p.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    p.add_argument("--n_folds", type=int, default=N_FOLDS_DEFAULT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt_dir", default="outputs/checkpoints/realdata_cv")
    p.add_argument("--out_dir", default="results/realdata_cv")
    p.add_argument("--model_ckpt_dir", default="outputs/checkpoints")
    p.add_argument("--skip_gru", action="store_true")
    p.add_argument("--skip_lstm", action="store_true")
    p.add_argument("--skip_tcn", action="store_true")
    p.add_argument(
        "--skip_transformer", action="store_true", help="Skip Transformer model"
    )

    p.add_argument(
        "--skip_rf",
        action="store_true",
        help="Skip Random Forest model (also skips IG/DynaMask/ExtremalMask/"
        "TIMING for it regardless, since it isn't differentiable)",
    )

    p.add_argument("--skip_karma", action="store_true")
    p.add_argument("--skip_ig", action="store_true")
    p.add_argument("--skip_timeshap", action="store_true")
    p.add_argument("--skip_timing", action="store_true")
    p.add_argument("--skip_tsmule", action="store_true")
    p.add_argument("--skip_shaptime", action="store_true")
    p.add_argument("--lag_only", action="store_true")
    p.add_argument(
        "--complexity_only",
        action="store_true",
        help="Skip all removal-based metrics; compute complexity only",
    )
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--ts_nsamples", type=int, default=200)
    p.add_argument("--tsmule_nsamples", type=int, default=100)
    p.add_argument("--shaptime_tn", type=int, default=6)
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global seed for numpy/torch/random and KARMA's KernelEstimator",
    )
    p.add_argument(
        "--tau",
        type=str,
        default=None,
        help="Comma-separated forecast step indices for KARMA's joint multistep "
        "variable importance, e.g. '0,1,2'. Only takes effect on datasets whose "
        "checkpoint has forecast_steps > 1; defaults to pooling all forecast "
        "steps jointly. Ignored (single-step) when forecast_steps == 1.",
    )

    args = p.parse_args()
    args.tau = [int(t) for t in args.tau.split(",")] if args.tau else None

    set_global_seed(args.seed)

    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir)
    model_ckpt_dir = Path(args.model_ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}  |  folds: {args.n_folds}  |  seed: {args.seed}")

    all_results = []
    for ds_name in args.datasets:
        if ds_name not in DATASETS:
            print(f"Unknown dataset '{ds_name}'. Available: {list(DATASETS.keys())}")
            continue
        try:
            r = run_dataset_cv(ds_name, args, device, ckpt_dir, out_dir, model_ckpt_dir)
            all_results.append(r)
        except Exception as e:
            print(f"\nERROR on dataset '{ds_name}': {e}")
            traceback.print_exc()

    if all_results:
        print_summary_cv(
            all_results, complexity_only=args.complexity_only, lag_only=args.lag_only
        )
        summary_file = out_dir / f"summary_cv{args.n_folds}.json"
        with open(summary_file, "w") as fh:
            json.dump(all_results, fh, indent=2, default=float)
        print(f"\nSummary → {summary_file}")


if __name__ == "__main__":
    main()
