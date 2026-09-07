"""
Unified training pipeline for all datasets and model architectures.

Usage:
    python -m pipeline.training_pipeline --dataset etth1 --model lstm
    python -m pipeline.training_pipeline --dataset weather --model both --epochs 100
    python -m pipeline.training_pipeline --dataset exchange_rate --model tcn --seed 0
    python -m pipeline.training_pipeline --dataset etth1 --model transformer
    python -m pipeline.training_pipeline --dataset etth1 --model gbm
    python -m pipeline.training_pipeline --dataset etth1 --model rf
    python -m pipeline.training_pipeline --dataset etth1 --model xgboost
    python -m pipeline.training_pipeline --dataset etth1 --model all
"""

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from models.training_pipeline import TrainingConfig, TrainingPipeline

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
DATASETS = [
    "etth1", "etth2", "ettm1", "ettm2", "weather", "exchange_rate",
    "beijing", "electricity", "web_traffic",
]


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train_model(
    dataset_name: str,
    model_type: str,
    epochs: int = 50,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple:
    """
    Train a single model on a dataset using configs from YAML files.

    Parameters
    ----------
    dataset_name : one of DATASETS
    model_type   : 'lstm', 'tcn', 'transformer', 'gbm', 'rf', or 'xgboost'
    epochs       : training epochs (overrides training/default.yaml).
                   Ignored by 'gbm'/'rf'/'xgboost', which fit tree ensembles
                   to convergence in one pass rather than iterating epochs
                   over mini-batches.
    batch_size   : mini-batch size (ignored by 'gbm'/'rf'/'xgboost')
    seed         : random seed

    Returns
    -------
    (pipeline, test_results) for lstm/tcn/transformer;
    (regressor, test_results) for gbm/rf/xgboost — not a torch.nn.Module, so
    there's no TrainingPipeline to return.
    """
    dataset_cfg = _load_yaml(CONFIGS_DIR / "datasets" / f"{dataset_name}.yaml")

    if model_type == "gbm":
        return _train_gbm(dataset_name, dataset_cfg, seed=seed)
    if model_type == "rf":
        return _train_rf(dataset_name, dataset_cfg, seed=seed)
    if model_type == "xgboost":
        return _train_xgboost(dataset_name, dataset_cfg, seed=seed)

    model_cfg = _load_yaml(CONFIGS_DIR / "models" / f"{model_type}.yaml")
    train_cfg = _load_yaml(CONFIGS_DIR / "training" / "default.yaml")

    torch.manual_seed(seed)

    print(f"\n{'='*60}")
    print(f"TRAINING {model_type.upper()} — {dataset_name}")
    print("=" * 60)

    mod = importlib.import_module(dataset_cfg["dataset_module"])
    train_loader, val_loader, test_loader = mod.get_dataloaders(
        data_dir=dataset_cfg["data_dir"],
        batch_size=batch_size,
    )

    # Infer D from data when not fixed in config
    D = dataset_cfg.get("D")
    if D is None:
        sample_x, _ = next(iter(train_loader))
        D = sample_x.shape[-1]

    forecast_steps = dataset_cfg.get("forecast_steps", 1)

    config = TrainingConfig()
    config.model_type = model_type
    config.epochs = epochs
    config.batch_size = batch_size
    config.learning_rate = train_cfg["learning_rate"]
    config.use_scheduler = train_cfg.get("use_scheduler", True)
    config.scheduler_type = train_cfg.get("scheduler_type", "cosine")
    config.dropout = train_cfg.get("dropout", 0.2)
    config.checkpoint_dir = dataset_cfg["checkpoints"][model_type]
    if model_type == "tcn":
        config.clip_grad_norm = train_cfg.get("clip_grad_norm", 1.0)

    pipeline = TrainingPipeline(config)

    model_kwargs = dict(input_size=D, output_size=D, forecast_steps=forecast_steps)
    if model_type == "lstm":
        model_kwargs.update(
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
        )
    elif model_type == "tcn":
        model_kwargs.update(
            num_channels=model_cfg["num_channels"],
            kernel_size=model_cfg["kernel_size"],
        )
    else:  # transformer
        model_kwargs.update(
            d_model=model_cfg["d_model"],
            nhead=model_cfg["nhead"],
            num_layers=model_cfg["num_layers"],
            dim_feedforward=model_cfg["dim_feedforward"],
        )

    pipeline.create_model(**model_kwargs)
    optimizer = pipeline.setup_optimizer()
    scheduler = pipeline.setup_scheduler(optimizer, epochs)
    pipeline.setup_trainer(optimizer, scheduler)
    pipeline.train(train_loader, val_loader)

    test_results = pipeline.evaluate_final(
        test_loader,
        checkpoint_path=f"{config.checkpoint_dir}/best.pt",
    )

    if model_type == "transformer":
        # nhead isn't recoverable from checkpoint tensor shapes alone
        # (MultiheadAttention's in_proj_weight is (3*d_model, d_model)
        # regardless of head count) — comparison_realdata.py's
        # load_pretrained_transformer reads it back from here.
        with open(f"{config.checkpoint_dir}/model_config.json", "w") as f:
            json.dump({"nhead": model_cfg["nhead"]}, f, indent=2)

    print(f"\nCheckpoint saved: {config.checkpoint_dir}/best.pt")
    return pipeline, test_results


def _train_tree_ensemble(
    dataset_name: str,
    dataset_cfg: dict,
    model_key: str,
    label: str,
    base,
    wrap_multioutput: bool,
) -> tuple:
    """
    Shared fit/eval/checkpoint logic for sklearn-style tree-ensemble
    forecasters (gbm, rf, xgboost): flatten each (T, D) input window to a
    single (T*D)-length feature vector, fit `base` against the flattened
    (forecast_steps*D)-length target, evaluate, and save a joblib checkpoint.

    `base` is a single already-constructed regressor. `wrap_multioutput`
    controls whether it's wrapped in MultiOutputRegressor (one clone fit per
    output column — needed for regressors that only accept 1-D targets, e.g.
    GradientBoostingRegressor) or fit directly (regressors that natively
    handle 2-D targets in one call, e.g. RandomForestRegressor, or the
    XGBRegressor(multi_strategy="multi_output_tree") built by _train_xgboost).

    Not a torch.nn.Module — no autograd, no mini-batch loop — so this
    bypasses TrainingPipeline/Trainer entirely and fits on the full
    training array in one call. Saved as a joblib pickle (best.joblib)
    rather than a torch .pt checkpoint, since there's no state_dict.
    """
    import joblib

    print(f"\n{'='*60}")
    print(f"TRAINING {label} — {dataset_name}")
    print("=" * 60)

    mod = importlib.import_module(dataset_cfg["dataset_module"])
    train_loader, val_loader, test_loader = mod.get_dataloaders(
        data_dir=dataset_cfg["data_dir"],
        batch_size=256,
    )

    def _collect(loader) -> tuple:
        xs, ys = [], []
        for x, y in loader:
            xs.append(x.numpy())
            ys.append(y.numpy())
        X = np.concatenate(xs, axis=0)  # (N, T, D)
        Y = np.concatenate(ys, axis=0)
        if Y.ndim == 2:  # (N, D) -> (N, 1, D), forecast_steps=1
            Y = Y[:, None, :]
        return X, Y  # (N, T, D), (N, forecast_steps, D)

    X_train, y_train = _collect(train_loader)
    X_val, y_val = _collect(val_loader)
    X_test, y_test = _collect(test_loader)
    N_tr, T, D = X_train.shape
    forecast_steps = y_train.shape[1]

    X_train_flat = X_train.reshape(N_tr, T * D)
    X_val_flat = X_val.reshape(len(X_val), T * D)
    X_test_flat = X_test.reshape(len(X_test), T * D)
    y_train_flat = y_train.reshape(len(y_train), -1)
    y_val_flat = y_val.reshape(len(y_val), -1)
    y_test_flat = y_test.reshape(len(y_test), -1)

    if wrap_multioutput:
        from sklearn.multioutput import MultiOutputRegressor

        regressor = MultiOutputRegressor(base, n_jobs=-1)
        fit_desc = f"MultiOutputRegressor({type(base).__name__})"
    else:
        regressor = base
        fit_desc = type(base).__name__
    print(
        f"  Fitting {fit_desc} on {N_tr} samples, {T * D} input features, "
        f"{y_train_flat.shape[1]} outputs …"
    )
    regressor.fit(X_train_flat, y_train_flat)

    def _eval(X_flat, y_flat, name: str) -> dict:
        pred = regressor.predict(X_flat)
        mse = float(np.mean((pred - y_flat) ** 2))
        mae = float(np.mean(np.abs(pred - y_flat)))
        ss_res = float(np.sum((y_flat - pred) ** 2))
        ss_tot = float(np.sum((y_flat - y_flat.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        results = {"mse": mse, "rmse": mse**0.5, "mae": mae, "r2": r2}
        print(
            f"  {name}: MSE={results['mse']:.6f}  RMSE={results['rmse']:.6f}  "
            f"MAE={results['mae']:.6f}  R2={results['r2']:.6f}"
        )
        return results

    _eval(X_train_flat, y_train_flat, "Train")
    val_results = _eval(X_val_flat, y_val_flat, "Val")
    test_results = _eval(X_test_flat, y_test_flat, "Test")
    test_results["val_loss"] = val_results["mse"]

    checkpoint_dir = Path(dataset_cfg["checkpoints"][model_key])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "best.joblib"
    joblib.dump(
        {"regressor": regressor, "T": T, "D": D, "forecast_steps": forecast_steps},
        ckpt_path,
    )
    with open(checkpoint_dir / "evaluation_results.json", "w") as f:
        json.dump(test_results, f, indent=2, default=str)

    print(f"\nCheckpoint saved: {ckpt_path}")
    return regressor, test_results


def _train_gbm(dataset_name: str, dataset_cfg: dict, seed: int = 42) -> tuple:
    """One GradientBoostingRegressor per output column (MultiOutputRegressor
    wrapped, since it only accepts 1-D targets)."""
    from sklearn.ensemble import GradientBoostingRegressor

    model_cfg = _load_yaml(CONFIGS_DIR / "models" / "gbm.yaml")
    base = GradientBoostingRegressor(
        n_estimators=model_cfg["n_estimators"],
        max_depth=model_cfg["max_depth"],
        learning_rate=model_cfg["learning_rate"],
        subsample=model_cfg.get("subsample", 1.0),
        random_state=seed,
    )
    return _train_tree_ensemble(
        dataset_name, dataset_cfg, "gbm", "GBM", base, wrap_multioutput=True
    )


def _train_rf(dataset_name: str, dataset_cfg: dict, seed: int = 42) -> tuple:
    """A single RandomForestRegressor fit jointly across all output columns —
    it accepts 2-D targets natively, so no MultiOutputRegressor wrapper."""
    from sklearn.ensemble import RandomForestRegressor

    model_cfg = _load_yaml(CONFIGS_DIR / "models" / "rf.yaml")
    base = RandomForestRegressor(
        n_estimators=model_cfg["n_estimators"],
        max_depth=model_cfg.get("max_depth"),
        min_samples_leaf=model_cfg.get("min_samples_leaf", 1),
        n_jobs=-1,
        random_state=seed,
    )
    return _train_tree_ensemble(
        dataset_name, dataset_cfg, "rf", "Random Forest", base, wrap_multioutput=False
    )


def _train_xgboost(dataset_name: str, dataset_cfg: dict, seed: int = 42) -> tuple:
    """A single XGBRegressor with multi_strategy="multi_output_tree", fit
    jointly across all output columns via one set of trees per boosting
    round — no MultiOutputRegressor wrapper (requires tree_method="hist")."""
    from xgboost import XGBRegressor

    model_cfg = _load_yaml(CONFIGS_DIR / "models" / "xgboost.yaml")
    base = XGBRegressor(
        n_estimators=model_cfg["n_estimators"],
        max_depth=model_cfg["max_depth"],
        learning_rate=model_cfg["learning_rate"],
        subsample=model_cfg.get("subsample", 1.0),
        colsample_bytree=model_cfg.get("colsample_bytree", 1.0),
        tree_method="hist",
        multi_strategy="multi_output_tree",
        random_state=seed,
    )
    return _train_tree_ensemble(
        dataset_name, dataset_cfg, "xgboost", "XGBoost", base, wrap_multioutput=False
    )


def main():
    p = argparse.ArgumentParser(description="Train time series forecasting models")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument(
        "--model",
        default="both",
        choices=["lstm", "tcn", "transformer", "gbm", "rf", "xgboost", "both", "all"],
        help="'both' = lstm+tcn (legacy default); 'all' = every model type",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.model == "both":
        models = ["lstm", "tcn"]
    elif args.model == "all":
        models = ["lstm", "tcn", "transformer", "gbm", "rf", "xgboost"]
    else:
        models = [args.model]
    for m in models:
        train_model(
            dataset_name=args.dataset,
            model_type=m,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
