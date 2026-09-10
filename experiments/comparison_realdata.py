#!/usr/bin/env python
"""
Lag AUC and Lag prediction-change comparison on real forecasting datasets.


Datasets
--------
  etth1, exchange_rate, beijing_pm25, web_traffic, electricity

Lag AUC metric
----------
  At step k (k=1..n_steps), remove the top-(k/n_steps) fraction of time steps
  in descending importance order (replace with training-set mean).  Measure
  mean absolute prediction change.  AUC = trapezoidal integral over that curve.

Usage
-----
  python -m experiments.comparison_realdata
  python -m experiments.comparison_realdata --datasets etth1 exchange_rate
  python -m experiments.comparison_realdata --skip_karma --skip_winit
"""

from __future__ import annotations
import abc
import argparse
import json
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

if not hasattr(np, "trapz"):  # NumPy ≥ 2.0 removed np.trapz
    np.trapz = np.trapezoid
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset

from captum.attr import IntegratedGradients

from karma.causal_recovery.edge_contribution import (
    compute_variable_importance,
    compute_variable_importance_multistep,
)
from karma.markov_approximation.markov_surrogacy import select_K_and_baseline
from karma.utils.discretiser import Discretiser
from karma.utils.kernel_estimator import TreeKernelEstimator
from karma.utils.sampling import SuffixPool

_ROOT = Path(__file__).parent.parent
torch.solve = lambda B, A: (torch.linalg.solve(A, B), None)

sys.path.insert(0, str(_ROOT))
for _ext in [
    "TIMING",
    "ts-mule",
]:
    _ext_p = str(_ROOT / _ext)
    if _ext_p not in sys.path:
        sys.path.insert(0, _ext_p)


def set_global_seed(seed: int) -> None:
    """Seed python's random, numpy's legacy global RNG, and torch (CPU + CUDA).

    Covers every randomness source in this pipeline that reads global RNG
    state rather than an explicit generator: TS-MuLe's perturbation sampling
    (np.random.choice), TIMING's segment sampling (torch.rand/randint), FIT's
    counterfactual sampling (torch.randn), and fresh GRU/LSTM/TCN training
    (DataLoader shuffling + weight init) when no checkpoint is loaded. Does
    NOT cover KARMA's TreeKernelEstimator, which takes its own explicit
    np.random.Generator — run_karma seeds that separately via its own `seed`
    argument.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class DatasetConfig:
    name: str
    data_dir: str
    D: int
    T: int  # sequence_length
    pred_horizon: int
    ckpt_prefix: str = ""  # folder prefix for LSTM/TCN checkpoints; defaults to name
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 64
    num_epochs: int = 50
    # KARMA params
    N: int = 2
    eps: float = 0.05
    lam: float = 0.01
    M: int = 100
    n_pool_min: int = 3
    K_max: int = 3

    def __post_init__(self):
        if not self.ckpt_prefix:
            self.ckpt_prefix = self.name


DATASETS: dict[str, DatasetConfig] = {
    "etth1": DatasetConfig(
        name="etth1",
        data_dir="data/generated/etth1",
        D=7,
        T=24,
        pred_horizon=7,
        M=200,
        K_max=24,
        lam=0.05,
        eps=0.0001,
    ),
    "exchange_rate": DatasetConfig(
        name="exchange_rate",
        data_dir="data/generated/exchange_rate",
        D=8,
        T=14,
        pred_horizon=7,
        M=100,
        K_max=10,
        lam=0.125,  # 0.15 for tcn, 0.01 for gru, 0.1 for lstm
        eps=0.005,
    ),
    "beijing_pm25": DatasetConfig(
        name="beijing_pm25",
        data_dir="data/generated/beijing_pm25",
        D=11,
        T=24,
        pred_horizon=1,
        M=100,
        K_max=24,
        ckpt_prefix="bpm25",
        lam=0.05,
        eps=0.001,
    ),
    "web_traffic": DatasetConfig(
        name="web_traffic",
        data_dir="data/generated/web_traffic",
        D=100,
        T=14,
        pred_horizon=7,
        N=2,
        hidden_size=128,
        num_layers=2,
        M=200,
        K_max=14,
        lam=0.1,
        eps=0.001,
    ),
    "etth2": DatasetConfig(
        name="etth2",
        data_dir="data/generated/etth2",
        D=7,
        T=24,
        pred_horizon=12,
        M=100,
        K_max=24,
        lam=0.15,
        eps=0.001,
    ),
    "ettm1": DatasetConfig(
        name="ettm1",
        data_dir="data/generated/ettm1",
        D=7,
        T=48,
        pred_horizon=1,
        M=100,
        K_max=48,
        lam=0.05,
        eps=0.001,
    ),
    "ettm2": DatasetConfig(
        name="ettm2",
        data_dir="data/generated/ettm2",
        D=7,
        T=48,
        pred_horizon=12,
        M=200,
        K_max=48,
        lam=0.05,
        eps=0.001,
    ),
    "electricity": DatasetConfig(
        name="electricity",
        data_dir="data/generated/electricity",
        D=20,
        T=48,
        N=3,
        pred_horizon=1,
        hidden_size=128,
        num_layers=2,
        M=100,
        K_max=45,
        lam=0.05,
        eps=0.001,
    ),
}

MAX_KARMA_D = 101  # skip KARMA for D > this (state space too large)
N_AUC_STEPS = 10  # progressive removal steps
N_TEST_MAX = 200  # max test samples for attribution (capped for speed)
N_TEST_LARGE_D = 50  # further reduced cap when D > 20
NON_DIFFERENTIABLE_ARCHS = {"rf"}  # sklearn/xgboost tree ensembles


class TorchModel(nn.Module, abc.ABC):
    """
    Class extends torch.nn.Module. Mainly for user to specify the forward with a ``return_all``
    option. The model is supposed to accept inputs of shape (num_samples, num_features, num_times).
    If return_all is True, the output should be of shape (num_samples, num_states, num_times).
    Otherwise, the output should be of shape (num_samples, num_states)
    """

    def __init__(self, feature_size, num_states, hidden_size, device):
        """
        Constructor

        Args:
            feature_size:
               The number of features the model is accepting.
            num_states:
               The number of output nodes.
            hidden_size:
               The hidden size of the model
            device:
               The torch device the model is on.
        """
        super().__init__()
        self.feature_size = feature_size
        self.num_states = num_states
        self.hidden_size = hidden_size
        if self.num_states > 1:
            activation = torch.nn.Softmax(dim=1)
        else:
            activation = torch.nn.Sigmoid()
        self.activation = activation
        self.device = device

    @abc.abstractmethod
    def forward(self, input, return_all=True):
        """
        Specify the forward function for this torch.nn.Module. The forward function should not
        include the activation function at the end. i.e. the output should be in logit space.

        Args:
            input:
                Shape = (num_samples, num_features, num_times)
            return_all:
                True if we want to get the output of the model only at the last timestep.

        Returns:
            A tensor of shape (num_samples, num_states, num_times) if return_all is True. Otherwise,
            a tensor of shape (num_samples, num_states) is returned.
        """

    def predict(self, input, return_all=True):
        """
        Apply the activation after the forward function.

            input:
                Shape = (num_samples, num_features, num_times)
            return_all:
                True if we want to get the output of the model only at the last timestep.

        Returns:
            A tensor of shape (num_samples, num_states, num_times) if return_all is True. Otherwise,
            a tensor of shape (num_samples, num_states) is returned.
        """
        return self.activation(self.forward(input, return_all=return_all))


class GRUForecastModel(TorchModel):
    """GRU regression model supporting single- and multi-step forecasting.

    forward(x: (B, D, T), return_all=False) → (B, D)       [next-step, all callers]
    forward(x: (B, D, T), return_all=True)  → (B, D, T)    [rolling, WinIT/DynaMask]
    predict_multistep(x: (B, D, T))         → (B, H, D)    [KARMA multi-step oracle]
    """

    def __init__(
        self,
        D: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        forecast_steps: int = 1,
        device="cpu",
    ):
        super().__init__(
            feature_size=D, num_states=D, hidden_size=hidden_size, device=device
        )
        self.D = D
        self.forecast_steps = forecast_steps
        self.gru = nn.GRU(
            D,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, forecast_steps * D)
        self.activation = nn.Identity()

    def forward(self, x: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        x_td = x.permute(0, 2, 1)  # (B, T, D)
        out, _ = self.gru(x_td)  # (B, T, hidden)
        if return_all:
            return self.fc(out)[..., : self.D].permute(0, 2, 1)  # (B, D, T)
        return self.fc(out[:, -1, :])[..., : self.D]  # (B, D)

    def predict_multistep(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, forecast_steps, D) from the last hidden state."""
        x_td = x.permute(0, 2, 1)
        out, _ = self.gru(x_td)
        raw = self.fc(out[:, -1, :])  # (B, forecast_steps * D)
        return raw.view(x.shape[0], self.forecast_steps, self.D)


class LSTMForecastModel(TorchModel):
    """Stacked LSTM with LayerNorm head.  Matches checkpoint architecture.

    forward(x: (B, D, T)) → (B, D)  [return_all=False, next-step prediction]
    forward(x: (B, D, T)) → (B, D, T)  [return_all=True, prediction at every t]
    """

    def __init__(
        self,
        D: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        forecast_steps: int = 1,
        device="cpu",
    ):
        super().__init__(
            feature_size=D, num_states=D, hidden_size=hidden_size, device=device
        )
        self.forecast_steps = forecast_steps
        self.D = D
        self.lstm = nn.LSTM(
            D,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, forecast_steps * D),
        )
        self.activation = nn.Identity()

    def _apply_head(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., hidden) → (..., D)  [first forecast step only]"""
        out = self.head(self.dropout_layer(self.layer_norm(h)))
        return out[..., : self.D]  # take first forecast step's D outputs

    def forward(self, x: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        x_td = x.permute(0, 2, 1)  # (B, T, D)
        lstm_out, _ = self.lstm(x_td)  # (B, T, hidden)
        if return_all:
            preds = self._apply_head(lstm_out)  # (B, T, D)
            return preds.permute(0, 2, 1)  # (B, D, T)
        return self._apply_head(lstm_out[:, -1, :])  # (B, D)

    def predict_multistep(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, forecast_steps, D) from the last hidden state."""
        x_td = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_td)
        h = self.dropout_layer(self.layer_norm(lstm_out[:, -1, :]))
        raw = self.head(h)  # (B, forecast_steps * D)
        return raw.view(x.shape[0], self.forecast_steps, self.D)


class _Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class _TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        from torch.nn.utils import weight_norm as wn

        self.conv1 = wn(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        )
        self.conv2 = wn(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        )
        self.net = nn.Sequential(
            self.conv1,
            _Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            self.conv2,
            _Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(self.net(x) + res)


class TCNForecastModel(TorchModel):
    """Dilated causal TCN.  Matches checkpoint architecture.

    forward(x: (B, D, T)) → (B, D)  [return_all=False]
    forward(x: (B, D, T)) → (B, D, T)  [return_all=True]
    """

    def __init__(
        self,
        D: int,
        num_channels: list[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        forecast_steps: int = 1,
        device="cpu",
    ):
        num_channels = num_channels or [64, 64, 128]
        super().__init__(
            feature_size=D, num_states=D, hidden_size=num_channels[-1], device=device
        )
        self.forecast_steps = forecast_steps
        self.D = D
        blocks = []
        for i, out_ch in enumerate(num_channels):
            in_ch = D if i == 0 else num_channels[i - 1]
            blocks.append(
                _TemporalBlock(
                    in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout
                )
            )
        self.network = nn.Sequential(*blocks)
        last_ch = num_channels[-1]
        self.head = nn.Sequential(
            nn.Linear(last_ch, last_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(last_ch, forecast_steps * D),
        )
        self.activation = nn.Identity()

    def _apply_head(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., channels) → (..., D)"""
        out = self.head(h)
        return out[..., : self.D]

    def forward(self, x: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        y = self.network(x)  # (B, channels, T)
        if return_all:
            h = y.permute(0, 2, 1)  # (B, T, channels)
            preds = self._apply_head(h)  # (B, T, D)
            return preds.permute(0, 2, 1)  # (B, D, T)
        h_last = y[:, :, -1]  # (B, channels)
        return self._apply_head(h_last)  # (B, D)

    def predict_multistep(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, forecast_steps, D) from the last temporal position."""
        y = self.network(x)
        raw = self.head(y[:, :, -1])  # (B, forecast_steps * D)
        return raw.view(x.shape[0], self.forecast_steps, self.D)


class TransformerForecastModel(TorchModel):
    """Causal Transformer encoder.  Matches checkpoint architecture from
    models/architectures/transformer_architecture.py (same nn.TransformerEncoder
    hyperparameters produce identical state_dict keys regardless of which file
    instantiates them, exactly like LSTMForecastModel/TCNForecastModel above).

    forward(x: (B, D, T)) → (B, D)  [return_all=False, next-step prediction]
    forward(x: (B, D, T)) → (B, D, T)  [return_all=True, causal prediction at every t]
    """

    def __init__(
        self,
        D: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        forecast_steps: int = 1,
        device="cpu",
    ):
        super().__init__(
            feature_size=D, num_states=D, hidden_size=d_model, device=device
        )
        self.forecast_steps = forecast_steps
        self.D = D
        self.d_model = d_model

        self.input_proj = nn.Linear(D, d_model)
        self.register_buffer(
            "pe", self._build_pe(d_model, max_len=2048), persistent=False
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, forecast_steps * D),
        )
        self.activation = nn.Identity()

    @staticmethod
    def _build_pe(d_model: int, max_len: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, T) -> (B, T, d_model), causal self-attention."""
        x_td = x.permute(0, 2, 1)  # (B, T, D)
        T = x_td.shape[1]
        h = self.input_proj(x_td) + self.pe[:, :T]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        return self.encoder(h, mask=mask, is_causal=True)  # (B, T, d_model)

    def _apply_head(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., d_model) → (..., D)  [first forecast step only]"""
        out = self.head(self.dropout_layer(self.layer_norm(h)))
        return out[..., : self.D]

    def forward(self, x: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        h = self._encode(x)  # (B, T, d_model)
        if return_all:
            preds = self._apply_head(h)  # (B, T, D)
            return preds.permute(0, 2, 1)  # (B, D, T)
        return self._apply_head(h[:, -1, :])  # (B, D)

    def predict_multistep(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, forecast_steps, D) from the last position's representation."""
        h = self._encode(x)
        last = self.dropout_layer(self.layer_norm(h[:, -1, :]))
        raw = self.head(last)  # (B, forecast_steps * D)
        return raw.view(x.shape[0], self.forecast_steps, self.D)


class TreeForecastModel(TorchModel):

    def __init__(self, D: int, T: int, forecast_steps: int = 1, device="cpu"):
        super().__init__(feature_size=D, num_states=D, hidden_size=0, device=device)
        self.D = D
        self.T = T
        self.forecast_steps = forecast_steps
        self.activation = nn.Identity()
        self.regressor = (
            None  # sklearn/xgboost regressor; set via load_pretrained_{gbm,rf,xgboost}
        )

    def _flatten(self, x: torch.Tensor) -> np.ndarray:
        # (B, D, T) -> (B, T, D) -> (B, T*D), matching the flattening
        # convention _train_tree_ensemble uses at fit time
        # (pipeline/training_pipeline.py).
        return x.detach().cpu().numpy().transpose(0, 2, 1).reshape(x.shape[0], -1)

    def forward(self, x: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        if return_all:
            raise NotImplementedError(
                "TreeForecastModel has no gradient/full-sequence path; only "
                "return_all=False and predict_multistep() are supported."
            )
        if self.regressor is None:
            raise RuntimeError(
                "Load a checkpoint via load_pretrained_{gbm,rf,xgboost}() first."
            )
        flat = self._flatten(x)
        pred = self.regressor.predict(flat).reshape(
            x.shape[0], self.forecast_steps, self.D
        )
        return torch.from_numpy(pred[:, 0, :].astype(np.float32)).to(x.device)

    def predict_multistep(self, x: torch.Tensor) -> torch.Tensor:
        if self.regressor is None:
            raise RuntimeError(
                "Load a checkpoint via load_pretrained_{gbm,rf,xgboost}() first."
            )
        flat = self._flatten(x)
        pred = self.regressor.predict(flat).reshape(
            x.shape[0], self.forecast_steps, self.D
        )
        return torch.from_numpy(pred.astype(np.float32)).to(x.device)


def load_dataset(cfg: DatasetConfig):
    base = Path(cfg.data_dir)
    X_train = np.load(base / "X_train.npy")  # (N, T, D)
    X_val = np.load(base / "X_val.npy")
    X_test = np.load(base / "X_test.npy")
    y_train = np.load(base / "y_train.npy")  # (N, pred_horizon, D)
    y_val = np.load(base / "y_val.npy")
    y_test = np.load(base / "y_test.npy")
    return X_train, X_val, X_test, y_train, y_val, y_test


def reconstruct_raw_series(X_windows: np.ndarray) -> np.ndarray:
    """Reconstruct raw (T_total, D) time series from (N, W, D) windows.

    Windows are assumed to be consecutive (stride=1).  X_windows[0] gives the
    first W values; each subsequent window adds one new observation at the end.
    """
    first = X_windows[0]  # (W, D)
    rest = X_windows[1:, -1, :]  # (N-1, D)
    return np.concatenate([first, rest], axis=0)  # (W + N - 1, D)


def train_gru(
    model: GRUForecastModel,
    X_train: np.ndarray,  # (N, T, D)
    y_train: np.ndarray,  # (N, pred_horizon, D)
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: DatasetConfig,
    device,
    ckpt_path: Optional[Path] = None,
) -> GRUForecastModel:
    multistep = int(y_train.shape[1]) > 1  # driven by actual data, not cfg
    X_tr = torch.from_numpy(X_train.transpose(0, 2, 1).astype(np.float32))  # (N, D, T)
    y_tr = torch.from_numpy(
        y_train.astype(np.float32) if multistep else y_train[:, 0, :].astype(np.float32)
    )  # (N, actual_horizon, D) or (N, D)
    X_v = torch.from_numpy(X_val.transpose(0, 2, 1).astype(np.float32))
    y_v = torch.from_numpy(
        y_val.astype(np.float32) if multistep else y_val[:, 0, :].astype(np.float32)
    )

    tr_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_v, y_v), batch_size=cfg.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    best_val = float("inf")
    best_state = None
    patience = 10

    model = model.to(device)
    for epoch in range(cfg.num_epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = (
                model.predict_multistep(xb)
                if multistep
                else model(xb, return_all=False)
            )
            loss = F.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = (
                    model.predict_multistep(xb)
                    if multistep
                    else model(xb, return_all=False)
                )
                val_losses.append(F.mse_loss(pred, yb).item())
        vl = float(np.mean(val_losses))
        scheduler.step(vl)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 10
            if ckpt_path:
                torch.save(best_state, ckpt_path)
        else:
            patience -= 1
            if patience == 0:
                break

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}: val_mse={vl:.5f}")

    model.load_state_dict(best_state)
    return model


def get_or_train_gru(
    cfg: DatasetConfig,
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    ckpt_dir: Path,
) -> GRUForecastModel:
    ckpt_path = ckpt_dir / f"gru_{cfg.name}.pt"
    actual_horizon = int(y_train.shape[1])  # ground truth from data, not cfg
    model = GRUForecastModel(
        D=cfg.D,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        forecast_steps=actual_horizon,
        device=device,
    )
    if ckpt_path.exists():
        print(f"  Loading GRU from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device)
    else:
        print(f"  Training GRU ({cfg.num_epochs} epochs max)…")
        model = train_gru(model, X_train, y_train, X_val, y_val, cfg, device, ckpt_path)
    model.eval()
    return model


def load_pretrained_lstm(cfg: DatasetConfig, device, model_ckpt_dir: Path):
    """Load LSTM from outputs/checkpoints/{ckpt_prefix}_lstm/best.pt."""
    ckpt_path = model_ckpt_dir / f"{cfg.ckpt_prefix}_lstm" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"LSTM checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    hidden_size = state["lstm.weight_hh_l0"].shape[1]
    num_layers = sum(1 for k in state if k.startswith("lstm.weight_hh"))
    forecast_steps = state["head.3.weight"].shape[0] // cfg.D
    model = LSTMForecastModel(
        D=cfg.D,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.0,
        forecast_steps=forecast_steps,
        device=device,
    )
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(
        f"  Loaded LSTM from {ckpt_path}  (hidden={hidden_size}, forecast_steps={forecast_steps})"
    )
    return model


def load_pretrained_tcn(cfg: DatasetConfig, device, model_ckpt_dir: Path):
    """Load TCN from outputs/checkpoints/{ckpt_prefix}_tcn/best.pt."""
    ckpt_path = model_ckpt_dir / f"{cfg.ckpt_prefix}_tcn" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"TCN checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    n_blocks = max(int(k.split(".")[1]) for k in state if k.startswith("network.")) + 1
    num_channels = [state[f"network.{i}.conv2.bias"].shape[0] for i in range(n_blocks)]
    # Support both weight_norm checkpoints (weight_v/weight_g) and plain checkpoints (weight)
    uses_weight_norm = any(k.endswith(".weight_v") for k in state)
    wkey = "weight_v" if uses_weight_norm else "weight"
    kernel_size = state[f"network.0.conv1.{wkey}"].shape[2]
    forecast_steps = state["head.3.weight"].shape[0] // cfg.D
    model = TCNForecastModel(
        D=cfg.D,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=0.0,
        forecast_steps=forecast_steps,
        device=device,
    )
    if not uses_weight_norm:
        # Checkpoint was saved without weight_norm; strip it from the model so keys match
        for block in model.network:
            nn.utils.remove_weight_norm(block.conv1)
            nn.utils.remove_weight_norm(block.conv2)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(
        f"  Loaded TCN from {ckpt_path}  (channels={num_channels}, kernel={kernel_size}, forecast_steps={forecast_steps})"
    )
    return model


def load_pretrained_transformer(cfg: DatasetConfig, device, model_ckpt_dir: Path):
    """Load Transformer from outputs/checkpoints/{ckpt_prefix}_transformer/best.pt.

    d_model, num_layers, dim_feedforward, and forecast_steps are all
    recoverable from the checkpoint's tensor shapes (same trick load_pretrained_
    lstm/tcn use); nhead is not — MultiheadAttention's in_proj_weight is always
    (3*d_model, d_model) regardless of head count, so it's read from a small
    companion model_config.json that pipeline/training_pipeline.py writes
    alongside best.pt.
    """
    ckpt_dir = model_ckpt_dir / f"{cfg.ckpt_prefix}_transformer"
    ckpt_path = ckpt_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Transformer checkpoint not found: {ckpt_path}")
    config_path = ckpt_dir / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Transformer model_config.json not found: {config_path} "
            "(nhead can't be inferred from checkpoint tensor shapes alone)"
        )
    nhead = json.loads(config_path.read_text())["nhead"]

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    d_model = state["input_proj.weight"].shape[0]
    num_layers = (
        max(int(k.split(".")[2]) for k in state if k.startswith("encoder.layers.")) + 1
    )
    dim_feedforward = state["encoder.layers.0.linear1.weight"].shape[0]
    forecast_steps = state["head.3.weight"].shape[0] // cfg.D
    model = TransformerForecastModel(
        D=cfg.D,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=0.0,
        forecast_steps=forecast_steps,
        device=device,
    )
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(
        f"  Loaded Transformer from {ckpt_path}  (d_model={d_model}, nhead={nhead}, "
        f"num_layers={num_layers}, forecast_steps={forecast_steps})"
    )
    return model


def _load_pretrained_tree(
    label: str, model_key: str, cfg: DatasetConfig, device, model_ckpt_dir: Path
):
    """Load a tree-ensemble checkpoint from
    outputs/checkpoints/{ckpt_prefix}_{model_key}/best.joblib.

    Not a torch checkpoint — _train_tree_ensemble (pipeline/training_pipeline.py)
    saves a joblib pickle of {regressor, T, D, forecast_steps} since
    sklearn/xgboost regressors have no state_dict. Used by load_pretrained_rf.
    """
    import joblib

    ckpt_path = model_ckpt_dir / f"{cfg.ckpt_prefix}_{model_key}" / "best.joblib"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{label} checkpoint not found: {ckpt_path}")
    saved = joblib.load(ckpt_path)
    model = TreeForecastModel(
        D=saved["D"],
        T=saved["T"],
        forecast_steps=saved["forecast_steps"],
        device=device,
    )
    model.regressor = saved["regressor"]
    model = model.to(device)
    model.eval()
    print(
        f"  Loaded {label} from {ckpt_path}  (T={saved['T']}, D={saved['D']}, "
        f"forecast_steps={saved['forecast_steps']})"
    )
    return model


def load_pretrained_rf(cfg: DatasetConfig, device, model_ckpt_dir: Path):
    """Load Random Forest from outputs/checkpoints/{ckpt_prefix}_rf/best.joblib."""
    return _load_pretrained_tree("Random Forest", "rf", cfg, device, model_ckpt_dir)


def _orig_preds(x_test: np.ndarray, model: GRUForecastModel, device) -> np.ndarray:
    """Predict on x_test → (B, D) for single-step models.

    For multistep-trained models (forecast_steps > 1), uses
    model.predict_multistep() to get the full (B, H, D) horizon instead of
    model(..., return_all=False), which — even on a multistep-trained model —
    silently truncates to just the first forecast step (see
    LSTMForecastModel._apply_head / TCNForecastModel._apply_head: "take
    first forecast step's D outputs"). Every removal-based metric below
    (feature_removal_auc, timestep_removal_auc, ...) calls this for both the
    unmasked and masked predictions, so as long as both
    sides go through the same branch here, the H>1 case falls out for free —
    np.abs(a - b).mean() and MSE both work unchanged on (B, H, D) arrays.
    """
    x = torch.from_numpy(x_test.astype(np.float32)).to(device)
    with torch.no_grad():
        if getattr(model, "forecast_steps", 1) > 1 and hasattr(
            model, "predict_multistep"
        ):
            return model.predict_multistep(x).cpu().numpy()  # (B, H, D)
        return model(x, return_all=False).cpu().numpy()  # (B, D)


def select_var_order(x_train: np.ndarray, max_K: int = 8) -> int:
    """BIC-based VAR order selection.  x_train: (N, D, T)."""
    N, D, T = x_train.shape
    rng = np.random.default_rng(0)
    xtr = x_train[rng.choice(N, size=min(N, 500), replace=False)]
    n = len(xtr)
    best_K, best_bic = 1, np.inf
    for K in range(1, min(max_K + 1, T // 2)):
        X_in = np.array(
            [xtr[i, :, t - K : t].T.flatten() for i in range(n) for t in range(K, T)],
            dtype=np.float32,
        )
        X_out = np.array(
            [xtr[i, :, t] for i in range(n) for t in range(K, T)], dtype=np.float32
        )
        lr = Ridge(alpha=1e-3).fit(X_in, X_out)
        rss = float(np.sum((X_out - lr.predict(X_in)) ** 2))
        n_obs = X_in.shape[0] * D
        bic = n_obs * np.log(max(rss / n_obs, 1e-12)) + (K * D * D + D) * np.log(n_obs)
        if bic < best_bic:
            best_bic, best_K = bic, K
    return best_K


def fit_var_imputer(x_train: np.ndarray, K: int) -> Ridge:
    """Fit VAR(K) Ridge regressor for conditional imputation.  x_train: (N, D, T)."""
    N, D, T = x_train.shape
    xtr = x_train[np.random.default_rng(1).choice(N, size=min(N, 2000), replace=False)]
    n = len(xtr)
    X_in = np.array(
        [xtr[i, :, t - K : t].T.flatten() for i in range(n) for t in range(K, T)],
        dtype=np.float32,
    )
    X_out = np.array(
        [xtr[i, :, t] for i in range(n) for t in range(K, T)], dtype=np.float32
    )
    return Ridge(alpha=1e-3).fit(X_in, X_out)


def _impute_col(
    x: np.ndarray,  # (B, D, T) — partially masked window
    t_idx: int,
    var_imputer: Ridge | None,
    var_K: int,
    fallback: np.ndarray,  # (D,) marginal training mean for this column
    karma_b_star: np.ndarray | None = None,  # (T-K*, D) KARMA certified baseline
) -> np.ndarray:  # (B, D)
    """Impute a single time column using the best available method."""
    B, D, _ = x.shape

    if karma_b_star is not None and t_idx < len(karma_b_star):
        return np.broadcast_to(karma_b_star[t_idx], (B, D)).copy()
    if var_imputer is not None and t_idx >= var_K:
        ctx = x[:, :, t_idx - var_K : t_idx]  # (B, D, K)
        ctx_flat = ctx.transpose(0, 2, 1).reshape(B, var_K * D)  # (B, K*D) time-major
        return var_imputer.predict(ctx_flat).astype(np.float32)
    return np.broadcast_to(fallback, (B, D)).copy()


def attr_to_time_scores(attr: np.ndarray) -> np.ndarray:
    """(B, D, T) → (T,): mean |attribution| over batch and features per timestep."""
    return np.abs(attr).mean(axis=(0, 1))


def karma_edges_to_time_scores(edges: list, T: int) -> np.ndarray:
    """Sum ρ values per time index from KARMA edges.

    Edge with lag l maps to time index T-l.  Sums ρ across all edges sharing a
    time index (so a timestep targeted by many causal edges scores higher).
    """
    scores = np.zeros(T, dtype=np.float32)
    for e in edges:
        t_idx = T - e["lag"]
        if 0 <= t_idx < T:
            scores[t_idx] += e["rho"]
    return scores


def _segment_topk_mask(time_scores: np.ndarray, target_k: float) -> np.ndarray:
    """Boolean (T,) mask selecting whole contiguous runs of exactly-equal
    time_scores values, ranked by run value descending, greedily included
    until the running timestep count reaches target_k.

    Runs (not individual timesteps) are the atomic unit — the run that would
    push the running count past target_k is still included in full, never
    split (same boundary rule as removal_drop_at_frac_segmented's whole-
    segment inclusion). This matters for methods like ShapTime whose
    per-chunk value is broadcast identically across every timestep in the
    chunk (so a plain per-timestep argsort has no real signal to break ties
    with inside a chunk — see run_shaptime/run_tsmule) and, to a lesser,
    data-dependent extent, TS-MuLe.

    For methods without exact ties — the common case, since independently-
    computed real-valued attributions essentially never coincide exactly —
    every run has length 1 and this reduces to the previous flat top-k
    behaviour exactly, so this is a strict improvement, not a behaviour
    change, for every non-segment-flat method (KARMA, IG, FO, TIMING, ...).
    """
    T = len(time_scores)
    runs = []  # (value, start, end) half-open [start, end)
    start = 0
    for t in range(1, T + 1):
        if t == T or time_scores[t] != time_scores[start]:
            runs.append((time_scores[start], start, t))
            start = t
    runs.sort(key=lambda r: -r[0])

    mask = np.zeros(T, dtype=bool)
    running = 0
    for _, s, e in runs:
        if running >= target_k:
            break
        mask[s:e] = True
        running += e - s
    return mask


def timestep_removal_auc(
    x_test: np.ndarray,  # (B, D, T)
    time_scores: np.ndarray,  # (T,) importance per time index
    model,
    x_train_mean: np.ndarray,  # (D, T)
    device,
    n_steps: int = N_AUC_STEPS,
    var_imputer: Ridge | None = None,
    var_K: int = 0,
    karma_b_star: np.ndarray | None = None,  # (T-K*, D) KARMA certified baseline
) -> tuple[float, list[float]]:
    """AUC under progressive whole-timestep removal conditioned on the Markov blanket.

    Each masked timestep is imputed via VAR(K) conditional on its K predecessors
    (preserving temporal correlation) rather than replaced with the marginal mean.
    For KARMA, positions outside the identified blanket use the certified b* baseline.

    Removal order respects whole contiguous-equal-value runs in time_scores
    (see _segment_topk_mask) rather than a flat per-timestep ranking.
    """
    _, D, T = x_test.shape

    orig = _orig_preds(x_test, model, device)
    changes = []
    with torch.no_grad():
        for step in range(1, n_steps + 1):
            k = max(1, step * T // n_steps)
            mask = _segment_topk_mask(time_scores, k)
            x_masked = x_test.copy()
            # Impute in temporal order so each position can condition on
            # already-imputed predecessors (AR-consistent chain).
            for t_idx in np.where(mask)[0]:
                x_masked[:, :, t_idx] = _impute_col(
                    x_masked,
                    t_idx,
                    var_imputer,
                    var_K,
                    x_train_mean[:, t_idx],
                    karma_b_star,
                )
            new_pred = _orig_preds(x_masked, model, device)
            changes.append(float(np.abs(orig - new_pred).mean()))

    xs = np.linspace(1 / n_steps, 1.0, n_steps)
    return float(np.trapz(changes, xs)), changes


def timestep_drop_at_frac(
    x_test: np.ndarray,
    time_scores: np.ndarray,
    model,
    x_train_mean: np.ndarray,
    device,
    frac: float = 0.25,
    var_imputer: Ridge | None = None,
    var_K: int = 0,
    karma_b_star: np.ndarray | None = None,
) -> float:
    """Mean |Δpred| after blanking the top-frac fraction of timesteps.

    Removal set respects whole contiguous-equal-value runs in time_scores
    (see _segment_topk_mask) rather than a flat per-timestep ranking.
    """
    _, D, T = x_test.shape
    mask = _segment_topk_mask(time_scores, frac * T)
    x_masked = x_test.copy()
    for t_idx in np.where(mask)[0]:
        x_masked[:, :, t_idx] = _impute_col(
            x_masked,
            t_idx,
            var_imputer,
            var_K,
            x_train_mean[:, t_idx],
            karma_b_star,
        )
    orig = _orig_preds(x_test, model, device)
    new_pred = _orig_preds(x_masked, model, device)
    return float(np.abs(orig - new_pred).mean())


def attribution_complexity(attr: np.ndarray) -> float:
    """Fractional-contribution entropy (Bhatt et al. 2020).

    Computes the Shannon entropy of the normalized absolute attribution
    distribution over all (D × T) cells, averaged over the batch.

    Implements the complexity metric described in:
        Bhatt et al., "Evaluating and Aggregating Feature-based Model
        Explanations", IJCAI 2020.

    Note: captum 0.9 does not export this metric; we follow the same
    formula used internally by captum's evaluation suite.

    attr: (B, D, T)
    Returns mean entropy in nats (lower = simpler / more concentrated).
    """
    flat = np.abs(attr).reshape(len(attr), -1).astype(np.float64)  # (B, N)
    total = flat.sum(axis=1, keepdims=True).clip(min=1e-12)
    p = flat / total
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(p > 0, -p * np.log(p), 0.0)
    return float(h.sum(axis=1).mean())


def karma_edges_to_attr(edges: list, D: int, T: int) -> np.ndarray:
    """Convert KARMA edges to a (D, T) attribution matrix.

    Edge {"src": s, "lag": l, "rho": r} maps to window position (s, T-l).
    Multiple edges sharing the same (src, lag) are combined by max ρ.
    Returns a (D, T) array broadcastable to (B, D, T) for feature_removal_auc.
    """
    attr = np.zeros((D, T), dtype=np.float32)
    for e in edges:
        src, lag, rho = e["src"], e["lag"], e["rho"]
        t_idx = T - lag  # lag=1 → last timestep, lag=2 → second-to-last, …
        if 0 <= t_idx < T:
            attr[src, t_idx] = max(attr[src, t_idx], rho)
    return attr


def _multistep_target(model, x: torch.Tensor, tau: list = None) -> torch.Tensor:
    """(B, D) target for black-box/gradient explainers that only ever call the
    model with return_all=False and expect a single (B, D) tensor.

    For forecast_steps > 1 models, model(x, return_all=False) silently
    truncates to just the first forecast step (see LSTMForecastModel/
    TCNForecastModel._apply_head). This instead uses predict_multistep and
    averages over the tau forecast steps (default: all steps), so the
    resulting target genuinely reflects the whole horizon rather than step 0
    alone. Falls back to model(x, return_all=False) for single-step models.
    """
    if getattr(model, "forecast_steps", 1) > 1 and hasattr(model, "predict_multistep"):
        multi = model.predict_multistep(x)  # (B, forecast_steps, D)
        steps = tau if tau is not None else list(range(multi.shape[1]))
        return multi[:, steps, :].mean(dim=1)  # (B, D)
    return model(x, return_all=False)


def run_ig(
    x_test: np.ndarray, model: GRUForecastModel, device, tau: list = None
) -> np.ndarray:
    """Integrated Gradients → (B, D, T).  Zero baseline, sum over D outputs.

    For forecast_steps > 1 models, targets _multistep_target (mean over tau
    forecast steps via predict_multistep) instead of model(return_all=False),
    which would otherwise silently attribute only the first forecast step.
    predict_multistep is a plain differentiable forward pass, so it works
    unchanged under IG's path integral (Captum evaluates it at every
    interpolated point along the baseline->input path, not just the input).
    """
    model.eval()

    def _scalar_forward(x: torch.Tensor) -> torch.Tensor:
        return _multistep_target(model, x, tau).sum(dim=-1)  # (B,)

    ig = IntegratedGradients(_scalar_forward)
    x_t = torch.from_numpy(x_test.astype(np.float32)).to(device)
    orig_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    attr = ig.attribute(x_t, baselines=torch.zeros_like(x_t))  # (B, D, T)
    torch.backends.cudnn.enabled = orig_cudnn
    return np.abs(attr.detach().cpu().numpy())


def run_timeshap(
    x_test: np.ndarray,  # (B, D, T)
    model: GRUForecastModel,
    device,
    nsamples: int = 200,
    tau: list = None,
) -> np.ndarray:
    """TimeShap cell-level attribution → (B, D, T)."""
    from timeshap.explainer.kernel import TimeShapKernel

    B, D_feat, T = x_test.shape
    x_td = x_test.transpose(0, 2, 1).astype(np.float32)  # (B, T, D)
    background = np.zeros((1, T, D_feat), dtype=np.float32)
    model.eval()

    # LassoLarsIC (used inside TimeShap's kernel solver) requires
    # nsamples > n_features = T * D.  Silently floor up when needed.
    n_features = T * D_feat
    if nsamples <= n_features:
        nsamples_eff = n_features + max(50, n_features // 4)
        print(
            f"    [TimeShap] nsamples={nsamples} < n_features={n_features}; "
            f"raising to {nsamples_eff}"
        )
        nsamples = nsamples_eff

    def model_fn(x: np.ndarray) -> np.ndarray:
        # x: (N, T, D) numpy → (N, D, T) torch → scalar (N,)
        x_dt = torch.from_numpy(x.transpose(0, 2, 1)).to(device)
        with torch.no_grad():
            return _multistep_target(model, x_dt, tau).mean(dim=-1).cpu().numpy()

    varying = (list(range(T)), list(range(D_feat)))
    attr = np.zeros((B, D_feat, T), dtype=np.float32)
    for b in range(B):
        kernel = TimeShapKernel(
            model_fn, background, rs=42, mode="cell", varying=varying
        )
        sv = kernel.shap_values(x_td[b : b + 1], pruning_idx=0, nsamples=nsamples)
        # sv: (T*D,) ordered (t0_d0, t0_d1, …, t1_d0, …) → reshape (T, D) → (D, T)
        attr[b] = np.abs(sv.reshape(T, D_feat).T)
    return attr


def _timing_segment_ig(
    forward_func,
    x: torch.Tensor,
    target_channel: int,
    n_samples: int,
    num_segments: int,
    min_seg_len: int,
    max_seg_len: int | None,
) -> torch.Tensor:
    """Segment-based, temporality-aware Integrated Gradients: the core algorithm behind TIMING
    (Yoon et al., "TIMING: Temporality-Aware Integrated Gradients for Time Series Explanation",
    ICML 2025 Spotlight -- github.com/drumpt/TIMING, attribution/explainers.py::OUR). Ported
    from their `attribute_random_time_segments_one_dim_same_for_batch`, generalized from
    "gather one class logit" (classification) to "read one regression output coordinate" --
    the same segment-masked-interpolation/gradient-accumulation core, not a simplification.

    Instead of IG's usual per-coordinate straight-line path, interpolation steps randomly hold
    out whole contiguous (feature, time-segment) blocks at the input value rather than the
    interpolated one, so the path respects temporal structure instead of moving every point
    independently.

    x: (batch, D, T). Returns the (batch, D, T) attribution for output coordinate
    `target_channel`.
    """
    batch, D, T = x.shape
    baselines = x.mean(dim=0, keepdim=True).expand_as(x)
    max_seg_len = T if max_seg_len is None else min(T, max_seg_len)

    dev = x.device
    alphas = torch.linspace(0, 1 - 1 / n_samples, n_samples, device=dev).view(
        -1, 1, 1, 1
    )
    expanded_inputs = x.unsqueeze(0)
    expanded_baselines = baselines.unsqueeze(0)
    interpolated = expanded_baselines + alphas * (expanded_inputs - expanded_baselines)

    dims = torch.randint(0, D, (n_samples, batch, num_segments), device=dev)
    seg_lens = torch.randint(
        min_seg_len, max_seg_len + 1, (n_samples, batch, num_segments), device=dev
    )
    t_starts = (
        torch.rand(n_samples, batch, num_segments, device=dev) * (T - seg_lens)
    ).long()

    time_mask = torch.ones_like(interpolated)
    batch_indices = torch.arange(batch, device=dev)
    sample_indices = torch.arange(n_samples, device=dev)
    for s in range(num_segments):
        max_len = int(seg_lens[:, :, s].max().item())
        base_range = torch.arange(max_len, device=dev).view(1, 1, -1)
        indices = t_starts[:, :, s].unsqueeze(-1) + base_range
        end_points = (t_starts[:, :, s] + seg_lens[:, :, s]).unsqueeze(-1)
        valid = (indices < end_points) & (indices < T)
        idx = (indices * valid).clamp(0, T - 1)
        time_mask[
            sample_indices.view(-1, 1, 1),
            batch_indices.view(1, -1, 1),
            dims[:, :, s].unsqueeze(-1),
            idx,
        ] = 0

    fixed_inputs = expanded_inputs.detach()
    masked_inputs = time_mask * interpolated + (1 - time_mask) * fixed_inputs
    masked_inputs.requires_grad_(True)

    preds = forward_func(masked_inputs.reshape(-1, D, T))
    preds = preds.reshape(n_samples, batch, -1)
    target = preds[:, :, target_channel]
    total = target.sum()

    (grad,) = torch.autograd.grad(outputs=total, inputs=masked_inputs)
    grad = grad * time_mask
    grads = grad.sum(dim=0)
    denom = time_mask.sum(dim=0) + 1e-8
    return grads * (x - baselines) / denom


def run_timing(
    x_test: np.ndarray,  # (B, D, T)
    model,
    device,
    n_samples: int = 50,
    tau: list = None,
) -> np.ndarray:
    """TIMING attribution → (B, D, T) (Jang et al., ICML 2025).

    Temporality-aware integrated gradients: randomly fixes positions during
    path integration so gradients are conditioned on temporal context.
    Adapted here for multi-output regression by summing all outputs.

    For forecast_steps > 1 models, forward_func targets _multistep_target
    (mean over tau forecast steps) instead of the model directly, which
    would otherwise silently attribute only the first forecast step —
    _timing_segment_ig only ever calls forward_func(x) -> (N, D) and reads
    off target_channel, so the (B, D)-shaped _multistep_target output is a
    drop-in replacement.
    """
    model.eval()
    _, D, T = x_test.shape

    x = torch.from_numpy(x_test.astype(np.float32)).to(device)  # (B, D, T)
    forward_func = lambda xin: _multistep_target(model, xin, tau)  # noqa: E731
    per_output = []
    for d_out in range(D):
        attr = _timing_segment_ig(
            forward_func,
            x,
            d_out,
            n_samples,
            num_segments=3,
            min_seg_len=1,
            max_seg_len=1,
        )
        scores_dt = attr.abs().mean(dim=0).detach().cpu().numpy()
        per_output.append(scores_dt)
    return np.stack(per_output, axis=0)  # (D, T)


def run_tsmule(
    x_test: np.ndarray,  # (B, D, T)
    model,
    device,
    n_samples: int = 100,
    tau: list = None,
) -> tuple[np.ndarray, np.ndarray]:
    """TS-MuLe attribution → (attr, seg_labels), both (B, D, T) (Schlegel et al.,
    "Ts-mule: Local Interpretable Model-Agnostic Explanations for Time Series
    Forecast Models", github.com/dbvis-ukon/ts-mule).

    LIME for time series: matrix-profile segments each instance's window, perturbs
    segments on/off, and fits a linear surrogate mapping segment on/off vectors to
    the (scalar) model output. Operates on one instance (T, D) at a time, so we loop
    over the batch — mirrors run_timeshap's per-sample loop below.

    Note: ts-mule's own default `segmentation_method='slopes-max'` (xai/lime.py)
    is not one of the methods MatrixProfileSegmentation.segment() actually
    implements ('slopes-sorted' | 'slopes-not-sorted' | 'bins-max' | 'bins-min'),
    so calling .explain() with defaults raises ValueError unconditionally; we pass
    'slopes-sorted' explicitly to work around their bug.

    seg_labels[b]: each cell's matrix-profile segment id for instance b, recovered
    by independently re-calling the same (deterministic) segmenter explain() used
    internally — LimeBase.explain() computes this segmentation but never exposes
    it (its segment_coef/coef properties are dead code in this library version,
    never assigned). Lets callers do segment-respecting top-k selection (see
    removal_drop_at_frac_segmented) instead of a flat cell-level ranking that can
    arbitrarily split TS-MuLe's actually-uniform segments — every cell in a
    segment shares the exact same value in `attr` (see to_original() in the
    vendored lime.py), so a plain per-cell argsort ranking has no real signal to
    break ties with; segment id lets a caller treat the segment as the atomic unit.
    """
    from tsmule.xai.lime import LimeTS

    model.eval()

    def f(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return _multistep_target(model, x, tau).sum(dim=-1)  # (B,)

    B, D, T = x_test.shape

    def _predict_fn(x_td: np.ndarray) -> float:
        # x_td: (T, D) single instance -> (1, D, T) native model input
        x_dt = torch.from_numpy(x_td.T[np.newaxis].astype(np.float32)).to(device)
        return float(f(x_dt).item())

    attr = np.zeros((B, D, T), dtype=np.float32)
    seg_labels = np.zeros((B, D, T), dtype=np.int64)
    for b in range(B):
        explainer = LimeTS(n_samples=n_samples)
        x_td = x_test[b].T.astype(np.float64)  # (T, D)
        try:
            coef = explainer.explain(
                x_td, _predict_fn, segmentation_method="slopes-not-sorted"
            )
            seg_m = explainer._segmenter.segment(
                x_td, segmentation_method="slopes-not-sorted"
            )  # (T, D) int, deterministic — same call explain() made internally
        except Exception as e:
            print(f"    [TS-MuLe] sample {b} failed: {e}")
            coef = np.zeros((T, D))
            seg_m = np.arange(T * D).reshape(
                T, D
            )  # neutral: every cell its own segment
        attr[b] = np.abs(coef).T  # (D, T)
        seg_labels[b] = seg_m.T  # (D, T)
    return attr, seg_labels


def _shaptime_segment_shapley(
    forward_func,  # (B, D, T) tensor -> (B,) scalar tensor
    x: torch.Tensor,  # (B, D, T)
    baseline: torch.Tensor,  # (B, D, T) replacement value for masked-out timesteps
    Tn: int,
) -> np.ndarray:  # (B, Tn) exact Shapley value per time-chunk
    """Exact Shapley values over Tn contiguous time-chunks ("super-time" segments) —
    the core algorithm behind ShapTime (Zhang et al., "ShapTime", IntelliSys 2023
    submission, github.com/Zhangyuyi-0825/ShapTime,
    Training/RNN-based/ShapTimeRNN.py). Ported from their get_sub_set /
    ValFunction / ShapleyValues (full power-set enumeration plus the
    |S|!(N-|S|-1)!/N! Shapley weight), generalized from their dataset-level
    chronological chunks (their `supertime` slices a whole stacked dataset along
    the sample axis, so one Shapley value describes an entire test corpus) to
    within-instance temporal chunks of a single forecast window, so every
    instance gets its own (D, T)-shaped attribution comparable to every other
    method here. Value function v(S) = model prediction with only the timesteps
    belonging to segments in S kept at their true values (everything else
    replaced by `baseline`) — the same inclusion-based ValFunction, chunk unit
    swapped from dataset rows to window timesteps.
    """
    import math
    from itertools import combinations

    B, D, T = x.shape
    Tn = max(1, min(Tn, T))
    bounds = np.linspace(0, T, Tn + 1).round().astype(int)
    chunks = [list(range(bounds[i], bounds[i + 1])) for i in range(Tn)]

    all_subsets = []
    for r in range(Tn + 1):
        all_subsets.extend(combinations(range(Tn), r))

    def _mask_for(subset) -> torch.Tensor:
        m = torch.zeros(T, dtype=torch.bool, device=x.device)
        for c in subset:
            for t in chunks[c]:
                m[t] = True
        return m

    with torch.no_grad():
        v = {}
        for subset in all_subsets:
            mask = _mask_for(subset).view(1, 1, T)
            x_masked = torch.where(mask, x, baseline)
            v[subset] = forward_func(x_masked)  # (B,)

    shapley = torch.zeros(B, Tn, device=x.device)
    N = Tn
    for i in range(Tn):
        for subset in all_subsets:
            if i in subset:
                continue
            S_num = len(subset)
            weight = (
                math.factorial(S_num) * math.factorial(N - S_num - 1)
            ) / math.factorial(N)
            s_with_i = tuple(sorted(subset + (i,)))
            shapley[:, i] += weight * (v[s_with_i] - v[subset])

    return shapley.detach().cpu().numpy()  # (B, Tn)


def run_shaptime(
    x_test: np.ndarray,  # (B, D, T)
    model,
    device,
    Tn: int = 6,
    tau: list = None,
) -> tuple[np.ndarray, np.ndarray]:
    """ShapTime attribution → (attr, seg_labels), both (B, D, T) (Zhang et al.,
    "ShapTime", IntelliSys 2023 submission, github.com/Zhangyuyi-0825/ShapTime).

    See _shaptime_segment_shapley for the within-instance reinterpretation of
    ShapTime's exact-Shapley-over-time-chunks algorithm (their published version
    computes one Shapley value per chronological chunk of the whole dataset, not
    per instance — not directly comparable to this framework's per-cell AUC
    metrics). Each of the Tn time-chunk Shapley values is broadcast to every
    (feature, time) cell it covers — same flat-tie structure as TS-MuLe's
    segments, just fixed-width time chunks (identical across every d and every
    instance b, unlike TS-MuLe's per-instance matrix-profile segments) rather
    than data-dependent ones. seg_labels lets callers do segment-respecting
    top-k selection (see removal_drop_at_frac_segmented) instead of a flat
    cell-level ranking that can arbitrarily split a chunk.
    """
    model.eval()

    def f(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return _multistep_target(model, x, tau).sum(dim=-1)  # (B,)

    B, D, T = x_test.shape
    x = torch.from_numpy(x_test.astype(np.float32)).to(device)
    baseline = torch.zeros_like(x)

    time_scores = _shaptime_segment_shapley(f, x, baseline, Tn)  # (B, Tn_actual)
    Tn_actual = time_scores.shape[1]
    bounds = np.linspace(0, T, Tn_actual + 1).round().astype(int)
    attr = np.zeros((B, D, T), dtype=np.float32)
    seg_labels = np.zeros((B, D, T), dtype=np.int64)
    for i in range(Tn_actual):
        attr[:, :, bounds[i] : bounds[i + 1]] = np.abs(time_scores[:, i])[:, None, None]
        seg_labels[:, :, bounds[i] : bounds[i + 1]] = i  # same chunk id for every b, d
    return attr, seg_labels


def run_karma(
    X_train: np.ndarray,  # (N, T, D) windowed
    X_val: np.ndarray,  # (N, T, D) windowed
    model: GRUForecastModel,
    cfg: DatasetConfig,
    device,
    verbose: bool = True,
    seed: int = 42,
    tau: list = None,
) -> tuple[list, int, np.ndarray]:
    """KARMA pipeline.  Returns (edges, K_star, b_star).

    b_star: (W-K*, D) certified baseline for non-blanket prefix positions.

    tau : forecast step indices for the joint multistep variable importance
          (compute_variable_importance_multistep). Only used when the fitted
          estimator's horizon > 1 (i.e. the oracle's forecast_steps > 1 —
          NOT cfg.pred_horizon, which is a display-only field that can be
          stale relative to the checkpoint actually loaded). None pools all
          horizon steps jointly. Ignored when horizon == 1.
    """
    X_train_raw = reconstruct_raw_series(X_train)  # (T_raw, D)
    X_val_raw = reconstruct_raw_series(X_val)
    W = cfg.T  # use dataset window size as KARMA oracle window

    model.eval()

    def f_oracle(window: np.ndarray) -> np.ndarray:
        is_single = window.ndim == 2
        w = window[np.newaxis] if is_single else window  # (B, W, D)
        x = torch.from_numpy(w.transpose(0, 2, 1).astype(np.float32)).to(device)
        with torch.no_grad():
            if getattr(model, "forecast_steps", 1) > 1:
                pred = (
                    model.predict_multistep(x).cpu().numpy()
                )  # (B, forecast_steps, D)
            else:
                pred = model(x, return_all=False).cpu().numpy()  # (B, D)
        return pred[0] if is_single else pred

    disc = Discretiser(N=cfg.N)
    disc.fit(X_train_raw)

    T_val_raw = len(X_val_raw)
    X_val_windows = np.stack([X_val_raw[t : t + W] for t in range(T_val_raw - W)])

    result = select_K_and_baseline(
        f=f_oracle,
        X_train=X_train_raw,
        X_val=X_val_windows,
        disc=disc,
        W=W,
        eps=cfg.eps,
        K_max=cfg.K_max,
        loss="regression",
        verbose=verbose,
        seed=seed,
    )
    K_star = result["K_star"]
    b_star = result["b_star"]
    pi_star = result["pi_star"]
    if verbose:
        print(f"  K* = {K_star},  Δ_pred = {result['Delta_A1']:.4f}")

    pool = SuffixPool(disc=disc, K=K_star, W=W)
    pool.build(X_train_raw)

    tree = TreeKernelEstimator(
        disc,
        K=K_star,
        W=W,
        M=cfg.M,
        n_pool=cfg.n_pool_min,
        pool=pool,
        b_star=b_star,
        rng=np.random.default_rng(seed),
    )
    tree.fit(
        f=f_oracle,
        pi_star=pi_star,
        verbose=verbose,
        mega_batch=torch.cuda.is_available(),
    )

    horizon = getattr(tree, "horizon", 1)
    if horizon > 1:
        vi = compute_variable_importance_multistep(
            tree, pi_star, disc, tau=tau, lam=cfg.lam
        )
    else:
        vi = compute_variable_importance(tree, pi_star, disc, lam=cfg.lam)
    return vi["edges"], K_star, b_star


def _run_methods_for_model(
    arch: str,
    model,
    X_test_dt: np.ndarray,
    X_tr_dt: np.ndarray,
    X_val_dt: np.ndarray,
    X_train: np.ndarray,
    X_val: np.ndarray,
    cfg: "DatasetConfig",
    args,
    ckpt_dir: Path,
    x_train_mean: np.ndarray,
    device: torch.device,
    var_imputer: Ridge | None = None,
    var_K: int = 0,
) -> dict:
    """Run all attribution methods for one model; keys are prefixed with arch."""
    r: dict = {}
    p = f"{arch}_"  # key prefix
    T = X_test_dt.shape[2]

    edges_k, K_star_k, b_star_k = None, None, None
    if not args.skip_karma and cfg.D <= MAX_KARMA_D:
        edges_k, K_star_k, b_star_k = run_karma(
            X_train,
            X_val,
            model,
            cfg,
            device,
            verbose=not args.quiet,
            seed=args.seed,
            tau=args.tau,
        )

    def _lag_auc(ts):
        return timestep_removal_auc(
            X_test_dt,
            ts,
            model,
            x_train_mean,
            device,
            var_imputer=var_imputer,
            var_K=var_K,
            karma_b_star=b_star_k,
        )

    def _lag_d25(ts):
        return timestep_drop_at_frac(
            X_test_dt,
            ts,
            model,
            x_train_mean,
            device,
            frac=0.25,
            var_imputer=var_imputer,
            var_K=var_K,
            karma_b_star=b_star_k,
        )

    def _store(m, attr, time_scores):
        lag_auc, lag_ch = _lag_auc(time_scores)
        lag_d25 = _lag_d25(time_scores)
        r.update(
            {
                f"{p}{m}_lag_auc": lag_auc,
                f"{p}{m}_lag_changes": lag_ch,
                f"{p}{m}_lag_drop25": lag_d25,
            }
        )
        msg = f"    lag_AUC={lag_auc:.4f}  lag_drop@25%={lag_d25:.4f}"
        print(msg)

    def _store_local(m, attr):
        attr = np.abs(attr).mean(axis=1)
        lag_auc, lag_d25 = 0, 0
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

    if not args.skip_karma and cfg.D <= MAX_KARMA_D:
        print("\n  [KARMA]")
        attr_k = karma_edges_to_attr(edges_k, cfg.D, cfg.T)
        attr_k_batch = np.broadcast_to(attr_k[np.newaxis], X_test_dt.shape).copy()
        ts_k = karma_edges_to_time_scores(edges_k, T)
        _store("karma", attr_k_batch, ts_k)
        r[f"{p}karma_K_star"] = K_star_k
        r[f"{p}karma_n_edges"] = len(edges_k)
        print(f"    K*={K_star_k}  edges={len(edges_k)}")
    elif cfg.D > MAX_KARMA_D:
        r[f"{p}karma_auc"] = None
        r[f"{p}karma_lag_auc"] = None

    if is_differentiable and not args.skip_ig:
        print("\n  [IG]")
        attr = run_ig(X_test_dt, model, device, tau=args.tau)
        _store_local("ig", attr)

    if not args.skip_timeshap:
        print("\n  [TimeShap] (nsamples={args.ts_nsamples})")
        attr = run_timeshap(
            X_test_dt, model, device, nsamples=args.ts_nsamples, tau=args.tau
        )
        _store_local("timeshap", attr)

    if is_differentiable and not args.skip_timing:
        print("\n  [TIMING]")
        try:
            attr = run_timing(X_test_dt, model, device, tau=args.tau)
            _store_local("timing", attr)
        except Exception as e:
            print(f"    TIMING failed: {e}")

    return r


def run_dataset(
    name: str,
    args,
    device: torch.device,
    ckpt_dir: Path,
    out_dir: Path,
    model_ckpt_dir: Path,
) -> dict:
    cfg = DATASETS[name]
    print(f"\n{'='*60}")
    print(f"Dataset: {name}  D={cfg.D}  T={cfg.T}  pred_horizon={cfg.pred_horizon}")
    print(f"{'='*60}")

    X_train, X_val, X_test, y_train, y_val, _ = load_dataset(cfg)
    print(f"  Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

    n_test = min(N_TEST_LARGE_D if cfg.D > 20 else N_TEST_MAX, len(X_test))
    X_test_s = X_test[:n_test]
    X_tr_dt = X_train.transpose(0, 2, 1)
    X_val_dt = X_val.transpose(0, 2, 1)
    X_test_dt = X_test_s.transpose(0, 2, 1).astype(np.float32)  # (B, D, T)
    x_train_mean = X_tr_dt.mean(axis=0)  # (D, T)

    print("  Fitting VAR imputer for conditional timestep imputation …")
    var_K = select_var_order(X_tr_dt, max_K=8)
    var_imputer = fit_var_imputer(X_tr_dt, var_K)
    print(f"  VAR order K={var_K}")

    results = {
        "dataset": name,
        "D": cfg.D,
        "T": cfg.T,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": n_test,
        "var_K": var_K,
    }

    if not args.skip_gru:
        print("\n── GRU ──")
        gru = get_or_train_gru(cfg, X_train, y_train, X_val, y_val, device, ckpt_dir)
        results.update(
            _run_methods_for_model(
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
            )
        )

    if not args.skip_lstm:
        print("\n── LSTM ──")
        try:
            lstm = load_pretrained_lstm(cfg, device, model_ckpt_dir)
            results.update(
                _run_methods_for_model(
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
                )
            )
        except FileNotFoundError as e:
            print(f"  Skipping LSTM: {e}")

    if not args.skip_tcn:
        print("\n── TCN ──")
        try:
            tcn = load_pretrained_tcn(cfg, device, model_ckpt_dir)
            results.update(
                _run_methods_for_model(
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
                )
            )
        except FileNotFoundError as e:
            print(f"  Skipping TCN: {e}")

    if not args.skip_transformer:
        print("\n── Transformer ──")
        try:
            transformer = load_pretrained_transformer(cfg, device, model_ckpt_dir)
            results.update(
                _run_methods_for_model(
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
                )
            )
        except FileNotFoundError as e:
            print(f"  Skipping Transformer: {e}")

    if not args.skip_rf:
        print("\n── Random Forest ──")
        try:
            rf = load_pretrained_rf(cfg, device, model_ckpt_dir)
            results.update(
                _run_methods_for_model(
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
                )
            )
        except FileNotFoundError as e:
            print(f"  Skipping Random Forest: {e}")

    # Save per-dataset JSON
    out_file = out_dir / f"{name}_results.json"
    with open(out_file, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\n  → {out_file}")
    return results


def print_summary(all_results: list[dict]) -> None:
    archs = ["gru", "lstm", "tcn", "transformer", "rf"]
    methods = [
        "ig",
        "timeshap",
        "karma",
        "timing",
    ]
    col_w = 10
    ds_w = 18
    arch_w = 6
    width = ds_w + arch_w + col_w * len(methods)
    sep = "=" * width
    header = f"{'Dataset':<{ds_w}}{'Arch':<{arch_w}}" + "".join(
        f"{m.upper():>{col_w}}" for m in methods
    )

    metrics = [
        (
            "lag_auc",
            "Lag AUC      — whole-timestep removal  (higher = better, tests temporal structure)",
        ),
        ("lag_drop25", "Lag drop@25% — whole-timestep removal  (higher = better)"),
    ]
    for metric, label in metrics:
        print(f"\n{sep}")
        print(label)
        print(sep)
        print(header)
        print("-" * width)
        for r in all_results:
            for arch in archs:
                if not any(r.get(f"{arch}_{m}_{metric}") is not None for m in methods):
                    continue
                row = f"{r['dataset']:<{ds_w}}{arch:<{arch_w}}"
                for m in methods:
                    val = r.get(f"{arch}_{m}_{metric}")
                    row += f"{'—':>{col_w}}" if val is None else f"{val:>{col_w}.4f}"
                print(row)
        print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="AUC comparison on real forecasting datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        help="Datasets to evaluate (default: all)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--ckpt_dir", default="outputs/checkpoints/realdata")
    parser.add_argument("--out_dir", default="results/realdata")
    parser.add_argument("--skip_karma", action="store_true")
    parser.add_argument("--skip_ig", action="store_true")
    parser.add_argument("--skip_timeshap", action="store_true")
    parser.add_argument(
        "--ts_nsamples",
        type=int,
        default=200,
        help="KernelSHAP coalitions per window (TimeShap)",
    )
    parser.add_argument(
        "--lag_only",
        action="store_true",
        help="Compute only lag (whole-timestep) metrics; skip cell-level AUC",
    )
    parser.add_argument("--skip_lstm", action="store_true", help="Skip LSTM model")
    parser.add_argument("--skip_gru", action="store_true", help="Skip GRU model")
    parser.add_argument("--skip_tcn", action="store_true", help="Skip TCN model")
    parser.add_argument(
        "--skip_transformer", action="store_true", help="Skip Transformer model"
    )

    parser.add_argument(
        "--model_ckpt_dir",
        default="outputs/checkpoints",
        help="Root dir containing {prefix}_lstm/ and {prefix}_tcn/ checkpoint folders",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress KARMA verbosity")
    parser.add_argument("--skip_timing", action="store_true")
    parser.add_argument("--fit_epochs", type=int, default=100)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global seed for numpy/torch/random and KARMA's TreeKernelEstimator",
    )
    parser.add_argument(
        "--tau",
        type=str,
        default=None,
        help="Comma-separated forecast step indices for KARMA's joint multistep "
        "variable importance, e.g. '0,1,2'. Only takes effect on datasets whose "
        "checkpoint has forecast_steps > 1; defaults to pooling all forecast "
        "steps jointly. Ignored (single-step) when forecast_steps == 1.",
    )
    args = parser.parse_args()

    args.tau = [int(t) for t in args.tau.split(",")] if args.tau else None

    set_global_seed(args.seed)

    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir)
    model_ckpt_dir = Path(args.model_ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}  |  seed: {args.seed}")

    all_results = []
    for ds_name in args.datasets:
        if ds_name not in DATASETS:
            print(f"Unknown dataset '{ds_name}'. Available: {list(DATASETS.keys())}")
            continue
        try:
            r = run_dataset(ds_name, args, device, ckpt_dir, out_dir, model_ckpt_dir)
            all_results.append(r)
        except Exception as e:
            print(f"\nERROR on dataset '{ds_name}': {e}")
            traceback.print_exc()

    if all_results:
        print_summary(all_results)
        summary_path = out_dir / "summary.json"
        with open(summary_path, "w") as fh:
            json.dump(all_results, fh, indent=2, default=float)
        print(f"\nFull summary → {summary_path}")


if __name__ == "__main__":
    main()
