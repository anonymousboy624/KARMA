#!/usr/bin/env python
"""
experiments/comparison_var_multi.py
──────────────────────────────────────────────────────────────────────────────
Multi-dataset comparison: KARMA vs  TS-MuLe, TimeSHAP, ShapTime, IG and TIMING on VAR processes of
increasing dimensionality and lag order.

Each dataset uses a randomly generated stationary VAR(K) process with a fixed
seed.  All methods explain the same analytical oracle (true DGP, no model
error).  Comparison metric: Kendall's τ vs G* (ground-truth coefficient
magnitudes) and pairwise between methods.

Configurations
--------------
  tiny    D=2  K=1  n_edges=2   T_train=2_000
  small   D=4  K=2  n_edges=5   T_train=5_000
  medium  D=4  K=3  n_edges=7   T_train=5_000
  large   D=6  K=3  n_edges=9   T_train=15_000
  xlarge  D=8  K=4  n_edges=14  T_train=30_000

Usage
-----
  python -m experiments.comparison_var_multi
  python -m experiments.comparison_var_multi --configs tiny small medium
"""

from __future__ import annotations

import abc
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import logging
import numpy as np
import torch
from torch.optim import Optimizer

torch.solve = lambda B, A: (torch.linalg.solve(A, B), None)
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import MultivariateNormal, constraints
from time import time
import itertools

from dataset.var import (
    build_coef_tensor,
    check_stationarity,
    simulate_var,
    edge_metrics,
)
from karma.causal_recovery.edge_contribution import compute_variable_importance
from karma.markov_approximation.markov_surrogacy import select_K_and_baseline
from karma.utils import check_design
from karma.utils.discretiser import Discretiser
from karma.utils.kernel_estimator import TreeKernelEstimator
from karma.utils.sampling import SuffixPool
from captum.attr import IntegratedGradients
from timeshap.explainer.kernel import TimeShapKernel

_ROOT = Path(__file__).parent.parent


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


def set_global_seed(seed: int) -> None:
    """Seed python's random, numpy's legacy global RNG, and torch (CPU + CUDA).

    Covers every randomness source in this pipeline that reads global RNG
    state rather than an explicit generator: TS-MuLe's perturbation sampling
    (np.random.choice), TIMING's segment sampling (torch.rand/randint), FIT's
    counterfactual sampling (torch.randn), and the VAR generator's own noise.
    Does NOT cover KARMA's TreeKernelEstimator, which takes its own explicit
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
class VARConfig:
    name: str
    D: int
    K: int
    n_edges: int
    T_train: int
    T_test: int = 1000
    noise_std: float = 0.3
    coef: float = 0.25
    seed: int = 42
    # Derived KARMA params — set after init
    N: int = 3  # discretiser bins (2 for large D to limit state space)
    W: int = 0  # oracle window; 0 = auto (K*3 + 2, min 8)
    M: int = 200  # MC draws per history
    n_pool_min: int = 5
    K_max: int = 0  # 0 = auto (K + 2)

    def __post_init__(self):
        if self.W == 0:
            self.W = max(self.K * 3 + 2, 8)
        if self.K_max == 0:
            self.K_max = self.K + 2
        # Fewer bins for high-D to keep state space tractable
        if self.D > 4 and self.N == 3:
            self.N = 2
        if self.D >= 6:
            self.M = max(50, 200 // self.D * 2)


ALL_CONFIGS: dict[str, VARConfig] = {
    "tiny": VARConfig("tiny", D=2, K=1, n_edges=2, T_train=2_000, T_test=500),
    "small": VARConfig("small", D=4, K=2, n_edges=5, T_train=5_000, T_test=1000),
    "medium": VARConfig("medium", D=4, K=3, n_edges=7, T_train=5_000, T_test=1000),
    "large": VARConfig("large", D=6, K=3, n_edges=9, T_train=15_000, T_test=2000),
    "xlarge": VARConfig("xlarge", D=8, K=4, n_edges=14, T_train=30_000, T_test=3000),
}


def generate_stationary_var(
    cfg: VARConfig,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """
    Generate a random stationary VAR(K) with D variables and n_edges causal
    edges.  Scales down coefficient magnitude until spectral radius < 0.97.
    Returns (A (D,D,K), true_edges).
    """
    rng = np.random.default_rng(cfg.seed)
    candidates = [
        (s, t, k)
        for s in range(cfg.D)
        for t in range(cfg.D)
        for k in range(1, cfg.K + 1)
    ]

    for attempt in range(50):
        idx = rng.choice(len(candidates), size=cfg.n_edges, replace=False)
        edges = [candidates[i] for i in sorted(idx)]

        coef = cfg.coef
        for _ in range(30):
            A = build_coef_tensor(edges, cfg.D, cfg.K, coef)
            _, stationary = check_stationarity(A, cfg.D, cfg.K)
            if stationary:
                return A, edges
            coef *= 0.85

        rng = np.random.default_rng(cfg.seed + attempt + 1)

    raise RuntimeError(f"Could not generate stationary VAR for config '{cfg.name}'")


class VARTorchModel(TorchModel):
    """Differentiable VAR(K) oracle.  Input (B, D, T), output (B, D) or (B, D, T)."""

    def __init__(self, A: np.ndarray, device):
        D_, K_ = A.shape[0], A.shape[2]
        super().__init__(feature_size=D_, num_states=D_, hidden_size=0, device=device)
        self.register_buffer("A_coef", torch.tensor(A, dtype=torch.float32))
        self.K_true = K_
        self.activation = nn.Identity()

    def forward(self, x: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        B, D_feat, T = x.shape
        K = min(self.K_true, T)
        if not return_all:
            pred = torch.zeros(B, D_feat, device=x.device, dtype=x.dtype)
            for k in range(K):
                pred = pred + x[:, :, T - 1 - k] @ self.A_coef[:, :, k]
            return pred
        else:
            out = torch.zeros(B, D_feat, T, device=x.device, dtype=x.dtype)
            for t in range(K, T):
                p = torch.zeros(B, D_feat, device=x.device, dtype=x.dtype)
                for k in range(K):
                    p = p + x[:, :, t - 1 - k] @ self.A_coef[:, :, k]
                out[:, :, t] = p
            return out


def make_numpy_oracle(A: np.ndarray):
    K = A.shape[2]

    def f_oracle(window: np.ndarray) -> np.ndarray:
        single = window.ndim == 2
        w = window[np.newaxis] if single else window
        B, W, _D = w.shape
        pred = np.zeros((B, _D))
        for k in range(min(K, W)):
            pred += w[:, W - 1 - k, :] @ A[:, :, k]
        return pred[0] if single else pred

    return f_oracle


def _mse_loss_multiple(Y_pred: torch.Tensor, Y_target: torch.Tensor) -> torch.Tensor:
    """MSE per sample: (N_area, B, T, D) → (B,)."""
    return ((Y_pred - Y_target) ** 2).mean(dim=[0, 2, 3])


def make_sliding_windows(X: np.ndarray, W: int):
    T, _D = X.shape
    xs = np.stack([X[t - W : t, :].T for t in range(W, T)]).astype(np.float32)
    ys = np.stack([X[t, :] for t in range(W, T)]).astype(np.float32)
    return xs, ys


def make_loaders(X: np.ndarray, W: int, batch_size: int = 64, val_frac: float = 0.1):
    x_all, y_all = make_sliding_windows(X, W)
    n_val = max(1, int(len(x_all) * val_frac))
    x_tr, y_tr = x_all[:-n_val], y_all[:-n_val]
    x_va, y_va = x_all[-n_val:], y_all[-n_val:]

    tr_batch_size = min(batch_size, len(x_tr))
    tr = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
        batch_size=tr_batch_size,
        shuffle=True,
        drop_last=len(x_tr) > tr_batch_size,
    )
    va = DataLoader(
        TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va)),
        batch_size=batch_size,
        shuffle=False,
    )
    return tr, va


def ground_truth_phi(A: np.ndarray, K_max: int) -> np.ndarray:
    D = A.shape[0]
    phi = np.zeros((D, K_max))
    phi[:, : A.shape[2]] = np.abs(A).sum(axis=1)
    return phi


def aggregate_to_phi(attr: np.ndarray, K_max: int, W: int) -> np.ndarray:
    D_src = attr.shape[1]
    phi = np.zeros((D_src, K_max))
    for k in range(1, K_max + 1):
        t_idx = W - k
        if 0 <= t_idx < W:
            phi[:, k - 1] = np.abs(attr[:, :, t_idx]).mean(axis=0)
    return phi


def aggregate_winit_to_phi(attr: np.ndarray, K_max: int) -> np.ndarray:
    D_src, window_size = attr.shape[1], attr.shape[3]
    phi = np.zeros((D_src, K_max))
    for k in range(1, K_max + 1):
        w_idx = window_size - k
        if 0 <= w_idx < window_size:
            phi[:, k - 1] = np.abs(attr[:, :, -1, w_idx]).mean(axis=0)
    return phi


def tau_vs_gt(phi: np.ndarray, phi_gt: np.ndarray) -> tuple[float, float]:
    r1 = phi.flatten()
    r2 = phi_gt.flatten()
    tau, p = kendalltau(r1, r2)
    return round(float(tau), 3), round(float(p), 4)


def run_karma(cfg, X_train, X_val, A, true_edges, lam, seed, verbose) -> dict:
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
        eps=0.00001,
        K_max=cfg.K_max,
        loss="regression",
        verbose=verbose,
        seed=seed,
    )
    K_star = result["K_star"]
    b_star, pi_star = result["b_star"], result["pi_star"]
    if verbose:
        print(f"    K* = {K_star},  Δ_pred = {result['delta_pred']:.4f}")

    check_design(
        disc,
        K=K_star,
        T_train=len(X_train),
        n_pool_min=cfg.n_pool_min,
        lam=lam,
        W=cfg.W,
    )

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
        rng=np.random.default_rng(seed),
    )
    tree.fit(f=f_oracle, pi_star=pi_star, verbose=verbose)

    vi = compute_variable_importance(tree, pi_star, disc, lam=lam)
    phi = np.zeros((cfg.D, cfg.K_max))
    phi[:, :K_star] = vi["phi"]
    em = edge_metrics(vi["edges"], true_edges, cfg.D, K_star)
    return {"phi": phi, "K_star": K_star, "edge_metrics": em}


def run_ig(x_test: np.ndarray, model: VARTorchModel, device) -> np.ndarray:
    """Integrated Gradients (zero baseline, sum over D outputs) → (B, D, W)."""
    model.eval()

    def _scalar_forward(x: torch.Tensor) -> torch.Tensor:
        return model(x, return_all=False).sum(dim=-1)  # (B,)

    ig = IntegratedGradients(_scalar_forward)
    x_t = torch.from_numpy(x_test.astype(np.float32)).to(device)
    orig_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    attr = ig.attribute(x_t, baselines=torch.zeros_like(x_t))  # (B, D, W)
    torch.backends.cudnn.enabled = orig_cudnn
    return np.abs(attr.detach().cpu().numpy())


def run_timeshap(
    x_test: np.ndarray,
    model: VARTorchModel,
    device,
    nsamples: int = 200,
) -> np.ndarray:
    """TimeShap cell-level attribution → (B, D, W).

    Explains each test window independently.  Cell SHAP values are reshaped
    (W, D) and transposed to (D, W) to match the (B, D, W) convention.
    """
    B, D_feat, W = x_test.shape
    x_td = x_test.transpose(0, 2, 1).astype(np.float32)  # (B, W, D)
    background = np.zeros((1, W, D_feat), dtype=np.float32)
    model.eval()

    def model_fn(x: np.ndarray) -> np.ndarray:
        x_dt = torch.from_numpy(x.transpose(0, 2, 1)).to(device)
        with torch.no_grad():
            return model(x_dt, return_all=False).mean(dim=-1).cpu().numpy()

    varying = (list(range(W)), list(range(D_feat)))
    attr = np.zeros((B, D_feat, W), dtype=np.float32)
    for b in range(B):
        kernel = TimeShapKernel(
            model_fn, background, rs=42, mode="cell", varying=varying
        )
        sv = kernel.shap_values(x_td[b : b + 1], pruning_idx=0, nsamples=nsamples)
        # sv: (W*D,) ordered (t0_d0, t0_d1, …, tW_dD) → reshape (W, D) → (D, W)
        attr[b] = np.abs(sv.reshape(W, D_feat).T)
    return attr


def _scalar_forward(model: VARTorchModel, device):
    """Wrap a (B, D, T) VAR model as (B, T, D) → (B,) scalar for mask learners."""
    model.eval()

    def f(x_td: torch.Tensor) -> torch.Tensor:
        x_dt = x_td.permute(0, 2, 1).to(device)
        with torch.no_grad():
            return model(x_dt, return_all=False).sum(dim=-1)

    return f


for _ext in ["TIMING", "ts-mule"]:
    _ext_p = str(_ROOT / _ext)
    if _ext_p not in sys.path:
        sys.path.insert(0, _ext_p)


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

    alphas = torch.linspace(0, 1 - 1 / n_samples, n_samples).view(-1, 1, 1, 1)
    expanded_inputs = x.unsqueeze(0)
    expanded_baselines = baselines.unsqueeze(0)
    interpolated = expanded_baselines + alphas * (expanded_inputs - expanded_baselines)

    dims = torch.randint(0, D, (n_samples, batch, num_segments))
    seg_lens = torch.randint(
        min_seg_len, max_seg_len + 1, (n_samples, batch, num_segments)
    )
    t_starts = (torch.rand(n_samples, batch, num_segments) * (T - seg_lens)).long()

    time_mask = torch.ones_like(interpolated)
    batch_indices = torch.arange(batch)
    sample_indices = torch.arange(n_samples)
    for s in range(num_segments):
        max_len = int(seg_lens[:, :, s].max().item())
        base_range = torch.arange(max_len).view(1, 1, -1)
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
) -> np.ndarray:
    """TIMING attribution → (B, D, T) (Jang et al., ICML 2025).

    Temporality-aware integrated gradients: randomly fixes positions during
    path integration so gradients are conditioned on temporal context.
    Adapted here for multi-output regression by summing all outputs.
    """
    model.eval()
    _, D, T = x_test.shape

    x = torch.from_numpy(x_test.astype(np.float32)).to(device)  # (B, D, T)
    per_output = []
    for d_out in range(D):
        attr = _timing_segment_ig(
            model,
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
) -> np.ndarray:
    """TS-MuLe attribution → (B, D, T) (Schlegel et al., "Ts-mule: Local Interpretable
    Model-Agnostic Explanations for Time Series Forecast Models",
    github.com/dbvis-ukon/ts-mule).

    LIME for time series: matrix-profile segments each instance's window, perturbs
    segments on/off, and fits a linear surrogate mapping segment on/off vectors to
    the (scalar) model output. Operates on one instance (T, D) at a time, so we
    loop over the batch.

    Note: ts-mule's own default `segmentation_method='slopes-max'` (xai/lime.py)
    is not one of the methods MatrixProfileSegmentation.segment() actually
    implements ('slopes-sorted' | 'slopes-not-sorted' | 'bins-max' | 'bins-min'),
    so calling .explain() with defaults raises ValueError unconditionally; we pass
    'slopes-sorted' explicitly to work around their bug.
    """
    from tsmule.xai.lime import LimeTS

    model.eval()
    B, D, T = x_test.shape
    f = _scalar_forward(model, device)  # (B, W, D) -> (B,)

    def _predict_fn(x_td: np.ndarray) -> float:
        x_td_t = torch.from_numpy(x_td[np.newaxis].astype(np.float32)).to(device)
        return float(f(x_td_t).item())

    attr = np.zeros((B, D, T), dtype=np.float32)
    for b in range(B):
        explainer = LimeTS(n_samples=n_samples)
        x_td = x_test[b].T.astype(np.float64)  # (T, D)
        try:
            coef = explainer.explain(
                x_td, _predict_fn, segmentation_method="slopes-sorted"
            )
        except Exception as e:
            print(f"    [TS-MuLe] sample {b} failed: {e}")
            coef = np.zeros((T, D))
        attr[b] = np.abs(coef).T  # (D, T)
    return attr


def _shaptime_segment_shapley(
    forward_func,  # (B, D, T) tensor -> (B,) scalar tensor
    x: torch.Tensor,  # (B, D, T)
    baseline: torch.Tensor,  # (B, D, T) replacement value for masked-out timesteps
    Tn: int,
) -> np.ndarray:  # (B, Tn_actual) exact Shapley value per time-chunk
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
) -> np.ndarray:
    """ShapTime attribution → (B, D, T) (Zhang et al., "ShapTime", IntelliSys 2023
    submission, github.com/Zhangyuyi-0825/ShapTime).

    See _shaptime_segment_shapley for the within-instance reinterpretation of
    ShapTime's exact-Shapley-over-time-chunks algorithm (their published version
    computes one Shapley value per chronological chunk of the whole dataset, not
    per instance — not directly comparable to this framework's per-cell metrics).
    Each of the Tn time-chunk Shapley values is broadcast to every (feature, time)
    cell it covers.
    """
    model.eval()

    def f(x_dt: torch.Tensor) -> torch.Tensor:
        # x_dt is (B, D, T) — VARTorchModel's native layout, no permute needed
        # (unlike _scalar_forward above, which bridges from mask nets' (B, T, D)).
        with torch.no_grad():
            return model(x_dt, return_all=False).sum(dim=-1)  # (B,)

    B, D, T = x_test.shape
    x = torch.from_numpy(x_test.astype(np.float32)).to(device)
    baseline = torch.zeros_like(x)

    time_scores = _shaptime_segment_shapley(f, x, baseline, Tn)  # (B, Tn_actual)
    Tn_actual = time_scores.shape[1]
    bounds = np.linspace(0, T, Tn_actual + 1).round().astype(int)
    attr = np.zeros((B, D, T), dtype=np.float32)
    for i in range(Tn_actual):
        attr[:, :, bounds[i] : bounds[i + 1]] = np.abs(time_scores[:, i])[:, None, None]
    return attr


def run_config(
    cfg: VARConfig,
    args,
    device: str,
    output_dir: Path,
) -> dict:
    print(f"\n{'═'*70}")
    print(
        f"  {cfg.name.upper()}   D={cfg.D}  K={cfg.K}  n_edges={cfg.n_edges}"
        f"  T_train={cfg.T_train:,}  N={cfg.N}  W={cfg.W}"
    )
    print(f"{'═'*70}")

    rng = np.random.default_rng(cfg.seed)
    A, true_edges = generate_stationary_var(cfg)

    X = simulate_var(A, cfg.D, cfg.K, cfg.T_train + cfg.T_test, cfg.noise_std, rng)
    X_train, X_test = X[: cfg.T_train], X[cfg.T_train :]

    phi_gt = ground_truth_phi(A, cfg.K_max)
    phis: dict[str, np.ndarray] = {"ground_truth": phi_gt}

    var_model = VARTorchModel(A, device=device).to(device)
    x_all, _ = make_sliding_windows(X_test, cfg.W)
    n_test = min(args.n_test, len(x_all))
    x_test_np = x_all[:n_test]

    train_loader, val_loader = make_loaders(X_train, cfg.W, batch_size=64)
    ckpt_dir = output_dir / cfg.name / "generators"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_karma:
        print("\n  KARMA")
        karma_out = run_karma(
            cfg,
            X_train,
            X_test,
            A,
            true_edges,
            lam=args.lam,
            seed=args.seed,
            verbose=args.verbose,
        )
        phis["karma"] = karma_out["phi"]
        em = karma_out["edge_metrics"]
        print(f"    P={em['precision']:.2f}  R={em['recall']:.2f}  F1={em['f1']:.2f}")

    if not args.skip_ig:
        print("\n  IG")
        attr_ig = run_ig(x_test_np, var_model, device)
        phis["ig"] = aggregate_to_phi(attr_ig, cfg.K_max, cfg.W)

    if not args.skip_timeshap:
        print("\n  TimeShap")
        attr_ts = run_timeshap(x_test_np, var_model, device, nsamples=args.ts_nsamples)
        phis["timeshap"] = aggregate_to_phi(attr_ts, cfg.K_max, cfg.W)

    if not args.skip_timing:
        print("\n  TIMING")
        attr_ti = run_timing(x_test_np, var_model, device)
        phis["timing"] = aggregate_to_phi(attr_ti, cfg.K_max, cfg.W)

    if not args.skip_tsmule:
        print("\n  TS-MuLe")
        try:
            attr_tsm = run_tsmule(
                x_test_np, var_model, device, n_samples=args.tsmule_samples
            )
            phis["tsmule"] = aggregate_to_phi(attr_tsm, cfg.K_max, cfg.W)
        except Exception as e:
            print(f"    TS-MuLe failed: {e}")
    if not args.skip_shaptime:
        print("\n  ShapTime")
        try:
            attr_st = run_shaptime(x_test_np, var_model, device, Tn=args.shaptime_Tn)
            phis["shaptime"] = aggregate_to_phi(attr_st, cfg.K_max, cfg.W)
        except Exception as e:
            print(f"    ShapTime failed: {e}")

    taus = {}
    for method, phi in phis.items():
        if method == "ground_truth":
            continue
        tau, p = tau_vs_gt(phi, phi_gt)
        taus[method] = (tau, p)

    return {
        "config": {
            "name": cfg.name,
            "D": cfg.D,
            "K": cfg.K,
            "n_edges": cfg.n_edges,
            "T_train": cfg.T_train,
        },
        "true_edges": [list(e) for e in true_edges],
        "karma_edge_metrics": (
            karma_out["edge_metrics"] if not args.skip_karma else None
        ),
        "karma_K_star": karma_out["K_star"] if not args.skip_karma else None,
        "phis": {m: p.tolist() for m, p in phis.items()},
        "tau_vs_gt": {m: {"tau": t, "p": pv} for m, (t, pv) in taus.items()},
    }


def print_summary(all_results: list[dict]) -> None:
    methods = [
        "karma",
        "ig",
        "timeshap",
        "timing",
        "tsmule",
        "shaptime",
    ]
    col_w = 14

    print(f"\n{'═'*70}")
    print("  Kendall's τ vs G*   (* p < 0.05)")
    print(f"{'═'*70}")

    hdr = f"  {'config':>8s}  {'D':>3s}  {'K':>3s}  {'T':>7s}"
    for m in methods:
        hdr += f"  {m:>{col_w}s}"
    print(hdr)
    print(f"  {'-'*66}")

    for res in all_results:
        cfg_info = res["config"]
        row = (
            f"  {cfg_info['name']:>8s}  {cfg_info['D']:>3d}  {cfg_info['K']:>3d}"
            f"  {cfg_info['T_train']:>7,d}"
        )
        for m in methods:
            entry = res["tau_vs_gt"].get(m)
            if entry is None:
                row += f"  {'—':>{col_w}s}"
            else:
                tau, p = entry["tau"], entry["p"]
                mark = "*" if p < 0.05 else " "
                row += f"  {tau:>+.3f}{mark}{'':>{col_w-7}s}"
        print(row)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--configs",
        nargs="+",
        default=list(ALL_CONFIGS.keys()),
        choices=list(ALL_CONFIGS.keys()),
        help="Which configs to run (default: all)",
    )
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--n_test", type=int, default=100)
    p.add_argument("--gen_epochs", type=int, default=50)
    p.add_argument("--dyn_epochs", type=int, default=100)
    p.add_argument("--winit_samples", type=int, default=3)
    p.add_argument("--skip_karma", action="store_true")
    p.add_argument(
        "--load_generators",
        action="store_true",
        help="Load saved WinIT generators instead of retraining",
    )
    p.add_argument("--skip_ig", action="store_true")
    p.add_argument("--skip_timeshap", action="store_true")
    p.add_argument(
        "--ts_nsamples",
        type=int,
        default=200,
        help="KernelSHAP coalitions per test window (TimeShap)",
    )
    p.add_argument("--skip_timing", action="store_true")
    p.add_argument("--skip_tsmule", action="store_true")
    p.add_argument(
        "--tsmule_samples",
        type=int,
        default=100,
        help="Perturbation samples per instance (TS-MuLe)",
    )
    p.add_argument("--skip_shaptime", action="store_true")
    p.add_argument(
        "--shaptime_Tn",
        type=int,
        default=6,
        help="Number of contiguous time-chunks for exact Shapley (ShapTime)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="results/comparison_var_multi")
    p.add_argument("--verbose", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  seed: {args.seed}")

    all_results = []
    for name in args.configs:
        cfg = ALL_CONFIGS[name]
        result = run_config(cfg, args, device, output_dir)
        all_results.append(result)

        # Save per-config JSON immediately
        with open(output_dir / f"{name}.json", "w") as fh:
            json.dump(result, fh, indent=2)

    print_summary(all_results)

    with open(output_dir / "summary.json", "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nResults → {output_dir}/")


if __name__ == "__main__":
    main()
