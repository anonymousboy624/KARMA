"""
scm_generator.py
----------------
SCM (Structural Causal Model) data generator for KARMA synthetic benchmark.

Generates multivariate time series from a time-series SCM with nonlinear
functional mechanisms, additive noise, and a known ground-truth causal DAG G*.
Uses tigramite PCMCI for G* recovery to enable the three-way comparison
required by the KARMA paper: G* vs PCMCI estimate vs KARMA G_f.

Key differences from var_generator.py:
  - Functional mechanisms can be nonlinear (sigmoid, threshold, product)
  - Noise can be non-Gaussian (Laplace, uniform) per variable
  - Causal structure is defined via explicit structural equations, not a
    coefficient matrix — closer to the do-calculus framing in KARMA Section 6.4
  - Spurious edge injected via a confounding latent variable (common factor)
    rather than a direct coefficient — tests KARMA's faithfulness under
    the common factor failure mode described in Section 8.1

Usage
-----
    python scm_generator.py

Outputs (saved to ./data/)
---------
    G_star_scm.npy          — true adjacency tensor  (D, D, K_true)  bool
    G_star_scm_edges.csv    — human-readable edge list
    train_scm.npy           — training series        (T_train, D)
    test_scm.npy            — test series            (T_test,  D)
    pcmci_scm_edges.csv     — edges recovered by PCMCI on train data
    metadata_scm.json       — all generation parameters
"""

import numpy as np

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

# ── parameters ───────────────────────────────────────────────────────────────
D = 4  # number of observed variables
K_TRUE = 3  # maximum lag in structural equations
T_TRAIN = 5_000
T_TEST = 1_000
BURN_IN = 500

VAR_NAMES = ["SPX", "SX5E", "NKY", "FTSE"]


def sigmoid(x, scale=1.0):
    return scale / (1.0 + np.exp(-x))


def soft_threshold(x, theta=0.5, scale=0.3):
    """Piecewise linear threshold — models regime-switching behaviour."""
    return scale * np.where(np.abs(x) > theta, np.sign(x) * (np.abs(x) - theta), 0.0)


def linear(x, coef=0.25):
    return coef * x


def product_interaction(x, y, coef=0.15):
    """Multiplicative interaction — models volatility amplification."""
    return coef * x * y


# Each variable X_d(t) is defined by a structural equation:
#   X_d(t) = f_d( parents ) + noise_d(t)
#
# True causal edges (mirroring VAR generator for comparability):
#   SPX(t-1)  -> SPX(t)    momentum          linear
#   SX5E(t-2) -> SPX(t)    transatlantic     sigmoid
#   NKY(t-1)  -> SX5E(t)   asian spillover   linear
#   FTSE(t-1) -> FTSE(t)   vol clustering    soft_threshold
#   SPX(t-3)  -> NKY(t)    cross-lag         linear
#   NKY(t-2)  -> FTSE(t)   cross-lag         linear
#   SX5E(t-3) -> SX5E(t)   mean reversion    sigmoid (sign-flipped)
#
# Spurious association via LATENT COMMON FACTOR Z(t):
#   Z(t) = 0.4*Z(t-1) + noise_z — AR(1) latent factor present only in train
#   Z(t-1) -> FTSE(t) and Z(t-1) -> NKY(t) during train period
#   This induces apparent FTSE(t-1) -> NKY(t) correlation without a direct edge,
#   testing the common-factor faithfulness failure mode from KARMA Section 8.1.
# ─────────────────────────────────────────────────────────────────────────────

TRUE_EDGES = [
    (0, 0, 1),  # SPX  -> SPX  lag 1
    (1, 0, 2),  # SX5E -> SPX  lag 2
    (2, 1, 1),  # NKY  -> SX5E lag 1
    (3, 3, 1),  # FTSE -> FTSE lag 1
    (0, 2, 3),  # SPX  -> NKY  lag 3
    (2, 3, 2),  # NKY  -> FTSE lag 2
    (1, 1, 3),  # SX5E -> SX5E lag 3
]

# The spurious association is NOT a direct edge — it is mediated by latent Z.
# So it does not appear in G* but may appear in PCMCI / G_f results.
SPURIOUS_LATENT = True  # toggle to disable the latent factor


def noise(size, dist="gaussian", scale=0.3, rng=rng):
    if dist == "gaussian":
        return rng.normal(0, scale, size)
    elif dist == "laplace":
        return rng.laplace(0, scale / np.sqrt(2), size)
    elif dist == "uniform":
        return rng.uniform(-scale * np.sqrt(3), scale * np.sqrt(3), size)
    raise ValueError(f"Unknown dist: {dist}")


def simulate_scm(T, with_latent=True, rng=rng):
    """
    Simulate T+BURN_IN steps, return (T, D) array after burn-in.

    Variable noise distributions are intentionally heterogeneous to
    distinguish the SCM from a Gaussian VAR:
      SPX  — Gaussian   (liquid, many small shocks)
      SX5E — Laplace    (fat-tailed, jump-like contagion)
      NKY  — Gaussian
      FTSE — Gaussian
    """
    total = T + BURN_IN
    X = np.zeros((total, D))
    Z = np.zeros(total)  # latent common factor

    # initialise
    X[:K_TRUE] = rng.normal(0, 0.3, (K_TRUE, D))
    Z[:K_TRUE] = rng.normal(0, 0.2, K_TRUE)

    for t in range(K_TRUE, total):
        # latent factor — AR(1), only affects NKY and FTSE during simulation
        Z[t] = 0.4 * Z[t - 1] + noise(1, "gaussian", 0.2, rng)[0]

        latent_effect_nky = 0.20 * Z[t - 1] if with_latent else 0.0
        latent_effect_ftse = 0.18 * Z[t - 1] if with_latent else 0.0

        # SPX(t): momentum (lag 1, linear) + transatlantic (lag 2, sigmoid)
        spx = (
            linear(X[t - 1, 0], coef=0.28)
            + sigmoid(X[t - 2, 1], scale=0.20)
            + noise(1, "gaussian", 0.30, rng)[0]
        )

        # SX5E(t): asian spillover (lag 1, linear) + mean reversion (lag 3, sigmoid flip)
        sx5e = (
            linear(X[t - 1, 2], coef=0.22)
            + sigmoid(-X[t - 3, 1], scale=0.18)  # sign flip = mean reversion
            + noise(1, "laplace", 0.25, rng)[0]
        )

        # NKY(t): cross-lag from SPX (lag 3, linear) + latent factor
        nky = (
            linear(X[t - 3, 0], coef=0.20)
            + latent_effect_nky
            + noise(1, "gaussian", 0.28, rng)[0]
        )

        # FTSE(t): vol clustering (lag 1, soft threshold) + cross-lag NKY (lag 2)
        #          + latent factor
        ftse = (
            soft_threshold(X[t - 1, 3], theta=0.4, scale=0.25)
            + linear(X[t - 2, 2], coef=0.18)
            + latent_effect_ftse
            + noise(1, "gaussian", 0.30, rng)[0]
        )

        X[t] = [spx, sx5e, nky, ftse]

    return X[BURN_IN:]


# def run_pcmci(data, K, var_names, alpha=0.05):
#     dataframe = pp.DataFrame(data, datatime=np.arange(len(data)), var_names=var_names)
#     pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
#     results = pcmci.run_pcmci(tau_max=K, pc_alpha=alpha)

#     edges = []
#     p_matrix = results["p_matrix"]
#     val_matrix = results["val_matrix"]
#     for tgt in range(data.shape[1]):
#         for src in range(data.shape[1]):
#             for lag in range(1, K + 1):
#                 p = p_matrix[src, tgt, lag]
#                 val = val_matrix[src, tgt, lag]
#                 if p < alpha:
#                     edges.append(
#                         {
#                             "src": src,
#                             "tgt": tgt,
#                             "lag": lag,
#                             "src_name": var_names[src],
#                             "tgt_name": var_names[tgt],
#                             "p_value": round(float(p), 5),
#                             "coef": round(float(val), 5),
#                         }
#                     )
#     return edges


# def edge_metrics(recovered_edges, true_edges):
#     true_set = {(e["src"], e["tgt"], e["lag"]) for e in true_edges}
#     rec_set = {(e["src"], e["tgt"], e["lag"]) for e in recovered_edges}
#     tp = len(true_set & rec_set)
#     fp = len(rec_set - true_set)
#     fn = len(true_set - rec_set)
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#     f1 = (
#         2 * precision * recall / (precision + recall)
#         if (precision + recall) > 0
#         else 0.0
#     )
#     return {
#         "precision": round(precision, 3),
#         "recall": round(recall, 3),
#         "f1": round(f1, 3),
#         "tp": tp,
#         "fp": fp,
#         "fn": fn,
#     }


# class SCM:
#     def __init__():
#         pass


# def main():
#     save_directory = "data/generated/scm"
#     os.makedirs(save_directory, exist_ok=True)

#     # ── simulate ─────────────────────────────────────────────────────────────
#     # Train: latent common factor active (induces spurious FTSE->NKY association)
#     # Test:  latent factor absent (spurious association disappears)
#     print("Simulating SCM train series (with latent factor) ...")
#     train = simulate_scm(T_TRAIN, with_latent=SPURIOUS_LATENT, rng=rng)
#     print("Simulating SCM test series  (no latent factor)   ...")
#     test = simulate_scm(T_TEST, with_latent=False, rng=rng)

#     print(f"\nTrain shape: {train.shape}   Test shape: {test.shape}")
#     print(f"Train mean:  {train.mean(axis=0).round(3)}")
#     print(f"Train std:   {train.std(axis=0).round(3)}")

#     # ── save arrays ──────────────────────────────────────────────────────────
#     np.save(save_directory + "/train_scm.npy", train)
#     np.save(save_directory + "/test_scm.npy", test)

#     # ── G* adjacency tensor ──────────────────────────────────────────────────
#     G_star = np.zeros((D, D, K_TRUE), dtype=bool)
#     for s, t, l in TRUE_EDGES:
#         G_star[s, t, l - 1] = True
#     np.save(save_directory + "/G_star_scm.npy", G_star)

#     edge_df = pd.DataFrame(
#         [
#             {
#                 "src": s,
#                 "tgt": t,
#                 "lag": l,
#                 "src_name": VAR_NAMES[s],
#                 "tgt_name": VAR_NAMES[t],
#                 "mechanism": [
#                     "linear",
#                     "sigmoid",
#                     "linear",
#                     "soft_threshold",
#                     "linear",
#                     "linear",
#                     "sigmoid_flip",
#                 ][i],
#                 "spurious": False,
#             }
#             for i, (s, t, l) in enumerate(TRUE_EDGES)
#         ]
#     )
#     edge_df.to_csv(save_directory + "/G_star_scm_edges.csv", index=False)
#     print(f"\nG* edges ({len(TRUE_EDGES)} true structural equations):")
#     print(edge_df.to_string(index=False))
#     print(f"\nLatent common factor active in train: {SPURIOUS_LATENT}")
#     print("  -> induces apparent FTSE(t-1)->NKY(t) via Z(t-1) [common factor]")
#     print("  -> absent from G* (no direct edge); tests faithfulness violation")

#     # ── PCMCI ────────────────────────────────────────────────────────────────
#     print("\nRunning PCMCI on SCM train data ...")
#     pcmci_edges = run_pcmci(train, K_TRUE, VAR_NAMES, alpha=0.05)
#     pcmci_df = pd.DataFrame(pcmci_edges)
#     pcmci_df.to_csv(save_directory + "/pcmci_scm_edges.csv", index=False)
#     print(f"PCMCI recovered {len(pcmci_edges)} edges:")
#     if len(pcmci_edges) > 0:
#         print(pcmci_df.to_string(index=False))

#     # check spurious latent-induced edge
#     spurious_recovered = any(
#         e["src"] == 3 and e["tgt"] == 2 and e["lag"] == 1 for e in pcmci_edges
#     )
#     print(f"\nSpurious FTSE->NKY (lag 1) recovered by PCMCI: {spurious_recovered}")
#     print("  (expected: possibly True — latent Z creates marginal correlation)")
#     print("  A model trained on this data may learn it; KARMA should report it in G_f.")

#     metrics = edge_metrics(pcmci_edges, TRUE_EDGES)
#     print(f"\nPCMCI vs G* (true structural edges only): {metrics}")

#     # ── nonlinearity diagnostics ─────────────────────────────────────────────
#     print("\nNonlinearity check — variable kurtosis (Gaussian = 3):")
#     for d, name in enumerate(VAR_NAMES):
#         m4 = np.mean((train[:, d] - train[:, d].mean()) ** 4)
#         s4 = train[:, d].std() ** 4
#         kurt = m4 / s4
#         print(
#             f"  {name}: kurtosis = {kurt:.2f}  "
#             f"{'fat-tailed' if kurt > 4 else 'approx Gaussian'}"
#         )

#     # ── metadata ─────────────────────────────────────────────────────────────
#     meta = {
#         "generator": "SCM",
#         "D": D,
#         "K_true": K_TRUE,
#         "T_train": T_TRAIN,
#         "T_test": T_TEST,
#         "burn_in": BURN_IN,
#         "mechanisms": [
#             "linear",
#             "sigmoid",
#             "linear",
#             "soft_threshold",
#             "linear",
#             "linear",
#             "sigmoid_flip",
#         ],
#         "noise_dists": {
#             "SPX": "gaussian",
#             "SX5E": "laplace",
#             "NKY": "gaussian",
#             "FTSE": "gaussian",
#         },
#         "spurious_via_latent": SPURIOUS_LATENT,
#         "spurious_description": "Latent AR(1) factor Z drives both NKY and FTSE, "
#         "inducing apparent FTSE(t-1)->NKY(t) in train only",
#         "true_edges": [list(e) for e in TRUE_EDGES],
#         "var_names": VAR_NAMES,
#         "seed": SEED,
#         "pcmci_metrics": metrics,
#         "pcmci_spurious_recovered": spurious_recovered,
#     }
#     with open(save_directory + "/metadata_scm.json", "w") as f:
#         json.dump(meta, f, indent=2)

#     print(
#         "\nSaved to "
#         + save_directory
#         + ": train_scm.npy, test_scm.npy, G_star_scm.npy, "
#         "G_star_scm_edges.csv, pcmci_scm_edges.csv, metadata_scm.json"
#     )


# if __name__ == "__main__":
#     main()
