import numpy as np
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

SEED = 42
rng = np.random.default_rng(SEED)


def lorenz96(x, F=8.0):
    """
    Lorenz-96 model (cyclic indices).

    dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    """
    N = len(x)
    dx = np.zeros(N)

    for i in range(N):
        dx[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F

    return dx


def simulate_l96(N=5, T=1000, dt=0.01, F=8.0, noise_std=0.0, rng=rng):
    """
    Simulate Lorenz-96 time series.

    Args:
        N: number of variables (dimensions)
        T: number of time steps
        dt: integration step size
        F: forcing constant (chaos ~8)
        noise_std: Gaussian noise level

    Returns:
        X: shape (T, N)
    """
    X = np.zeros((T, N))

    # initial condition (small perturbation)
    x = F * np.ones(N)
    x[0] += 0.01

    for t in range(T):
        # Runge-Kutta 4 (RK4)
        k1 = lorenz96(x, F)
        k2 = lorenz96(x + 0.5 * dt * k1, F)
        k3 = lorenz96(x + 0.5 * dt * k2, F)
        k4 = lorenz96(x + dt * k3, F)

        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if noise_std > 0:
            x += rng.normal(0, noise_std, size=N)

        X[t] = x

    return X


def run_pcmci(data, K, var_names, alpha=0.05):
    dataframe = pp.DataFrame(data, datatime=np.arange(len(data)), var_names=var_names)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmci(tau_max=K, pc_alpha=alpha)

    edges = []
    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]
    for tgt in range(data.shape[1]):
        for src in range(data.shape[1]):
            for lag in range(1, K + 1):
                p = p_matrix[src, tgt, lag]
                val = val_matrix[src, tgt, lag]
                if p < alpha:
                    edges.append(
                        {
                            "src": src,
                            "tgt": tgt,
                            "lag": lag,
                            "src_name": var_names[src],
                            "tgt_name": var_names[tgt],
                            "p_value": round(float(p), 5),
                            "coef": round(float(val), 5),
                        }
                    )
    return edges
