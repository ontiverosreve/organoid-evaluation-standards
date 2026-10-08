"""Shared honest-controls audit library (mirrors Stage C canonical pipeline).

Canonical setup (DO NOT change):
  - 150 reservoir nodes, spectral radius 0.9, sparsity 0.1
  - 50 ms bins, 10 s blocks (200 bins)
  - leak-rate alphas [0.005, 0.01, 0.02], seeds [7, 11, 22]
  - quality-weighted ridge readout, lambda 1e-3
  - label-shuffle nulls: 20 reps (release convention)
"""
import numpy as np

N_RESERVOIR = 150
SEEDS = [7, 11, 22]
ALPHAS = [0.005, 0.01, 0.02]
LAM = 1e-3
N_SHUFFLE = 20


def build_reservoir(seed, n_channels, n_reservoir=N_RESERVOIR,
                    spectral_radius=0.9, sparsity=0.1):
    rng = np.random.default_rng(seed)
    W_res = rng.uniform(-1, 1, (n_reservoir, n_reservoir))
    W_res *= rng.random((n_reservoir, n_reservoir)) < sparsity
    W_res *= spectral_radius / np.max(np.abs(np.linalg.eigvals(W_res)))
    W_in = rng.uniform(-1, 1, (n_reservoir, n_channels))
    return W_res, W_in


def run_leaky_reservoir(block, W_res, W_in, alpha):
    n_samples, n_reservoir = block.shape[0], W_res.shape[0]
    states = np.zeros((n_samples, n_reservoir))
    for i in range(1, n_samples):
        u = block[i - 1]
        pre = np.tanh(W_res @ states[i - 1] + W_in @ u)
        states[i] = (1 - alpha) * states[i - 1] + alpha * pre
    return states.mean(axis=0)


def block_quality_weight(block):
    channel_std = block.std(axis=0)
    dead_fraction = np.mean(channel_std < 1e-3)
    return max(1.0 - dead_fraction, 0.1)


def ridge_fit(X_train, y_train, w_train, lam=LAM):
    """Weighted ridge regression to one-hot targets. Returns W_out."""
    Y = np.eye(2)[y_train]
    W_sqrt = np.sqrt(w_train)[:, None]
    return np.linalg.solve(
        (X_train * W_sqrt).T @ (X_train * W_sqrt) + lam * np.eye(X_train.shape[1]),
        (X_train * W_sqrt).T @ (Y * W_sqrt),
    )


def accuracy(blocks, labels, weights, train_idx, test_idx, alpha, seed):
    """Full pipeline for one (alpha, seed): reservoir states + ridge readout."""
    W_res, W_in = build_reservoir(seed, blocks.shape[2])
    states = np.array([run_leaky_reservoir(b, W_res, W_in, alpha) for b in blocks])
    W_out = ridge_fit(states[train_idx], labels[train_idx], weights[train_idx])
    preds = np.argmax(states[test_idx] @ W_out, axis=1)
    return float(np.mean(preds == labels[test_idx])), states, W_out


def interleave_split(n):
    tr = np.arange(n) % 3 != 0
    return tr, ~tr


def chronological_split(meta, frac=2 / 3):
    """Train on first frac of blocks within each file tag, test on the rest."""
    tr = np.zeros(len(meta), dtype=bool)
    tags = {}
    for i, m in enumerate(meta):
        tags.setdefault(m[0], []).append(i)
    for tag, idx in tags.items():
        idx = np.array(idx)
        order = np.argsort([meta[i][2] for i in idx])  # by start time
        k = int(len(idx) * frac)
        tr[idx[order[:k]]] = True
    return tr, ~tr


def sweep_alphas(blocks, labels, weights, meta, split_fn, verbose=False):
    """Run alpha x seed sweep. Returns dict alpha -> dict with accs, states, W_outs."""
    n = len(blocks)
    train_idx, test_idx = split_fn(n) if split_fn.__name__ == '<lambda>' else split_fn(meta)
    out = {}
    for alpha in ALPHAS:
        accs, states_l, wouts = [], [], []
        for seed in SEEDS:
            acc, states, W_out = accuracy(blocks, labels, weights,
                                          train_idx, test_idx, alpha, seed)
            accs.append(acc); states_l.append(states); wouts.append(W_out)
        out[alpha] = {"acc": accs, "states": states_l, "W_outs": wouts,
                      "train_idx": train_idx, "test_idx": test_idx}
        if verbose:
            print(f"  alpha={alpha}: {np.mean(accs):.3f} +/- {np.std(accs):.3f} "
                  f"(seeds {[f'{a:.3f}' for a in accs]})", flush=True)
    return out


def label_shuffle_null(blocks, labels, weights, train_idx, test_idx,
                       alpha, seed=7, n_reps=N_SHUFFLE, rng_seed=0):
    """20-rep label-shuffle null at fixed (alpha, seed). Returns list of accs."""
    rng = np.random.default_rng(rng_seed)
    W_res, W_in = build_reservoir(seed, blocks.shape[2])
    states = np.array([run_leaky_reservoir(b, W_res, W_in, alpha) for b in blocks])
    accs = []
    for _ in range(n_reps):
        yl = rng.permutation(labels)
        W_out = ridge_fit(states[train_idx], yl[train_idx], weights[train_idx])
        preds = np.argmax(states[test_idx] @ W_out, axis=1)
        accs.append(float(np.mean(preds == yl[test_idx])))
    return accs
