"""Step 3: stimulation-amplitude decoding with honest controls (FS369).

Cohort: 120 whole-array stim timestamps (d1=d2=300, polarity=0, shape=0,
nb_pulse=2, period=10000), amplitudes {0.8, 1.0, 1.5} uA, 40 trials each,
4 days (2025-02-20/21/28, 2025-03-06), 30 trials/day. Chance = 1/3.

Canonical pipeline (audit_lib) ADAPTATIONS (documented here):
  1. Trials replace 10 s blocks: each trial is 50 one-ms bins x 32 channels
     (canonical used 50 ms bins x 200 bins). Reservoir (150 nodes, rho 0.9,
     sparsity 0.1) and run_leaky_reservoir are used UNCHANGED; the mean
     state over the 50 bins is the trial feature.
  2. ridge_fit is generalized from binary (np.eye(2)) to K-class one-hot
     (np.eye(K), K=3); lambda 1e-3 and quality weighting unchanged.
  3. label_shuffle_null is adapted to stratified 5-fold trial CV (20 reps).

Analyses:
  A. pooled 5-fold stratified CV, reservoir (alpha x seed) + logreg (secondary)
  B. per-day 5-fold stratified CV, reservoir
  C. artifact control: [10,50] ms window (0-10 ms blanked), reservoir + logreg
  D. baseline null: [-550,-500] ms window, reservoir + logreg (must be ~chance)
  E. label-shuffle null, 20 reps, fixed (alpha=0.01, seed=7), pooled CV
  F. cross-day transfer (§3.7): train 02-20 -> test 02-28, and reverse
  G. locality: per-electrode (50x1) pooled CV vs all-32 (50x32)

Outputs: results/decoding_amplitude_fs369.csv, results/cross_day_transfer_fs369.csv
"""
import numpy as np, pandas as pd, sys, time, csv
sys.path.insert(0, "/home/hatch/workspace/reservoir-eeg/work/audit_finalspark/code")
from audit_lib import (build_reservoir, run_leaky_reservoir, block_quality_weight,
                       ALPHAS, SEEDS, LAM, N_SHUFFLE)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

R = "/home/hatch/workspace/reservoir-eeg/work/audit_finalspark/results/"
npz = np.load(R + "fs369_trial_features.npz", allow_pickle=True)
X_full = npz["X_full"].astype(np.float64)   # (120, 50, 32)
X_base = npz["X_base"].astype(np.float64)   # (120, 50, 32)
y = npz["y"].astype(int)
days = npz["day"].astype(str)
DAYLIST = sorted(set(days.tolist()))
K = 3
CHANCE = 1.0 / K
print(f"trials={len(y)} classes={K} chance={CHANCE:.4f} days={DAYLIST}", flush=True)


# ---- adapted multiclass ridge (canonical adaptation #2) ----
def ridge_fit_K(X_train, y_train, w_train, lam=LAM):
    Y = np.eye(K)[y_train]
    W_sqrt = np.sqrt(w_train)[:, None]
    return np.linalg.solve(
        (X_train * W_sqrt).T @ (X_train * W_sqrt) + lam * np.eye(X_train.shape[1]),
        (X_train * W_sqrt).T @ (Y * W_sqrt),
    )


def trial_states(X, alpha, seed):
    W_res, W_in = build_reservoir(seed, X.shape[2])
    return np.array([run_leaky_reservoir(b, W_res, W_in, alpha) for b in X])


def reservoir_cv(X, y_idx, alpha, seed, n_splits=5, rs=7):
    """Mean/sd accuracy over stratified folds; quality-weighted canonical readout."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
    states = trial_states(X, alpha, seed)
    w = np.array([block_quality_weight(b) for b in X])
    accs = []
    for tr, te in skf.split(states, y_idx):
        W_out = ridge_fit_K(states[tr], y_idx[tr], w[tr])
        pred = np.argmax(states[te] @ W_out, axis=1)
        accs.append(float(np.mean(pred == y_idx[te])))
    return float(np.mean(accs)), float(np.std(accs)), accs


def logreg_cv(X, y_idx, n_splits=5, rs=7):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
    Xf = X.reshape(len(X), -1)
    accs = []
    for tr, te in skf.split(Xf, y_idx):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0, random_state=rs))
        clf.fit(Xf[tr], y_idx[tr])
        accs.append(float(clf.score(Xf[te], y_idx[te])))
    return float(np.mean(accs)), float(np.std(accs))


def shuffle_null_cv(X, y_idx, alpha, seed, n_reps=N_SHUFFLE, rs=7):
    """Label-shuffle null at fixed (alpha, seed): same pooled 5-fold CV, permuted labels."""
    rng = np.random.default_rng(0)
    states = trial_states(X, alpha, seed)
    w = np.array([block_quality_weight(b) for b in X])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
    folds = list(skf.split(states, y_idx))
    out = []
    for _ in range(n_reps):
        yl = rng.permutation(y_idx)
        a = []
        for tr, te in folds:
            W_out = ridge_fit_K(states[tr], yl[tr], w[tr])
            pred = np.argmax(states[te] @ W_out, axis=1)
            a.append(float(np.mean(pred == yl[te])))
        out.append(float(np.mean(a)))
    return out


rows = []   # decoding_amplitude_fs369.csv
xrows = []  # cross_day_transfer_fs369.csv
t0 = time.time()


def add(analysis, window, model, day, alpha, seed, n_trials, acc_mean, acc_sd,
        electrode="", null_mean="", null_sd=""):
    rows.append({"analysis": analysis, "window": window, "model": model, "day": day,
                 "electrode": electrode, "alpha": alpha, "seed": seed,
                 "n_trials": n_trials, "chance": round(CHANCE, 4),
                 "acc_mean": round(acc_mean, 4), "acc_sd": round(acc_sd, 4),
                 "null_mean": null_mean, "null_sd": null_sd})


def run_reservoir_grid(analysis, X, y_idx, day_label, n_trials, win_label):
    for alpha in ALPHAS:
        for seed in SEEDS:
            m, s, _ = reservoir_cv(X, y_idx, alpha, seed)
            add(analysis, win_label, "reservoir_ridge", day_label, alpha, seed,
                n_trials, m, s)
            print(f"  {analysis} {day_label} a={alpha} s={seed}: {m:.3f}+/-{s:.3f}", flush=True)
    # summary row across the 9 combos
    vals = [r["acc_mean"] for r in rows
            if r["analysis"] == analysis and r["day"] == day_label
            and r["model"] == "reservoir_ridge" and r["window"] == win_label]
    add(analysis, win_label, "reservoir_ridge", day_label, "ALL", "ALL",
        n_trials, float(np.mean(vals)), float(np.std(vals)))


# ---------- A. pooled CV ----------
print("A. pooled 5-fold CV, reservoir", flush=True)
run_reservoir_grid("pooled_cv", X_full, y, "all", len(y), "[0,50]ms")
m, s = logreg_cv(X_full, y)
add("pooled_cv", "[0,50]ms", "logreg", "all", "", "", len(y), m, s)
print(f"  pooled logreg: {m:.3f}+/-{s:.3f}", flush=True)

# ---------- B. per-day CV ----------
print("B. per-day 5-fold CV, reservoir", flush=True)
for d in DAYLIST:
    idx = np.nonzero(days == d)[0]
    run_reservoir_grid("perday_cv", X_full[idx], y[idx], d, len(idx), "[0,50]ms")
    m, s = logreg_cv(X_full[idx], y[idx])
    add("perday_cv", "[0,50]ms", "logreg", d, "", "", len(idx), m, s)
    print(f"  perday {d} logreg: {m:.3f}+/-{s:.3f}", flush=True)

# ---------- C. artifact control [10,50]ms ----------
print("C. artifact control [10,50]ms", flush=True)
X_art = X_full[:, 10:, :]
run_reservoir_grid("artifact_blanked", X_art, y, "all", len(y), "[10,50]ms")
m, s = logreg_cv(X_art, y)
add("artifact_blanked", "[10,50]ms", "logreg", "all", "", "", len(y), m, s)
print(f"  artifact logreg: {m:.3f}+/-{s:.3f}", flush=True)

# ---------- D. baseline null ----------
print("D. baseline null [-550,-500]ms", flush=True)
run_reservoir_grid("baseline_null", X_base, y, "all", len(y), "[-550,-500]ms")
m, s = logreg_cv(X_base, y)
add("baseline_null", "[-550,-500]ms", "logreg", "all", "", "", len(y), m, s)
print(f"  baseline logreg: {m:.3f}+/-{s:.3f}", flush=True)

# ---------- E. label-shuffle null ----------
print("E. label-shuffle null (20 reps, alpha=0.01, seed=7)", flush=True)
nulls = shuffle_null_cv(X_full, y, alpha=0.01, seed=7)
m_true, s_true, _ = reservoir_cv(X_full, y, alpha=0.01, seed=7)
add("shuffle_null", "[0,50]ms", "reservoir_ridge", "all", 0.01, 7, len(y),
    m_true, s_true, null_mean=round(float(np.mean(nulls)), 4),
    null_sd=round(float(np.std(nulls)), 4))
print(f"  true={m_true:.3f} null={np.mean(nulls):.3f}+/-{np.std(nulls):.3f} "
      f"max={np.max(nulls):.3f}", flush=True)

# ---------- F. cross-day transfer (§3.7) ----------
print("F. cross-day transfer", flush=True)
for d_tr, d_te in [("2025-02-20", "2025-02-28"), ("2025-02-28", "2025-02-20")]:
    tr = np.nonzero(days == d_tr)[0]; te = np.nonzero(days == d_te)[0]
    for alpha in ALPHAS:
        for seed in SEEDS:
            states = trial_states(X_full, alpha, seed)
            w = np.array([block_quality_weight(b) for b in X_full])
            W_out = ridge_fit_K(states[tr], y[tr], w[tr])
            pred = np.argmax(states[te] @ W_out, axis=1)
            acc = float(np.mean(pred == y[te]))
            xrows.append({"direction": f"{d_tr}->{d_te}", "window": "[0,50]ms",
                          "model": "reservoir_ridge", "alpha": alpha, "seed": seed,
                          "n_train": len(tr), "n_test": len(te),
                          "chance": round(CHANCE, 4), "acc": round(acc, 4)})
    # logreg transfer
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(X_full[tr].reshape(len(tr), -1), y[tr])
    acc = float(clf.score(X_full[te].reshape(len(te), -1), y[te]))
    xrows.append({"direction": f"{d_tr}->{d_te}", "window": "[0,50]ms",
                  "model": "logreg", "alpha": "", "seed": "",
                  "n_train": len(tr), "n_test": len(te),
                  "chance": round(CHANCE, 4), "acc": round(acc, 4)})
    accs = [r["acc"] for r in xrows if r["direction"] == f"{d_tr}->{d_te}"
            and r["model"] == "reservoir_ridge"]
    print(f"  {d_tr}->{d_te}: reservoir {np.mean(accs):.3f}+/-{np.std(accs):.3f} "
          f"logreg {acc:.3f}", flush=True)

# ---------- G. locality: per-electrode vs all-32 ----------
print("G. per-electrode pooled CV, reservoir", flush=True)
for k in range(32):
    Xk = X_full[:, :, k:k + 1]
    vals = []
    for alpha in ALPHAS:
        for seed in SEEDS:
            m, s, _ = reservoir_cv(Xk, y, alpha, seed)
            vals.append(m)
    add("per_electrode", "[0,50]ms", "reservoir_ridge", "all", "ALL", "ALL",
        len(y), float(np.mean(vals)), float(np.std(vals)), electrode=str(64 + k))
print("  per-electrode done", flush=True)

cols = ["analysis", "window", "model", "day", "electrode", "alpha", "seed",
        "n_trials", "chance", "acc_mean", "acc_sd", "null_mean", "null_sd"]
with open(R + "decoding_amplitude_fs369.csv", "w", newline="") as f:
    csv.DictWriter(f, fieldnames=cols).writeheader()
    csv.DictWriter(f, fieldnames=cols).writerows(rows)
xcols = ["direction", "window", "model", "alpha", "seed", "n_train", "n_test",
         "chance", "acc"]
with open(R + "cross_day_transfer_fs369.csv", "w", newline="") as f:
    csv.DictWriter(f, fieldnames=xcols).writeheader()
    csv.DictWriter(f, fieldnames=xcols).writerows(xrows)
print("elapsed", round(time.time() - t0, 1), "s")
print("saved decoding_amplitude_fs369.csv, cross_day_transfer_fs369.csv")
