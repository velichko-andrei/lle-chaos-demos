#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline KNN-slope demo (self-contained)

This script shows the **basic principle** from the paper — estimating the
largest Lyapunov exponent (LLE) as the slope of log forecast error vs horizon
— implemented directly here **without using any external helper module**.

What it does
------------
1) Simulates the logistic map on an r-grid and computes the analytic ground-truth
   Lyapunov exponent λ_true from the derivative along the trajectory.
2) For each hyperparameter combo, builds KNN one-step-ahead predictors for a set
   of horizons (H = 1..HMAX), computes the geometric mean error (GME),
   takes log(GME) for each horizon, and fits a straight line y(h) ~ a*h + b.
   The estimated LLE is the slope 'a'.
3) Evaluates R² on the region where λ_true > 0, picks the best combo, and saves:
   - data_full.npz (r, full series, λ_true, λ_est for best combo, noise, SNR)
   - true_vs_est_best.csv
   - summary_all_combos.csv (R² per combo)
   - run_meta.json (metadata)
   - best_plot.png (best estimate vs ground truth)

Dependencies
------------
numpy, scipy, scikit-learn, matplotlib

Run
---
python baseline_knn_demo.py
"""

import os
import json
import time
import numpy as np
from typing import Tuple, Dict

from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import gmean


# =============================== KNN core =====================================

def get_lyap_knn(series: np.ndarray,
                 look_back: int = 10,
                 test_size: float = 0.20,
                 neighbors: int = 20,
                 horizon_min: int = 1,
                 horizon_max: int = 10,
                 horizon_step: int = 1):
    """
    Build KNN forecasters for horizons h ∈ [horizon_min..horizon_max] and compute
    the slope of log(GME) vs horizon.

    Returns:
        points : list of [h, log_gme] pairs (increasing h)
        slope  : float slope 'a' of linear fit y ~ a*h + b, or None on failure
    """
    horizon_range = range(horizon_min, horizon_max + 1, horizon_step)
    min_length = int(np.ceil(1.0 / test_size + look_back + max(horizon_range)))
    if len(series) < min_length:
        print(f"[WARN] Series too short, need at least {min_length} samples.")
        return [], None
    if len(horizon_range) < 2:
        print("[WARN] Need at least 2 horizons to fit a line.")
        return [], None

    points: list[list[float]] = []
    for h in horizon_range:
        # Build supervised dataset (no shuffling)
        X, Y = [], []
        for i in range(len(series) - look_back - h):
            X.append(series[i:(i + look_back)])
            Y.append(series[i + look_back + h - 1])
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        split = int(round(X.shape[0] * (1.0 - test_size)))
        if split <= 0 or split >= X.shape[0]:
            return [], None  # not enough samples for a proper split

        trainX, trainY = X[:split], Y[:split]
        testX,  testY  = X[split:], Y[split:]

        knn = KNeighborsRegressor(n_neighbors=neighbors, weights="distance", p=2, n_jobs=-1)
        knn.fit(trainX, trainY)

        predY = knn.predict(testX)

        # --- IMPORTANT: flatten to 1-D before gmean to avoid deprecation warnings
        err = np.abs(testY.ravel() - predY.ravel())
        # gmean returns a scalar for 1-D input; .item() ensures pure Python float
        test_gme = np.array(gmean(err), dtype=float).item()

        # Guard against log(0)
        if not np.isfinite(test_gme) or test_gme <= 1e-12:
            return [], 0.0

        points.append([float(h), float(np.log(test_gme))])

    line = np.asarray(points, dtype=float)  # shape (H, 2)
    # Linear fit: y = a*h + b  → slope 'a'
    reg = LinearRegression().fit(line[:, 0].reshape(-1, 1), line[:, 1])
    return points, float(reg.coef_[0])



# ============================ Logistic map + truth =============================

def simulate_logistic_and_true(x_init: float = 0.5,
                               x_len: int = 10_000,
                               x_warmup: int = 1_000,
                               r_min: float = 3.0,
                               r_max: float = 4.0,
                               r_num_points: int = 1_000,
                               bifur_num_points: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate the logistic map x_{t+1} = r * x_t * (1 - x_t) on an r-grid and compute:
      r            : shape (R,)
      bifur        : shape (bifur_num_points, R)  (last states; zeros if disabled)
      lyap_true    : shape (R,) analytic largest Lyapunov exponent
      full_series  : shape (x_len, R) trajectories after warm-up

    The analytic exponent is accumulated as:
      λ_true = (1 / x_len) * Σ log |r - 2*r*x_t|  (along the simulated path)
    """
    r = np.linspace(r_min, r_max, r_num_points)
    bifur = np.zeros((bifur_num_points, r_num_points), float) if bifur_num_points > 0 else np.zeros((0, r_num_points), float)
    full_series = np.zeros((x_len, r_num_points), float)
    lyap = np.zeros(r_num_points, float)
    x = np.ones(r_num_points, float) * x_init

    for i in range(x_len + x_warmup):
        x = r * x * (1.0 - x)

        if i >= x_warmup:
            lyap += np.log(np.abs(r - 2.0 * r * x))
            full_series[i - x_warmup, :] = x

            j = i - (x_len + x_warmup - bifur_num_points)
            if bifur_num_points > 0 and j >= 0:
                bifur[j, :] = x

    lyap /= float(x_len)
    return r, bifur, lyap, full_series


# ================================ Utilities ====================================

def compute_snr_db(original: np.ndarray, noise: np.ndarray) -> float:
    """SNR in dB for a pair of equal-length vectors."""
    original = original.astype(np.float64)
    noise = noise.astype(np.float64)
    sp = np.mean(original ** 2)
    npow = np.mean(noise ** 2)
    if npow == 0.0:
        return float("inf")
    if sp == 0.0:
        return float("-inf")
    return 10.0 * np.log10(sp / npow)


# ============================== Experiment setup ==============================

# Time settings for a single run
X_LEN = 1000         # observed length (after warm-up)
X_WARMUP = 1000      # burn-in
X0 = 0.533           # initial state (same for all r)

# r-grid for evaluation
R_NUM = 100
R_MIN = 3.5
R_MAX = 3.98

# Optional noise injected into *input series* of the estimator only
NOISE_MEAN = 0.0
NOISE_STD = 0.00     # 0 disables noise
NOISE = np.random.normal(NOISE_MEAN, NOISE_STD, size=X_LEN)

# Hyperparameter grid (simple demo)
LB_LIST = [1]
NN_LIST = [1, 3]
HMAX_LIST = [7]
TS_LIST = [0.2, 0.4]


# ================================== Main ======================================

def main():
    # 1) Simulate logistic map and analytic truth
    r, bifur, lyap_true, full_series = simulate_logistic_and_true(
        x_init=X0,
        x_len=X_LEN,
        x_warmup=X_WARMUP,
        r_min=R_MIN,
        r_max=R_MAX,
        r_num_points=R_NUM,
        bifur_num_points=0,
    )

    # Mask where ground-truth exponent is positive (used for R² metric)
    pos = lyap_true > 0

    # Mean SNR across r-columns for the chosen NOISE (for info only)
    snrs = [compute_snr_db(full_series[:, u], NOISE) for u in range(R_NUM)]
    mean_snr_db = float(np.mean(snrs))
    print(f"SNR (mean over r): {mean_snr_db:.3f} dB")

    # 2) Evaluate the grid
    results: Dict[tuple, np.ndarray] = {}
    best_combo = None
    best_r2 = -np.inf

    for lb in LB_LIST:
        for nn in NN_LIST:
            for hM in HMAX_LIST:
                for ts in TS_LIST:
                    est_curve = np.zeros(R_NUM, float)
                    for u in range(R_NUM):
                        series_in = full_series[:, u] + NOISE
                        _, slope = get_lyap_knn(series=series_in,
                                                look_back=lb,
                                                test_size=ts,
                                                neighbors=nn,
                                                horizon_min=1,
                                                horizon_max=hM,
                                                horizon_step=1)
                        est_curve[u] = np.nan if slope is None else float(slope)

                    results[(lb, nn, hM, ts)] = est_curve

                    r2_pos = r2_score(lyap_true[pos], est_curve[pos])
                    r2_all = r2_score(lyap_true, est_curve)
                    print(f"lb={lb} nn={nn} hmax={hM} ts={ts}  |  R2(true>0)={r2_pos:.6f}  R2(all)={r2_all:.6f}")

                    if np.isfinite(r2_pos) and r2_pos > best_r2:
                        best_r2 = float(r2_pos)
                        best_combo = (lb, nn, hM, ts)
                        best_curve = est_curve.copy()

    if best_combo is None:
        raise RuntimeError("No valid combo found.")

    lb, nn, hM, ts = best_combo
    r2_pos_best = r2_score(lyap_true[pos], best_curve[pos])
    r2_all_best = r2_score(lyap_true, best_curve)
    print(f"\nBEST: lb={lb} nn={nn} hmax={hM} ts={ts}  |  R2(true>0)={r2_pos_best:.6f}  R2(all)={r2_all_best:.6f}")

    # 3) Save outputs
    unix_ts = int(time.time())
    run_dir = os.path.join("results_demo_basic_principle", str(unix_ts))
    os.makedirs(run_dir, exist_ok=True)

    # npz with data
    npz_path = os.path.join(run_dir, "data_full.npz")
    np.savez_compressed(
        npz_path,
        r=r,
        full_series=full_series,
        lyap_true=lyap_true,
        lyap_est_best=best_curve,
        noise=NOISE,
        mean_snr_db=mean_snr_db,
        best_combo=np.array(best_combo, dtype=float),
    )

    # CSV: truth vs best estimate
    csv_best = os.path.join(run_dir, "true_vs_est_best.csv")
    with open(csv_best, "w", encoding="utf-8") as fcsv:
        fcsv.write("r,lyap_true,lyap_est,is_true_positive\n")
        for i in range(R_NUM):
            fcsv.write(f"{r[i]},{lyap_true[i]},{best_curve[i]},{bool(pos[i])}\n")

    # CSV: summary for all combos
    csv_sum = os.path.join(run_dir, "summary_all_combos.csv")
    with open(csv_sum, "w", encoding="utf-8") as fsum:
        fsum.write("look_back,neighbors,hmax,test_size,R2_true_pos,R2_all\n")
        for (lb_, nn_, hM_, ts_), arr in results.items():
            fsum.write(f"{lb_},{nn_},{hM_},{ts_},{r2_score(lyap_true[pos], arr[pos])},{r2_score(lyap_true, arr)}\n")

    # JSON meta
    meta = {
        "best_combo": {"look_back": lb, "neighbors": nn, "hmax": hM, "test_size": ts},
        "metrics": {"R2_true_pos_only": r2_pos_best, "R2_all": r2_all_best, "mean_SNR_dB": mean_snr_db},
        "shapes": {"full_series": list(full_series.shape), "r_points": int(R_NUM)},
        "paths": {
            "npz": os.path.abspath(npz_path),
            "csv_true_vs_est": os.path.abspath(csv_best),
            "csv_summary": os.path.abspath(csv_sum),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as fj:
        json.dump(meta, fj, ensure_ascii=False, indent=2)

    # Plot: best vs truth
    plt.figure(figsize=(9, 5))
    plt.plot(r, lyap_true, lw=2.2, color="k", label="Ground truth (analytic)")
    plt.plot(r, best_curve, lw=2.0, color="#1f77b4",
             label=f"KNN best (lb={lb}, nn={nn}, H={hM}, ts={ts})\nR²>0={r2_pos_best:.4f}  R²(all)={r2_all_best:.4f}")
    plt.xlabel("r")
    plt.ylabel("Lyapunov exponent")
    plt.title("Best KNN-slope estimate vs ground truth (logistic map)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "best_plot.png"), dpi=170)
    plt.close()

    print("\nSaved to:", os.path.abspath(run_dir))
    print("Best combo:", best_combo,
          "| R2(true>0) =", r2_pos_best,
          "| R2(all) =", r2_all_best)


if __name__ == "__main__":
    main()
