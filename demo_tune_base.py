#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Velichko A.
Paper:
  "A novel approach for estimating largest Lyapunov exponents in one-dimensional chaotic
   time series using machine learning"
   Chaos 35, 101101 (2025) — https://doi.org/10.1063/5.0289352
  Authors: Andrei Velichko (Corresponding Author); Maksim Belyaev; Petr Boriskov
  
Simple production tuner for LLE (BASE: slope of log-GME over horizons 1..HMAX).

What it does
------------
1) For each map in MAPS:
   - Simulates series and analytic ground-truth Lyapunov λ_true via simulate_map_and_lyapunov.
2) Grid-searches BASE hyperparameters (see CONFIG) and picks the combo with
   the highest R² over r-points where λ_true > 0 (masking non-positive truth).
3) Saves artifacts into runs_tune/<UNIX_TS>/:
   - <map>_grid_<STAMP>.csv      : all combos with their R²(true>0)
   - <map>_base_vs_true_<STAMP>.png : final curve vs analytic truth (dense r-grid)
   - results_<STAMP>.json        : summary with best params and metrics
   - <map>_final_<STAMP>.npz     : r, λ_true, λ_est for the final dense grid

Notes
-----
• BASE estimator builds y(h) for horizons h=1..HMAX (hmin=1, step=1) using KNN
  forecast error with geometric mean, then returns the linear slope on 1..HMAX.
• We match your baseline: WARMUP=1000, SERIES_LEN=1000, X0=0.533.
"""

from __future__ import annotations

import os
import csv
import json
import time
from typing import List, Optional, Tuple

import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Quiet some noisy warnings (optional)
try:
    from sklearn.exceptions import InconsistentVersionWarning, UndefinedMetricWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning, module=r"^sklearn\.base$")
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning,      module=r"^sklearn\.metrics\._regression$")
except Exception:
    pass
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"^nolds(\.|$)")
warnings.filterwarnings("ignore", message=".*pkg_resources.*slated for removal.*")

# Your module must be next to this script
from lyapunov_kit import simulate_map_and_lyapunov, get_lyap_knn


# ============================== CONFIG ========================================

# Maps to tune. Leave only what you need.
MAPS = [
    ("logistic",   "r*x*(1-x)",        3.5,  3.98, None),
    ("sine",       "r*sin(pi*x)",      0.85, 1.0,  None),
    ("cubic_map",  "r*x*(1.0 - x**2)", 2.3,  3.0,  None),
    ("chebyshev",  "cos(r*acos(x))",   1.0, 10.0,  None),
]

# Hyperparameter grid (as requested)
GRID_LOOK_BACK = [1, 2, 3]
GRID_NEIGHBORS = [1, 3, 10]
GRID_TEST_SIZE = [0.2, 0.4]
GRID_HMAX      = [2, 3, 7]   # HMAX limits prefix; slope is computed on 1..HMAX

# Simulation and output
R_POINTS_TUNE:  int = 100     # r-points for the tuning grid
R_POINTS_FINAL: int = 1000    # r-points for the final dense evaluation/plot
SERIES_LEN:     int = 1000    # observed length after warmup
WARMUP:         int = 1000    # burn-in (aligned with your baseline)
X0:             float = 0.533

OUT_ROOT = "results_demo_tune_base"
UNIX_TS  = int(time.time())
RUN_DIR  = os.path.join(OUT_ROOT, str(UNIX_TS))
STAMP    = time.strftime("%Y%m%d-%H%M%S")
os.makedirs(RUN_DIR, exist_ok=True)


# =========================== BASE ESTIMATOR PIECES =============================

def _y_from_points(points: List[List[float]], hmax: int) -> np.ndarray:
    """[[h, y], ...] -> vector y(h=1..hmax); missing horizons become NaN."""
    h2y = {int(h): float(y) for (h, y) in (points or [])}
    return np.asarray([h2y.get(h, float("nan")) for h in range(1, hmax + 1)], float)

def _slope_on_prefix(y: np.ndarray, n: int) -> Optional[float]:
    """Linear slope of y vs horizon on 1..n; NaN-safe; None if <2 finite points."""
    if n < 2:
        return None
    yy = np.asarray(y[:n], float)
    hh = np.arange(1, n + 1, dtype=float)
    m = np.isfinite(yy)
    if m.sum() < 2:
        return None
    a, _b = np.polyfit(hh[m], yy[m], 1)
    return float(a)

def _build_knn_yvec(series: np.ndarray,
                    look_back: int,
                    neighbors: int,
                    test_size: float,
                    *,
                    hmax: int,
                    hmin: int = 1,
                    hstep: int = 1) -> np.ndarray:
    """Build KNN GME points and compress to y(h=1..hmax) with hmin=1."""
    if hmax < 2:
        raise ValueError("hmax must be >= 2")
    points, _ = get_lyap_knn(
        series=series,
        look_back=look_back,
        test_size=test_size,
        neighbors=neighbors,
        horizon_min=hmin,
        horizon_max=hmax,
        horizon_step=hstep,
    )
    return _y_from_points(points, hmax=hmax)

def LLE_BASE(series: np.ndarray,
             look_back: int,
             neighbors: int,
             test_size: float,
             *,
             hmax: int,
             hmin: int = 1,
             hstep: int = 1) -> Optional[float]:
    """
    BASE estimator: build y(h) for h=1..HMAX (hmin=1), return linear slope over 1..HMAX.
    """
    if hmax < 2:
        return None
    yvec = _build_knn_yvec(series, look_back, neighbors, test_size,
                           hmax=hmax, hmin=hmin, hstep=hstep)
    return _slope_on_prefix(yvec, hmax)

def r2_pos_true_only(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² computed only over pairs where y_true > 0; returns NaN if <2 pairs."""
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    m = np.isfinite(yt) & np.isfinite(yp) & (yt > 0)
    if m.sum() < 2:
        return float("nan")
    return float(r2_score(yt[m], yp[m]))

def _plot_curves(r, y_true, y_est, title_est, out_path):
    """Plot truth (black) and estimate (blue)."""
    plt.figure(figsize=(9, 5))
    plt.plot(r, y_true, lw=2.2, color="k", label="Ground truth (analytic)")
    plt.plot(r, y_est,  lw=2.0, color="#1f77b4", label=title_est)
    plt.xlabel("r"); plt.ylabel("Lyapunov exponent")
    plt.title(title_est + " vs ground truth")
    plt.grid(True, alpha=0.3); plt.legend(loc="best")
    vals = np.concatenate([np.asarray(y_true)[np.isfinite(y_true)],
                           np.asarray(y_est)[np.isfinite(y_est)]])
    if vals.size:
        ymin, ymax = float(np.min(vals)), float(np.max(vals))
        if not np.isfinite(ymin) or not np.isfinite(ymax):
            ymin, ymax = -0.1, 0.6
        pad = 0.08 * max(1.0, ymax - ymin)
        plt.ylim(ymin - pad, ymax + pad)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()

def _evaluate_base_on_grid(series_matrix: np.ndarray,
                           lyap_true: np.ndarray,
                           *,
                           look_back: int,
                           neighbors: int,
                           test_size: float,
                           hmax: int,
                           print_prefix: str = "",
                           print_every: int = 10) -> Tuple[float, np.ndarray]:
    """
    Compute R²(true>0) for a given combo over all r-columns (series_matrix.shape[1]).
    Returns (R², estimated curve).
    """
    Rn = series_matrix.shape[1]
    est = np.full(Rn, np.nan, float)
    step = max(1, Rn // max(1, print_every))

    for i in range(Rn):
        v = LLE_BASE(series_matrix[:, i],
                     look_back=look_back,
                     neighbors=neighbors,
                     test_size=test_size,
                     hmax=hmax, hmin=1, hstep=1)
        est[i] = np.nan if (v is None or not np.isfinite(v)) else float(v)
        if print_prefix and (((i + 1) % step == 0) or ((i + 1) == Rn)):
            pct = (i + 1) / Rn
            print(f"{print_prefix} r: {i+1}/{Rn} ({pct:.0%})", flush=True)

    return r2_pos_true_only(lyap_true, est), est


# ============================== TUNING PIPELINE ================================

def tune_for_map(name: str,
                 expr: str,
                 r_min: float, r_max: float,
                 clip_domain,
                 *,
                 r_points_tune: int,
                 r_points_final: int,
                 x0: float,
                 warmup: int,
                 series_len: int,
                 out_dir: str,
                 stamp: str) -> dict:
    """
    Full pipeline for one map:
      - simulate tuning grid (r_points_tune), get ground truth and series,
      - grid-search BASE and select the best combo by R²(true>0),
      - simulate final dense grid (r_points_final) and evaluate best combo,
      - save CSV/PNG/JSON/NPZ artifacts, return a result dict.
    """
    # 1) Simulate tuning grid
    r_tune, _bif_t, lyap_true_tune, full_series_tune = simulate_map_and_lyapunov(
        expr_str=expr,
        x_init=x0,
        x_len=series_len,
        x_warmup=warmup,
        r_min=r_min,
        r_max=r_max,
        r_num_points=r_points_tune,
        bifur_num_points=0,
        clip_domain=clip_domain,
    )
    print(f"[SIM] {name}: tuning grid simulated | full_series={full_series_tune.shape}")

    # 2) Grid-search
    combos = [(lb, nn, ts, hM)
              for lb in GRID_LOOK_BACK
              for nn in GRID_NEIGHBORS
              for ts in GRID_TEST_SIZE
              for hM in GRID_HMAX]
    total = len(combos)
    print(f"[TUNE] {name}: total combos = {total}")

    grid_csv = os.path.join(out_dir, f"{name}_grid_{stamp}.csv")
    with open(grid_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["look_back", "neighbors", "test_size", "hmax", "R2_true_pos"])

        best_r2 = -np.inf
        best_params = None
        best_curve = None

        for idx, (lb, nn, ts, hM) in enumerate(combos, start=1):
            prefix = f"[TUNE] {name} combo {idx}/{total} (lb={lb}, nn={nn}, ts={ts}, hmax={hM})"
            r2v, est_curve = _evaluate_base_on_grid(
                full_series_tune, lyap_true_tune,
                look_back=lb, neighbors=nn, test_size=ts, hmax=hM,
                print_prefix=prefix, print_every=10
            )
            writer.writerow([lb, nn, ts, hM, r2v])

            if np.isfinite(r2v) and (r2v > best_r2):
                best_r2 = float(r2v)
                best_params = {"look_back": lb, "neighbors": nn, "test_size": ts,
                               "hmax": hM, "hmin": 1, "hstep": 1}
                best_curve = est_curve
                print(f"[BEST] {name}: R2(true>0)={best_r2:.6f} with {best_params}")

    print(f"[SAVE] {name}: grid CSV -> {os.path.basename(grid_csv)}")

    # 3) Final evaluation on dense grid
    r_fin, _bif_f, lyap_true_fin, full_series_fin = simulate_map_and_lyapunov(
        expr_str=expr,
        x_init=x0,
        x_len=series_len,
        x_warmup=warmup,
        r_min=r_min,
        r_max=r_max,
        r_num_points=r_points_final,
        bifur_num_points=0,
        clip_domain=clip_domain,
    )

    est_final = np.full(r_points_final, np.nan, float)
    step = max(1, int(0.05 * r_points_final))
    for i in range(r_points_final):
        v = LLE_BASE(full_series_fin[:, i],
                     look_back=best_params["look_back"],
                     neighbors=best_params["neighbors"],
                     test_size=best_params["test_size"],
                     hmax=best_params["hmax"],
                     hmin=best_params["hmin"],
                     hstep=best_params["hstep"])
        est_final[i] = np.nan if (v is None or not np.isfinite(v)) else float(v)
        if ((i + 1) % step == 0) or ((i + 1) == r_points_final):
            print(f"[FINAL] {name} progress: {i+1}/{r_points_final} ({(i+1)/r_points_final:.0%})")

    r2_final = r2_pos_true_only(lyap_true_fin, est_final)
    print(f"[FINAL] {name}: R2(true>0) on final grid = {r2_final:.6f}")

    # 4) Save plot and NPZ
    png_base = os.path.join(out_dir, f"{name}_base_vs_true_{stamp}.png")
    _plot_curves(r_fin, lyap_true_fin, est_final, f"{name}: LLE_BASE (tuned)", png_base)
    print(f"[SAVE] {name}: plot -> {os.path.basename(png_base)}")

    npz_final = os.path.join(out_dir, f"{name}_final_{STAMP}.npz")
    np.savez_compressed(npz_final,
                        r=r_fin,
                        lyap_true=lyap_true_fin,
                        lyap_est=est_final,
                        best_params=np.array([best_params["look_back"],
                                              best_params["neighbors"],
                                              best_params["hmax"],
                                              best_params["test_size"]], dtype=float))
    print(f"[SAVE] {name}: NPZ -> {os.path.basename(npz_final)}")

    # 5) Pack result
    entry = {
        "map": name,
        "expr": expr,
        "r_range": [float(r_min), float(r_max)],
        "R_POINTS_TUNE": r_points_tune,
        "R_POINTS_FINAL": r_points_final,
        "best_base": {
            "params": best_params,
            "r2_tune_true_pos_only": (None if not np.isfinite(best_r2) else float(best_r2)),
            "r2_final_true_pos_only": (None if not np.isfinite(r2_final) else float(r2_final)),
            "grid_csv": os.path.abspath(grid_csv),
            "plot": os.path.abspath(png_base),
            "final_npz": os.path.abspath(npz_final),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return entry


# ================================== MAIN ======================================

def main():
    print("Run directory:", os.path.abspath(RUN_DIR))
    config = {
        "maps": [m[0] for m in MAPS],
        "R_POINTS_TUNE": R_POINTS_TUNE,
        "R_POINTS_FINAL": R_POINTS_FINAL,
        "SERIES_LEN": SERIES_LEN,
        "WARMUP": WARMUP,
        "X0": X0,
        "grid": {
            "look_back": GRID_LOOK_BACK,
            "neighbors": GRID_NEIGHBORS,
            "test_size": GRID_TEST_SIZE,
            "hmax": GRID_HMAX,
            "hmin": 1,
            "hstep": 1
        },
        "run_dir": os.path.abspath(RUN_DIR),
        "stamp": STAMP,
        "unix": UNIX_TS
    }

    results = []
    for name, expr, r_min, r_max, clip_domain in MAPS:
        print(f"\n=== MAP: {name} | expr: {expr} | r∈[{r_min}, {r_max}] | "
              f"SERIES_LEN={SERIES_LEN} WARMUP={WARMUP} ===", flush=True)

        entry = tune_for_map(
            name, expr, r_min, r_max, clip_domain,
            r_points_tune=R_POINTS_TUNE, r_points_final=R_POINTS_FINAL,
            x0=X0, warmup=WARMUP, series_len=SERIES_LEN,
            out_dir=RUN_DIR, stamp=STAMP
        )
        results.append(entry)

    # Save JSON summary
    summary = {"config": config, "maps": results}
    json_path = os.path.join(RUN_DIR, f"results_{STAMP}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Saved JSON:", os.path.abspath(json_path))

    # Save compact R² summary CSV
    csv_rows = [["map", "BASE_tune_R2", "BASE_final_R2"]]
    for m in results:
        csv_rows.append([
            m["map"],
            m["best_base"]["r2_tune_true_pos_only"],
            m["best_base"]["r2_final_true_pos_only"],
        ])
    csv_path = os.path.join(RUN_DIR, f"r2_summary_{STAMP}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(csv_rows)
    print("Saved R2 CSV:", os.path.abspath(csv_path))

if __name__ == "__main__":
    main()
