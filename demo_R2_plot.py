#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Velichko A.
Paper:
  "A novel approach for estimating largest Lyapunov exponents in one-dimensional chaotic
   time series using machine learning"
   Chaos 35, 101101 (2025) — https://doi.org/10.1063/5.0289352
  Authors: Andrei Velichko (Corresponding Author); Maksim Belyaev; Petr Boriskov
  
Minimal production demo: LLE over multiple 1D maps (grid of r)

Two methods (same public API):
    • LLE_BASE(series, **optional_overrides) -> float|None
        Uses BASE_* parameters (configurable at the top). You may override
        them per-call, but ML parameters remain unaffected.
    • LLE_ML1(series) -> float|None
        Uses fixed ML constants: KNN=(1,3,0.40), horizons=1..10 (DO NOT CHANGE).

Workflow:
  - For each map: build r-grid, simulate noiseless ground truth (analytic Lyapunov).
  - Optionally add normalized Gaussian noise to series ONLY for estimator input.
  - Compute R² on region where true LLE > 0.
  - Save plots and incrementally update a JSON file per map.
"""

from __future__ import annotations

import os
import csv
import json
import time
from typing import List, Optional, Tuple

import numpy as np

# Silence noisy warnings (optional)
import warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning, UndefinedMetricWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning, module=r"^sklearn\.base$")
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning,      module=r"^sklearn\.metrics\._regression$")
except Exception:
    pass
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"^nolds(\.|$)")
warnings.filterwarnings("ignore", message=".*pkg_resources.*slated for removal.*")

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Core tools
from lyapunov_kit import (
    simulate_map_and_lyapunov,  # NOISE-FREE truth + series generator
    get_lyap_knn,
    get_lyap_ml_simple,
)

# ============================= Configuration ===================================

# SELECT METHOD: "BASE" or "ML1"
METHOD: str = "BASE"

# Maps: (name, expression, r_min, r_max, clip_domain)
MAPS = [
    ("logistic",           "r*x*(1-x)",                 3.5, 3.98,  None),
    ("chebyshev",          "cos(r*acos(x))",            1.0, 10.0,  None),

    ("beta_shift",        "Mod(r*x, 1.0)",                                     1.1, 10.0, (0.0, 1.0)),
    ("skew_tent",         "Piecewise((x/r, x < r), ((1.0 - x)/(1.0 - r), True))",
                                                                              0.05, 0.95, (0.0, 1.0)),
    ("ulam_abs",          "1.0 - r*Abs(x)",                                     0.0,  2.0,  (-1.0, 1.0)),
    ("gauss_boole",       "Mod(1.0/(x) + r, 1.0)",                              0.0,  1.0,  (1e-8, 1.0 - 1e-8)),
    ("folded_sawtooth",   "1.0 - Abs(Mod(r*x, 2.0) - 1.0)",                     1.0,  2.0,  (0.0, 1.0)),
    ("chebyshev_sine",    "sin(r*asin(x))",                                     1.0,  10.0, (-1.0, 1.0)),
    ("mod_quadratic",     "Mod(r*x*(1.0 - x), 1.0)",                            2.0,  8.0,  (0.0, 1.0)),
    ("saturated_mod",     "Mod(r*Piecewise((0.0, x<0.0), (x, x<=1.0), (1.0, True)), 1.0)",
                                                                              1.1, 10.0, (0.0, 1.0)),
    ("mobius_mod",        "Mod((x + r) / (1.0 + r*Abs(x)), 1.0)",                0.1,  5.0,  (0.0, 1.0)),
]

PROGRESS_CHUNK: float = 0.05
R_POINTS: int = 1000
WARMUP: int = 1000
SERIES_LEN: int = 1000
BIFUR_POINTS: int = 800
X0: float = 0.533

# Noise for estimator input only (ground truth is always noiseless)
NOISE_STD: float = 0.006
NOISE_SEED: int | None = None

# --------- Per-run folder ------------------------------------------------------
UNIX_TS = int(time.time())
OUT_ROOT = "results_demo_R2_plot"
RUN_DIR = os.path.join(OUT_ROOT, f"{METHOD.lower()}_noise{NOISE_STD:g}_len{SERIES_LEN}_{UNIX_TS}")
os.makedirs(RUN_DIR, exist_ok=True)
STAMP = time.strftime("%Y%m%d-%H%M%S")

# ========================= Parameter ideology (IMPORTANT) ======================
# BASE — tunable, separate from ML:
BASE_LOOK_BACK: int = 1
BASE_NEIGHBORS: int = 1
BASE_TEST_SIZE: float = 0.40
BASE_HMIN: int = 1
BASE_HMAX: int = 7          # slope will be computed on horizons 1..BASE_HMAX
BASE_HSTEP: int = 1

# ML — fixed constants (DO NOT CHANGE):
ML_LOOK_BACK: int = 1
ML_NEIGHBORS: int = 3
ML_TEST_SIZE: float = 0.40
ML_HMIN: int = 1
ML_HMAX: int = 10           # model expects exactly 10 points
ML_HSTEP: int = 1

# ============================= Helpers =========================================

def _y_from_points(points: List[List[float]], *, hmin: int, hmax: int) -> np.ndarray:
    """[[h, y], ...] → vector y(h=hmin..hmax); fill missing with NaN."""
    h2y = {int(h): float(y) for (h, y) in (points or [])}
    return np.asarray([h2y.get(h, float("nan")) for h in range(hmin, hmax + 1)], float)

def _slope_on_first_n(y: np.ndarray, n: int) -> Optional[float]:
    """Linear slope of y vs horizon index on first n points; None if <2 finite."""
    if n < 2:
        return None
    yy = np.asarray(y[:n], float)
    hh = np.arange(1, n + 1, dtype=float)
    m = np.isfinite(yy)
    if m.sum() < 2:
        return None
    a, _b = np.polyfit(hh[m], yy[m], 1)
    return float(a)

def _build_knn_yvec(series: np.ndarray, *, look_back: int, neighbors: int,
                    test_size: float, hmin: int, hmax: int, hstep: int) -> np.ndarray:
    """Generic KNN point builder with explicit params."""
    points, _ = get_lyap_knn(
        series=series,
        look_back=look_back,
        test_size=test_size,
        neighbors=neighbors,
        horizon_min=hmin,
        horizon_max=hmax,
        horizon_step=hstep,
    )
    return _y_from_points(points, hmin=hmin, hmax=hmax)

# ============================= Public API functions ============================

def LLE_BASE(series: np.ndarray,
             look_back: Optional[int] = None,
             neighbors: Optional[int] = None,
             test_size: Optional[float] = None,
             hmin: Optional[int] = None,
             hmax: Optional[int] = None,
             hstep: Optional[int] = None) -> Optional[float]:
    """
    General-purpose KNN method (BASE):
      - Defaults come from BASE_* constants above.
      - You may override any of the parameters per-call.
      - Slope is computed across horizons 1..hmax (with hmin fixed to 1 by default).
    Returns float or None if slope cannot be computed.
    """
    lb  = BASE_LOOK_BACK if look_back is None else int(look_back)
    nn  = BASE_NEIGHBORS if neighbors is None else int(neighbors)
    ts  = BASE_TEST_SIZE if test_size is None else float(test_size)
    hm0 = BASE_HMIN      if hmin is None else int(hmin)
    hm1 = BASE_HMAX      if hmax is None else int(hmax)
    stp = BASE_HSTEP     if hstep is None else int(hstep)

    if hm1 < 2:
        return None
    yvec = _build_knn_yvec(series, look_back=lb, neighbors=nn, test_size=ts,
                           hmin=hm0, hmax=hm1, hstep=stp)
    return _slope_on_first_n(yvec, hm1 - hm0 + 1)

def LLE_ML1(series: np.ndarray) -> Optional[float]:
    """
    ML #1 ("simple"): requires exactly 10 horizons (1..10) and fixed KNN=(1,3,0.40).
    Returns float or None on failure/missing model.
    """
    try:
        yvec = _build_knn_yvec(series,
                               look_back=ML_LOOK_BACK,
                               neighbors=ML_NEIGHBORS,
                               test_size=ML_TEST_SIZE,
                               hmin=ML_HMIN, hmax=ML_HMAX, hstep=ML_HSTEP)
        lam, _info = get_lyap_ml_simple(yvec)  # model expects length-10 vector
        return float(lam) if np.isfinite(lam) else None
    except Exception:
        return None

# ============================= Metrics / Plotting ==============================

def _r2_pos_true_only(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² using only pairs where lle_true > 0; returns NaN if <2 valid pairs."""
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    m = np.isfinite(yt) & np.isfinite(yp) & (yt > 0)
    if m.sum() < 2:
        return float("nan")
    return float(r2_score(yt[m], yp[m]))

def _plot_curves(r, y_true, y_est, title_est, out_path):
    """Plot estimate vs analytic truth (only region where analytic LLE > 0)."""
    r = np.asarray(r, float)
    yt = np.asarray(y_true, float)
    ye = np.asarray(y_est, float)
    m = np.isfinite(yt) & (yt > 0)
    yt_plot = np.where(m, yt, np.nan)
    ye_plot = np.where(m, ye, np.nan)

    plt.figure(figsize=(9, 5))
    plt.plot(r, yt_plot, lw=2.2, color="k", label="Ground truth (true>0)")
    plt.plot(r, ye_plot, lw=2.0, color="#1f77b4", label=title_est + " (on true>0)")
    plt.xlabel("r"); plt.ylabel("Lyapunov exponent")
    plt.title(title_est + " vs ground truth")
    plt.grid(True, alpha=0.3); plt.legend(loc="best")

    ymin_fixed = -0.05
    vals = np.concatenate([yt_plot[np.isfinite(yt_plot)], ye_plot[np.isfinite(ye_plot)]])
    if vals.size:
        ymax = float(np.max(vals)); pad = 0.08 * max(1.0, ymax - ymin_fixed)
        plt.ylim(ymin_fixed, ymax + pad)
    else:
        plt.ylim(ymin_fixed, 0.6)

    plt.axhline(0.0, color="gray", lw=1.0, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()

def _plot_bifurcation(r: np.ndarray,
                      bifur: np.ndarray,
                      name: str,
                      clip_domain: Optional[Tuple[float, float] | str],
                      out_path: str,
                      *, point_size: float = 0.2, alpha: float = 0.35) -> None:
    """Bifurcation diagram: scatter of the last BIFUR_POINTS states vs r."""
    R = np.repeat(r[np.newaxis, :], bifur.shape[0], axis=0).ravel()
    X = bifur.ravel()
    plt.figure(figsize=(9.5, 5.5))
    plt.scatter(R, X, s=point_size, c="k", alpha=alpha, linewidths=0)

    if clip_domain == "01":
        plt.ylim(0.0, 1.0)
    elif clip_domain == "-11":
        plt.ylim(-1.0, 1.0)
    elif isinstance(clip_domain, (tuple, list)) and len(clip_domain) == 2:
        lo, hi = float(clip_domain[0]), float(clip_domain[1])
        plt.ylim(lo, hi)

    plt.xlabel("r")
    plt.ylabel("x (last states)")
    plt.title(f"{name}: bifurcation diagram (last {bifur.shape[0]} states per r)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

# ============================= JSON results ====================================

def _json_path() -> str:
    return os.path.join(RUN_DIR, f"results_{STAMP}.json")

def _load_or_init_results() -> dict:
    p = _json_path()
    if os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "runs" in data:
                return data
        except Exception:
            pass
    return {
        "stamp": STAMP,
        "unix": UNIX_TS,
        "run_dir": os.path.abspath(RUN_DIR),
        "method": METHOD,
        "config": {
            "R_POINTS": R_POINTS,
            "WARMUP": WARMUP,
            "SERIES_LEN": SERIES_LEN,
            "BIFUR_POINTS": BIFUR_POINTS,
            "X0": X0,
            "NOISE_STD": NOISE_STD,
            "NOISE_SEED": NOISE_SEED,
            "BASE": {
                "look_back": BASE_LOOK_BACK, "neighbors": BASE_NEIGHBORS, "test_size": BASE_TEST_SIZE,
                "hmin": BASE_HMIN, "hmax": BASE_HMAX, "hstep": BASE_HSTEP
            },
            "ML1": {
                "look_back": ML_LOOK_BACK, "neighbors": ML_NEIGHBORS, "test_size": ML_TEST_SIZE,
                "hmin": ML_HMIN, "hmax": ML_HMAX, "hstep": ML_HSTEP
            }
        },
        "runs": []
    }

def _save_json_atomic(data: dict) -> None:
    path = _json_path()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def _append_run_to_json(run_entry: dict) -> None:
    data = _load_or_init_results()
    data["runs"].append(run_entry)
    _save_json_atomic(data)

# ============================= Noise injection =================================

def _amplitude_scale_from_domain(clip_domain) -> float:
    if clip_domain == "01":
        return 0.5
    if clip_domain == "-11":
        return 1.0
    if isinstance(clip_domain, (tuple, list)) and len(clip_domain) == 2:
        lo, hi = float(clip_domain[0]), float(clip_domain[1])
        return max(0.5 * (hi - lo), 1e-12)
    return np.nan

def _maybe_add_noise(full_series: np.ndarray, clip_domain, *, noise_std: float,
                     rng: np.random.Generator) -> np.ndarray:
    if not (noise_std and float(noise_std) > 0.0):
        return full_series
    T, N = full_series.shape
    out = full_series.copy()
    base_scale = _amplitude_scale_from_domain(clip_domain)
    if np.isnan(base_scale):
        col_scale = np.maximum(1e-12, np.nanmean(np.abs(out), axis=0))
        noise = rng.normal(loc=0.0, scale=noise_std * col_scale, size=(T, N))
    else:
        noise = rng.normal(loc=0.0, scale=noise_std * base_scale, size=(T, N))
    out += noise
    if clip_domain == "01":
        np.clip(out, 0.0, 1.0, out=out)
    elif clip_domain == "-11":
        np.clip(out, -1.0, 1.0, out=out)
    elif isinstance(clip_domain, (tuple, list)) and len(clip_domain) == 2:
        lo, hi = float(clip_domain[0]), float(clip_domain[1])
        np.clip(out, lo, hi, out=out)
    return out

# ============================= Main ===========================================

def main():
    rng = np.random.default_rng(NOISE_SEED) if (NOISE_SEED is not None) else np.random.default_rng()

    def method_dispatch(series: np.ndarray) -> Optional[float]:
        return LLE_ML1(series) if METHOD == "ML1" else LLE_BASE(series)

    summary_rows = [["map", f"R2_{METHOD} (true>0 only)"]]

    for name, expr, r_min, r_max, clip_domain in MAPS:
        print(f"\n=== MAP: {name} | expr: {expr} | r∈[{r_min}, {r_max}] | SERIES_LEN={SERIES_LEN} WARMUP={WARMUP} | METHOD={METHOD} ===", flush=True)

        # (1) Truth & trajectories (noiseless)
        r, bifur, lyap_true, full_series = simulate_map_and_lyapunov(
            expr_str=expr,
            x_init=X0,
            x_len=SERIES_LEN,
            x_warmup=WARMUP,
            r_min=r_min,
            r_max=r_max,
            r_num_points=R_POINTS,
            bifur_num_points=BIFUR_POINTS,
            clip_domain=clip_domain,
        )

        # (2) Estimator input (optional noise)
        series_for_est = _maybe_add_noise(full_series, clip_domain, noise_std=NOISE_STD, rng=rng)

        # (3) Per-r estimates
        est = np.full(R_POINTS, np.nan, float)
        step = max(1, int(R_POINTS * PROGRESS_CHUNK))
        for i in range(R_POINTS):
            v = method_dispatch(series_for_est[:, i])
            est[i] = np.nan if (v is None or not np.isfinite(v)) else float(v)
            if ((i + 1) % step == 0) or ((i + 1) == R_POINTS):
                print(f"  progress [{name}/{METHOD}]: {i + 1}/{R_POINTS} ({(i + 1)/R_POINTS:.0%})", flush=True)

        # (4) R² metric (true>0 only)
        r2_val = _r2_pos_true_only(lyap_true, est)
        r2_str = "NA" if not np.isfinite(r2_val) else f"{r2_val:.4f}"
        print(f"[R2(true>0)] {name:9s}  {METHOD}={r2_str}", flush=True)
        summary_rows.append([name, None if not np.isfinite(r2_val) else float(r2_val)])

        # (5) Plots
        png_lle = os.path.join(RUN_DIR, f"{name}_{METHOD.lower()}_vs_true_{STAMP}.png")
        _plot_curves(r, lyap_true, est, f"{name}: LLE_{METHOD}", png_lle)
        print(f"Saved plot: {os.path.basename(png_lle)}", flush=True)

        png_bif = os.path.join(RUN_DIR, f"{name}_bifur_{STAMP}.png")
        if bifur is not None and bifur.size:
            _plot_bifurcation(r, bifur, name, clip_domain, png_bif)
            print(f"Saved bifurcation: {os.path.basename(png_bif)}", flush=True)
        else:
            print("No bifurcation data (bifur_num_points=0) — skipped.", flush=True)

        # (6) JSON
        run_entry = {
            "map": name,
            "expr": expr,
            "r_range": [float(r_min), float(r_max)],
            "R_POINTS": R_POINTS,
            "clip_domain": clip_domain if isinstance(clip_domain, str)
                            else (list(clip_domain) if clip_domain is not None else None),
            "method": METHOD,
            "metrics": {"r2_true_pos_only": (None if not np.isfinite(r2_val) else float(r2_val))},
            "artifacts": {
                "plot_lle": os.path.abspath(png_lle),
                "plot_bifur": os.path.abspath(png_bif) if (bifur is not None and bifur.size) else None,
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _append_run_to_json(run_entry)
        print(f"JSON updated: {os.path.basename(_json_path())}", flush=True)

    # (7) Summary CSV
    csv_path = os.path.join(RUN_DIR, f"r2_summary_{METHOD.lower()}_{STAMP}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(summary_rows)
    print("Saved R2 CSV:", csv_path, flush=True)
    print("Run directory:", os.path.abspath(RUN_DIR), flush=True)

if __name__ == "__main__":
    main()
