#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Velichko A.
Paper:
  "A novel approach for estimating largest Lyapunov exponents in one-dimensional chaotic
   time series using machine learning"
   Chaos 35, 101101 (2025) — https://doi.org/10.1063/5.0289352
  Authors: Andrei Velichko (Corresponding Author); Maksim Belyaev; Petr Boriskov
  
Readable demo: estimate the largest Lyapunov exponent (LLE) for multiple 1D maps
with five clear functions:

    LLE_BASE(series)     # General-purpose KNN slope on horizons BASE_HMIN..BASE_HMAX
    LLE_ML1(series)      # ML (simple)   — fixed KNN + fixed 10 horizons (1..10)
    LLE_ML2(series)      # ML (extended) — fixed KNN + fixed 10 horizons (1..10)
    LLE_ML3(series)      # ML (corr_n3)  — fixed KNN + fixed 10 horizons (1..10)
    LLE_NOLD(series, ...)# Rosenstein lyap_r (nolds)



JSON output is compact (no KNN points). Console prints one line per trial.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Tuple, Optional

import numpy as np

# (optional) silence noisy warnings
import warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning, UndefinedMetricWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning, module=r"^sklearn\.base$")
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning,      module=r"^sklearn\.metrics\._regression$")
except Exception:
    pass
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"^nolds(\.|$)")
warnings.filterwarnings("ignore", message=".*pkg_resources.*slated for removal.*")

# core toolkit
from lyapunov_kit import (
    simulate_map_and_lyapunov,
    get_lyap_knn,
    get_lyap_ml_simple,
    get_lyap_ml_extended,
    get_lyap_ml_corr_n3,
    get_lyap_nolds,
)

# ----------------------------- MAP LIST (edit as you like) -----------------------------

# (name, expression, r_min, r_max, clip_domain)
MAPS = [
    ("logistic",   "r*x*(1-x)",      3.5, 3.98, None),
    ("sine",       "r*sin(pi*x)",    0.85, 1.0, None),
    ("chebyshev",  "cos(r*acos(x))", 1.0, 10.0, None),
]

TRIALS_PER_MAP: int = 10

# ----------------------------- GLOBAL SIM SETTINGS -------------------------------------

WARMUP: int = 10_000           # burn-in length
LEN_MIN, LEN_MAX = 200, 1400   # random observed length range per trial
RNG_SEED: int = 12345

# ====================== PARAMETER IDEOLOGY (IMPORTANT) =========================
# BASE — tunable (change here as needed)
BASE_LOOK_BACK: int = 1
BASE_NEIGHBORS: int = 1
BASE_TEST_SIZE: float = 0.40
BASE_HMIN: int = 1
BASE_HMAX: int = 7        # slope uses horizons h = BASE_HMIN..BASE_HMAX (inclusive)
BASE_HSTEP: int = 1

# ML — fixed constants (DO NOT CHANGE; models expect 10 horizons 1..10)
ML_LOOK_BACK: int = 1
ML_NEIGHBORS: int = 3
ML_TEST_SIZE: float = 0.40
ML_HMIN: int = 1
ML_HMAX: int = 10
ML_HSTEP: int = 1

# ----- Default Rosenstein (nolds) parameters
NOLDS_DEFAULTS = dict(
    emb_dim=2,
    trajectory_len=7,
    lag=1,
    min_tsep=10,
    auto_lag=False,       # fixed lag=1 per spec
    lag_max=1000,
    fallback_to_zero=True,
    silence_warnings=True,
)

# Output
OUT_DIR = "results_demo_lle"
os.makedirs(OUT_DIR, exist_ok=True)
STAMP = time.strftime("%Y%m%d-%H%M%S")
OUT_JSON = os.path.join(OUT_DIR, f"random_maps_runs_{STAMP}.json")


# ============================= SMALL UTILITIES =========================================

def _x0_range_for_map(name: str) -> Tuple[float, float]:
    """Pick a safe initial domain for x0 per map."""
    n = name.lower()
    if n in ("logistic", "sine"):
        return (1e-6, 1.0 - 1e-6)
    if n in ("chebyshev", "cubic"):
        return (-0.999, 0.999)
    return (0.0, 1.0)

def _y_from_points(points: List[List[float]], *, hmin: int, hmax: int) -> np.ndarray:
    """[[h, y], ...] → vector y for horizons h=hmin..hmax; fill missing with NaN."""
    h2y = {int(h): float(y) for (h, y) in (points or [])}
    return np.asarray([h2y.get(h, float("nan")) for h in range(hmin, hmax + 1)], float)

def _slope_on_horizons(y: np.ndarray, *, hmin: int, hmax: int) -> Optional[float]:
    """Linear slope of y(h) vs actual horizon values h=hmin..hmax. NaN-safe; None if <2 valid points."""
    if (hmax - hmin + 1) < 2:
        return None
    yy = np.asarray(y, float)
    hh = np.arange(hmin, hmax + 1, dtype=float)
    m = np.isfinite(yy)
    if m.sum() < 2:
        return None
    a, _b = np.polyfit(hh[m], yy[m], 1)
    return float(a)

def _fmt(x: Optional[float], nd=4) -> str:
    """Pretty print float or None."""
    return "NA" if (x is None or not np.isfinite(x)) else f"{x:.{nd}f}"

def _build_knn_yvec(series: np.ndarray, *,
                    look_back: int, neighbors: int, test_size: float,
                    hmin: int, hmax: int, hstep: int) -> np.ndarray:
    """Build KNN GME points with explicit params and compress to y(h=hmin..hmax)."""
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


# ============================= FIVE PUBLIC API FUNCTIONS ================================

def LLE_BASE(series: np.ndarray) -> Optional[float]:
    """
    BASE (general-purpose): KNN forecast-error slope on horizons h = BASE_HMIN..BASE_HMAX.
    Uses BASE_* constants (separate from ML). Returns λ̂ or None if not computable.
    """
    if BASE_HMAX < BASE_HMIN + 1:
        return None
    yvec = _build_knn_yvec(series,
                           look_back=BASE_LOOK_BACK,
                           neighbors=BASE_NEIGHBORS,
                           test_size=BASE_TEST_SIZE,
                           hmin=BASE_HMIN, hmax=BASE_HMAX, hstep=BASE_HSTEP)
    return _slope_on_horizons(yvec, hmin=BASE_HMIN, hmax=BASE_HMAX)


def LLE_ML1(series: np.ndarray) -> Optional[float]:
    """
    ML #1 ("simple"): classifier on raw y(1..10).
    Uses fixed ML_* constants; returns λ̂ or None on failure/missing model.
    """
    try:
        yvec = _build_knn_yvec(series,
                               look_back=ML_LOOK_BACK,
                               neighbors=ML_NEIGHBORS,
                               test_size=ML_TEST_SIZE,
                               hmin=ML_HMIN, hmax=ML_HMAX, hstep=ML_HSTEP)
        lam, _info = get_lyap_ml_simple(yvec)
        return float(lam) if np.isfinite(lam) else None
    except Exception:
        return None


def LLE_ML2(series: np.ndarray) -> Optional[float]:
    """
    ML #2 ("extended"): classifier on engineered features from y(1..10).
    Uses fixed ML_* constants; returns λ̂ or None on failure.
    """
    try:
        yvec = _build_knn_yvec(series,
                               look_back=ML_LOOK_BACK,
                               neighbors=ML_NEIGHBORS,
                               test_size=ML_TEST_SIZE,
                               hmin=ML_HMIN, hmax=ML_HMAX, hstep=ML_HSTEP)
        lam, _info = get_lyap_ml_extended(yvec)
        return float(lam) if np.isfinite(lam) else None
    except Exception:
        return None


def LLE_ML3(series: np.ndarray) -> Optional[float]:
    """
    ML #3 ("corr_n3"): regressor that predicts a correction for slope@n=3 using y(1..10).
    Uses fixed ML_* constants; returns λ̂ or None on failure.
    """
    try:
        yvec = _build_knn_yvec(series,
                               look_back=ML_LOOK_BACK,
                               neighbors=ML_NEIGHBORS,
                               test_size=ML_TEST_SIZE,
                               hmin=ML_HMIN, hmax=ML_HMAX, hstep=ML_HSTEP)
        lam, _info = get_lyap_ml_corr_n3(yvec)
        return float(lam) if np.isfinite(lam) else None
    except Exception:
        return None


def LLE_NOLD(series: np.ndarray,
             emb_dim: int = NOLDS_DEFAULTS["emb_dim"],
             trajectory_len: int = NOLDS_DEFAULTS["trajectory_len"],
             lag: Optional[int] = NOLDS_DEFAULTS["lag"],
             min_tsep: Optional[int] = NOLDS_DEFAULTS["min_tsep"],
             auto_lag: bool = NOLDS_DEFAULTS["auto_lag"],
             lag_max: int = NOLDS_DEFAULTS["lag_max"],
             fallback_to_zero: bool = NOLDS_DEFAULTS["fallback_to_zero"],
             silence_warnings: bool = NOLDS_DEFAULTS["silence_warnings"]) -> float:
    """
    Rosenstein LLE via nolds.lyap_r (classic baseline).
    Returns 0.0 if fallback triggers or computation is unstable.
    """
    return float(get_lyap_nolds(
        series,
        emb_dim=emb_dim,
        trajectory_len=trajectory_len,
        min_tsep=min_tsep,
        lag=lag,
        auto_lag=auto_lag,
        lag_max=lag_max,
        fallback_to_zero=fallback_to_zero,
        silence_warnings=silence_warnings,
    ))


# ============================= GENERATION + REPORTING ================================

def _simulate_one_series(expr: str, r: float, x0: float, length: int,
                         warmup: int, clip_domain: Optional[str]) -> Tuple[np.ndarray, float]:
    """
    Simulate a single series for a fixed r using the exact map expression.
    Returns:
        series (np.ndarray), lle_true (float from analytic derivative)
    """
    _r_arr, _bifur, lyap_true_arr, full_series = simulate_map_and_lyapunov(
        expr_str=expr,
        x_init=float(x0),
        x_len=length,
        x_warmup=warmup,
        r_min=float(r),
        r_max=float(r),
        r_num_points=1,
        bifur_num_points=0,
        clip_domain=clip_domain,
    )
    return full_series[:, 0], float(lyap_true_arr[0])


def _print_trial_line(map_name: str, t: int, r: float, L: int,
                      true_lle: float,
                      base: Optional[float], ml1: Optional[float],
                      ml2: Optional[float], ml3: Optional[float],
                      nold: float):
    """One concise console line per trial."""
    print(
        f"[{map_name:9s} | trial {t:02d}] r={r:.5f}  L={L:4d}  "
        f"λ_true={_fmt(true_lle)}  "
        f"LLE_BASE={_fmt(base)}  "
        f"LLE_ML1={_fmt(ml1)}  "
        f"LLE_ML2={_fmt(ml2)}  "
        f"LLE_ML3={_fmt(ml3)}  "
        f"LLE_NOLD={_fmt(nold)}"
    )


def main():
    rng = np.random.default_rng(RNG_SEED)

    meta = {
        "trials_per_map": TRIALS_PER_MAP,
        "warmup": WARMUP,
        "length_range": [LEN_MIN, LEN_MAX],
        "BASE_params": dict(
            look_back=BASE_LOOK_BACK, neighbors=BASE_NEIGHBORS, test_size=BASE_TEST_SIZE,
            horizon_min=BASE_HMIN, horizon_max=BASE_HMAX, horizon_step=BASE_HSTEP
        ),
        "ML_params_fixed": dict(
            look_back=ML_LOOK_BACK, neighbors=ML_NEIGHBORS, test_size=ML_TEST_SIZE,
            horizon_min=ML_HMIN, horizon_max=ML_HMAX, horizon_step=ML_HSTEP
        ),
        "nolds_params": NOLDS_DEFAULTS,
        "rng_seed": RNG_SEED,
        "timestamp": STAMP,
        "note": "JSON is compact (no KNN points). ML values may be None if models are missing.",
    }

    all_results = {"meta": meta, "maps": []}

    for name, expr, r_min, r_max, clip_domain in MAPS:
        print(f"\n=== MAP: {name} | expr: {expr} | r∈[{r_min}, {r_max}] | clip={clip_domain} ===")
        trials: List[Dict] = []

        # choose a safe x0 domain per map
        xlo, xhi = _x0_range_for_map(name)

        for t in range(1, TRIALS_PER_MAP + 1):
            r = float(rng.uniform(r_min, r_max))
            x0 = float(rng.uniform(xlo, xhi))
            L  = int(rng.integers(LEN_MIN, LEN_MAX + 1))

            series, lle_true = _simulate_one_series(expr, r, x0, L, WARMUP, clip_domain)

            # --- 5 methods
            lle_base = LLE_BASE(series)
            lle_ml1  = LLE_ML1(series)
            lle_ml2  = LLE_ML2(series)
            lle_ml3  = LLE_ML3(series)
            lle_nold = LLE_NOLD(series)

            _print_trial_line(name, t, r, L, lle_true, lle_base, lle_ml1, lle_ml2, lle_ml3, lle_nold)

            trials.append({
                "trial_id": t,
                "r": r,
                "x0": x0,
                "length": L,
                "lambda_true": lle_true,
                "methods": {
                    "LLE_BASE":   None if (lle_base is None or not np.isfinite(lle_base)) else float(lle_base),
                    "LLE_ML1":    None if (lle_ml1  is None or not np.isfinite(lle_ml1 )) else float(lle_ml1),
                    "LLE_ML2":    None if (lle_ml2  is None or not np.isfinite(lle_ml2 )) else float(lle_ml2),
                    "LLE_ML3":    None if (lle_ml3  is None or not np.isfinite(lle_ml3 )) else float(lle_ml3),
                    "LLE_NOLD":   float(lle_nold) if np.isfinite(lle_nold) else None,
                }
            })

        all_results["maps"].append({
            "name": name,
            "expr": expr,
            "r_min": r_min,
            "r_max": r_max,
            "clip_domain": clip_domain,
            "trials": trials,
        })

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\nSaved JSON:", OUT_JSON)


if __name__ == "__main__":
    main()
