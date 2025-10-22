# lyapunov_kit.py
# Version: 1.0.0 (production)
# Description:
#   Utilities for building 1D discrete maps from analytic formulas (with analytic derivative),
#   vectorized simulation over parameter grids, and multiple Lyapunov-exponent estimators:
#   - KNN-based slope-from-forecast-error (points over horizons)
#   - Rosenstein method via nolds.lyap_r
#   - ML-based inference over KNN points (model loader + 3 predictors)
#
# Requirements: numpy, sympy, scikit-learn, scipy, nolds
# Paper (credit): "A novel approach for estimating largest Lyapunov exponents in one-dimensional
#                 chaotic time series using machine learning", Chaos 35, 101101 (2025),
#                 doi:10.1063/5.0289352
# Authors: Andrei Velichko (Corresponding Author), Maksim Belyaev, Petr Boriskov

from __future__ import annotations
import warnings
import numpy as np
import sympy as sp
import nolds
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from scipy.stats import gmean

__version__ = "1.0.0"

__all__ = [
    "build_map_functions",
    "simulate_map_and_lyapunov",
    "get_lyap_knn",
    "get_lyap_nolds",
    "estimate_lag_acf",
    # ↓↓↓ NEW: export ML inference helpers and model loader
    "load_ml_models",
    "get_lyap_ml_simple",
    "get_lyap_ml_extended",
    "get_lyap_ml_corr_n3",
]


# =============================================================================
# utils: sympy -> numpy
# =============================================================================


def build_map_functions(expr_str: str):
    """
    Build vectorized NumPy callables f(x,r) and dfdx(x,r) from string.
    - Accepts SymPy names: sin, cos, tanh, Abs, Mod, Piecewise, pi, ...
    - If analytic derivative can't be lambdified (e.g., due to Mod/Piecewise),
      falls back to robust numerical derivative (central difference).
    """
    x, r = sp.symbols("x r")

    ns = {
        "pi": sp.pi,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "tanh": sp.tanh,
        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
        "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
        "Abs": sp.Abs, "Mod": sp.Mod, "Piecewise": sp.Piecewise,
        "min": sp.Min, "max": sp.Max,
    }

    # Guard against accidental 'np.' in string expressions
    s = expr_str.replace("np.", "")
    expr = sp.sympify(s, locals=ns)

    # forward map
    f = sp.lambdify((x, r), expr, modules="numpy")

    # try analytic derivative → fallback to numerical if it fails
    try:
        dfdx_expr = sp.diff(expr, x)
        dfdx = sp.lambdify((x, r), dfdx_expr, modules="numpy")
        _ = dfdx(0.123, 2.0)  # smoke test
        return f, dfdx
    except Exception as e:
        warnings.warn(
            f"[build_map_functions] Analytic derivative failed for '{expr_str}'. "
            f"Using numerical derivative. Reason: {type(e).__name__}: {e}",
            RuntimeWarning
        )

        def dfdx_num(xv, rv, h=1e-7):
            xv = np.asarray(xv, dtype=float)
            rv = np.asarray(rv, dtype=float)
            xp = f(xv + h, rv)
            xm = f(xv - h, rv)
            out = (xp - xm) / (2.0 * h)
            return np.where(np.isfinite(out), out, 0.0)

        return f, dfdx_num

# =============================================================================
# simulation (ground truth)
# =============================================================================

def simulate_map_and_lyapunov(
    expr_str: str,
    *,
    x_init: float = 0.5,
    x_len: int = 10_000,
    x_warmup: int = 1_000,
    r_min: float = 3.0,
    r_max: float = 4.0,
    r_num_points: int = 1_000,
    bifur_num_points: int = 500,
    clip_domain: str | None = None,  # '01' → clip to [0,1]; '-11' → clip to [-1,1]
    eps: float = 1e-12,
    dtype=np.float64,
):
    """
    Vectorized simulation of x_{t+1} = f(x_t, r) over a grid of r-values.

    Returns:
        r              : (r_num_points,) grid
        bifur          : (bifur_num_points, r_num_points) last states (for bifurcation plot)
        lyap           : (r_num_points,) theoretical Lyapunov exponent (mean log|f'(x)|)
        full_series    : (x_len, r_num_points) simulated series after warmup

    Notes:
      - expr_str is parsed symbolically; derivative f'(x) is computed analytically.
      - clip_domain:
          '01'  → clip to [0, 1]
          '-11' → clip to [-1, 1]
          None  → no clipping
    """
    f, dfdx = build_map_functions(expr_str)

    r = np.linspace(r_min, r_max, r_num_points, dtype=dtype)
    lyap = np.zeros(r_num_points, dtype=dtype)
    full_series = np.zeros((x_len, r_num_points), dtype=dtype)
    bifur = np.zeros((max(0, bifur_num_points), r_num_points), dtype=dtype)

    x = np.ones(r_num_points, dtype=dtype) * dtype(x_init)

    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(x_len + x_warmup):
            # avoid division-by-zero explosions for maps like 1/x
            x_safe = np.where(np.abs(x) < 1e-12, np.sign(x) * 1e-12 + (x == 0) * 1e-12, x)
            x = f(x_safe, r)

            if clip_domain == "01":
                x = np.clip(x, 0.0, 1.0)
            elif clip_domain == "-11":
                x = np.clip(x, -1.0, 1.0)
            elif isinstance(clip_domain, (tuple, list)) and len(clip_domain) == 2:
                lo, hi = float(clip_domain[0]), float(clip_domain[1])
                x = np.clip(x, lo, hi)


            if i >= x_warmup:
                deriv = dfdx(x, r)
                # guard against 0/NaN/complex
                deriv_abs = np.maximum(np.abs(np.real(deriv)), eps)
                lyap += np.log(deriv_abs)
                full_series[i - x_warmup, :] = x

                if bifur_num_points > 0:
                    j = i - (x_len + x_warmup - bifur_num_points)
                    if j >= 0:
                        bifur[j, :] = x

    lyap = lyap / float(x_len)
    return r, bifur, lyap, full_series

# =============================================================================
# KNN-based estimator: build points over forecast horizons
# =============================================================================

def get_lyap_knn(
    series=None,
    look_back: int = 10,
    test_size: float = 0.2,
    neighbors: int = 20,
    horizon_min: int = 1,
    horizon_max: int = 10,
    horizon_step: int = 1,
):
    """
    Estimate λ from the slope of log(GME(error)) vs forecast horizon (KNN forecaster).

    Returns:
      points: list of [horizon, log(GME(error))]
      slope : linear regression coefficient (λ estimate) or None if not computable
    """
    if series is None:
        series = []

    horizon_range = list(range(horizon_min, horizon_max + 1, horizon_step))
    if len(horizon_range) < 2:
        print("Should be at least 2 points to make regression line")
        return [], None

    min_length = int(np.ceil(1 / max(test_size, 1e-12) + look_back + max(horizon_range)))
    if len(series) < min_length:
        print(f"Should be at least {min_length} elements in data series")
        return [], None

    points = []
    series = np.asarray(series, dtype=float)

    for horizon in horizon_range:
        dataX, dataY = [], []
        last = len(series) - look_back - horizon
        if last <= 1:
            continue

        for i in range(last):
            a = series[i : (i + look_back)]
            dataX.append(a)
            dataY.append(series[i + look_back + horizon - 1])

        X = np.asarray(dataX, dtype=float)
        Y = np.asarray(dataY, dtype=float).reshape(-1, 1)

        split_point = int(round(X.shape[0] * (1 - test_size)))
        if split_point <= 0 or split_point >= X.shape[0]:
            # no valid train/test split
            continue

        trainX, trainY = X[:split_point], Y[:split_point].ravel()
        testX, testY   = X[split_point:], Y[split_point:]

        # robustness for small train size: cap k by #train samples
        k_eff = int(max(1, min(neighbors, trainX.shape[0])))

        knn = KNeighborsRegressor(n_neighbors=k_eff, weights="distance", p=2, n_jobs=-1)
        knn.fit(trainX, trainY)
        predY = knn.predict(testX).reshape(-1, 1)

        err = np.abs(testY - predY).ravel()
        testGME = float(gmean(np.maximum(err, 1e-12)))  # scalar

        if not np.isfinite(testGME) or testGME <= 1e-12:
            # edge case: exact predictions or numeric issues
            points.append([horizon, np.log(1e-12)])
        else:
            points.append([horizon, np.log(testGME)])

    if len(points) < 2:
        return points, None

    line = np.asarray(points, dtype=float)
    reg = LinearRegression().fit(line[:, 0].reshape(-1, 1), line[:, 1])
    return points, float(reg.coef_[0])

# =============================================================================
# Rosenstein estimator via nolds.lyap_r (with optional auto-lag)
# =============================================================================

def estimate_lag_acf(series, max_lag=1000, threshold=1/np.e) -> int:
    """
    Estimate lag from autocorrelation: first lag where ACF < threshold.
    If none found: return min(max_lag, 0.05*len(series)), at least 1.
    """
    x = np.asarray(series, dtype=float)
    n = x.size
    if n < 3:
        return 1
    x = x - np.mean(x)
    max_lag = int(min(max_lag, n - 2))
    if max_lag < 1:
        return 1

    # fast ACF via tail correlation (normalized by zero-lag)
    acf = np.correlate(x, x, mode="full")[n-1:n+max_lag]
    if acf[0] == 0:
        return 1
    acf = acf / acf[0]

    idx = np.where(acf < threshold)[0]
    if idx.size == 0:
        return max(1, min(max_lag, int(0.05 * n)))
    return int(idx[0])


def get_lyap_nolds(
    series,
    *,
    emb_dim: int = 10,
    trajectory_len: int = 20,
    min_tsep: int | None = None,
    lag: int | None = None,
    auto_lag: bool = True,
    lag_max: int = 1000,
    fallback_to_zero: bool = True,
    silence_warnings: bool = True,
):
    """
    Lyapunov exponent via Rosenstein et al., implemented as nolds.lyap_r.

    Behavior:
      - If lag=None and auto_lag=True → estimate lag from ACF (estimate_lag_acf).
      - If min_tsep=None → use max(2*lag, 10) to decouple time neighbors.
      - With silence_warnings=True, suppress RuntimeWarning from nolds.

    Returns:
      float scalar; on issues returns 0.0 (if fallback_to_zero=True) or NaN.
    """
    try:
        x = np.asarray(series, dtype=float)
        if x.size < (emb_dim + trajectory_len + 5):
            # too short series → nolds may misbehave
            return 0.0 if fallback_to_zero else np.nan

        if (lag is None) and auto_lag:
            lag = estimate_lag_acf(x, max_lag=lag_max)
        if min_tsep is None:
            min_tsep = max(2 * (lag or 1), 10)

        if silence_warnings:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="nolds")
                lle = nolds.lyap_r(
                    x, emb_dim=emb_dim, lag=lag, min_tsep=min_tsep, trajectory_len=trajectory_len
                )
        else:
            lle = nolds.lyap_r(
                x, emb_dim=emb_dim, lag=lag, min_tsep=min_tsep, trajectory_len=trajectory_len
            )

        if not np.isfinite(lle):
            return 0.0 if fallback_to_zero else np.nan
        return float(lle)

    except Exception:
        return 0.0 if fallback_to_zero else np.nan


# ============================ ML inference for KNN points ======================
# Models are stored as pickles in ./models. Paths are read from a .env file
# (located alongside this module). Fallback: pick the most recent by mtime.

import os, glob, pickle
from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np

# --- Constants synchronized with the training script (feature layout matters!) ---
_HMAX = 10
_HOR = np.arange(1, _HMAX + 1, dtype=float)  # 1..10
EPS = 1e-12
RMSE_ONE_SE_DELTA = 0.05
STAB_TAU = 0.05
STAB_RUN = 2

# Cache for loaded models (avoid reloads)
__ML_CACHE = {"simple": None, "extended": None, "corr": None}
__ML_PATHS = {"simple": None, "extended": None, "corr": None}
__WARN_ONCE = {
    "simple": False, "extended": False, "corr": False,
    "env": False, "compat_ext": False, "compat_corr": False
}

def _warn_once(tag: str, msg: str):
    if not __WARN_ONCE.get(tag, False):
        __WARN_ONCE[tag] = True
        print(f"[ML-WARN:{tag}] {msg}", flush=True)

def _info_once(tag: str, msg: str):
    if not __WARN_ONCE.get(tag, False):
        __WARN_ONCE[tag] = True
        print(f"[ML-INFO:{tag}] {msg}", flush=True)

# ---------- Read model names from .env (module folder) ----------
# Expected keys in .env:
#   CLS_SIMPLE_PKL=cls_simple_h10_....pkl
#   CLS_EXTENDED_PKL=cls_extended_h10_....pkl
#   REG_CORR_N3_PKL=reg_corr_n3_h10_....pkl
_ENV_KEYS = {
    "simple": "CLS_SIMPLE_PKL",
    "extended": "CLS_EXTENDED_PKL",
    "corr": "REG_CORR_N3_PKL",
}

def _read_env_model_paths(models_dir: str) -> Dict[str, Optional[str]]:
    base_dir = Path(__file__).resolve().parent
    env_path = base_dir / ".env"
    env_vals: Dict[str, Optional[str]] = {"simple": None, "extended": None, "corr": None}

    loaded_any = False
    if env_path.exists():
        # no dependency on external dotenv: parse manually
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip(); v = v.strip().strip('"\'')

            for tag, env_key in _ENV_KEYS.items():
                if k == env_key and v:
                    # allow absolute path or filename under models/
                    p = Path(v)
                    if not p.is_absolute():
                        p = Path(models_dir) / v
                    env_vals[tag] = str(p)
                    loaded_any = True
        if not loaded_any:
            _warn_once("env", f".env found ({env_path.name}) but expected keys {_ENV_KEYS.values()} are missing")
    else:
        _warn_once("env", f".env not found next to module ({env_path}). Fallback by mtime will be used.")

    # filter-out non-existing files
    for tag, p in env_vals.items():
        if p is not None and not os.path.isfile(p):
            _warn_once("env", f".env refers to a non-existing file for {tag}: {p}")
            env_vals[tag] = None
    return env_vals

def _latest_model(models_dir: str, prefix: str) -> Optional[str]:
    """Pick the most recent *.pkl matching `<prefix>_h10_*.pkl` (fallback)."""
    pat = os.path.join(models_dir, f"{prefix}_h{_HMAX}_*.pkl")
    cands = glob.glob(pat)
    if not cands:
        return None
    cands.sort(key=os.path.getmtime, reverse=True)
    return cands[0]

def load_ml_models(
    models_dir: str = "models",
    *,
    cls_simple_path: str | None = None,
    cls_extended_path: str | None = None,
    reg_corr_n3_path: str | None = None,
    force_reload: bool = False,
) -> Dict[str, object]:
    """
    Load/cache the 3 models from `models_dir`.
    Priority of paths:
      1) explicit *_path arguments,
      2) names from .env (CLS_SIMPLE_PKL / CLS_EXTENDED_PKL / REG_CORR_N3_PKL),
      3) fallback: most recent file by modification time.
    """
    # 1) explicit
    paths = {
        "simple": cls_simple_path,
        "extended": cls_extended_path,
        "corr": reg_corr_n3_path,
    }

    # 2) from .env (only if not provided explicitly)
    env_paths = _read_env_model_paths(models_dir)
    for k in paths:
        if paths[k] is None:
            paths[k] = env_paths.get(k)

    # 3) fallback by modification time
    for k, prefix in (("simple", "cls_simple"), ("extended", "cls_extended"), ("corr", "reg_corr_n3")):
        if paths[k] is None:
            paths[k] = _latest_model(models_dir, prefix)

    # load
    for key, p in paths.items():
        if p is None:
            __ML_CACHE[key] = None
            __ML_PATHS[key] = None
            _warn_once(key, f"model file not found (neither .env nor by date) in '{models_dir}'")
            continue
        if (not force_reload) and (__ML_PATHS.get(key) == p) and (__ML_CACHE.get(key) is not None):
            continue
        with open(p, "rb") as f:
            __ML_CACHE[key] = pickle.load(f)
            __ML_PATHS[key] = p
            _info_once(key, f"loaded model: {p}")

    return {k: __ML_CACHE[k] for k in ("simple", "extended", "corr")}

# ---------- Helpers: polyfit, prefix slopes, features --------------------------
def _polyfit_masked(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Linear regression ignoring NaNs; if <2 points → (nan, nan)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan"), float("nan")
    a, b = np.polyfit(x[m], y[m], 1)
    return float(a), float(b)

def _slopes_from_points(y: np.ndarray) -> np.ndarray:
    """
    y: length-10 array (log GME values for horizons h=1..10).
    Returns [s_2, s_3, ..., s_10] (length 9): slope of linear fit on prefix 1..n.
    """
    y = np.asarray(y, float)
    if y.shape[0] != _HMAX:
        raise ValueError(f"expected points of length {_HMAX}, got {y.shape[0]}")
    out = np.full(_HMAX - 1, np.nan, float)  # indices 0..8 => n=2..10
    for idx, n in enumerate(range(2, _HMAX + 1)):
        a, b = _polyfit_masked(_HOR[:n], y[:n])
        out[idx] = a
    return out

def _features_simple(y: np.ndarray) -> np.ndarray:
    """Simple features: raw y(1..10)."""
    y = np.asarray(y, float)
    if y.shape[0] != _HMAX:
        raise ValueError(f"expected points of length {_HMAX}, got {y.shape[0]}")
    return y.reshape(1, -1)

# ---- IMPORTANT: extended features exactly match the training pipeline (83 features) ----
def _features_extended(y: np.ndarray) -> np.ndarray:
    """
    Build 83 features exactly as in training:
      [ y(10), y_norm0(10), y_normR(10),
        dy(9), ddy(8),
        y0,yL, yr, mean(dy), std(dy),
        neg_dy, pos_dy, frac_neg_dy, frac_pos_dy,
        frac_neg_ddy, frac_pos_ddy, mean_ddy, std_ddy,
        dy0, dy1, dyn2, dyn1, dy_ratio_10,
        s_seq(9), s0, s1, s_last, s_mean, s_std, ds0, dsn1,
        one_se_n, stab_n ]
    The feature order matches the training pipeline 1:1.
    """
    y = np.asarray(y, float)
    if y.shape[0] != _HMAX:
        raise ValueError(f"expected points of length {_HMAX}, got {y.shape[0]}")
    x = _HOR

    dy  = np.diff(y)
    ddy = np.diff(dy) if y.size >= 3 else np.array([])

    y0, yL = y[0], y[-1]
    yr = float(np.nanmax(y) - np.nanmin(y)) if np.all(np.isfinite(y)) else 0.0
    y_norm0 = y - y0
    rng = max(abs(yL - y0), EPS)
    y_normR = y_norm0 / rng

    # monotonicity / derivative stats
    neg_dy = int(np.sum(dy < 0));  pos_dy = int(np.sum(dy > 0))
    frac_neg_dy = float(neg_dy) / max(dy.size, 1)
    frac_pos_dy = float(pos_dy) / max(dy.size, 1)

    if ddy.size:
        neg_ddy = int(np.sum(ddy < 0)); pos_ddy = int(np.sum(ddy > 0))
        frac_neg_ddy = float(neg_ddy) / max(ddy.size, 1)
        frac_pos_ddy = float(pos_ddy) / max(ddy.size, 1)
        mean_ddy = float(np.nanmean(ddy)); std_ddy = float(np.nanstd(ddy))
    else:
        frac_neg_ddy = frac_pos_ddy = 0.0
        mean_ddy = std_ddy = 0.0

    dy0 = float(dy[0]) if dy.size else 0.0
    dy1 = float(dy[1]) if dy.size >= 2 else 0.0
    dyn1 = float(dy[-1]) if dy.size else 0.0
    dyn2 = float(dy[-2]) if dy.size >= 2 else 0.0
    dy_ratio_10 = (dy1 / max(abs(dy0), EPS)) if dy.size >= 2 else 0.0

    # sequence of prefix slopes s_n
    s_seq = []
    for n in range(2, _HMAX + 1):
        a, b = _polyfit_masked(x[:n], y[:n])
        s_seq.append(a)
    s_seq = np.asarray(s_seq, float)
    ds = np.diff(s_seq) if s_seq.size >= 2 else np.array([])
    s0 = float(s_seq[0]) if s_seq.size else 0.0
    s1 = float(s_seq[1]) if s_seq.size >= 2 else s0
    s_last = float(s_seq[-1]) if s_seq.size else s0
    s_mean = float(np.nanmean(s_seq)) if s_seq.size else 0.0
    s_std  = float(np.nanstd(s_seq))  if s_seq.size else 0.0
    ds0  = float(ds[0]) if ds.size else 0.0
    dsn1 = float(ds[-1]) if ds.size else 0.0

    # RMSE over prefixes -> rmin -> one-SE index
    rm = []
    for n in range(2, _HMAX + 1):
        a, b = _polyfit_masked(x[:n], y[:n])
        if not np.isfinite(a):
            rm.append(np.nan)
        else:
            yy = y[:n]
            yhat = a * x[:n] + b
            m = np.isfinite(yy) & np.isfinite(yhat)
            rm.append(float(np.sqrt(np.mean((yy[m] - yhat[m])**2))) if m.sum() else np.nan)
    rm = np.asarray(rm, float)
    rmin = np.nanmin(rm) if np.any(np.isfinite(rm)) else np.nan
    one_se_n = _HMAX
    if np.isfinite(rmin):
        for i, n in enumerate(range(2, _HMAX + 1)):
            if np.isfinite(rm[i]) and rm[i] <= rmin * (1.0 + RMSE_ONE_SE_DELTA + 1e-12):
                one_se_n = n
                break

    # "stability" of s_n (consecutive small changes for `runs` times)
    def _stab_n(s, tau=STAB_TAU, runs=STAB_RUN):
        s = np.asarray(s, float)
        if np.sum(np.isfinite(s)) < 2:
            return 2
        for i in range(1, s.size):
            ok = True
            for k in range(runs):
                j = i + k
                if j >= s.size:
                    ok = False; break
                sp, sn = s[j-1], s[j]
                if not (np.isfinite(sp) and np.isfinite(sn)):
                    ok = False; break
                denom = max(abs(sp), EPS)
                if abs(sn - sp) > tau * denom:
                    ok = False; break
            if ok:
                return i + 1
        return s.size + 1
    stab_n = _stab_n(s_seq)

    feats = []
    feats += list(y) + list(y_norm0) + list(y_normR)                       # 30
    feats += list(dy) + list(ddy)                                          # +9 +8 = 47
    feats += [y0, yL, yr, float(np.nanmean(dy)) if dy.size else 0.0,
              float(np.nanstd(dy))  if dy.size else 0.0]                   # +5  = 52
    feats += [neg_dy, pos_dy, frac_neg_dy, frac_pos_dy,
              frac_neg_ddy, frac_pos_ddy, mean_ddy, std_ddy]               # +8  = 60
    feats += [dy0, dy1, dyn2, dyn1, dy_ratio_10]                           # +5  = 65
    feats += list(s_seq)                                                   # +9  = 74
    feats += [s0, s1, s_last, s_mean, s_std, ds0, dsn1, one_se_n, stab_n]  # +9  = 83
    return np.asarray(feats, float).reshape(1, -1)

def _resize_to_model(X: np.ndarray, model, tag: str) -> np.ndarray:
    """Match `model.n_features_in_` by padding/truncation if needed (safety guard)."""
    exp = getattr(model, "n_features_in_", None)
    if exp is None or X.shape[1] == exp:
        return X
    cur = X.shape[1]
    if cur < exp:
        pad = np.zeros((X.shape[0], exp - cur), dtype=float)
        X = np.hstack([X, pad])
        _warn_once(tag, f"feature size {cur}→{exp}: padded with zeros")
    else:
        X = X[:, :exp]
        _warn_once(tag, f"feature size {cur}→{exp}: truncated tail features")
    return X

# ---------------------------- Public ML inference APIs -------------------------

def get_lyap_ml_simple(
    points_1to10: np.ndarray,
    *,
    models_dir: str = "models",
    model_path: str | None = None,
) -> Tuple[float, Dict]:
    """
    λ̂ via cls_simple (Pipeline: StandardScaler -> MLPClassifier):
      classify n ∈ {0,2..10} from raw y(1..10), then λ̂ = s_n.
    """
    models = load_ml_models(models_dir, cls_simple_path=model_path)
    cls = models["simple"]
    if cls is None:
        raise FileNotFoundError("Model cls_simple (*.pkl) not found.")
    y = np.asarray(points_1to10, float)
    X = _features_simple(y)
    try:
        X = _resize_to_model(X, cls, "simple")  # usually length 10; kept defensive
        n_pred = int(np.clip(cls.predict(X)[0], 0, _HMAX))
    except Exception as e:
        _warn_once("simple", f"predict error: {type(e).__name__}: {e}")
        n_pred = 3  # safe default
    slopes = _slopes_from_points(y)
    lam = 0.0 if n_pred == 0 else float(slopes[n_pred - 2])
    return lam, {"chosen_n": n_pred, "model_path": __ML_PATHS["simple"]}

def get_lyap_ml_extended(
    points_1to10: np.ndarray,
    *,
    models_dir: str = "models",
    model_path: str | None = None,
) -> Tuple[float, Dict]:
    """
    λ̂ via cls_extended (Pipeline: StandardScaler -> MLPClassifier):
      classify n ∈ {0,2..10} from 83 engineered features; then λ̂ = s_n.
    """
    models = load_ml_models(models_dir, cls_extended_path=model_path)
    cls = models["extended"]
    if cls is None:
        raise FileNotFoundError("Model cls_extended (*.pkl) not found.")
    y = np.asarray(points_1to10, float)
    X = _features_extended(y)
    X = _resize_to_model(X, cls, "extended")
    try:
        n_pred = int(np.clip(cls.predict(X)[0], 0, _HMAX))
    except Exception as e:
        _warn_once("extended", f"predict error: {type(e).__name__}: {e}")
        n_pred = 3
    slopes = _slopes_from_points(y)
    lam = 0.0 if n_pred == 0 else float(slopes[n_pred - 2])
    return lam, {"chosen_n": n_pred, "model_path": __ML_PATHS["extended"]}

def get_lyap_ml_corr_n3(
    points_1to10: np.ndarray,
    *,
    models_dir: str = "models",
    model_path: str | None = None,
    clip_factor: Tuple[float, float] = (0.25, 4.0),
    nan_if_invalid: bool = False,
) -> Tuple[float, Dict]:
    """
    λ̂ via reg_corr_n3 (Pipeline: StandardScaler -> MLPRegressor):
      predict a multiplicative factor for s3, then λ̂ = s3 * clip(factor_pred).
    """
    models = load_ml_models(models_dir, reg_corr_n3_path=model_path)
    reg = models["corr"]
    if reg is None:
        raise FileNotFoundError("Model reg_corr_n3 (*.pkl) not found.")
    y = np.asarray(points_1to10, float)
    slopes = _slopes_from_points(y)
    s3 = float(slopes[1])  # n=3 → index 1
    if not (np.isfinite(s3) and s3 > 0):
        return (np.nan if nan_if_invalid else 0.0), {
            "chosen_n": 3, "model_path": __ML_PATHS["corr"], "factor": np.nan
        }
    X = _features_extended(y)
    X = _resize_to_model(X, reg, "corr")
    try:
        fmin, fmax = clip_factor
        factor = float(np.clip(reg.predict(X)[0], fmin, fmax))
    except Exception as e:
        _warn_once("corr", f"predict error: {type(e).__name__}: {e}")
        factor = 1.0
    lam = s3 * factor
    return lam, {"chosen_n": 3, "model_path": __ML_PATHS["corr"], "factor": factor}
