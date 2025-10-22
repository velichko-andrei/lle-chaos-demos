# LLE demos & tuning (1D maps)

Small, readable demos for estimating the **largest Lyapunov exponent (LLE)** on classical 1-D maps.

There are two families of estimators:

* **BASE** — a parametric KNN slope method. **You choose the parameters** (look_back, neighbors, test split, horizon range). This reproduces the behavior shown in the paper’s figures.
* **ML methods (ML1/ML2/ML3)** — **no parameters to set**: you pass a series and get a number. These are designed to be robust and convenient for practical use. (If the trained models are not present, they simply return `None`.)

You can try both on your data and keep the one that works best.

> **Reference paper**
> *A novel approach for estimating largest Lyapunov exponents in one-dimensional chaotic time series using machine learning*
> Chaos 35, 101101 (2025)
> Authors: Andrei Velichko (Corresponding Author); Maksim Belyaev; Petr Boriskov

---

## Repository layout

```
.
├── lyapunov_kit.py            # Core toolkit: map simulator, KNN builder, ML wrappers, nolds wrapper
├── demo_tune_base.py          # Grid search for BASE on 4 maps; saves CSV/NPZ/PNG/JSON
├── demo_R2_plot.py            # LLE vs ground truth across many maps; METHOD="BASE" or "ML1"
├── demo_lle.py                # 10 random trials per map; compares BASE, ML1/ML2/ML3, NOLD
├── demo_basic_principle.py    # Baseline KNN principle **without** lyapunov_kit (paper-style)
├── models/                    # (optional) trained models for ML1/ML2/ML3; can be empty
├── results_demo_tune_base/    # Outputs from demo_tune_base.py (timestamped subfolders)
├── results_demo_R2_plot/      # Outputs from demo_R2_plot.py  (timestamped subfolders)
├── results_demo_lle/          # Outputs from demo_lle.py      (timestamped subfolders)
├── runs_tune/                 # Extra/legacy tuning outputs
├── *.bat                      # Windows helpers to run scripts
└── README.md                  # This file
```

> Timestamped folders under `results_*` are created automatically and contain plots, CSV/JSON, NPZ.

---

## Installation

Python ≥ 3.9 is recommended.

```bash
# (optional) create & activate a venv
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# install dependencies
pip install --upgrade pip wheel
pip install numpy scipy scikit-learn matplotlib nolds
```

**ML models:** if you want to use `LLE_ML1/2/3`, place the trained model files into `models/`.
If `models/` is empty, the ML functions will just return `None` (the rest still runs).

---

## Quick start

### 1) Tune BASE parameters on 4 maps

Runs a small grid search and saves artifacts.

```bash
python demo_tune_base.py
```

Outputs → `results_demo_tune_base/<UNIX_TS>/`

You get:

* `*_grid_<STAMP>.csv` — full grid with **R² on true>0 region**
* `*_base_vs_true_<STAMP>.png` — tuned curve vs analytic truth (dense r-grid)
* `*_final_<STAMP>.npz` — arrays `r`, `lyap_true`, `lyap_est`, `best_params`
* `results_<STAMP>.json` — best params + metrics
* `r2_summary_<STAMP>.csv` — compact R² summary

**Where to change the search grid?** At the top of `demo_tune_base.py`:

```python
GRID_LOOK_BACK = [1, 2, 3]
GRID_NEIGHBORS = [1, 3, 10]
GRID_TEST_SIZE = [0.2, 0.4]
GRID_HMAX      = [2, 3, 7]
```

> **Note:** `WARMUP`, `SERIES_LEN`, `R_POINTS_*`, and the r-range define the data you tune on.
> Changing them changes the trajectories and your final R².

---

### 2) Plot LLE vs ground truth across many maps

Switch between the methods with one flag.

```bash
python demo_R2_plot.py
```

Key knobs at the top:

```python
METHOD = "BASE"   # or "ML1"
NOISE_STD = 0.0   # normalized noise added ONLY to estimator input (truth stays clean)

# BASE parameters (user-tunable, independent of ML):
BASE_LOOK_BACK = 1
BASE_NEIGHBORS = 1
BASE_TEST_SIZE = 0.40
BASE_HMIN = 1
BASE_HMAX = 7
BASE_HSTEP = 1
```

* **BASE** uses the parameters above.
* **ML1** has **no user parameters** — just pass a series.

Outputs → `results_demo_R2_plot/<method>_noise<σ>_len<L>_<UNIX_TS>/`

* Per-map LLE plot `*_vs_true_*.png`
* Bifurcation plot `*_bifur_*.png` (if enabled)
* `r2_summary_<STAMP>.csv` (R² on true>0 only)
* Per-map JSON (incremental)

> **To match the tuner’s R² with BASE**, keep `WARMUP`, `SERIES_LEN`, `R_POINTS`, and r-range **identical**.

---

### 3) Random trial benchmark (5 methods)

Runs 10 random series per map and prints one line per trial.

```bash
python demo_lle.py
```

Each trial computes:

```python
LLE_BASE(series)       # uses BASE_* from this file
LLE_ML1(series)        # no user parameters
LLE_ML2(series)        # no user parameters
LLE_ML3(series)        # no user parameters
LLE_NOLD(series, ...)  # Rosenstein lyap_r (nolds)
```

Outputs → `results_demo_lle/<UNIX_TS>/random_maps_runs_<STAMP>.json`

---

### 4) Basic principle (paper-style, no toolkit)

This script re-implements the baseline KNN method **without** `lyapunov_kit` to show the core idea exactly as in the paper.

```bash
python demo_basic_principle.py
```

What it does:

* Simulates the logistic map and computes analytic ground truth.
* Implements its own `get_lyap_knn` (KNN regression across horizons) and linear slope of `log(GME)` vs horizon.
* Tries a small set of BASE parameter combos and reports **R² (true>0)** and **overall R²**.
* Saves the best curve, CSVs, and an NPZ with arrays and metadata under `runs_base/<UNIX_TS>/`.

Where to tweak:

```python
# experiment settings (series length, warmup, r-range, noise)
x_num_points = 1000
x_num_warmup = 1000
r_min, r_max = 3.5, 3.98
std = 0.00   # noise std (set to 0 for noiseless)

# parameter combos to test (look_back, neighbors, hmax, test_size)
for look_back in [1]:
    for neighbors in [1, 3]:
        for horizon_max in [7]:
            for test_size in [0.2, 0.4]:
                ...
```

---

## Method ideology (important)

* **BASE** is a **user-tunable** method. You control:

  * `look_back`, `neighbors`, `test_size`, and the horizon range `[hmin..hmax]`
  * the LLE estimate is the **slope** of `log(GME)` vs horizon over that range
* **ML methods (ML1/ML2/ML3)** are **parameter-free** from the user perspective:

  * you pass a 1-D series and get the predicted LLE
  * they are designed to be robust on real, noisy data
  * if models are not present, the functions return `None` gracefully

---

## Outputs & metrics

* **Ground truth** is always **noise-free**, computed analytically during simulation.
* **Estimator input** may include optional normalized Gaussian noise; **truth is never altered**.
* **R² metric** is computed **only where λ_true > 0** (matches the paper’s figures).
  The field name is `r2_true_pos_only`.

---

## Common pitfalls

* **Warm-up mismatch:** Different `WARMUP` ⇒ different trajectories ⇒ different KNN errors ⇒ different R².
  Match `WARMUP`, `SERIES_LEN`, `R_POINTS`, and r-range if you want identical numbers across scripts.
* **Missing ML models:** If `models/` is empty, `LLE_ML*` returns `None` (no crash).
* **Noise:** Noise is applied only to the estimator input. Ground truth stays clean.

---

## Windows helpers

Double-click the `.bat` files to run:

* `demo_tune_base.bat`
* `demo_R2_plot.bat`
* `demo_lle.bat`

(Each simply calls `python <script>.py`.)

---

## Citation

If you use these scripts, please cite the paper above.

---

This README keeps the user story simple: **BASE has knobs**, **ML has no knobs**, and there’s a **paper-style baseline script** (`demo_basic_principle.py`) for clarity.
