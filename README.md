# LLE demos & tuning (1D maps) ✨

Small, readable demos for estimating the **largest Lyapunov exponent (LLE)** on classical 1-D maps.

Two estimator families:

* **BASE** — a *parametric* KNN slope method. **You choose the parameters** (look_back, neighbors, test split, horizon range). This reproduces the behavior shown in the paper’s figures.
* **ML methods (ML1/ML2/ML3)** — **no user parameters**: you pass a series and get a number. Designed for robust, convenient use. *(If trained models are missing, they simply return `None`.)*

> **Reference**
> *A novel approach for estimating largest Lyapunov exponents in one-dimensional chaotic time series using machine learning*
> Chaos 35, 101101 (2025) — Andrei Velichko; Maksim Belyaev; Petr Boriskov
> DOI: [10.1063/5.0289352](https://doi.org/10.1063/5.0289352))
> 
Full text: https://www.researchgate.net/profile/Andrei-Velichko

🎥 Video demonstration

For a quick, intuitive explanation of how the Lyapunov exponent estimation works, check out the short demo:
👉 Watch the video here https://doigram.com/watch/vid_51

It visually walks through the forecasting-error idea and how both BASE and ML1 extract the slope — perfect if you prefer to see the method in action rather than just read about it.
---

## 📁 Repository layout (current)

```
.
├── lyapunov_kit.py            # Core toolkit: map simulator, KNN builder, ML wrappers, nolds wrapper
├── demo_tune_base.py          # Grid search for BASE on 4 maps; saves CSV/NPZ/PNG/JSON
├── demo_R2_plot.py            # LLE vs ground truth across many maps; METHOD="BASE" or "ML1"
├── demo_lle.py                # 10 random trials per map; compares BASE, ML1/ML2/ML3, NOLD
├── demo_basic_principle.py    # Baseline KNN principle WITHOUT lyapunov_kit (paper-style)
├── models/                    # (optional) trained models for ML1/ML2/ML3; can be empty
├── demo_example/              # ✅ Example artifacts committed to the repo (PNG/CSV/NPZ/JSON)
│   ├── demo_demo_basic_principle/
│   │   └── 1761061508/        # best_plot.png, data_full.npz, CSVs, JSON
│   ├── demo_demo_tune_base/
│   │   └── 1761051002/        # *_grid_*.csv, *_final_*.npz, *_base_vs_true_*.png, results_*.json
│   ├── demo_demo_R2_plot/
│   │   ├── base_noise0_len1000_1761055061/   # per-map LLE plots, bifurcation plots, CSV, JSON
│   │   ├── base_noise0.006_len1000_1761055332/
│   │   ├── ml1_noise0_len1000_1761055283/
│   │   └── ml1_noise0.006_len1000_1761055303/
│   └── demo_demo_lle/         # random_maps_runs_*.json
├── .env                       # local settings (not required)
├── requirements.txt
├── LICENSE (MIT)
└── README.md
```

> The `demo_example/` folder lets you **preview** typical outputs (plots, CSV, NPZ) without running anything. Open the PNGs and CSVs right away. 🖼️📄

---

## ⚙️ Installation

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

**ML models:** to use `LLE_ML1/2/3`, place the trained model files in `models/`.
If `models/` is empty, ML functions just return `None` (the rest still runs).

---

## 🚀 Quick start

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

**Change the search grid** at the top of `demo_tune_base.py`:

```python
GRID_LOOK_BACK = [1, 2, 3]
GRID_NEIGHBORS = [1, 3, 10]
GRID_TEST_SIZE = [0.2, 0.4]
GRID_HMAX      = [2, 3, 7]
```

> 🔎 **Note:** `WARMUP`, `SERIES_LEN`, `R_POINTS_*`, and the r-range define the data you tune on.
> Changing them changes the trajectories and your final R².

---

### 2) Plot LLE vs ground truth across many maps

Switch method with one flag.

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

> 🎯 To match the tuner’s R² with BASE, keep `WARMUP`, `SERIES_LEN`, `R_POINTS`, and r-range **identical**.

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

### 4) Basic principle (paper-style, no toolkit) 🧪

Re-implements the baseline KNN method **without** `lyapunov_kit` to show the core idea exactly as in the paper.

```bash
python demo_basic_principle.py
```

It:

* Simulates the logistic map and computes analytic ground truth.
* Implements its own `get_lyap_knn` (KNN regression across horizons) and linear slope of `log(GME)` vs horizon.
* Tries a small set of BASE parameter combos and reports **R² (true>0)** and **overall R²**.
* Saves the best curve, CSVs, and an NPZ under `runs_base/<UNIX_TS>/`.

Where to tweak:

```python
# experiment settings
x_num_points = 1000
x_num_warmup = 1000
r_min, r_max = 3.5, 3.98
std = 0.00   # noise std (0 for noiseless)

# parameter combos to test
for look_back in [1]:
    for neighbors in [1, 3]:
        for horizon_max in [7]:
            for test_size in [0.2, 0.4]:
                ...
```

---

## 🧠 Method ideology (important)

* **BASE** is **user-tunable**. You control:

  * `look_back`, `neighbors`, `test_size`, and the horizon range `[hmin..hmax]`
  * the LLE estimate is the **slope** of `log(GME)` vs horizon over that range
* **ML methods (ML1/ML2/ML3)** are **parameter-free** from the user perspective:

  * you pass a 1-D series and get the predicted LLE
  * designed to be robust on real, noisy data
  * if models are not present, the functions return `None` gracefully

---

## 📊 Outputs & metrics

* **Ground truth** is always **noise-free**, computed analytically during simulation.
* **Estimator input** may include optional normalized Gaussian noise; **truth is never altered**.
* **R² metric** is computed **only where λ_true > 0** (matches the paper’s figures) — field name: `r2_true_pos_only`.

---

## ❗ Common pitfalls

* **Warm-up mismatch:** different `WARMUP` ⇒ different trajectories ⇒ different KNN errors ⇒ different R².
  Keep `WARMUP`, `SERIES_LEN`, `R_POINTS`, and r-range aligned across scripts.
* **Missing ML models:** if `models/` is empty, `LLE_ML*` returns `None`.
* **Noise:** noise is applied only to the estimator input; the analytic truth remains clean.

---

## 🧾 License & citation

* License: **MIT** (see `LICENSE`).
* If you use these scripts, please cite the paper above. 🙏

---

This README keeps the user story simple: **BASE has knobs**, **ML has no knobs**, and there’s a **paper-style baseline** (`demo_basic_principle.py`) plus a **demo_example** folder with ready-made artifacts you can open immediately.
