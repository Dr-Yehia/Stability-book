#!/usr/bin/env python3
"""
CFS V18 Symbolic DSM Correction Pipeline (PySR + NSGA-III ranking)
===================================================================

Purpose
-------
Build a publication-oriented symbolic equation workflow for the CFS built-up
column project. The scientific formulation is deliberately *not* a free-form
``Pt/Py = f(all variables)`` model. Instead, the script anchors the proposed
symbolic equation to a physics/design baseline:

    Pt/Py = DSM_local * exp(g_symbolic)

where ``g_symbolic`` is a compact correction learned from leakage-controlled,
defensible design variables only.

What this script does
---------------------
1. Load the V17 final package ZIP, dataset, holdout predictions, and audit files.
2. Run the V17 inference model as a teacher when the optional ML dependencies
   are available; otherwise continue with the official holdout teacher values.
3. Create ``symbolic_dataset.csv`` with physics features and three log-correction
   targets:
      - actual:  log(PtPy_actual / DSM_local)
      - teacher: log(PtPy_teacher / DSM_local)
      - hybrid:  0.70 * actual + 0.30 * teacher
4. Optionally run PySR in staged mode on non-holdout rows only by default,
   preserving the official holdout for final candidate evaluation.
5. Evaluate candidate equations as ``DSM_local * exp(g_symbolic)`` on the
   holdout/test subset, including accuracy, MAPE, COV, bias, unsafe prediction,
   monotonicity, group fairness, complexity, and conservative calibration.
6. Apply Pareto/NSGA-style multi-objective filtering and select Simple,
   Balanced, Advanced, and Conservative recommendations.

Optional dependencies
---------------------
The dataset build and baseline evaluation only require the app dependencies:
``numpy``, ``pandas``, ``scikit-learn``, and ``joblib``.

For symbolic search/ranking install optional research dependencies:

    pip install pysr sympy pymoo

PySR also needs a working Julia runtime. This script keeps PySR optional so the
Streamlit deployment is not forced to install Julia-heavy research packages.

Examples
--------
Build the symbolic dataset and baseline reports only:

    python CFS_V18_Symbolic_DSM_NSGAIII.py --build-only

Run quick PySR searches for actual / teacher / hybrid correction targets:

    python CFS_V18_Symbolic_DSM_NSGAIII.py --run-pysr --preset quick

Evaluate an existing equations CSV exported by this script/PySR:

    python CFS_V18_Symbolic_DSM_NSGAIII.py --candidate-equations results_v18_symbolic/pysr_all_equations.csv

Scientific guardrails
---------------------
Excluded from symbolic equations by design:
``StudyID``, ``TestLabel``, true failure mode, fastener spacing, row labels, and
any target-derived features other than the explicit training target. PySR training
excludes holdout rows unless ``--allow-holdout-training`` is explicitly supplied
for debugging only.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# v18 self-contained bootstrap: install missing deps and fetch the V17 ZIP
# automatically when running on Kaggle / a fresh cloud notebook. Skip silently
# if everything is already importable. Heavy/optional packages (pysr, pymoo,
# sympy) install but the rest of the script will gracefully degrade if pysr's
# Julia runtime is unavailable.
# ---------------------------------------------------------------------------

_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/working").exists()


def _sanitize_argv_for_jupyter() -> None:
    """Strip IPython/Jupyter kernel-launcher flags injected into sys.argv.

    Kaggle/Colab/Jupyter run the script through ipykernel which injects flags
    like ``-f /tmp/.../kernel.json`` and ``--HistoryManager.hist_file=:memory:``
    that argparse would reject. We detect that pattern and reset sys.argv to
    just the program name so the rest of the pipeline (and argparse) sees a
    clean, empty CLI.
    """
    argv = sys.argv[1:]
    if not argv:
        return
    looks_like_kernel = False
    if any("kernel_launcher" in a or "ipykernel" in a for a in argv):
        looks_like_kernel = True
    for i, a in enumerate(argv):
        if a == "-f" and i + 1 < len(argv) and argv[i + 1].endswith(".json"):
            looks_like_kernel = True
            break
    if any(a.startswith("--HistoryManager") for a in argv):
        looks_like_kernel = True
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            looks_like_kernel = True
    except Exception:
        pass
    if looks_like_kernel:
        print(f"[BOOTSTRAP] Detected IPython/Jupyter kernel; stripping kernel argv: {argv}")
        sys.argv = [sys.argv[0]]


_sanitize_argv_for_jupyter()


def _ensure_packages(packages: list[str]) -> None:
    missing = []
    name_map = {"scikit-learn": "sklearn", "pysr": "pysr", "pymoo": "pymoo",
                "sympy": "sympy", "openpyxl": "openpyxl", "xgboost": "xgboost",
                "lightgbm": "lightgbm", "matplotlib": "matplotlib",
                "joblib": "joblib", "pandas": "pandas", "numpy": "numpy"}
    for pkg in packages:
        spec_name = name_map.get(pkg.split("==")[0], pkg.split("==")[0])
        if importlib.util.find_spec(spec_name) is None:
            missing.append(pkg)
    if not missing:
        return
    print(f"[BOOTSTRAP] Installing missing packages: {missing}")
    for pkg in missing:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
        except Exception as exc:
            print(f"[BOOTSTRAP WARN] Could not install {pkg}: {exc!r}")


# Always make sure the lightweight packages used by pure dataset/baseline mode
# are present. The optional research stack is installed too on Kaggle so that
# --run-pysr works out of the box without any additional uploads.
_ensure_packages(["pandas", "numpy", "scikit-learn==1.6.1", "joblib", "matplotlib",
                  "xgboost", "lightgbm", "openpyxl", "sympy", "pymoo"])
if _KAGGLE:
    # PySR requires a Julia runtime; Kaggle provides one transparently after
    # install. We attempt the install but never fail the whole pipeline if the
    # pysr install or its first-time Julia setup is unavailable.
    _ensure_packages(["pysr"])

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# v18 Kaggle/cloud auto-fetch helpers
# ---------------------------------------------------------------------------


def _candidate_zip_paths() -> list[Path]:
    candidates: list[Path] = []
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    candidates.append(here / "results v17" / "cfs_v17_FINAL_PACKAGE.zip")
    candidates.append(here / "cfs_v17_FINAL_PACKAGE.zip")
    candidates.append(Path.cwd() / "results v17" / "cfs_v17_FINAL_PACKAGE.zip")
    candidates.append(Path.cwd() / "cfs_v17_FINAL_PACKAGE.zip")
    candidates.append(Path("/kaggle/working/cfs_v17_FINAL_PACKAGE.zip"))
    candidates.append(Path("/kaggle/working/results v17/cfs_v17_FINAL_PACKAGE.zip"))
    candidates.append(Path("/kaggle/input/cfs-v17-final-package/cfs_v17_FINAL_PACKAGE.zip"))
    candidates += list(Path("/kaggle/input").glob("**/cfs_v17_FINAL_PACKAGE.zip")) if Path("/kaggle/input").exists() else []
    candidates.append(Path("results v17") / "cfs_v17_FINAL_PACKAGE.zip")
    return candidates


# URL-encoded path to the V17 package on the active feature branch.
V17_ZIP_URL = (
    "https://raw.githubusercontent.com/Dr-Yehia/Stability-book/"
    "claude/analyze-v16-results-AjbG2/results%20v17/cfs_v17_FINAL_PACKAGE.zip"
)


def auto_locate_v17_zip(preferred: Path | None = None) -> Path:
    """Return a usable path to cfs_v17_FINAL_PACKAGE.zip, downloading if needed."""
    if preferred is not None and Path(preferred).exists() and Path(preferred).stat().st_size > 1024:
        return Path(preferred)
    for cand in _candidate_zip_paths():
        if cand.exists() and cand.stat().st_size > 1024:
            print(f"[BOOTSTRAP] Found local V17 package: {cand}")
            return cand
    # Download to /kaggle/working when on Kaggle, otherwise current dir.
    target_dir = Path("/kaggle/working") if _KAGGLE and Path("/kaggle/working").exists() else Path.cwd()
    target = target_dir / "cfs_v17_FINAL_PACKAGE.zip"
    print(f"[BOOTSTRAP] Downloading V17 package from GitHub:\n  {V17_ZIP_URL}")
    req = urllib.request.Request(V17_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = resp.read()
    if len(data) < 1024:
        raise RuntimeError("Downloaded V17 package is suspiciously small; aborting.")
    target.write_bytes(data)
    print(f"[BOOTSTRAP] V17 package saved: {target} ({len(data)/1024:.1f} KB)")
    return target


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EPS = 1e-12
E_STEEL_MPA = 200_000.0
DEFAULT_ZIP = Path("results v17") / "cfs_v17_FINAL_PACKAGE.zip"
DEFAULT_ROOT_DATA = Path("CFS_Built-up_Columns_ML_Dataset.csv")
DEFAULT_OUT_DIR = Path("results_v18_symbolic")
MIN_PYSR_TRAIN_ROWS = 50

OFFICIAL_FEATURES = [
    "lambda_c",
    "lambda_led",
    "Pcrl_Py",
    "Pne_Py",
    "h_t",
    "b_t",
    "L_t",
    "A_t2",
    "Fy_E",
]

EXTENDED_AUDIT_FEATURES = [
    *OFFICIAL_FEATURES,
    "L_b",
    "DSM_local",
    "DSM_global",
    "base_geom",
    "base_min",
    "base_mean",
    "FMprob_F",
    "FMprob_LDF",
    "FMprob_entropy",
]

PYSR_TARGETS = {
    "actual": "target_logcorr_actual",
    "teacher": "target_logcorr_teacher",
    "hybrid": "target_logcorr_hybrid",
}

PRESETS = {
    "quick": {
        "niterations": 1000,
        "population_size": 80,
        "populations": 24,
        "maxsize": 35,
    },
    "strong": {
        "niterations": 5000,
        "population_size": 120,
        "populations": 40,
        "maxsize": 45,
    },
    "final": {
        "niterations": 15000,
        "population_size": 150,
        "populations": 60,
        "maxsize": 55,
    },
}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    seen: dict[str, int] = {}
    for i, col in enumerate(df.columns):
        raw = "" if col is None else str(col).strip()
        if raw == "" or raw.lower().startswith("unnamed"):
            continue
        if raw in seen:
            seen[raw] += 1
            raw = f"{raw}__dup{seen[raw]}"
        else:
            seen[raw] = 0
        cols.append((i, raw))
    out = df.iloc[:, [i for i, _ in cols]].copy()
    out.columns = [name for _, name in cols]
    return out


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    direct = {str(c): c for c in df.columns}
    for cand in candidates:
        if cand in direct:
            return direct[cand]
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def numeric_series(df: pd.DataFrame, candidates: Iterable[str], default: float = np.nan) -> pd.Series:
    col = find_col(df, candidates)
    if col is None:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def text_series(df: pd.DataFrame, candidates: Iterable[str], default: str = "UNKNOWN") -> pd.Series:
    col = find_col(df, candidates)
    if col is None:
        return pd.Series(default, index=df.index, dtype=object)
    return df[col].fillna(default).astype(str).str.strip().replace("", default)


def safe_div(a: Any, b: Any) -> np.ndarray:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    return aa / np.maximum(np.abs(bb), EPS)


def safe_log_ratio(numerator: Any, denominator: Any) -> np.ndarray:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    return np.log(np.maximum(num, EPS) / np.maximum(den, EPS))


def finite_clip(x: Any, low: float, high: float) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return np.clip(arr, low, high)


def json_dump(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Classical / DSM-style baseline functions
# ---------------------------------------------------------------------------


def curve_aisc_ssrc(lambda_c: Any) -> np.ndarray:
    lc = np.maximum(np.asarray(lambda_c, dtype=float), EPS)
    return np.where(lc <= 1.5, 0.658 ** (lc**2), 0.877 / (lc**2))


def curve_ecp_lrfd_2012(lambda_c: Any) -> np.ndarray:
    lc = np.maximum(np.asarray(lambda_c, dtype=float), EPS)
    out = np.where(lc <= 1.10, 1.0 - 0.384 * (lc**2), 0.648 / (lc**2))
    return np.clip(out, 0.0, 1.5)


def curve_ec3(lambda_c: Any, alpha: float = 0.34, gamma: float = 1.0) -> np.ndarray:
    lc = np.maximum(np.asarray(lambda_c, dtype=float), EPS)
    phi = 0.5 * (1.0 + alpha * (lc - 0.2) + lc**2)
    rad = np.maximum(phi**2 - lc**2, 0.0)
    chi = 1.0 / np.maximum(phi + np.sqrt(rad), EPS)
    return np.minimum(chi, 1.0) / gamma


def curve_euler_like(klr: Any) -> np.ndarray:
    x = np.maximum(np.asarray(klr, dtype=float), EPS)
    return np.clip((100.0 / x) ** 2, 0.0, 1.5)


def dsm_global(lambda_c: Any) -> np.ndarray:
    return curve_aisc_ssrc(lambda_c)


def dsm_local(lambda_led: Any, pne_py: Any) -> np.ndarray:
    lam = np.maximum(np.asarray(lambda_led, dtype=float), EPS)
    pne = np.asarray(pne_py, dtype=float)
    reduction = np.where(
        lam <= 0.776,
        1.0,
        (1.0 - 0.15 / (lam**0.8 + EPS)) / (lam**0.8 + EPS),
    )
    return np.clip(pne * reduction, 0.02, 1.5)


# ---------------------------------------------------------------------------
# Metrics and candidate evaluation
# ---------------------------------------------------------------------------


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp) & (np.abs(yt) > EPS) & (np.abs(yp) > EPS)
    yt = yt[mask]
    yp = yp[mask]
    if len(yt) < 2:
        return {}
    ratio = yt / np.maximum(yp, EPS)
    return {
        "N": int(len(yt)),
        "R2": float(r2_score(yt, yp)),
        "RMSE": float(mean_squared_error(yt, yp) ** 0.5),
        "MAE": float(mean_absolute_error(yt, yp)),
        "MAPE_%": float(np.mean(np.abs((yt - yp) / np.maximum(np.abs(yt), EPS))) * 100.0),
        "Mean_Test_over_Pred": float(np.mean(ratio)),
        "COV_Test_over_Pred": float(np.std(ratio) / np.maximum(np.mean(ratio), EPS)),
        "Unsafe_Overprediction_%": float(np.mean(yp > yt) * 100.0),
        "Unsafe_5_%": float(np.mean(yp > 1.05 * yt) * 100.0),
        "Unsafe_10_%": float(np.mean(yp > 1.10 * yt) * 100.0),
        "PtPy_gt_1p30_%": float(np.mean(yp > 1.30) * 100.0),
    }


def group_fairness_metrics(df: pd.DataFrame, pred: np.ndarray, group_col: str = "SG_design") -> dict[str, float]:
    if group_col not in df.columns:
        return {"Max_Group_MAPE_%": np.nan, "Std_Group_MAPE_%": np.nan}
    tmp = pd.DataFrame({"actual": df["PtPy_actual"].values, "pred": pred, "group": df[group_col].values})
    values = []
    for _, part in tmp.groupby("group"):
        met = compute_metrics(part["actual"], part["pred"])
        if met and met["N"] >= 3:
            values.append(met["MAPE_%"])
    if not values:
        return {"Max_Group_MAPE_%": np.nan, "Std_Group_MAPE_%": np.nan}
    return {"Max_Group_MAPE_%": float(np.max(values)), "Std_Group_MAPE_%": float(np.std(values))}


def monotonicity_penalty(df: pd.DataFrame, predict_fn, n_base: int = 80, n_grid: int = 8) -> float:
    """Approximate monotonicity violation rate for lambda_c and lambda_led.

    A violation occurs when increasing the chosen slenderness variable increases
    predicted Pt/Py beyond a small numerical tolerance while other variables are
    fixed at sampled row values. When lambda_led slides we also recompute the
    physics baselines (DSM_local, DSM_global, base_min/mean/geom) from the new
    slenderness so the test reflects the full Pt/Py = DSM_local * exp(g)
    expression rather than only the symbolic correction g.
    """
    if df.empty:
        return float("nan")
    sample = df.sample(n=min(n_base, len(df)), random_state=2026).copy()
    total = 0
    violations = 0
    for variable in ["lambda_c", "lambda_led"]:
        if variable not in sample.columns:
            continue
        low, high = np.nanpercentile(df[variable].astype(float), [5, 95])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            continue
        grid = np.linspace(low, high, n_grid)
        for _, row in sample.iterrows():
            block = pd.DataFrame([row.to_dict()] * n_grid)
            block[variable] = grid
            # Recompute physics-baseline columns that depend on the swept variable.
            if variable == "lambda_c":
                block["DSM_global"] = dsm_global(block["lambda_c"].astype(float).values)
            elif variable == "lambda_led":
                pne_py = block["Pne_Py"].astype(float).values if "Pne_Py" in block.columns else np.full(n_grid, np.nan)
                block["DSM_local"] = dsm_local(block["lambda_led"].astype(float).values, pne_py)
            # Refresh derived multi-baseline columns when present so the mono test
            # reflects DSM_local * exp(g) faithfully across the grid.
            cols_for_base = [c for c in ["DSM_global", "DSM_local", "Pne_Py", "Pcrl_Py"] if c in block.columns]
            if {"DSM_global", "DSM_local", "Pne_Py", "Pcrl_Py"}.issubset(block.columns):
                stack = block[["DSM_global", "DSM_local", "Pne_Py", "Pcrl_Py"]].astype(float).values
                block["base_min"] = np.nanmin(stack, axis=1)
                block["base_mean"] = np.nanmean(stack, axis=1)
                block["base_geom"] = np.exp(np.nanmean(np.log(np.maximum(stack, EPS)), axis=1))
            pred = np.asarray(predict_fn(block), dtype=float)
            diffs = np.diff(pred)
            total += len(diffs)
            violations += int(np.sum(diffs > 1e-4))
    if total == 0:
        return float("nan")
    return float(violations / total)


def conservative_calibration(y_true: Any, y_pred: Any, target_bias: tuple[float, float] = (1.03, 1.05)) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    records = []
    # Conservative design calibration should never increase nominal symbolic
    # predictions, so keep k <= 1.0. This makes the final design variant safer
    # even when the unconstrained symbolic expression is slightly biased.
    for k in np.round(np.arange(0.90, 1.0001, 0.005), 3):
        met = compute_metrics(yt, k * yp)
        if not met:
            continue
        mean_ratio = met["Mean_Test_over_Pred"]
        in_band = target_bias[0] <= mean_ratio <= target_bias[1]
        records.append({
            "k": float(k),
            "in_band": bool(in_band),
            "distance": float(min(abs(mean_ratio - target_bias[0]), abs(mean_ratio - target_bias[1])) if not in_band else 0.0),
            **met,
        })
    if not records:
        return {"k_conservative": 1.0}
    table = pd.DataFrame(records)
    table = table.sort_values(["in_band", "Unsafe_10_%", "distance", "MAPE_%"], ascending=[False, True, True, True])
    best = table.iloc[0].to_dict()
    return {
        "k_conservative": float(best["k"]),
        "Conservative_R2": float(best["R2"]),
        "Conservative_MAPE_%": float(best["MAPE_%"]),
        "Conservative_Mean_Test_over_Pred": float(best["Mean_Test_over_Pred"]),
        "Conservative_Unsafe_5_%": float(best["Unsafe_5_%"]),
        "Conservative_Unsafe_10_%": float(best["Unsafe_10_%"]),
    }


def hard_constraint_pass(row: pd.Series, max_complexity: float = 45.0) -> bool:
    return bool(
        row.get("R2", -999.0) >= 0.90
        and row.get("MAPE_%", 999.0) <= 10.0
        and row.get("COV_Test_over_Pred", 999.0) <= 0.15
        and 0.98 <= row.get("Mean_Test_over_Pred", -999.0) <= 1.12
        and row.get("Unsafe_10_%", 999.0) <= 15.0
        and row.get("Monotonicity_Violation", 999.0) <= 0.05
        and row.get("complexity", 999.0) <= max_complexity
    )


def normalize_for_score(s: pd.Series, higher_is_better: bool) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if vals.notna().sum() == 0:
        return pd.Series(0.0, index=s.index)
    vals = vals.fillna(vals.median())
    lo = vals.min()
    hi = vals.max()
    if abs(hi - lo) < EPS:
        return pd.Series(1.0, index=s.index)
    out = (vals - lo) / (hi - lo)
    return out if higher_is_better else 1.0 - out


def add_ranking_score(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    out = table.copy()
    out["Score"] = (
        0.25 * normalize_for_score(out["R2"], True)
        + 0.20 * normalize_for_score(out["MAPE_%"], False)
        + 0.15 * normalize_for_score(out["COV_Test_over_Pred"], False)
        + 0.15 * normalize_for_score(out["Unsafe_5_%"], False)
        + 0.10 * normalize_for_score(out["complexity"], False)
        + 0.10 * normalize_for_score(out["Monotonicity_Violation"], False)
        + 0.05 * normalize_for_score(out["Max_Group_MAPE_%"], False)
    )
    return out


def pareto_front_mask(objectives: np.ndarray) -> np.ndarray:
    """Return non-dominated mask for minimization objectives.

    Uses pymoo when available and falls back to a deterministic NumPy routine.
    All pymoo lookups are wrapped so a missing or broken pymoo install never
    crashes the pipeline.
    """
    try:
        if importlib.util.find_spec("pymoo.util.nds.non_dominated_sorting") is not None:
            module = importlib.import_module("pymoo.util.nds.non_dominated_sorting")
            fronts = module.NonDominatedSorting().do(objectives, only_non_dominated_front=True)
            mask = np.zeros(len(objectives), dtype=bool)
            mask[np.asarray(fronts, dtype=int)] = True
            return mask
    except (ModuleNotFoundError, ImportError, AttributeError, ValueError) as exc:
        print(f"[PARETO] pymoo unavailable, falling back to NumPy: {exc!r}")

    n = objectives.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominated_by_any = np.all(objectives <= objectives[i], axis=1) & np.any(objectives < objectives[i], axis=1)
        dominated_by_any[i] = False
        if dominated_by_any.any():
            mask[i] = False
    return mask


# ---------------------------------------------------------------------------
# Data/package loading
# ---------------------------------------------------------------------------


def extract_package(zip_path: Path, out_dir: Path) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    return out_dir


def load_package_tables(zip_path: Path, work_dir: Path) -> dict[str, Any]:
    pkg_dir = extract_package(zip_path, work_dir / "package")
    tables: dict[str, Any] = {"pkg_dir": pkg_dir}
    tables["summary"] = json.loads((pkg_dir / "v16_final_summary.json").read_text(encoding="utf-8"))
    tables["holdout"] = pd.read_csv(pkg_dir / "v16_holdout_predictions.csv")
    tables["cv_folds"] = pd.read_csv(pkg_dir / "v16_repeated_cv_folds.csv")
    data_path = pkg_dir / "data" / "CFS_Built-up_Columns_ML_Dataset.csv"
    tables["raw_data"] = clean_unnamed(pd.read_csv(data_path))
    return tables


def load_v17_teacher_predictions(pkg_dir: Path, raw_data: pd.DataFrame) -> pd.Series | None:
    """Return V17 teacher predictions when the local environment supports it.

    This intentionally uses importlib instead of import-time dependency forcing,
    so the dataset-building path remains usable in light environments.
    """
    sys.path.insert(0, str(pkg_dir))
    module = importlib.import_module("cfs_v17_predict")
    predictor_cls = getattr(module, "CFSV17Predictor")
    predictor = predictor_cls.load(str(pkg_dir / "cfs_v17_inference_bundle.joblib"))
    pred = predictor.predict(raw_data)
    if "PtPy_pred" not in pred.columns:
        raise ValueError("V17 predictor output does not include PtPy_pred")
    return pd.to_numeric(pred["PtPy_pred"], errors="coerce")


# ---------------------------------------------------------------------------
# Symbolic dataset
# ---------------------------------------------------------------------------


def design_group(section_type: pd.Series, sections: pd.Series) -> pd.Series:
    stype = section_type.fillna("").astype(str).str.upper().str.replace(" ", "", regex=False)
    sec = sections.fillna("").astype(str).str.upper()
    group = pd.Series("G7b_Rest", index=stype.index, dtype=object)
    group[stype.str.contains("O-2C", regex=False)] = "G1_O2C"
    group[stype.str.contains("O-2U", regex=False)] = "G2_O2U"
    group[stype.str.contains("C-U+C", regex=False) | stype.str.contains("CUC", regex=False)] = "G3_CUC"
    group[stype.str.contains("HC", regex=False)] = "G4_HC"
    group[stype.str.contains("C-2C", regex=False)] = "G5_C2C"
    group[sec.str.contains("OPEN", regex=False)] = group.where(group != "G7b_Rest", "G6_Open")
    group[sec.str.contains("CLOSED", regex=False) | sec.str.contains("BOX", regex=False)] = group.where(group != "G7b_Rest", "G7a_Closed_Box")
    return group


def build_symbolic_dataset(
    raw_data: pd.DataFrame,
    holdout: pd.DataFrame,
    teacher_pred: pd.Series | None,
) -> pd.DataFrame:
    df = raw_data.copy().reset_index(drop=True)
    out = pd.DataFrame(index=df.index)
    out["row_index_original"] = np.arange(len(df), dtype=int)

    out["SectionType"] = text_series(df, ["SectionType", "Section Types", "Section_Type"])
    out["Sections"] = text_series(df, ["Sections"])
    out["BC"] = text_series(df, ["BC"])
    out["SG_design"] = design_group(out["SectionType"], out["Sections"])

    out["L"] = numeric_series(df, ["L"])
    out["t"] = numeric_series(df, ["t"])
    out["h"] = numeric_series(df, ["h"])
    out["b"] = numeric_series(df, ["b"])
    out["A"] = numeric_series(df, ["A"])
    out["Fy"] = numeric_series(df, ["Fy"])
    out["Py"] = numeric_series(df, ["Py"])
    out["PtPy_actual"] = numeric_series(df, ["Pt/Py", "PtPy_actual", "PtPy", "Ptest/Py"])
    out["Pcrl"] = numeric_series(df, ["Pcrl", "P(crl,crd)", "Pcrl_crd"])
    out["Pne"] = numeric_series(df, ["Pne"])
    out["KLr"] = numeric_series(df, ["KLr", "KL_r", "KL/r"])
    out["lambda_c"] = numeric_series(df, ["lambda_c", "λc"])
    out["lambda_led"] = numeric_series(df, ["lambda_led", "λ(le-d)"])

    out["h_t"] = safe_div(out["h"], out["t"])
    out["b_t"] = safe_div(out["b"], out["t"])
    out["L_t"] = safe_div(out["L"], out["t"])
    out["L_b"] = safe_div(out["L"], out["b"])
    out["A_t2"] = safe_div(out["A"], np.maximum(out["t"], EPS) ** 2)
    out["Fy_E"] = safe_div(out["Fy"], E_STEEL_MPA)
    out["Pcrl_Py"] = safe_div(out["Pcrl"], out["Py"])
    out["Pne_Py"] = safe_div(out["Pne"], out["Py"])

    out["DSM_global"] = dsm_global(out["lambda_c"])
    out["DSM_local"] = dsm_local(out["lambda_led"], out["Pne_Py"])
    out["base_min"] = np.nanmin(out[["DSM_global", "DSM_local", "Pcrl_Py", "Pne_Py"]].values, axis=1)
    out["base_mean"] = np.nanmean(out[["DSM_global", "DSM_local", "Pcrl_Py", "Pne_Py"]].values, axis=1)
    out["base_geom"] = np.exp(np.nanmean(np.log(np.maximum(out[["DSM_global", "DSM_local", "Pcrl_Py", "Pne_Py"]].values, EPS)), axis=1))

    # Placeholders for optional FM-probability columns if the teacher bundle or
    # a future feature export provides them. They are audit-only by default and
    # are not included in OFFICIAL_FEATURES.
    for col in ["FMprob_F", "FMprob_LDF", "FMprob_entropy"]:
        out[col] = np.nan

    out["is_holdout"] = False
    out["PtPy_teacher"] = np.nan
    if not holdout.empty and "row_index_original" in holdout.columns:
        idx = pd.to_numeric(holdout["row_index_original"], errors="coerce").dropna().astype(int)
        valid_idx = idx[(idx >= 0) & (idx < len(out))]
        out.loc[valid_idx, "is_holdout"] = True
        if "PtPy_pred" in holdout.columns:
            teacher_map = holdout.set_index("row_index_original")["PtPy_pred"]
            out.loc[valid_idx, "PtPy_teacher"] = out.loc[valid_idx, "row_index_original"].map(teacher_map)

    if teacher_pred is not None and len(teacher_pred) == len(out):
        out["PtPy_teacher"] = pd.to_numeric(teacher_pred, errors="coerce").values

    out["target_logcorr_actual"] = safe_log_ratio(out["PtPy_actual"], out["DSM_local"])
    out["target_logcorr_teacher"] = safe_log_ratio(out["PtPy_teacher"], out["DSM_local"])
    out["target_logcorr_hybrid"] = 0.70 * out["target_logcorr_actual"] + 0.30 * out["target_logcorr_teacher"]

    required = ["PtPy_actual", "DSM_local", *OFFICIAL_FEATURES]
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=required).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# PySR
# ---------------------------------------------------------------------------


def pysr_loss_function(alpha: float = 1.5, beta: float = 4.0) -> str:
    """Julia loss for log-correction with asymmetric overprediction penalty.

    PySR custom loss receives prediction and target for the *log correction*.
    The DSM-local scaling is handled by final evaluation outside PySR. Because
    PySR's elementwise loss does not directly know DSM_local and PtPy_actual
    unless using a more advanced custom objective, this loss remains a robust
    Huber-like correction loss. Safety is enforced again during NSGA ranking.
    """
    return textwrap.dedent(
        f"""
        function loss(prediction, target)
            err = prediction - target
            abs_err = abs(err)
            huber = abs_err <= 0.05 ? 0.5 * err^2 : 0.05 * (abs_err - 0.025)
            over = max(err, 0.0)
            severe = max(err - log(1.05), 0.0)
            return huber + {alpha} * over^2 + {beta} * severe^2
        end
        """
    ).strip()


def run_pysr_stage(
    data: pd.DataFrame,
    target_name: str,
    out_dir: Path,
    preset: str,
    features: list[str],
    allow_holdout_training: bool = False,
) -> pd.DataFrame:
    pysr_module = importlib.import_module("pysr")
    PySRRegressor = getattr(pysr_module, "PySRRegressor")

    cfg = PRESETS[preset]
    stage_dir = ensure_dir(out_dir / f"pysr_{target_name}_{preset}")
    cols = features
    target_col = PYSR_TARGETS[target_name]

    train_source = data.copy() if allow_holdout_training else data[~data["is_holdout"]].copy()
    train_df = train_source.dropna(subset=cols + [target_col]).copy()
    if len(train_df) < MIN_PYSR_TRAIN_ROWS:
        mode = "all rows" if allow_holdout_training else "non-holdout rows"
        print(
            f"[WARN] Skipping PySR target={target_name}: only {len(train_df)} "
            f"usable {mode}; need at least {MIN_PYSR_TRAIN_ROWS}."
        )
        return pd.DataFrame()

    x_train = train_df[cols].astype(float)
    y_train = train_df[target_col].astype(float)

    model = PySRRegressor(
        niterations=cfg["niterations"],
        population_size=cfg["population_size"],
        populations=cfg["populations"],
        maxsize=cfg["maxsize"],
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "log1p", "square"],
        extra_sympy_mappings={},
        model_selection="best",
        elementwise_loss=pysr_loss_function(),
        parsimony=0.003,
        complexity_of_operators={"/": 3, "sqrt": 2, "log1p": 3, "square": 2},
        constraints={"/": (-1, 9), "sqrt": 9, "log1p": 9, "square": 9},
        nested_constraints={"sqrt": {"sqrt": 0, "log1p": 1}, "log1p": {"log1p": 0}, "square": {"square": 1}},
        # PySR rejects passing both temp_equation_file=True and a fixed
        # output_directory; we want the equations written into stage_dir,
        # so leave temp_equation_file at its default (False).
        output_directory=str(stage_dir),
        random_state=2026,
        deterministic=True,
        parallelism="serial",
        progress=True,
    )
    model.fit(x_train, y_train, variable_names=cols)
    equations = model.equations_.copy()
    equations.insert(0, "target_stage", target_name)
    equations.insert(1, "preset", preset)
    equations.to_csv(stage_dir / "equations.csv", index=False)
    return equations


# ---------------------------------------------------------------------------
# Equation evaluation / NSGA-style selection
# ---------------------------------------------------------------------------


def expression_to_callable(expression: str, features: list[str]):
    sympy_module = importlib.import_module("sympy")
    symbols = sympy_module.symbols(features)
    if not isinstance(symbols, tuple):
        symbols = (symbols,)
    locals_map = {name: sym for name, sym in zip(features, symbols)}
    # PySR exports can use either variable names or positional x0/x1/x2 names.
    locals_map.update({f"x{i}": sym for i, sym in enumerate(symbols)})
    locals_map.update({
        "sqrt": sympy_module.sqrt,
        "log": sympy_module.log,
        "log1p": lambda x: sympy_module.log(1 + x),
        "exp": sympy_module.exp,
        "pow": sympy_module.Pow,
        "square": lambda x: x**2,
        "cube": lambda x: x**3,
    })
    expr = sympy_module.sympify(str(expression), locals=locals_map)
    func = sympy_module.lambdify(symbols, expr, modules=["numpy"])

    def _predict_g(frame: pd.DataFrame) -> np.ndarray:
        args = [frame[name].astype(float).values for name in features]
        values = func(*args)
        return np.asarray(values, dtype=float)

    return _predict_g, str(expr)


def evaluate_equations(
    symbolic: pd.DataFrame,
    equations: pd.DataFrame,
    out_dir: Path,
    features: list[str],
    allow_no_holdout: bool = False,
) -> pd.DataFrame:
    holdout = symbolic[symbolic["is_holdout"]].copy()
    if holdout.empty:
        if not allow_no_holdout:
            raise RuntimeError(
                "evaluate_equations: no holdout rows were marked. The official "
                "holdout split is required for honest evaluation. Re-run after "
                "ensuring v16_holdout_predictions.csv contains 'row_index_original',"
                " or pass --allow-no-holdout for debugging only."
            )
        print("[WARN] evaluate_equations: no holdout rows, using full dataset (debug mode).")
        holdout = symbolic.copy()

    rows = []
    for i, eq_row in equations.reset_index(drop=True).iterrows():
        expr = eq_row.get("equation") or eq_row.get("sympy_format") or eq_row.get("lambda_format")
        if pd.isna(expr) or str(expr).strip() == "":
            continue
        target_stage = str(eq_row.get("target_stage", "unknown"))
        complexity = float(eq_row.get("complexity", np.nan)) if pd.notna(eq_row.get("complexity", np.nan)) else float(len(str(expr)))
        try:
            g_fn, sympy_expr = expression_to_callable(str(expr), features)
            def pred_fn(frame: pd.DataFrame) -> np.ndarray:
                g = finite_clip(g_fn(frame), -3.0, 3.0)
                return np.asarray(frame["DSM_local"], dtype=float) * np.exp(g)

            pred = finite_clip(pred_fn(holdout), 0.0, 1.5)
            met = compute_metrics(holdout["PtPy_actual"], pred)
            if not met:
                continue
            fairness = group_fairness_metrics(holdout, pred)
            mono = monotonicity_penalty(holdout, pred_fn)
            calib = conservative_calibration(holdout["PtPy_actual"], pred)
            rows.append({
                "candidate_id": f"EQ_{i:04d}",
                "target_stage": target_stage,
                "equation": str(expr),
                "sympy_equation": sympy_expr,
                "complexity": complexity,
                "pysr_loss": float(eq_row.get("loss", np.nan)) if pd.notna(eq_row.get("loss", np.nan)) else np.nan,
                **met,
                **fairness,
                "Monotonicity_Violation": mono,
                **calib,
            })
        except Exception as exc:
            rows.append({
                "candidate_id": f"EQ_{i:04d}",
                "target_stage": target_stage,
                "equation": str(expr),
                "complexity": complexity,
                "evaluation_error": str(exc),
            })

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    numeric_cols = [
        "R2", "MAPE_%", "COV_Test_over_Pred", "Mean_Test_over_Pred", "Unsafe_5_%",
        "Unsafe_10_%", "complexity", "Monotonicity_Violation", "Max_Group_MAPE_%",
    ]
    valid = table.dropna(subset=[c for c in numeric_cols if c in table.columns]).copy()
    if not valid.empty:
        valid["Hard_Constraint_Pass"] = valid.apply(hard_constraint_pass, axis=1)
        objectives = np.column_stack([
            1.0 - valid["R2"].values,
            valid["MAPE_%"].values,
            valid["COV_Test_over_Pred"].values,
            np.abs(valid["Mean_Test_over_Pred"].values - 1.03),
            valid["Unsafe_5_%"].values,
            valid["complexity"].values / 100.0,
            valid["Monotonicity_Violation"].values,
            valid["Max_Group_MAPE_%"].values,
        ])
        objectives = np.nan_to_num(objectives, nan=1e6, posinf=1e6, neginf=1e6)
        valid["Pareto_NonDominated"] = pareto_front_mask(objectives)
        valid = add_ranking_score(valid)
        table = table.merge(
            valid[["candidate_id", "Hard_Constraint_Pass", "Pareto_NonDominated", "Score"]],
            on="candidate_id",
            how="left",
        )
    table.to_csv(out_dir / "nsga3_candidate_evaluation.csv", index=False)
    return table


def select_recommendations(evaluated: pd.DataFrame) -> dict[str, Any]:
    if evaluated.empty or "R2" not in evaluated.columns:
        return {}
    table = evaluated.copy()
    table = table[table["R2"].notna()].copy()
    if table.empty:
        return {}

    pareto_mask = table["Pareto_NonDominated"].fillna(False) if "Pareto_NonDominated" in table.columns else pd.Series(False, index=table.index)
    pool = table[pareto_mask].copy()
    if pool.empty:
        pool = table.copy()
    hard_mask = pool["Hard_Constraint_Pass"].fillna(False) if "Hard_Constraint_Pass" in pool.columns else pd.Series(False, index=pool.index)
    constrained = pool[hard_mask].copy()
    if not constrained.empty:
        pool = constrained
    pool = add_ranking_score(pool).sort_values("Score", ascending=False)

    def pick(mask: pd.Series, fallback: pd.DataFrame) -> dict[str, Any]:
        subset = pool[mask].copy()
        if subset.empty:
            subset = fallback.copy()
        if subset.empty:
            return {}
        row = subset.sort_values("Score", ascending=False).iloc[0]
        keys = [
            "candidate_id", "target_stage", "equation", "complexity", "R2", "MAPE_%",
            "COV_Test_over_Pred", "Mean_Test_over_Pred", "Unsafe_5_%", "Unsafe_10_%",
            "Monotonicity_Violation", "Max_Group_MAPE_%", "Score", "k_conservative",
            "Conservative_R2", "Conservative_MAPE_%", "Conservative_Mean_Test_over_Pred",
            "Conservative_Unsafe_5_%", "Conservative_Unsafe_10_%",
        ]
        return {k: (None if pd.isna(row.get(k, np.nan)) else row.get(k)) for k in keys if k in row.index}

    recs = {
        "Simple": pick(pool["complexity"] <= 15, pool.nsmallest(5, "complexity")),
        "Balanced": pick((pool["complexity"] > 15) & (pool["complexity"] <= 30), pool),
        "Advanced": pick((pool["complexity"] > 30) & (pool["complexity"] <= 45), pool.sort_values("R2", ascending=False)),
    }

    # Conservative recommendation prefers calibrated safety after selection.
    conservative_pool = pool.copy()
    if "Conservative_Unsafe_10_%" in conservative_pool.columns:
        conservative_pool = conservative_pool.sort_values(
            ["Conservative_Unsafe_10_%", "Conservative_MAPE_%", "complexity"],
            ascending=[True, True, True],
        )
    if conservative_pool.empty:
        recs["Design_Conservative"] = {}
    else:
        conservative_row = conservative_pool.iloc[0]
        recs["Design_Conservative"] = {
            k: (None if pd.isna(conservative_row.get(k, np.nan)) else conservative_row.get(k))
            for k in [
                "candidate_id", "target_stage", "equation", "complexity", "R2", "MAPE_%",
                "COV_Test_over_Pred", "Mean_Test_over_Pred", "Unsafe_5_%", "Unsafe_10_%",
                "Monotonicity_Violation", "Max_Group_MAPE_%", "Score", "k_conservative",
                "Conservative_R2", "Conservative_MAPE_%", "Conservative_Mean_Test_over_Pred",
                "Conservative_Unsafe_5_%", "Conservative_Unsafe_10_%",
            ]
            if k in conservative_row.index
        }
    return recs


# ---------------------------------------------------------------------------
# Baseline comparisons
# ---------------------------------------------------------------------------


def evaluate_baselines(symbolic: pd.DataFrame, allow_no_holdout: bool = False) -> pd.DataFrame:
    holdout = symbolic[symbolic["is_holdout"]].copy()
    if holdout.empty:
        if not allow_no_holdout:
            raise RuntimeError(
                "evaluate_baselines: no holdout rows were marked. Pass "
                "--allow-no-holdout for debug runs only."
            )
        print("[WARN] evaluate_baselines: no holdout rows, using full dataset (debug mode).")
        holdout = symbolic.copy()
    methods = {
        "AISC_SSRC": curve_aisc_ssrc(holdout["lambda_c"]),
        "ECP_LRFD_2012": curve_ecp_lrfd_2012(holdout["lambda_c"]),
        "EC3_alpha_034": curve_ec3(holdout["lambda_c"], alpha=0.34),
        "EC3_alpha_021": curve_ec3(holdout["lambda_c"], alpha=0.21),
        "Euler_like": curve_euler_like(holdout["KLr"]),
        "DSM_global": holdout["DSM_global"],
        "DSM_local": holdout["DSM_local"],
        "base_geom": holdout["base_geom"],
    }
    if "PtPy_teacher" in holdout.columns and holdout["PtPy_teacher"].notna().any():
        methods["V17_teacher_or_holdout_prediction"] = holdout["PtPy_teacher"]
    rows = []
    for name, pred in methods.items():
        met = compute_metrics(holdout["PtPy_actual"], pred)
        if met:
            rows.append({"Method": name, **met})
    return pd.DataFrame(rows).sort_values("R2", ascending=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CFS V18 DSM-corrected symbolic regression pipeline.")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="Path to cfs_v17_FINAL_PACKAGE.zip.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick", help="PySR search preset.")
    parser.add_argument("--run-pysr", action="store_true", help="Run PySR staged symbolic regression.")
    parser.add_argument("--build-only", action="store_true", help="Only build symbolic dataset and baseline report.")
    parser.add_argument("--candidate-equations", type=Path, help="Existing candidate equations CSV to evaluate/rank.")
    parser.add_argument("--skip-teacher", action="store_true", help="Skip full V17 teacher inference; use official holdout teacher only.")
    parser.add_argument(
        "--allow-holdout-training",
        action="store_true",
        help="Debugging only: allow PySR to train on holdout rows. Do not use for publication runs.",
    )
    parser.add_argument("--allow-no-holdout", action="store_true", help="Debugging only: allow evaluation without a marked holdout split.")
    parser.add_argument("--features", choices=["official", "extended"], default="official", help="Feature set for PySR/equation evaluation.")
    parser.add_argument("--no-package", action="store_true", help="Skip auto-zipping the output directory at the end.")

    # Kaggle / cloud notebook ergonomics: when run inside an IPython kernel
    # (Kaggle, Colab, Jupyter), sys.argv contains kernel-launcher flags such
    # as ``-f /tmp/.../kernel.json`` that argparse would reject. Detect that
    # case and replace argv with our publication-friendly defaults so the
    # user can simply hit "Run" without any CLI arguments.
    raw_argv = sys.argv[1:]

    def _looks_like_ipykernel(argv: list[str]) -> bool:
        if not argv:
            return False
        if any("kernel_launcher" in a or "ipykernel" in a for a in argv):
            return True
        # Pattern: -f /tmp/<...>.json injected by Jupyter/IPython kernels.
        for i, a in enumerate(argv):
            if a == "-f" and i + 1 < len(argv) and argv[i + 1].endswith(".json"):
                return True
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                return True
        except Exception:
            pass
        return False

    use_defaults = (not raw_argv) or _looks_like_ipykernel(raw_argv) or _KAGGLE
    if use_defaults:
        out_default = Path("/kaggle/working/results_v18_symbolic") if Path("/kaggle/working").exists() else DEFAULT_OUT_DIR
        argv = ["--run-pysr", "--preset", "quick", "--features", "official",
                "--out-dir", str(out_default)]
        print(f"[BOOTSTRAP] Kaggle/IPython context detected -> auto-running with: {argv}")
    else:
        argv = raw_argv
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# v18 final packaging: produce a single downloadable ZIP next to the output
# directory so the user gets one click in the Kaggle Output panel.
# ---------------------------------------------------------------------------


def package_outputs_zip(out_dir: Path, archive_basename: str = "cfs_v18_FINAL_PACKAGE") -> Path | None:
    out_dir = Path(out_dir)
    if not out_dir.exists():
        print(f"[PACKAGE WARN] output dir missing: {out_dir}")
        return None
    parent = out_dir.parent if out_dir.parent.exists() else out_dir
    archive_root = str(parent / archive_basename)
    try:
        archive_path = shutil.make_archive(archive_root, "zip", root_dir=str(out_dir))
        print(f"\n[PACKAGE] Created: {archive_path}")
        return Path(archive_path)
    except Exception as exc:
        print(f"[PACKAGE WARN] make_archive failed: {exc!r}")
        try:
            zip_path = archive_root + ".zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in out_dir.rglob("*"):
                    if p.is_file():
                        zf.write(p, arcname=p.relative_to(out_dir))
            print(f"[PACKAGE] Created via zipfile fallback: {zip_path}")
            return Path(zip_path)
        except Exception as exc2:
            print(f"[PACKAGE ERR] zipfile fallback failed: {exc2!r}")
            return None


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    # v18 self-locating ZIP: try the user-provided path, then a list of common
    # local paths, finally fall back to a GitHub raw download.
    try:
        zip_path = auto_locate_v17_zip(args.zip if args.zip.exists() else None)
    except Exception as exc:
        raise FileNotFoundError(
            f"V17 package ZIP could not be located or downloaded ({exc!r}). "
            "Provide --zip <path> or place cfs_v17_FINAL_PACKAGE.zip next to the script."
        ) from exc
    args.zip = zip_path

    work_dir = ensure_dir(out_dir / "_work")
    tables = load_package_tables(args.zip, work_dir)
    raw_data = tables["raw_data"]
    holdout = tables["holdout"]
    summary = tables["summary"]

    teacher_pred = None
    teacher_error = None
    if not args.skip_teacher:
        try:
            teacher_pred = load_v17_teacher_predictions(tables["pkg_dir"], raw_data)
        except Exception as exc:
            teacher_error = str(exc)
            print(f"[WARN] Full V17 teacher inference unavailable; using holdout teacher only. Reason: {exc}")

    symbolic = build_symbolic_dataset(raw_data, holdout, teacher_pred)
    symbolic_path = out_dir / "symbolic_dataset.csv"
    symbolic.to_csv(symbolic_path, index=False)
    print(f"[OK] Wrote {symbolic_path} ({symbolic.shape[0]} rows, {symbolic.shape[1]} columns)")

    # Holdout guard: refuse to silently fall back to the full dataset.
    n_holdout = int(symbolic["is_holdout"].sum())
    print(f"[INFO] Holdout rows marked: {n_holdout} / {len(symbolic)}")
    if n_holdout == 0 and not args.allow_no_holdout:
        raise RuntimeError(
            "No holdout rows were marked in the symbolic dataset. The official "
            "holdout split is required for honest evaluation. Re-run after "
            "confirming v16_holdout_predictions.csv carries 'row_index_original',"
            " or pass --allow-no-holdout for debug runs only."
        )

    baseline = evaluate_baselines(symbolic, allow_no_holdout=args.allow_no_holdout)
    baseline_path = out_dir / "equation_vs_codes_table.csv"
    baseline.to_csv(baseline_path, index=False)
    print(f"[OK] Wrote {baseline_path}")

    features = OFFICIAL_FEATURES if args.features == "official" else EXTENDED_AUDIT_FEATURES
    features = [f for f in features if f in symbolic.columns and symbolic[f].notna().any()]

    all_equations = []
    if args.run_pysr:
        for target_name in PYSR_TARGETS:
            print(f"[INFO] Running PySR target={target_name}, preset={args.preset}, features={features}")
            eqs = run_pysr_stage(
                symbolic,
                target_name,
                out_dir,
                args.preset,
                features,
                allow_holdout_training=args.allow_holdout_training,
            )
            if not eqs.empty:
                all_equations.append(eqs)

    if args.candidate_equations is not None:
        all_equations.append(pd.read_csv(args.candidate_equations))

    evaluated = pd.DataFrame()
    recommendations = {}
    if all_equations:
        equations = pd.concat(all_equations, ignore_index=True)
        equations_path = out_dir / "pysr_all_equations.csv"
        equations.to_csv(equations_path, index=False)
        print(f"[OK] Wrote {equations_path}")

        evaluated = evaluate_equations(symbolic, equations, out_dir, features, allow_no_holdout=args.allow_no_holdout)
        recommendations = select_recommendations(evaluated)
        json_dump(recommendations, out_dir / "best_symbolic_equations.json")
        print(f"[OK] Wrote {out_dir / 'best_symbolic_equations.json'}")

    report = {
        "script": Path(__file__).name,
        "formulation": "Pt/Py = DSM_local * exp(g_symbolic)",
        "official_features": OFFICIAL_FEATURES,
        "extended_audit_features": EXTENDED_AUDIT_FEATURES,
        "active_features": features,
        "targets": PYSR_TARGETS,
        "outputs": {
            "symbolic_dataset": str(symbolic_path),
            "baseline_comparison": str(baseline_path),
            "candidate_evaluation": str(out_dir / "nsga3_candidate_evaluation.csv") if not evaluated.empty else None,
            "recommendations": str(out_dir / "best_symbolic_equations.json") if recommendations else None,
        },
        "data": {
            "n_rows_symbolic": int(len(symbolic)),
            "n_holdout": int(symbolic["is_holdout"].sum()),
            "teacher_inference_available": teacher_pred is not None,
            "teacher_error": teacher_error,
            "allow_holdout_training": bool(args.allow_holdout_training),
            "min_pysr_train_rows": MIN_PYSR_TRAIN_ROWS,
        },
        "official_v16_holdout_metrics": summary.get("holdout_test_metrics", {}),
        "hard_constraints": {
            "R2_test_min": 0.90,
            "MAPE_test_max_%": 10.0,
            "COV_max": 0.15,
            "Mean_Test_over_Pred_range": [0.98, 1.12],
            "Unsafe_10_max_%": 15.0,
            "Monotonic_violation_max": 0.05,
            "Official_complexity_max": 45,
            "Conservative_k_range": [0.90, 1.00],
        },
        "nsga3_objectives_minimized": [
            "1 - R2_test",
            "MAPE_test",
            "COV_test_over_pred",
            "abs(mean_test_over_pred - 1.03)",
            "unsafe_5_percent",
            "complexity / 100",
            "monotonicity_violation",
            "max_group_MAPE",
        ],
    }
    json_dump(report, out_dir / "symbolic_report.json")
    print(f"[OK] Wrote {out_dir / 'symbolic_report.json'}")

    if args.build_only:
        print("[DONE] Build-only mode complete.")
    elif not args.run_pysr and args.candidate_equations is None:
        print("[DONE] Dataset and baseline reports complete. Use --run-pysr or --candidate-equations for symbolic ranking.")
    else:
        print("[DONE] Symbolic workflow complete.")

    # v18 final packaging: a single downloadable ZIP next to the output dir.
    archive_path = None
    if not args.no_package:
        try:
            archive_path = package_outputs_zip(out_dir, archive_basename="cfs_v18_FINAL_PACKAGE")
        except Exception as exc:
            print(f"[PACKAGE WARN] failed to package outputs: {exc!r}")

    print("\n" + "=" * 100)
    print("V18 SUMMARY")
    print("=" * 100)
    print(f"Output directory : {out_dir}")
    print(f"Holdout rows     : {int(symbolic['is_holdout'].sum())} / {len(symbolic)}")
    if not evaluated.empty and "R2" in evaluated.columns:
        try:
            best_r2 = float(evaluated["R2"].max())
            print(f"Best candidate R²: {best_r2:.4f}")
        except Exception:
            pass
    if archive_path:
        print(f"\n>>> DOWNLOAD: {archive_path}")
        print(">>> In Kaggle, click this file in the Output panel to download the full V18 package.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
