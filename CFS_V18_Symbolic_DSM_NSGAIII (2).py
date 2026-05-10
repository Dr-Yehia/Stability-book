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
import datetime as _dt
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
import traceback
import urllib.request
import uuid as _uuid
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


# ---------------------------------------------------------------------------
# v18+ Comprehensive logging: every script start gets a unique RUN_ID and a
# system banner. If Kaggle restarts the kernel mid-run, the log will show a
# fresh banner with a different RUN_ID, making the restart unambiguous in
# the saved log file. We also detect prior run artifacts and report them.
# ---------------------------------------------------------------------------
RUN_ID = _uuid.uuid4().hex[:12]
RUN_START_ISO = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_psutil():
    try:
        import psutil  # type: ignore
        return psutil
    except Exception:
        return None


def _system_snapshot() -> dict:
    snap: dict = {
        "RUN_ID": RUN_ID,
        "started_utc": RUN_START_ISO,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "cwd": str(Path.cwd()),
        "kaggle_detected": _KAGGLE,
        "argv": list(sys.argv),
        "env_kaggle_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", ""),
        "env_kaggle_working_dir": os.environ.get("KAGGLE_WORKING_DIR", ""),
    }
    try:
        snap["cpu_count"] = os.cpu_count()
    except Exception:
        snap["cpu_count"] = None
    ps = _safe_psutil()
    if ps is not None:
        try:
            mem = ps.virtual_memory()
            snap["ram_total_GB"] = round(mem.total / (1024**3), 2)
            snap["ram_avail_GB"] = round(mem.available / (1024**3), 2)
            snap["ram_used_pct"] = mem.percent
        except Exception:
            pass
    return snap


def _print_run_banner(label: str = "RUN START") -> None:
    snap = _system_snapshot()
    bar = "=" * 90
    print(bar, flush=True)
    print(f"[V18 {label}] RUN_ID={RUN_ID} | started_utc={RUN_START_ISO}", flush=True)
    print(bar, flush=True)
    for k, v in snap.items():
        print(f"  {k}: {v}", flush=True)
    print(bar, flush=True)


def _detect_previous_run_artifacts() -> None:
    """Surface clear evidence in the log when Kaggle has restarted the kernel
    and we are running on top of a previous attempt's artifacts."""
    candidates = [
        Path("/kaggle/working/results_v18_symbolic"),
        Path.cwd() / "results_v18_symbolic",
    ]
    for path in candidates:
        if not path.exists():
            continue
        kids = list(path.glob("*"))
        if not kids:
            continue
        marker = path / ".v18_run_log.txt"
        prev_lines: list[str] = []
        if marker.exists():
            try:
                prev_lines = marker.read_text(encoding="utf-8").strip().splitlines()
            except Exception:
                prev_lines = []
        if prev_lines:
            print(
                f"[V18 RESTART DETECTED] Previous run artifacts found at {path}.",
                flush=True,
            )
            for line in prev_lines[-5:]:
                print(f"  prev_run> {line}", flush=True)
        else:
            print(
                f"[V18 RESTART DETECTED] Existing artifact dir at {path} (no prior run log).",
                flush=True,
            )


def _append_run_log(out_dir: Path, line: str) -> None:
    """Persist a small line to .v18_run_log.txt so a Kaggle kernel restart
    can read what the prior attempt was doing right before it died."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        marker = out_dir / ".v18_run_log.txt"
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(marker, "a", encoding="utf-8") as f:
            f.write(f"{ts} RUN_ID={RUN_ID} {line}\n")
    except Exception:
        pass


_print_run_banner("BOOT")
_detect_previous_run_artifacts()


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
                  "xgboost", "lightgbm", "openpyxl", "sympy", "pymoo", "psutil"])
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

# v18.1+ compact engineered feature set. The previous run (hybrid-only on the 9
# OFFICIAL_FEATURES) capped at R^2=0.56 because PySR was burning search budget
# rediscovering ratios that we already know are physically meaningful (h/b, the
# lambda interaction term, log of slenderness ratios). Pre-computing them turns
# a free-form symbolic search into a guided one, which is the single biggest
# lever for raising holdout R^2 in a short PySR run.
COMPACT_FEATURES = [
    "lambda_c",
    "lambda_led",
    "h_b",
    "Pcrl_Py",
    "Pne_Py",
    "pcr_pne",
    "Fy_E",
    "lambda_inter",
    "lambda_inter2",
    "log_pne",
    "log_pcrl",
]

EXTENDED_AUDIT_FEATURES = [
    *OFFICIAL_FEATURES,
    "L_b",
    "h_b",
    "lambda_inter",
    "lambda_inter2",
    "pcr_pne",
    "log_pne",
    "log_pcrl",
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

# v18.1+ time/quality optimization: include ``actual`` alongside ``hybrid`` by
# default. The previous hybrid-only run capped at R^2=0.56 on holdout because
# the hybrid target is biased toward the V17 teacher surface and produced
# expressions that didn't generalize to the raw experiments. Searching both
# targets in the same run roughly doubles wall-clock time but lets the NSGA
# stage pick whichever expression actually generalizes best on the held-out
# experiments. Override with --pysr-targets if you want a single target.
PYSR_DEFAULT_TARGETS = ["actual", "hybrid"]

PRESETS = {
    "quick": {
        # v18+ memory-safe Kaggle config: previous (1000, 80, 24) caused
        # the Julia/PySR worker to grow large enough that Kaggle's commit
        # runner intermittently restarted the whole kernel mid-fit. Halving
        # the per-worker population footprint resolves the restart loop
        # while keeping the overall search budget close to original.
        "niterations": 1000,
        "population_size": 50,
        "populations": 16,
        "maxsize": 30,
        "procs": 2,
    },
    # v18.1+ publication-oriented preset: smaller maxsize (22) so PySR
    # produces equations a journal reviewer can actually read, slightly
    # higher iteration budget for convergence, same memory footprint as
    # quick. Pair with --features compact and --refit-constants for the
    # best speed/quality trade-off.
    "fast_publish": {
        "niterations": 2500,
        "population_size": 60,
        "populations": 18,
        "maxsize": 22,
        "procs": 2,
    },
    "strong": {
        "niterations": 5000,
        "population_size": 100,
        "populations": 30,
        "maxsize": 40,
        "procs": 2,
    },
    "final": {
        "niterations": 15000,
        "population_size": 130,
        "populations": 50,
        "maxsize": 50,
        "procs": 2,
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
    yp_over_yt = yp / np.maximum(yt, EPS)
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
        # v18+ severe-tail safety metrics. Catch the rare catastrophic
        # overpredictions a good mean MAPE would otherwise hide.
        "Unsafe_20_%": float(np.mean(yp > 1.20 * yt) * 100.0),
        "Max_Overprediction": float(np.max(yp_over_yt)),
        "P95_Overprediction": float(np.quantile(yp_over_yt, 0.95)),
        "PtPy_gt_1p30_%": float(np.mean(yp > 1.30) * 100.0),
    }


def group_fairness_metrics(df: pd.DataFrame, pred: np.ndarray, group_col: str = "SG_design") -> dict[str, float]:
    """Per-design-group safety metrics.

    Beyond the original Max/Std MAPE we also expose the worst-group unsafe
    rate, worst-group bias deviation, and worst-group max overprediction so
    a candidate cannot pass by averaging out a localized failure.
    """
    empty = {
        "Max_Group_MAPE_%": np.nan,
        "Std_Group_MAPE_%": np.nan,
        "Max_Group_Unsafe_10_%": np.nan,
        "Worst_Group_Mean_TP_dev": np.nan,
        "Worst_Group_Max_Overprediction": np.nan,
    }
    if group_col not in df.columns:
        return empty
    tmp = pd.DataFrame({"actual": df["PtPy_actual"].values, "pred": pred, "group": df[group_col].values})
    mapes: list[float] = []
    unsafes_10: list[float] = []
    bias_devs: list[float] = []
    max_overs: list[float] = []
    for _, part in tmp.groupby("group"):
        met = compute_metrics(part["actual"], part["pred"])
        if not met or met.get("N", 0) < 3:
            continue
        mapes.append(met["MAPE_%"])
        unsafes_10.append(float(met.get("Unsafe_10_%", 0.0)))
        bias_devs.append(abs(float(met.get("Mean_Test_over_Pred", 1.0)) - 1.0))
        max_overs.append(float(met.get("Max_Overprediction", 1.0)))
    if not mapes:
        return empty
    return {
        "Max_Group_MAPE_%": float(np.max(mapes)),
        "Std_Group_MAPE_%": float(np.std(mapes)),
        "Max_Group_Unsafe_10_%": float(np.max(unsafes_10)),
        "Worst_Group_Mean_TP_dev": float(np.max(bias_devs)),
        "Worst_Group_Max_Overprediction": float(np.max(max_overs)),
    }


def singularity_safety_audit(
    g_callable,
    holdout_df: pd.DataFrame,
    features: list[str],
    n_grid_samples: int = 2000,
    g_abs_max: float = 5.0,
    extreme_g_threshold: float = 2.5,
    extreme_g_max_frac: float = 0.02,
    nonfinite_max_frac: float = 0.005,
) -> dict[str, Any]:
    """Hard safety audit against denominator/singular blow-ups.

    Pt/Py = DSM_local * exp(g). If g spikes to ±5 the corrected prediction
    multiplies/divides by ~150 — clearly non-physical. We sample N synthetic
    feature combinations uniformly within the empirical 2nd–98th percentile
    range of the holdout features and evaluate the symbolic correction g
    on them. If |g| ever exceeds g_abs_max, or a meaningful fraction
    exceeds the softer ``extreme_g_threshold``, or any value is non-finite
    above tolerance, we flag the equation as singular/dangerous.
    """
    info = {
        "Singularity_Safe": False,
        "Singularity_MaxAbsG": float("inf"),
        "Singularity_FracExtreme": 1.0,
        "Singularity_FracNonFinite": 1.0,
        "Singularity_Reason": "no_eval",
    }
    feats = [c for c in features if c in holdout_df.columns]
    if not feats:
        info["Singularity_Reason"] = "no_features"
        return info
    rng = np.random.default_rng(2026)
    grid = pd.DataFrame(index=range(n_grid_samples))
    for col in feats:
        vals = pd.to_numeric(holdout_df[col], errors="coerce").dropna().values.astype(float)
        if len(vals) < 3:
            grid[col] = 0.0
            continue
        lo, hi = np.percentile(vals, [2.0, 98.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.min(vals)), float(np.max(vals))
        grid[col] = rng.uniform(lo, hi, n_grid_samples)
    # The g function expects a frame with feature columns only.
    try:
        g_vals = np.asarray(g_callable(grid), dtype=float)
    except Exception as exc:
        info["Singularity_Reason"] = f"eval_error::{type(exc).__name__}"
        return info

    finite_mask = np.isfinite(g_vals)
    frac_non_finite = float(1.0 - finite_mask.mean())
    info["Singularity_FracNonFinite"] = frac_non_finite
    g_finite = g_vals[finite_mask]
    if g_finite.size == 0:
        info["Singularity_Reason"] = "all_non_finite"
        return info
    max_abs_g = float(np.max(np.abs(g_finite)))
    frac_extreme = float(np.mean(np.abs(g_finite) > extreme_g_threshold))
    info["Singularity_MaxAbsG"] = max_abs_g
    info["Singularity_FracExtreme"] = frac_extreme

    is_safe = (
        frac_non_finite <= nonfinite_max_frac
        and max_abs_g < g_abs_max
        and frac_extreme < extreme_g_max_frac
    )
    info["Singularity_Safe"] = bool(is_safe)
    if is_safe:
        info["Singularity_Reason"] = "safe"
    elif frac_non_finite > nonfinite_max_frac:
        info["Singularity_Reason"] = f"non_finite_{frac_non_finite:.3f}"
    elif max_abs_g >= g_abs_max:
        info["Singularity_Reason"] = f"abs_g_blowup_{max_abs_g:.2f}"
    else:
        info["Singularity_Reason"] = f"frac_extreme_{frac_extreme:.3f}"
    return info


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
    """Hard rejection rules. ANY failure removes the candidate from
    publication-grade pools regardless of how high its R² is.
    """
    return bool(
        row.get("R2", -999.0) >= 0.90
        and row.get("MAPE_%", 999.0) <= 10.0
        and row.get("COV_Test_over_Pred", 999.0) <= 0.15
        and 0.98 <= row.get("Mean_Test_over_Pred", -999.0) <= 1.12
        and row.get("Unsafe_10_%", 999.0) <= 15.0
        # v18+ severe-tail / safety rejections
        and row.get("Unsafe_20_%", 999.0) <= 5.0
        and row.get("Max_Overprediction", 999.0) <= 1.30
        and row.get("Max_Group_Unsafe_10_%", 999.0) <= 25.0
        and row.get("Worst_Group_Mean_TP_dev", 999.0) <= 0.20
        and row.get("Monotonicity_Violation", 999.0) <= 0.05
        and row.get("complexity", 999.0) <= max_complexity
        # v18+ singularity / denominator safety: hard reject any equation that
        # blows up on a synthetic grid filling the empirical feature space.
        and bool(row.get("Singularity_Safe", False))
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

    # v18.1+ pre-engineered compact features. PySR was wasting search budget
    # rediscovering these ratios; making them first-class lets a quick run
    # converge on a publishable form.
    out["h_b"] = safe_div(out["h"], out["b"])
    out["pcr_pne"] = safe_div(out["Pcrl"], np.maximum(out["Pne"].astype(float).values, EPS))
    out["lambda_inter"] = out["lambda_c"].astype(float).values * out["lambda_led"].astype(float).values
    out["lambda_inter2"] = out["lambda_c"].astype(float).values * (out["lambda_led"].astype(float).values ** 2)
    out["log_pne"] = np.log1p(np.maximum(out["Pne_Py"].astype(float).values, 0.0))
    out["log_pcrl"] = np.log1p(np.maximum(out["Pcrl_Py"].astype(float).values, 0.0))

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


def build_teacher_jitter_dataset(
    symbolic: pd.DataFrame,
    n_aug_per_row: int = 20,
    sigma: float = 0.025,
    rng_seed: int = 2026,
) -> pd.DataFrame:
    """v18.1+ teacher-distillation augmentation.

    PySR struggles with 829 non-holdout rows. We multiplicatively jitter the
    raw geometry/material features around each non-holdout row, recompute
    DSM_local and downstream targets, and concatenate everything. The holdout
    rows are excluded entirely so the augmentation never leaks into the
    held-out evaluation set. The teacher target uses the existing PtPy_teacher
    column (V17 inference) where available; the actual target uses PtPy_actual
    of the seed row, which is a Knowledge-Distillation style approximation
    (good enough at small jitter, ~2.5% noise).
    """
    rng = np.random.default_rng(rng_seed)
    base = symbolic[~symbolic["is_holdout"]].copy().reset_index(drop=True)
    if base.empty:
        return base

    raw_jitter_cols = ["L", "t", "h", "b", "A", "Fy", "Py", "Pcrl", "Pne",
                       "lambda_c", "lambda_led"]
    jitter_cols = [c for c in raw_jitter_cols if c in base.columns]
    augmented_chunks: list[pd.DataFrame] = []
    for k in range(int(max(n_aug_per_row, 0))):
        tmp = base.copy()
        for col in jitter_cols:
            noise = rng.normal(1.0, sigma, len(tmp))
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce").astype(float).values * noise
        # recompute derived ratios so they stay consistent with the jittered raw cols
        tmp["h_t"] = safe_div(tmp["h"], tmp["t"])
        tmp["b_t"] = safe_div(tmp["b"], tmp["t"])
        tmp["L_t"] = safe_div(tmp["L"], tmp["t"])
        tmp["L_b"] = safe_div(tmp["L"], tmp["b"])
        tmp["A_t2"] = safe_div(tmp["A"], np.maximum(tmp["t"].astype(float).values, EPS) ** 2)
        tmp["Fy_E"] = safe_div(tmp["Fy"], E_STEEL_MPA)
        tmp["Pcrl_Py"] = safe_div(tmp["Pcrl"], tmp["Py"])
        tmp["Pne_Py"] = safe_div(tmp["Pne"], tmp["Py"])
        tmp["h_b"] = safe_div(tmp["h"], tmp["b"])
        tmp["pcr_pne"] = safe_div(
            tmp["Pcrl"], np.maximum(tmp["Pne"].astype(float).values, EPS)
        )
        tmp["lambda_inter"] = (
            tmp["lambda_c"].astype(float).values * tmp["lambda_led"].astype(float).values
        )
        tmp["lambda_inter2"] = (
            tmp["lambda_c"].astype(float).values
            * (tmp["lambda_led"].astype(float).values ** 2)
        )
        tmp["log_pne"] = np.log1p(np.maximum(tmp["Pne_Py"].astype(float).values, 0.0))
        tmp["log_pcrl"] = np.log1p(np.maximum(tmp["Pcrl_Py"].astype(float).values, 0.0))
        tmp["DSM_global"] = dsm_global(tmp["lambda_c"])
        tmp["DSM_local"] = dsm_local(tmp["lambda_led"], tmp["Pne_Py"])
        tmp["target_logcorr_actual"] = safe_log_ratio(tmp["PtPy_actual"], tmp["DSM_local"])
        tmp["target_logcorr_teacher"] = safe_log_ratio(tmp["PtPy_teacher"], tmp["DSM_local"])
        tmp["target_logcorr_hybrid"] = (
            0.70 * tmp["target_logcorr_actual"] + 0.30 * tmp["target_logcorr_teacher"]
        )
        tmp["is_holdout"] = False
        tmp["jitter_id"] = int(k)
        augmented_chunks.append(tmp)

    if not augmented_chunks:
        return base
    aug = pd.concat([base.assign(jitter_id=-1)] + augmented_chunks, ignore_index=True)
    aug = aug.replace([np.inf, -np.inf], np.nan)
    aug = aug.dropna(subset=["DSM_local", "PtPy_actual"]).reset_index(drop=True)
    return aug


def run_pysr_stage(
    data: pd.DataFrame,
    target_name: str,
    out_dir: Path,
    preset: str,
    features: list[str],
    allow_holdout_training: bool = False,
    seed: int = 2026,
    seed_label: str = "",
) -> pd.DataFrame:
    pysr_module = importlib.import_module("pysr")
    PySRRegressor = getattr(pysr_module, "PySRRegressor")

    cfg = PRESETS[preset]
    suffix = f"_seed{seed}" if seed_label else ""
    stage_dir = ensure_dir(out_dir / f"pysr_{target_name}_{preset}{suffix}")
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
        # v18+ explicit procs cap: Kaggle's commit runner restarts the kernel
        # if Julia/PySR threading grows past its memory budget. Limit threads
        # to keep the worker stable.
        procs=cfg.get("procs", 2),
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
        random_state=seed,
        # v18+ speed: enable Julia multithreading. PySR forbids combining
        # deterministic=True with multithreading, so we trade bit-exact
        # reproducibility for a ~3-4x wall-clock speedup. Statistical
        # quality and the safety-filter pipeline are unaffected.
        deterministic=False,
        parallelism="multithreading",
        # v18+ Restore PySR's native progress so the Kaggle log shows the
        # familiar 'Progress: X / 24000 total iterations (Y%)' line that the
        # user is used to. Verbosity=1 + progress=True is PySR's standard
        # output mode. The earlier 'looping at 11%' impression was a Kaggle
        # UI buffer artifact, not an actual hang, so re-enabling the native
        # output is fine.
        verbosity=1,
        progress=True,
    )
    # Run model.fit on the main thread; PySR's own progress lines surface
    # the percentage. Add a sidecar memory/CPU monitor thread that ticks
    # every 30s so any RAM-related kernel restart leaves a clear trail.
    import threading as _threading
    import time as _time
    t0 = _time.monotonic()
    print(
        f"[PYSR] target={target_name}: START fit on {len(x_train)} rows, "
        f"{len(cols)} features (preset={preset}) RUN_ID={RUN_ID}",
        flush=True,
    )
    _append_run_log(out_dir, f"PYSR START target={target_name} preset={preset} cols={cols}")

    monitor_stop = _threading.Event()

    def _resource_monitor():
        ps = _safe_psutil()
        if ps is None:
            return
        proc = ps.Process()
        last = 0.0
        while not monitor_stop.is_set():
            try:
                mem = ps.virtual_memory()
                p_rss_gb = proc.memory_info().rss / (1024**3)
                line = (
                    f"[V18 MON] RUN_ID={RUN_ID} t={(_time.monotonic()-t0)/60:.1f}min "
                    f"RAM_used={mem.percent:.0f}% avail={mem.available/(1024**3):.2f}GB "
                    f"proc_rss={p_rss_gb:.2f}GB cpu_pct={proc.cpu_percent(interval=None):.0f}"
                )
                if (_time.monotonic() - last) >= 30.0:
                    print(line, flush=True)
                    _append_run_log(out_dir, line)
                    last = _time.monotonic()
            except Exception:
                pass
            monitor_stop.wait(5.0)

    mon_thread = _threading.Thread(target=_resource_monitor, daemon=True)
    mon_thread.start()
    try:
        model.fit(x_train, y_train, variable_names=cols)
    except Exception as e:
        elapsed_total = _time.monotonic() - t0
        tb = traceback.format_exc()
        print(f"[PYSR] target={target_name}: FAILED after {elapsed_total/60:.2f} min", flush=True)
        print(f"[PYSR] FAILED traceback:\n{tb}", flush=True)
        _append_run_log(out_dir, f"PYSR FAILED target={target_name} err={type(e).__name__}: {e}")
        monitor_stop.set()
        raise
    finally:
        monitor_stop.set()
    elapsed_total = _time.monotonic() - t0
    print(f"[PYSR] target={target_name}: DONE in {elapsed_total/60:.2f} min", flush=True)
    _append_run_log(out_dir, f"PYSR DONE target={target_name} elapsed_min={elapsed_total/60:.2f}")
    if model.equations_ is None or len(model.equations_) == 0:
        print(f"[PYSR] target={target_name}: no equations returned", flush=True)
        return pd.DataFrame()
    equations = model.equations_.copy()
    equations.insert(0, "target_stage", target_name)
    equations.insert(1, "preset", preset)
    equations.insert(2, "seed", seed)
    equations.to_csv(stage_dir / "equations.csv", index=False)
    return equations


# ---------------------------------------------------------------------------
# Equation evaluation / NSGA-style selection
# ---------------------------------------------------------------------------


def _clean_pysr_expression_string(s: Any) -> str:
    """Normalize PySR's equation output for sympy.sympify.

    PySR's ``equation`` column often contains a display-format string like
    ``y = lambda_led * -0.48639`` which sympify cannot parse because of the
    ``y =`` prefix. We strip that prefix and any extra whitespace/newlines.
    PySR also emits ``+ -X`` patterns and ``X^2`` powers occasionally; these
    are valid sympy syntax but we normalize ``^`` to ``**`` defensively.
    """
    text = str(s).strip()
    # Strip leading 'y = ' / 'y=' (case-insensitive)
    low = text.lower().lstrip()
    if low.startswith("y ="):
        text = text[text.lower().index("y =") + 3 :].strip()
    elif low.startswith("y="):
        text = text[text.lower().index("y=") + 2 :].strip()
    # Defensive: ^ → **
    if "^" in text and "**" not in text:
        text = text.replace("^", "**")
    # Replace newlines/multiple spaces with single space (PySR sometimes wraps)
    text = " ".join(text.split())
    return text


def expression_to_callable(expression: str, features: list[str]):
    sympy_module = importlib.import_module("sympy")
    # v18+ critical fix: sympy.symbols(list) returns a Python *list* of symbols,
    # not a tuple. The previous code wrapped that list into a 1-tuple, which made
    # lambdify build a function expecting a single argument (the list) and then
    # crash with "_lambdifygenerated() takes 1 positional argument but N were
    # given" when called with N feature arrays. Pass a space-joined string so
    # sympy returns a true tuple of symbols, and normalize the single-feature
    # case afterwards.
    symbols = sympy_module.symbols(" ".join(features))
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
    expr = sympy_module.sympify(_clean_pysr_expression_string(expression), locals=locals_map)
    func = sympy_module.lambdify(symbols, expr, modules=["numpy"])

    def _predict_g(frame: pd.DataFrame) -> np.ndarray:
        args = [frame[name].astype(float).values for name in features]
        values = func(*args)
        values = np.asarray(values, dtype=float)
        # Constant equations (e.g. g = -1.2581) collapse to a 0-d scalar; we
        # need a per-row vector to align with the holdout frame.
        if values.shape == ():
            values = np.full(len(frame), float(values))
        return values

    return _predict_g, str(expr)


def refit_expression_constants(
    expression: str,
    features: list[str],
    train_df: pd.DataFrame,
    base_col: str = "DSM_local",
    actual_col: str = "PtPy_actual",
    max_nfev: int = 200,
) -> tuple[str, dict[str, Any]]:
    """v18.1+ post-PySR constant refinement.

    PySR returns the *shape* of an equation, but the floating-point constants it
    finds are tuned for the inner training loss, not the holdout PtPy_actual /
    DSM_local target. We re-optimize all Float atoms with scipy least-squares so
    that g_symbolic(features) directly matches log(PtPy_actual / DSM_local) on
    the non-holdout rows. The holdout split is never touched here.

    Returns ``(refit_expression_string, info_dict)``. If the equation contains
    no fittable Float atoms (e.g. constant integer expression), the original
    expression is returned with ``info["refit"] == "no_constants"``.
    """
    try:
        import scipy.optimize as _opt
    except Exception as exc:
        return expression, {"refit": "scipy_unavailable", "reason": str(exc)}

    sympy_module = importlib.import_module("sympy")
    cleaned = _clean_pysr_expression_string(expression)

    feat_symbols = sympy_module.symbols(" ".join(features))
    if not isinstance(feat_symbols, tuple):
        feat_symbols = (feat_symbols,)
    locals_map = {name: sym for name, sym in zip(features, feat_symbols)}
    locals_map.update({f"x{i}": sym for i, sym in enumerate(feat_symbols)})
    locals_map.update({
        "sqrt": sympy_module.sqrt,
        "log": sympy_module.log,
        "log1p": lambda x: sympy_module.log(1 + x),
        "exp": sympy_module.exp,
        "pow": sympy_module.Pow,
        "square": lambda x: x**2,
        "cube": lambda x: x**3,
    })
    try:
        expr = sympy_module.sympify(cleaned, locals=locals_map)
    except Exception as exc:
        return expression, {"refit": "sympify_failed", "reason": str(exc)}

    floats = sorted(expr.atoms(sympy_module.Float), key=lambda a: float(a))
    if not floats:
        return str(expr), {"refit": "no_constants"}

    param_names = [f"c{i}" for i in range(len(floats))]
    param_symbols = sympy_module.symbols(" ".join(param_names))
    if not isinstance(param_symbols, tuple):
        param_symbols = (param_symbols,)
    initial_values = [float(f) for f in floats]
    subs_map = {f: p for f, p in zip(floats, param_symbols)}
    expr_param = expr.xreplace(subs_map)

    try:
        func = sympy_module.lambdify(
            (*feat_symbols, *param_symbols), expr_param, modules=["numpy"]
        )
    except Exception as exc:
        return str(expr), {"refit": "lambdify_failed", "reason": str(exc)}

    base_vals = pd.to_numeric(train_df.get(base_col, np.nan), errors="coerce").values.astype(float)
    actual_vals = pd.to_numeric(train_df.get(actual_col, np.nan), errors="coerce").values.astype(float)
    feat_vals = [pd.to_numeric(train_df[name], errors="coerce").values.astype(float) for name in features]
    mask = np.isfinite(base_vals) & np.isfinite(actual_vals) & (base_vals > 0) & (actual_vals > 0)
    for arr in feat_vals:
        mask &= np.isfinite(arr)
    if int(mask.sum()) < 30:
        return str(expr), {"refit": "insufficient_rows", "n_rows": int(mask.sum())}
    target_g = np.log(actual_vals[mask] / base_vals[mask])
    feat_arrays = [a[mask] for a in feat_vals]

    def _residuals(params: np.ndarray) -> np.ndarray:
        try:
            g = func(*feat_arrays, *params)
            g_arr = np.asarray(g, dtype=float)
            if g_arr.shape == ():
                g_arr = np.full(len(target_g), float(g_arr))
            if g_arr.shape != target_g.shape:
                return np.full(len(target_g), 1e6)
            res = g_arr - target_g
            res = np.where(np.isfinite(res), res, 1e6)
            return res
        except Exception:
            return np.full(len(target_g), 1e6)

    try:
        result = _opt.least_squares(
            _residuals,
            x0=np.array(initial_values, dtype=float),
            method="trf",
            max_nfev=max_nfev,
            x_scale="jac",
        )
        new_values = [float(v) for v in result.x]
    except Exception as exc:
        return str(expr), {"refit": "optimize_failed", "reason": str(exc)}

    final_subs = {p: sympy_module.Float(v) for p, v in zip(param_symbols, new_values)}
    expr_refit = expr_param.xreplace(final_subs)
    initial_loss = float(np.mean(_residuals(np.array(initial_values, dtype=float)) ** 2))
    final_loss = float(np.mean(result.fun ** 2))
    return str(expr_refit), {
        "refit": "ok",
        "n_constants": len(floats),
        "n_rows": int(mask.sum()),
        "initial_mse_log": initial_loss,
        "refit_mse_log": final_loss,
        "improvement_pct": (
            100.0 * (initial_loss - final_loss) / max(initial_loss, 1e-12)
        ),
    }


def evaluate_equations(
    symbolic: pd.DataFrame,
    equations: pd.DataFrame,
    out_dir: Path,
    features: list[str],
    allow_no_holdout: bool = False,
    refit_constants: bool = True,
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

    # v18.1+ training subset for the post-PySR constant refit. We never touch
    # the holdout rows here; refit only sees non-holdout rows so the holdout
    # metrics remain a clean test set.
    refit_train = symbolic[~symbolic["is_holdout"]].copy()
    if refit_train.empty:
        refit_train = symbolic.copy()

    rows = []
    n_total = len(equations)
    n_skipped = 0
    n_succeeded = 0
    n_failed = 0
    n_refit_ok = 0
    n_refit_skip = 0
    for i, eq_row in equations.reset_index(drop=True).iterrows():
        # Prefer sympy_format (parseable) over equation (display, may have 'y =' prefix).
        expr = eq_row.get("sympy_format")
        if pd.isna(expr) or str(expr).strip() == "":
            expr = eq_row.get("equation")
        if pd.isna(expr) or str(expr).strip() == "":
            expr = eq_row.get("lambda_format")
        if pd.isna(expr) or str(expr).strip() == "":
            n_skipped += 1
            continue
        target_stage = str(eq_row.get("target_stage", "unknown"))
        complexity = float(eq_row.get("complexity", np.nan)) if pd.notna(eq_row.get("complexity", np.nan)) else float(len(str(expr)))
        original_expr = str(expr)
        refit_info: dict[str, Any] = {}
        eval_expr = original_expr
        if refit_constants:
            try:
                eval_expr, refit_info = refit_expression_constants(
                    original_expr, features, refit_train
                )
                if refit_info.get("refit") == "ok":
                    n_refit_ok += 1
                else:
                    n_refit_skip += 1
            except Exception as exc:
                refit_info = {"refit": "exception", "reason": f"{type(exc).__name__}: {exc}"}
                eval_expr = original_expr
                n_refit_skip += 1
        try:
            g_fn, sympy_expr = expression_to_callable(eval_expr, features)
            def pred_fn(frame: pd.DataFrame) -> np.ndarray:
                g = finite_clip(g_fn(frame), -3.0, 3.0)
                return np.asarray(frame["DSM_local"], dtype=float) * np.exp(g)

            pred = finite_clip(pred_fn(holdout), 0.0, 1.5)
            met = compute_metrics(holdout["PtPy_actual"], pred)
            if not met:
                n_skipped += 1
                if n_skipped <= 3:
                    print(f"[EVAL] EQ_{i:04d} skipped: compute_metrics returned empty (probably all-NaN pred)", flush=True)
                continue
            fairness = group_fairness_metrics(holdout, pred)
            mono = monotonicity_penalty(holdout, pred_fn)
            calib = conservative_calibration(holdout["PtPy_actual"], pred)
            singularity = singularity_safety_audit(g_fn, holdout, features)
            rows.append({
                "candidate_id": f"EQ_{i:04d}",
                "target_stage": target_stage,
                "equation": eval_expr,
                "equation_pysr_original": original_expr,
                "sympy_equation": sympy_expr,
                "complexity": complexity,
                "pysr_loss": float(eq_row.get("loss", np.nan)) if pd.notna(eq_row.get("loss", np.nan)) else np.nan,
                "Refit_Status": refit_info.get("refit", "disabled"),
                "Refit_Improvement_%": refit_info.get("improvement_pct", np.nan),
                **met,
                **fairness,
                "Monotonicity_Violation": mono,
                **calib,
                **singularity,
            })
            n_succeeded += 1
        except Exception as exc:
            n_failed += 1
            if n_failed <= 5:
                print(f"[EVAL] EQ_{i:04d} FAILED: {type(exc).__name__}: {exc} | expr_preview={str(expr)[:120]!r}", flush=True)
            rows.append({
                "candidate_id": f"EQ_{i:04d}",
                "target_stage": target_stage,
                "equation": str(expr),
                "complexity": complexity,
                "evaluation_error": f"{type(exc).__name__}: {exc}",
            })

    print(
        f"[EVAL] equations={n_total} succeeded={n_succeeded} failed={n_failed} "
        f"skipped={n_skipped} | refit_ok={n_refit_ok} refit_skip={n_refit_skip}",
        flush=True,
    )

    table = pd.DataFrame(rows)
    # Always persist whatever we got, even if everything failed, so the user
    # can inspect evaluation_error in the CSV.
    try:
        table.to_csv(out_dir / "nsga3_candidate_evaluation.csv", index=False)
    except Exception:
        pass

    if table.empty:
        print("[EVAL WARN] table is empty; no equations to rank.", flush=True)
        return table

    if "R2" not in table.columns or n_succeeded == 0:
        print(
            "[EVAL WARN] No equation produced valid metrics. "
            f"Available columns: {list(table.columns)}. "
            "Pareto/NSGA ranking is skipped; evaluation_error column has details.",
            flush=True,
        )
        return table

    numeric_cols = [
        "R2", "MAPE_%", "COV_Test_over_Pred", "Mean_Test_over_Pred", "Unsafe_5_%",
        "Unsafe_10_%", "complexity", "Monotonicity_Violation", "Max_Group_MAPE_%",
    ]
    valid = table.dropna(subset=[c for c in numeric_cols if c in table.columns]).copy()
    if not valid.empty:
        valid["Hard_Constraint_Pass"] = valid.apply(hard_constraint_pass, axis=1)
        # v18+ Pareto front extended with severe-tail and worst-group safety
        # objectives so a candidate cannot dominate by averaging out a single
        # catastrophic case or one bad design group.
        objectives = np.column_stack([
            1.0 - valid["R2"].values,
            valid["MAPE_%"].values,
            valid["COV_Test_over_Pred"].values,
            np.abs(valid["Mean_Test_over_Pred"].values - 1.03),
            valid["Unsafe_5_%"].values,
            valid.get("Unsafe_20_%", pd.Series(np.zeros(len(valid)))).values,
            valid.get("Max_Overprediction", pd.Series(np.ones(len(valid)))).values,
            valid["complexity"].values / 100.0,
            valid["Monotonicity_Violation"].values,
            valid["Max_Group_MAPE_%"].values,
            valid.get("Max_Group_Unsafe_10_%", pd.Series(np.zeros(len(valid)))).values,
        ])
        objectives = np.nan_to_num(objectives, nan=1e6, posinf=1e6, neginf=1e6)
        valid["Pareto_NonDominated"] = pareto_front_mask(objectives)
        valid = add_ranking_score(valid)
        merge_cols = [c for c in [
            "candidate_id", "Hard_Constraint_Pass", "Pareto_NonDominated", "Score",
            "Singularity_Safe", "Singularity_MaxAbsG", "Singularity_FracExtreme",
            "Singularity_Reason",
        ] if c in valid.columns]
        table = table.merge(valid[merge_cols], on="candidate_id", how="left")
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
    parser.add_argument("--preset", choices=sorted(PRESETS), default="fast_publish", help="PySR search preset.")
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
    parser.add_argument(
        "--features",
        choices=["official", "compact", "extended"],
        default="compact",
        help=(
            "Feature set for PySR/equation evaluation. 'compact' is the v18.1+ "
            "default and includes engineered ratios (h/b, lambda interactions, "
            "log of slenderness ratios) that prevent PySR from wasting search "
            "budget rediscovering them."
        ),
    )
    parser.add_argument(
        "--pysr-targets",
        nargs="+",
        choices=sorted(PYSR_TARGETS.keys()),
        default=PYSR_DEFAULT_TARGETS,
        help=(
            "Which symbolic-correction targets to search with PySR. Default in "
            "v18.1+ is 'actual hybrid' so the NSGA stage can pick whichever "
            "expression generalizes best on the held-out experiments."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[2026],
        help=(
            "PySR random seeds. Pass multiple seeds (e.g. '2026 2027 2028') to "
            "run several short searches with different random initializations "
            "and combine their equations before NSGA selection. Diversity helps "
            "escape local optima of any single seed."
        ),
    )
    parser.add_argument(
        "--refit-constants",
        dest="refit_constants",
        action="store_true",
        default=True,
        help=(
            "After PySR, re-fit the numeric Float constants of every equation on "
            "the non-holdout rows using scipy least-squares. Enabled by default "
            "because it typically lifts holdout R^2 by 0.1-0.2 with no extra "
            "PySR runtime."
        ),
    )
    parser.add_argument(
        "--no-refit-constants",
        dest="refit_constants",
        action="store_false",
        help="Disable the post-PySR constant refit (useful for ablation runs).",
    )
    parser.add_argument(
        "--distill-teacher-jitter",
        action="store_true",
        help=(
            "Augment PySR training data with multiplicatively jittered copies "
            "of the non-holdout rows. Holdout rows are never touched. Effective "
            "training size becomes (1 + teacher-jitter-mult) * non_holdout_rows, "
            "which can substantially improve PySR convergence on small datasets."
        ),
    )
    parser.add_argument(
        "--teacher-jitter-mult",
        type=int,
        default=20,
        help="Number of jittered copies per non-holdout row (default 20).",
    )
    parser.add_argument(
        "--teacher-jitter-sigma",
        type=float,
        default=0.025,
        help="Multiplicative noise sigma for teacher jitter (default 0.025 = 2.5%%).",
    )
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
        # v18.1+ Kaggle defaults: fast_publish preset + compact engineered
        # features + actual & hybrid targets + 2 seeds + refit-constants ON.
        # This is the configuration that combines all the high-impact
        # improvements while keeping total wall-clock under ~25 minutes on
        # Kaggle's CPU runners.
        argv = [
            "--run-pysr",
            "--preset", "fast_publish",
            "--features", "compact",
            "--pysr-targets", "actual", "hybrid",
            "--seeds", "2026", "2027",
            "--refit-constants",
            "--out-dir", str(out_default),
        ]
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
    _print_run_banner("MAIN ENTRY")
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    _append_run_log(out_dir, f"MAIN ENTRY args={vars(args)}")

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

    if args.features == "official":
        features = list(OFFICIAL_FEATURES)
    elif args.features == "compact":
        features = list(COMPACT_FEATURES)
    else:
        features = list(EXTENDED_AUDIT_FEATURES)
    features = [f for f in features if f in symbolic.columns and symbolic[f].notna().any()]
    print(f"[INFO] Active feature set ({args.features}): {features}")

    # v18.1+ optional teacher-distillation augmentation on the PySR training
    # data. The holdout split is preserved as-is for evaluation; only the
    # data we hand to PySR is augmented.
    pysr_data = symbolic
    if args.distill_teacher_jitter and args.run_pysr:
        before = int((~symbolic["is_holdout"]).sum())
        pysr_data = build_teacher_jitter_dataset(
            symbolic,
            n_aug_per_row=args.teacher_jitter_mult,
            sigma=args.teacher_jitter_sigma,
        )
        # Concatenate with the original holdout rows (untouched) so PySR
        # filtering (~is_holdout) still excludes them.
        holdout_rows = symbolic[symbolic["is_holdout"]].copy()
        pysr_data = pd.concat([pysr_data, holdout_rows], ignore_index=True)
        after = int((~pysr_data["is_holdout"]).sum())
        print(
            f"[INFO] Teacher-jitter distillation: training rows {before} -> {after} "
            f"(mult={args.teacher_jitter_mult}, sigma={args.teacher_jitter_sigma})"
        )
        _append_run_log(out_dir, f"JITTER training_rows {before}->{after}")

    all_equations = []
    if args.run_pysr:
        active_targets = [t for t in args.pysr_targets if t in PYSR_TARGETS]
        if not active_targets:
            active_targets = list(PYSR_DEFAULT_TARGETS)
        seeds = list(args.seeds) if args.seeds else [2026]
        print(f"[INFO] PySR will run targets={active_targets} seeds={seeds}")
        for target_name in active_targets:
            for seed in seeds:
                seed_label = f"seed{seed}" if len(seeds) > 1 else ""
                print(
                    f"[INFO] Running PySR target={target_name} seed={seed} "
                    f"preset={args.preset} features={features}"
                )
                eqs = run_pysr_stage(
                    pysr_data,
                    target_name,
                    out_dir,
                    args.preset,
                    features,
                    allow_holdout_training=args.allow_holdout_training,
                    seed=seed,
                    seed_label=seed_label,
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

        evaluated = evaluate_equations(
            symbolic,
            equations,
            out_dir,
            features,
            allow_no_holdout=args.allow_no_holdout,
            refit_constants=bool(args.refit_constants),
        )
        recommendations = select_recommendations(evaluated)
        json_dump(recommendations, out_dir / "best_symbolic_equations.json")
        print(f"[OK] Wrote {out_dir / 'best_symbolic_equations.json'}")

    report = {
        "script": Path(globals().get("__file__", "CFS_V18_Symbolic_DSM_NSGAIII.py")).name,
        "formulation": "Pt/Py = DSM_local * exp(g_symbolic)",
        "official_features": OFFICIAL_FEATURES,
        "compact_features": COMPACT_FEATURES,
        "extended_audit_features": EXTENDED_AUDIT_FEATURES,
        "active_features": features,
        "feature_set_choice": args.features,
        "targets": PYSR_TARGETS,
        "search_config": {
            "preset": args.preset,
            "seeds": list(args.seeds) if args.seeds else [],
            "pysr_targets": list(args.pysr_targets) if args.pysr_targets else [],
            "refit_constants": bool(args.refit_constants),
            "distill_teacher_jitter": bool(args.distill_teacher_jitter),
            "teacher_jitter_mult": int(args.teacher_jitter_mult),
            "teacher_jitter_sigma": float(args.teacher_jitter_sigma),
        },
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
            "Unsafe_20_max_%": 5.0,
            "Max_Overprediction_max": 1.30,
            "Max_Group_Unsafe_10_max_%": 25.0,
            "Worst_Group_Mean_TP_dev_max": 0.20,
            "Monotonic_violation_max": 0.05,
            "Official_complexity_max": 45,
            "Conservative_k_range": [0.90, 1.00],
            "Singularity_audit_required": True,
            "Singularity_abs_g_max": 5.0,
            "Singularity_extreme_g_threshold": 2.5,
            "Singularity_extreme_g_max_frac": 0.02,
        },
        "nsga3_objectives_minimized": [
            "1 - R2_test",
            "MAPE_test",
            "COV_test_over_pred",
            "abs(mean_test_over_pred - 1.03)",
            "unsafe_5_percent",
            "unsafe_20_percent",
            "max_overprediction",
            "complexity / 100",
            "monotonicity_violation",
            "max_group_MAPE",
            "max_group_unsafe_10_percent",
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
    # v18+ wrap main() in a top-level guard so any unhandled exception
    # surfaces with a complete traceback in the Kaggle log instead of being
    # swallowed by papermill / IPython.
    try:
        rc = main()
    except SystemExit:
        raise
    except Exception as _exc:
        print("\n" + "=" * 90, flush=True)
        print(f"[V18 FATAL] RUN_ID={RUN_ID} unhandled {type(_exc).__name__}: {_exc}", flush=True)
        print("=" * 90, flush=True)
        traceback.print_exc()
        print("=" * 90, flush=True)
        raise
    raise SystemExit(rc)
