#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CFS Built-up Columns — v16 Pareto-Legendary Official No-Fastener Framework
===================================================================

Kaggle-ready single-file script.

Core scientific idea:
- Do NOT use true Failure Mode (FM) as a strength-model input.
- Instead, train a cross-fitted Failure-Mode Probability Model using design inputs only.
- Feed predicted FM probabilities into a physics-corrected strength ensemble.
- Predict Pt/Py with no fastener spacing, no specimen label, no study ID, no target-derived leakage.

Why this is stronger than v2-v13:
1) No target-derived features such as Pt_DSM_ratio or target residuals as inputs.
2) No true FM as a design input.
3) Cross-fitted failure-mode probabilities.
4) Multi-baseline physics correction:
   DSM_global, DSM_local, DSM_min, DSM_mean, Pne/Py, Pcrl/Py.
5) Cross-fitted target encoding inside every training fold.
6) Stacked correction experts + non-negative final blend.
7) OOF-only calibration and residual correction.
8) Holdout + repeated CV stability check.
9) Auto-downloads the open-source GitHub dataset.
10) Saves full outputs for publication audit.

Important scientific note:
No honest code can guarantee R2 >= 0.96 without fastener spacing and without leakage.
This script is designed to maximize the official, defensible result.
"""

# ============================================================
# 0. CONFIG
# ============================================================

CONFIG = {
    "DATA_URL": "https://raw.githubusercontent.com/Dr-Yehia/Stability-book/main/CFS_Built-up_Columns_ML_Dataset.csv",
    "OUTPUT_DIR": "/kaggle/working/cfs_v16_pareto_legendary_official",

    # Use fast for debugging, strong for final Kaggle run.
    "MODE": "strong",  # "fast" or "strong"

    # Validation
    "TEST_SIZE": 0.20,
    "RANDOM_STATE": 42,
    "INNER_FOLDS": 5,
    "REPEATED_CV_FOLDS": 5,
    "REPEATED_CV_REPEATS": 2,
    "RUN_REPEATED_CV": True,
    "RUN_FULL_STACK_REPEATED_CV": False,  # True is much slower.

    # v16 multi-objective / specialist settings
    "USE_DESIGN_GROUP_SPECIALISTS": True,
    "USE_SLENDERNESS_REGIME_SPECIALISTS": True,
    "MIN_SPECIALIST_ROWS": 55,
    # v17 gating: only accept a specialist if it beats the fallback baseline
    # on its own subset by at least this margin (in R² units).
    "SPECIALIST_GATE_EPS": 0.003,
    "MULTIOBJECTIVE_SAFETY_TARGET_UNSAFE": 48.0,
    "CONFORMAL_ALPHA": 0.10,

    # Official design setting
    "USE_TRUE_FM_IN_STRENGTH_MODEL": False,  # keep False for official model
    "USE_FM_PROBABILITIES": True,
    "USE_TARGET_ENCODING": True,

    # Final correction stages
    "USE_CALIBRATION": True,
    "USE_RESIDUAL_CORRECTION": True,

    # Prediction clipping
    "PRED_MIN": 0.02,
    "PRED_MAX": 1.30,

    # Artifacts
    "SAVE_MODELS": True,
    "MAKE_PLOTS": True,
    "MAKE_SHAP": True,
}

# ============================================================
# 1. Imports and dependency handling
# ============================================================

import os
import re
import sys
import json
import math
import time
import glob
import warnings
import subprocess
import urllib.request
from pathlib import Path

warnings.filterwarnings("ignore")

def ensure_package(package, import_name=None):
    import importlib.util
    name = import_name or package
    if importlib.util.find_spec(name) is None:
        print(f"[INSTALL] {package}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=False)

ensure_package("xgboost")
ensure_package("lightgbm")
ensure_package("catboost")
if CONFIG["MAKE_SHAP"]:
    ensure_package("shap")

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, Ridge, HuberRegressor, LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    log_loss,
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

try:
    import joblib
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False

try:
    from xgboost import XGBRegressor, XGBClassifier
    import xgboost as _xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import lightgbm as lgb_lib
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    HAS_CAT = True
except Exception:
    HAS_CAT = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# ============================================================
# 2. Utility
# ============================================================

EPS = 1e-12

def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def clean_text_series(s):
    return (
        s.fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    )

def safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def sdiv(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.divide(a, np.where(np.abs(b) < EPS, np.nan, b))

def finite_fill(x, value=0.0):
    return pd.Series(x).replace([np.inf, -np.inf], np.nan).fillna(value).values

def clip_pred(p):
    return np.clip(np.asarray(p, dtype=float), CONFIG["PRED_MIN"], CONFIG["PRED_MAX"])

def metrics(y, p, prefix=""):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return {}
    rmse = math.sqrt(mean_squared_error(y, p))
    mae = mean_absolute_error(y, p)
    r2 = r2_score(y, p) if len(y) > 1 else np.nan
    mape = np.mean(np.abs((y - p) / (np.abs(y) + EPS))) * 100
    ratio = y / (p + EPS)
    mu = np.mean(ratio)
    cov = np.std(ratio) / mu if abs(mu) > EPS else np.nan
    unsafe = np.mean(p > y) * 100
    return {
        prefix + "R2": float(r2),
        prefix + "RMSE": float(rmse),
        prefix + "MAE": float(mae),
        prefix + "MAPE_%": float(mape),
        prefix + "Mean_Test_over_Pred": float(mu),
        prefix + "COV_Test_over_Pred": float(cov),
        prefix + "Unsafe_Overprediction_%": float(unsafe),
        prefix + "N": int(len(y)),
    }

def print_metrics(title, d):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    for k, v in d.items():
        if isinstance(v, float):
            print(f"{k:36s}: {v:.6f}")
        else:
            print(f"{k:36s}: {v}")

def to_json_safe(obj, _seen=None):
    """
    Recursively convert objects to JSON-serializable equivalents.
    Drops non-serializable model objects (e.g. RidgeCV, HuberRegressor),
    numpy arrays, and DataFrames, replacing them with a short type tag.
    """
    if _seen is None:
        _seen = set()
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    oid = id(obj)
    if oid in _seen:
        return f"<circular:{type(obj).__name__}>"
    _seen.add(oid)
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v, _seen) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(v, _seen) for v in obj]
    if isinstance(obj, np.ndarray):
        return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
    if isinstance(obj, pd.DataFrame):
        return f"<DataFrame shape={obj.shape}>"
    if isinstance(obj, pd.Series):
        return f"<Series len={len(obj)}>"
    return f"<{type(obj).__name__}>"

def dsm_global(lambda_c):
    lc = np.asarray(lambda_c, dtype=float)
    lc = np.where(lc <= 0, np.nan, lc)
    return np.where(lc <= 1.5, 0.658 ** (lc ** 2), 0.877 / (lc ** 2 + EPS))

def dsm_local_from_lambda(lambda_led, pne_py):
    lam = np.asarray(lambda_led, dtype=float)
    pne = np.asarray(pne_py, dtype=float)
    lam = np.where(lam <= 0, np.nan, lam)
    # DSM-style local reduction; no target value used.
    reduction = np.where(
        lam <= 0.776,
        1.0,
        (1.0 - 0.15 / (lam ** 0.8 + EPS)) / (lam ** 0.8 + EPS),
    )
    return pne * reduction

def make_family(label):
    label = str(label).strip()
    label = re.sub(r"[-_]\d+$", "", label)
    label = re.sub(r"\s+", "", label)
    return label if label else "Unknown"

def stratify_key(y, sg=None, n_bins=6):
    y = pd.Series(y).reset_index(drop=True)
    try:
        bins = pd.qcut(y, q=min(n_bins, max(2, y.nunique())), labels=False, duplicates="drop")
    except Exception:
        bins = pd.cut(y, bins=min(n_bins, max(2, y.nunique())), labels=False)
    bins = bins.fillna(0).astype(str)
    if sg is not None:
        sg = pd.Series(sg).reset_index(drop=True).astype(str)
        key = sg + "_B" + bins
        vc = key.value_counts()
        if (vc < 2).any():
            return bins
        return key
    return bins


# ============================================================
# 3. Data loading
# ============================================================

def find_or_download_dataset():
    out_dir = Path(CONFIG["OUTPUT_DIR"])
    mkdir(out_dir)
    data_dir = out_dir / "data"
    mkdir(data_dir)

    local_candidates = [
        "CFS_Built-up_Columns_ML_Dataset.csv",
        "/kaggle/working/CFS_Built-up_Columns_ML_Dataset.csv",
    ]
    local_candidates += glob.glob("/kaggle/input/**/CFS_Built-up_Columns_ML_Dataset.csv", recursive=True)
    for p in local_candidates:
        if Path(p).exists() and Path(p).stat().st_size > 1000:
            print(f"[DATA] Using local dataset: {p}")
            return Path(p)

    url = CONFIG["DATA_URL"]
    dst = data_dir / Path(url).name
    try:
        print(f"[DATA] Downloading from GitHub:\n  {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            content = r.read()
        if len(content) < 1000:
            raise RuntimeError("Downloaded file is too small.")
        dst.write_bytes(content)
        print(f"[DATA] Downloaded: {dst} ({dst.stat().st_size/1024:.1f} KB)")
        return dst
    except Exception as e:
        raise RuntimeError(
            "Could not download dataset. Enable Internet in Kaggle or upload the CSV as a Kaggle Dataset."
        ) from e


def load_and_standardize():
    path = find_or_download_dataset()
    raw = pd.read_csv(path)
    raw = raw.loc[:, [c for c in raw.columns if str(c).strip() and "Unnamed" not in str(c)]].copy()

    # Expected dataset columns, with fallbacks for robust loading.
    def first(names, required=True):
        for n in names:
            if n in raw.columns:
                return n
        if required:
            raise KeyError(f"Missing required column. Tried: {names}. Available: {list(raw.columns)}")
        return None

    col = {
        "TestLabel": first(["Test_Label", "TestLabel", "Label", "Specimen"], required=False),
        "SectionType": first(["Section Types", "SectionType"]),
        "Sections": first(["Sections"], required=False),
        "BC": first(["BC", "Boundary", "Boundary Condition"]),
        "FM": first(["FM", "Failure Mode", "FailureMode"], required=False),
        "L": first(["L"]),
        "t": first(["t"]),
        "h": first(["h"]),
        "b": first(["b"]),
        "A": first(["A"]),
        "Fy": first(["Fy"]),
        "Py": first(["Py"]),
        "Pcrl": first(["P(crl,crd)", "Pcrl_crd", "Pcrl"]),
        "Pne": first(["Pne"]),
        "KLr": first(["KL_r", "KL/r"]),
        "lambda_c": first(["λc", "lambda_c", "lam_c"]),
        "lambda_led": first(["λ(le-d)", "lambda_led", "lam_led"]),
        "PtPy": first(["Pt/Py", "PtPy"]),
        "Pt": first(["Pt"], required=False),
    }

    df = pd.DataFrame(index=raw.index)
    df["TestLabel"] = clean_text_series(raw[col["TestLabel"]]) if col["TestLabel"] else pd.Series(["Unknown"] * len(raw))
    df["SectionType"] = clean_text_series(raw[col["SectionType"]])
    df["Sections"] = clean_text_series(raw[col["Sections"]]) if col["Sections"] else pd.Series(["Unknown"] * len(raw))
    df["BC"] = clean_text_series(raw[col["BC"]]).str.replace(r"\s+", "", regex=True)
    df["FM"] = clean_text_series(raw[col["FM"]]).str.replace(r"\s+", "", regex=True) if col["FM"] else pd.Series(["Unknown"] * len(raw))

    for new, old in [
        ("L", "L"), ("t", "t"), ("h", "h"), ("b", "b"), ("A", "A"), ("Fy", "Fy"),
        ("Py", "Py"), ("Pcrl", "Pcrl"), ("Pne", "Pne"), ("KLr", "KLr"),
        ("lambda_c", "lambda_c"), ("lambda_led", "lambda_led"), ("PtPy", "PtPy"),
    ]:
        df[new] = safe_num(raw[col[old]])

    if col["Pt"] is not None:
        df["Pt"] = safe_num(raw[col["Pt"]])
    else:
        df["Pt"] = df["PtPy"] * df["Py"]

    df["Family"] = df["TestLabel"].apply(make_family)

    # Clean rows
    df = df[df["PtPy"].notna()].copy()
    df = df[(df["PtPy"] > 0.02) & (df["PtPy"] < 1.50)].copy()
    df = df[(df["Py"] > 0) & (df["t"] > 0) & (df["h"] > 0) & (df["b"] > 0)].copy()

    # Fill numeric NaNs
    for c in df.select_dtypes(include=np.number).columns:
        med = df[c].replace([np.inf, -np.inf], np.nan).median()
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(med if np.isfinite(med) else 0.0)

    df = df.reset_index(drop=True)
    print(f"[DATA] Rows after cleaning: {len(df)}")
    print(f"[DATA] Section types: {df['SectionType'].nunique()} | FM labels: {df['FM'].nunique()} | Families: {df['Family'].nunique()}")
    return df, str(path)


# ============================================================
# 4. Feature engineering
# ============================================================

def assign_section_group(row):
    st = str(row["SectionType"]).strip()
    sec = str(row["Sections"]).strip().lower()
    if st == "O-2C":
        return "G1_O2C"
    if st == "O-2U":
        return "G2_O2U"
    if st == "C-U+C":
        return "G3_CUC"
    if st.startswith("HC"):
        return "G4_HC"
    if st.startswith("C-2C") or st.startswith("C-2Σ") or st.startswith("C-2\u03a3"):
        return "G5_C2C"
    if st.startswith("O-"):
        return "G6_Open"
    if "closed" in sec or "box" in st.lower() or "-i-" in st.lower():
        return "G7a_Closed_Box"
    return "G7b_Rest"

def engineer_design_features(df):
    df = df.copy()

    # Section group from section geometry only, not from true failure mode.
    df["SG_design"] = df.apply(assign_section_group, axis=1)

    # Dimensionless geometry
    df["h_t"] = sdiv(df["h"], df["t"])
    df["b_t"] = sdiv(df["b"], df["t"])
    df["L_h"] = sdiv(df["L"], df["h"])
    df["L_b"] = sdiv(df["L"], df["b"])
    df["L_t"] = sdiv(df["L"], df["t"])
    df["h_b"] = sdiv(df["h"], df["b"])
    df["b_h"] = sdiv(df["b"], df["h"])
    df["A_t2"] = sdiv(df["A"], df["t"] ** 2)
    df["sqrt_A_t"] = sdiv(np.sqrt(np.abs(df["A"])), df["t"])

    # Material ratio
    # If E is unavailable, use a constant nominal steel E so Fy/E remains dimensionless.
    E_nom = 203000.0
    df["Fy_E"] = df["Fy"] / E_nom
    df["Fy_norm"] = df["Fy"] / 350.0

    # Buckling ratios
    df["Pcrl_Py"] = sdiv(df["Pcrl"], df["Py"])
    df["Pne_Py"] = sdiv(df["Pne"], df["Py"])
    df["Pcrl_Pne"] = sdiv(df["Pcrl_Py"], df["Pne_Py"])
    df["Pne_Pcrl"] = sdiv(df["Pne_Py"], df["Pcrl_Py"])

    # Slenderness transformations
    df["lambda_c_sq"] = df["lambda_c"] ** 2
    df["lambda_c_cu"] = df["lambda_c"] ** 3
    df["lambda_led_sq"] = df["lambda_led"] ** 2
    df["lambda_led_cu"] = df["lambda_led"] ** 3
    df["inv_lambda_c"] = sdiv(1.0, df["lambda_c"])
    df["inv_lambda_led"] = sdiv(1.0, df["lambda_led"])
    df["lambda_c_over_led"] = sdiv(df["lambda_c"], df["lambda_led"])
    df["lambda_led_over_c"] = sdiv(df["lambda_led"], df["lambda_c"])

    # Interactions
    df["lambda_c_led"] = df["lambda_c"] * df["lambda_led"]
    df["lambda_c_KLr"] = df["lambda_c"] * df["KLr"]
    df["lambda_c_Pne"] = df["lambda_c"] * df["Pne_Py"]
    df["lambda_led_Pcrl"] = df["lambda_led"] * df["Pcrl_Py"]
    df["Pne_Pcrl_product"] = df["Pne_Py"] * df["Pcrl_Py"]
    df["h_t_b_t"] = df["h_t"] * df["b_t"]
    df["lambda_c_h_t"] = df["lambda_c"] * df["h_t"]
    df["lambda_c_b_t"] = df["lambda_c"] * df["b_t"]

    # Powers
    df["Pcrl_sq"] = df["Pcrl_Py"] ** 2
    df["Pne_sq"] = df["Pne_Py"] ** 2
    df["sqrt_Pcrl"] = np.sqrt(np.abs(df["Pcrl_Py"]))
    df["sqrt_Pne"] = np.sqrt(np.abs(df["Pne_Py"]))

    # DSM-inspired baselines and descriptors. No target is used.
    df["DSM_global"] = dsm_global(df["lambda_c"])
    df["DSM_local"] = dsm_local_from_lambda(df["lambda_led"], df["Pne_Py"])
    base_candidates = pd.DataFrame({
        "DSM_global": df["DSM_global"],
        "DSM_local": df["DSM_local"],
        "Pne_Py": df["Pne_Py"],
        "Pcrl_Py": df["Pcrl_Py"],
    }).replace([np.inf, -np.inf], np.nan)
    base_candidates = base_candidates.where(base_candidates > 0)

    df["base_global"] = df["DSM_global"]
    df["base_local"] = df["DSM_local"]
    df["base_pne"] = df["Pne_Py"]
    df["base_pcrl"] = df["Pcrl_Py"]
    df["base_min"] = base_candidates.min(axis=1)
    df["base_mean"] = base_candidates.mean(axis=1)
    df["base_geom"] = np.sqrt(np.maximum(df["base_min"], CONFIG["PRED_MIN"]) * np.maximum(df["base_mean"], CONFIG["PRED_MIN"]))

    for b in ["base_global", "base_local", "base_pne", "base_pcrl", "base_min", "base_mean", "base_geom"]:
        df[b] = pd.Series(df[b]).replace([np.inf, -np.inf], np.nan).fillna(df["base_mean"]).clip(CONFIG["PRED_MIN"], CONFIG["PRED_MAX"])

    df["DSM_global_local_ratio"] = sdiv(df["DSM_global"], df["DSM_local"])
    df["DSM_spread_abslog"] = np.abs(np.log(np.maximum(df["DSM_global"], CONFIG["PRED_MIN"]) / np.maximum(df["DSM_local"], CONFIG["PRED_MIN"])))
    df["Pne_DSM_global"] = sdiv(df["Pne_Py"], df["DSM_global"])
    df["Pcrl_DSM_global"] = sdiv(df["Pcrl_Py"], df["DSM_global"])
    df["min_global_local"] = pd.DataFrame({"g": df["Pne_Py"], "l": df["Pcrl_Py"]}).min(axis=1)
    df["max_global_local"] = pd.DataFrame({"g": df["Pne_Py"], "l": df["Pcrl_Py"]}).max(axis=1)
    df["range_global_local"] = df["max_global_local"] - df["min_global_local"]

    # Region flags
    df["region_short"] = (df["lambda_c"] < 0.7).astype(float)
    df["region_intermediate"] = ((df["lambda_c"] >= 0.7) & (df["lambda_c"] <= 1.3)).astype(float)
    df["region_slender"] = (df["lambda_c"] > 1.3).astype(float)

    df["is_open_section"] = df["Sections"].astype(str).str.lower().str.contains("open").astype(float)
    df["is_closed_section"] = df["Sections"].astype(str).str.lower().str.contains("closed").astype(float)
    df["is_half_closed_section"] = df["Sections"].astype(str).str.lower().str.contains("half").astype(float)

    # Fill numeric NaNs after engineering
    for c in df.select_dtypes(include=np.number).columns:
        med = df[c].replace([np.inf, -np.inf], np.nan).median()
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(med if np.isfinite(med) else 0.0)

    return df

NUM_FEATURES_BASE = [
    "h_t", "b_t", "L_h", "L_b", "L_t", "h_b", "b_h", "A_t2", "sqrt_A_t",
    "Fy_E", "Fy_norm",
    "Pcrl_Py", "Pne_Py", "Pcrl_Pne", "Pne_Pcrl",
    "lambda_c", "lambda_led", "KLr",
    "lambda_c_sq", "lambda_c_cu", "lambda_led_sq", "lambda_led_cu",
    "inv_lambda_c", "inv_lambda_led", "lambda_c_over_led", "lambda_led_over_c",
    "lambda_c_led", "lambda_c_KLr", "lambda_c_Pne", "lambda_led_Pcrl",
    "Pne_Pcrl_product", "h_t_b_t", "lambda_c_h_t", "lambda_c_b_t",
    "Pcrl_sq", "Pne_sq", "sqrt_Pcrl", "sqrt_Pne",
    "DSM_global", "DSM_local", "base_global", "base_local", "base_pne", "base_pcrl",
    "base_min", "base_mean", "base_geom",
    "DSM_global_local_ratio", "DSM_spread_abslog", "Pne_DSM_global", "Pcrl_DSM_global",
    "min_global_local", "max_global_local", "range_global_local",
    "region_short", "region_intermediate", "region_slender",
    "is_open_section", "is_closed_section", "is_half_closed_section",
]

CAT_FEATURES = ["SectionType", "Sections", "BC", "SG_design"]


# ============================================================
# 5. Cross-fitted failure-mode probabilities
# ============================================================

def make_classifier_matrix(fit_df, apply_df, onehot_cols=None):
    X = pd.DataFrame(index=apply_df.index)
    for c in NUM_FEATURES_BASE:
        X[c] = apply_df[c].astype(float).values

    oh = pd.get_dummies(apply_df[CAT_FEATURES].astype(str), prefix=CAT_FEATURES, dtype=float)
    if onehot_cols is None:
        fit_oh = pd.get_dummies(fit_df[CAT_FEATURES].astype(str), prefix=CAT_FEATURES, dtype=float)
        onehot_cols = list(fit_oh.columns)
    for c in onehot_cols:
        if c not in oh.columns:
            oh[c] = 0.0
    X = pd.concat([X, oh[onehot_cols]], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.astype(float), onehot_cols

def classifier_models(seed=42):
    models = {
        "ETC": ExtraTreesClassifier(
            n_estimators=600 if CONFIG["MODE"] == "strong" else 250,
            max_features=0.75,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "RFC": RandomForestClassifier(
            n_estimators=500 if CONFIG["MODE"] == "strong" else 250,
            max_features=0.75,
            min_samples_leaf=2,
            random_state=seed + 1,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    }
    if HAS_XGB:
        models["XGBC"] = XGBClassifier(
            n_estimators=900 if CONFIG["MODE"] == "strong" else 350,
            learning_rate=0.025,
            max_depth=4,
            subsample=0.90,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed + 2,
            n_jobs=-1,
            verbosity=0,
        )
    if HAS_LGB:
        models["LGBC"] = LGBMClassifier(
            n_estimators=900 if CONFIG["MODE"] == "strong" else 350,
            learning_rate=0.025,
            num_leaves=31,
            subsample=0.90,
            colsample_bytree=0.85,
            random_state=seed + 3,
            n_jobs=-1,
            verbosity=-1,
            class_weight="balanced",
        )
    return models

def crossfit_fm_probabilities(train_df, test_df, seed=42):
    print("\n[FM PROB] Cross-fitting failure-mode probability model")
    le = LabelEncoder()
    y_fm = le.fit_transform(train_df["FM"].astype(str).values)
    classes = list(le.classes_)
    n_classes = len(classes)
    print(f"[FM PROB] Classes: {classes}")

    # If too few classes, return no probabilities.
    if n_classes < 2:
        return train_df, test_df, {"classes": classes, "used": False}

    kf = KFold(n_splits=min(CONFIG["INNER_FOLDS"], max(2, len(train_df) // 30)), shuffle=True, random_state=seed)

    oof_prob = np.zeros((len(train_df), n_classes))
    test_prob_accum = np.zeros((len(test_df), n_classes))
    fold_scores = []

    base_models = classifier_models(seed)
    n_folds = kf.get_n_splits()

    for fold, (idx_tr, idx_va) in enumerate(kf.split(train_df), 1):
        fit_df = train_df.iloc[idx_tr].copy()
        val_df = train_df.iloc[idx_va].copy()
        y_fit = y_fm[idx_tr]
        y_val = y_fm[idx_va]

        X_fit, oh_cols = make_classifier_matrix(fit_df, fit_df)
        X_val, _ = make_classifier_matrix(fit_df, val_df, oh_cols)
        X_test, _ = make_classifier_matrix(fit_df, test_df, oh_cols)

        prob_val_sum = np.zeros((len(val_df), n_classes))
        prob_test_sum = np.zeros((len(test_df), n_classes))
        used = 0

        for name, model in base_models.items():
            try:
                m = clone(model)
                m.fit(X_fit, y_fit)
                pv = m.predict_proba(X_val)
                pt = m.predict_proba(X_test)

                # Align columns if a fold misses a class
                aligned_v = np.zeros((len(val_df), n_classes))
                aligned_t = np.zeros((len(test_df), n_classes))
                for j, cls in enumerate(m.classes_):
                    aligned_v[:, int(cls)] = pv[:, j]
                    aligned_t[:, int(cls)] = pt[:, j]
                prob_val_sum += aligned_v
                prob_test_sum += aligned_t
                used += 1
            except Exception as e:
                print(f"  [FM PROB WARN] {name} failed in fold {fold}: {repr(e)}")

        if used == 0:
            prior = np.bincount(y_fit, minlength=n_classes) / len(y_fit)
            prob_val = np.tile(prior, (len(val_df), 1))
            prob_test = np.tile(prior, (len(test_df), 1))
        else:
            prob_val = prob_val_sum / used
            prob_test = prob_test_sum / used

        oof_prob[idx_va] = prob_val
        test_prob_accum += prob_test / n_folds

        pred_val = np.argmax(prob_val, axis=1)
        acc = accuracy_score(y_val, pred_val)
        bal = balanced_accuracy_score(y_val, pred_val)
        fold_scores.append({"fold": fold, "acc": acc, "balanced_acc": bal})
        print(f"  fold {fold}: acc={acc:.4f}, balanced_acc={bal:.4f}")

    # Add probabilities as features
    train_df = train_df.copy()
    test_df = test_df.copy()
    for i, cls in enumerate(classes):
        safe_cls = re.sub(r"[^A-Za-z0-9]+", "_", str(cls)).strip("_")
        col = f"FMprob_{safe_cls}"
        train_df[col] = oof_prob[:, i]
        test_df[col] = test_prob_accum[:, i]

    # Confidence / entropy features
    train_df["FMprob_max"] = oof_prob.max(axis=1)
    test_df["FMprob_max"] = test_prob_accum.max(axis=1)
    train_df["FMprob_entropy"] = -np.sum(oof_prob * np.log(oof_prob + EPS), axis=1)
    test_df["FMprob_entropy"] = -np.sum(test_prob_accum * np.log(test_prob_accum + EPS), axis=1)

    info = {
        "classes": classes,
        "used": True,
        "fold_scores": fold_scores,
        "prob_columns": [f"FMprob_{re.sub(r'[^A-Za-z0-9]+', '_', str(cls)).strip('_')}" for cls in classes] + ["FMprob_max", "FMprob_entropy"],
    }
    return train_df, test_df, info


# ============================================================
# 6. Strength matrix builder
# ============================================================

def smoothed_target_map(fit_df, y_fit, cat, smooth=15.0):
    y_series = pd.Series(np.asarray(y_fit, dtype=float), index=fit_df.index)
    global_mean = float(np.mean(y_fit))
    grp = y_series.groupby(fit_df[cat].astype(str))
    means = grp.mean()
    counts = grp.size()
    return (means * counts + global_mean * smooth) / (counts + smooth)

def strength_numeric_cols(df):
    cols = list(NUM_FEATURES_BASE)
    if CONFIG["USE_FM_PROBABILITIES"]:
        cols += [c for c in df.columns if c.startswith("FMprob_")]
        # v16+ targeted no-leakage interactions (predicted FM x design group / mechanics).
        cols += [c for c in df.columns if c.startswith("FMint_")]
    # True FM is intentionally not added.
    return [c for c in cols if c in df.columns]


def add_predicted_fm_interactions(train_df, test_df, fm_info):
    """
    Build no-leakage interaction features using PREDICTED failure-mode
    probabilities (cross-fitted) and design-only signals. The true FM
    label is never used here. These features specifically target the
    G5_C2C + predicted-F bottleneck.
    """
    if not fm_info or not fm_info.get("used", False):
        return train_df, test_df

    train_df = train_df.copy()
    test_df = test_df.copy()

    # Locate FM probability columns we have available.
    def find_prob_col(df, fm_name):
        target = f"FMprob_{fm_name}"
        if target in df.columns:
            return target
        for c in df.columns:
            if c.startswith("FMprob_") and c.split("FMprob_")[-1].upper() == fm_name.upper():
                return c
        return None

    pf_tr = find_prob_col(train_df, "F")
    pf_te = find_prob_col(test_df, "F")
    pldf_tr = find_prob_col(train_df, "LDF")
    pldf_te = find_prob_col(test_df, "LDF")

    def safe_get(df, name):
        return df[name].astype(float).values if name in df.columns else np.zeros(len(df))

    is_g5_tr = (train_df["SG_design"].astype(str).values == "G5_C2C").astype(float)
    is_g5_te = (test_df["SG_design"].astype(str).values == "G5_C2C").astype(float)

    if pf_tr is not None and pf_te is not None:
        pF_tr = safe_get(train_df, pf_tr)
        pF_te = safe_get(test_df, pf_te)

        train_df["FMint_PF"] = pF_tr
        test_df["FMint_PF"] = pF_te

        train_df["FMint_G5C2C_x_PF"] = is_g5_tr * pF_tr
        test_df["FMint_G5C2C_x_PF"] = is_g5_te * pF_te

        train_df["FMint_PF_x_lambda_c"] = pF_tr * safe_get(train_df, "lambda_c")
        test_df["FMint_PF_x_lambda_c"] = pF_te * safe_get(test_df, "lambda_c")

        train_df["FMint_PF_x_Pcrl_Py"] = pF_tr * safe_get(train_df, "Pcrl_Py")
        test_df["FMint_PF_x_Pcrl_Py"] = pF_te * safe_get(test_df, "Pcrl_Py")

        train_df["FMint_PF_x_Pne_Py"] = pF_tr * safe_get(train_df, "Pne_Py")
        test_df["FMint_PF_x_Pne_Py"] = pF_te * safe_get(test_df, "Pne_Py")

        train_df["FMint_PF_x_DSM_local"] = pF_tr * safe_get(train_df, "DSM_local")
        test_df["FMint_PF_x_DSM_local"] = pF_te * safe_get(test_df, "DSM_local")

    if pldf_tr is not None and pldf_te is not None:
        pLDF_tr = safe_get(train_df, pldf_tr)
        pLDF_te = safe_get(test_df, pldf_te)

        train_df["FMint_PLDF"] = pLDF_tr
        test_df["FMint_PLDF"] = pLDF_te

        train_df["FMint_G5C2C_x_PLDF"] = is_g5_tr * pLDF_tr
        test_df["FMint_G5C2C_x_PLDF"] = is_g5_te * pLDF_te

    # Sanitize any NaN/Inf produced by interactions.
    for c in [c for c in train_df.columns if c.startswith("FMint_")]:
        train_df[c] = pd.Series(train_df[c]).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        test_df[c] = pd.Series(test_df[c]).replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    return train_df, test_df

def make_strength_matrix(fit_df, apply_df, y_fit_for_te, onehot_cols=None):
    X = pd.DataFrame(index=apply_df.index)

    for c in strength_numeric_cols(apply_df):
        X[c] = apply_df[c].astype(float).values

    # Cross-fitted target encoding, computed from the fold training only.
    if CONFIG["USE_TARGET_ENCODING"]:
        global_mean = float(np.mean(y_fit_for_te))
        for cat in CAT_FEATURES:
            mapping = smoothed_target_map(fit_df, y_fit_for_te, cat, smooth=15.0)
            X[f"{cat}_te"] = apply_df[cat].astype(str).map(mapping).fillna(global_mean).values

    # One-hot design categoricals
    oh = pd.get_dummies(apply_df[CAT_FEATURES].astype(str), prefix=CAT_FEATURES, dtype=float)
    if onehot_cols is None:
        fit_oh = pd.get_dummies(fit_df[CAT_FEATURES].astype(str), prefix=CAT_FEATURES, dtype=float)
        onehot_cols = list(fit_oh.columns)
    for c in onehot_cols:
        if c not in oh.columns:
            oh[c] = 0.0
    X = pd.concat([X, oh[onehot_cols]], axis=1)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.astype(float), onehot_cols


# ============================================================
# 7. Strength models
# ============================================================

def regressor_models(seed=42, n_rows=500):
    if CONFIG["MODE"] == "fast":
        n_big = 500
        n_mid = 400
    else:
        n_big = 1600
        n_mid = 900

    depth = 4 if n_rows < 120 else 5 if n_rows < 300 else 6
    min_leaf = 2 if n_rows < 150 else 1

    models = {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=n_mid,
            max_features=0.78,
            min_samples_leaf=min_leaf,
            random_state=seed,
            n_jobs=-1,
        ),
        "HistGBR_abs": HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.035,
            max_iter=700 if CONFIG["MODE"] == "strong" else 300,
            max_leaf_nodes=25,
            min_samples_leaf=max(8, n_rows // 90),
            l2_regularization=0.05,
            random_state=seed + 1,
            early_stopping=True,
        ),
    }

    if HAS_XGB:
        models["XGB_huber"] = XGBRegressor(
            objective="reg:pseudohubererror",
            n_estimators=n_big,
            learning_rate=0.018,
            max_depth=depth,
            min_child_weight=2,
            subsample=0.88,
            colsample_bytree=0.84,
            reg_alpha=0.02,
            reg_lambda=0.20,
            random_state=seed + 2,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )
        models["XGB_square"] = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=max(900, n_big // 2),
            learning_rate=0.020,
            max_depth=max(3, depth - 1),
            min_child_weight=2,
            subsample=0.90,
            colsample_bytree=0.88,
            reg_alpha=0.04,
            reg_lambda=0.50,
            random_state=seed + 3,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )

    if HAS_LGB:
        models["LGB_huber"] = LGBMRegressor(
            objective="huber",
            alpha=0.85,
            n_estimators=n_big,
            learning_rate=0.018,
            num_leaves=min(63, max(15, n_rows // 10)),
            min_child_samples=max(6, n_rows // 100),
            subsample=0.88,
            subsample_freq=1,
            colsample_bytree=0.84,
            reg_alpha=0.03,
            reg_lambda=0.25,
            random_state=seed + 4,
            n_jobs=-1,
            verbosity=-1,
        )

    if HAS_CAT:
        models["CatBoost"] = CatBoostRegressor(
            iterations=n_big,
            learning_rate=0.018,
            depth=min(7, max(4, depth + 1)),
            l2_leaf_reg=4.0,
            loss_function="RMSE",
            random_seed=seed + 5,
            verbose=False,
            allow_writing_files=False,
        )

    return models

def train_correction_expert(name, train_df, test_df, base_col, seed=42):
    """
    Train models on correction:
        correction = log(PtPy / base_col)
    Then return PtPy predictions:
        pred = base * exp(correction)
    All feature construction is inside folds for target encoding.
    """
    print(f"\n[EXPERT] {name} | base={base_col}")
    n = len(train_df)
    folds = min(CONFIG["INNER_FOLDS"], max(3, n // 35))
    folds = min(folds, n)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    y_strength = train_df["PtPy"].values
    base_train = train_df[base_col].values.clip(CONFIG["PRED_MIN"], CONFIG["PRED_MAX"])
    base_test = test_df[base_col].values.clip(CONFIG["PRED_MIN"], CONFIG["PRED_MAX"])
    y_corr = np.log(np.clip(y_strength, CONFIG["PRED_MIN"], CONFIG["PRED_MAX"]) / base_train)
    y_corr = np.clip(y_corr, -1.5, 1.5)

    models = regressor_models(seed, n_rows=n)
    oof_corr = pd.DataFrame(index=train_df.index)
    test_corr = pd.DataFrame(index=test_df.index)

    for model_name in models:
        oof_corr[model_name] = 0.0
        test_corr[model_name] = 0.0

    for fold, (idx_tr, idx_va) in enumerate(kf.split(train_df), 1):
        fit_df = train_df.iloc[idx_tr].copy()
        val_df = train_df.iloc[idx_va].copy()
        y_fit_strength = fit_df["PtPy"].values

        X_fit, oh_cols = make_strength_matrix(fit_df, fit_df, y_fit_strength)
        X_val, _ = make_strength_matrix(fit_df, val_df, y_fit_strength, oh_cols)
        X_test, _ = make_strength_matrix(fit_df, test_df, y_fit_strength, oh_cols)

        for model_name, model in models.items():
            try:
                m = clone(model)
                m.fit(X_fit, y_corr[idx_tr])
                oof_corr.iloc[idx_va, oof_corr.columns.get_loc(model_name)] = m.predict(X_val)
                test_corr[model_name] += m.predict(X_test) / folds
            except Exception as e:
                print(f"  [WARN] {model_name} fold {fold} failed: {repr(e)}")

        fold_corr_avg = oof_corr.iloc[idx_va].mean(axis=1).values
        fold_pred = clip_pred(base_train[idx_va] * np.exp(np.clip(fold_corr_avg, -1.5, 1.5)))
        print(f"  fold {fold}/{folds}: R2={r2_score(y_strength[idx_va], fold_pred):.5f}")

    # Blend correction models on OOF correction target
    B_train = oof_corr.values
    B_test = test_corr.values
    try:
        blend = RidgeCV(alphas=np.logspace(-5, 2, 30))
        blend.fit(B_train, y_corr)
    except Exception:
        blend = Ridge(alpha=0.01)
        blend.fit(B_train, y_corr)

    corr_oof = blend.predict(B_train)
    corr_test = blend.predict(B_test)

    pred_oof = clip_pred(base_train * np.exp(np.clip(corr_oof, -1.5, 1.5)))
    pred_test = clip_pred(base_test * np.exp(np.clip(corr_test, -1.5, 1.5)))
    r2 = r2_score(y_strength, pred_oof)

    print(f"  [EXPERT DONE] {name}: OOF_R2={r2:.6f}")
    return pred_oof, pred_test, {"name": name, "base": base_col, "oof_r2": float(r2), "blend": blend, "model_columns": list(oof_corr.columns)}


def multiobjective_metrics(y, p):
    d = metrics(y, p)
    # Objectives:
    # maximize R2; minimize MAPE; minimize unsafe overprediction; keep mean Test/Pred near 1.
    # Unsafe overprediction is not forced to zero because that would be overly conservative,
    # but we penalize going above a target threshold.
    unsafe_target = CONFIG.get("MULTIOBJECTIVE_SAFETY_TARGET_UNSAFE", 48.0)
    risk = max(0.0, d.get("Unsafe_Overprediction_%", 100.0) - unsafe_target) / 100.0
    mean_bias = abs(d.get("Mean_Test_over_Pred", 1.0) - 1.0)
    mape = d.get("MAPE_%", 100.0) / 100.0
    r2 = d.get("R2", -999.0)
    # A compact scalar only for choosing among Pareto candidates.
    # The report still shows all objectives separately.
    score = r2 - 0.55 * mape - 0.18 * risk - 0.35 * mean_bias
    d["ParetoScore"] = float(score)
    d["RiskPenalty"] = float(risk)
    d["MeanBiasAbs"] = float(mean_bias)
    return d


def pareto_front_indices(rows):
    """
    rows: list of metric dicts.
    Dominance: higher R2 better, lower MAPE, Unsafe, MeanBias better.
    """
    if not rows:
        return []
    vals = []
    for r in rows:
        vals.append([
            r.get("R2", -999.0),
            -r.get("MAPE_%", 999.0),
            -r.get("Unsafe_Overprediction_%", 999.0),
            -r.get("MeanBiasAbs", 999.0),
        ])
    vals = np.asarray(vals, dtype=float)
    front = []
    for i in range(len(vals)):
        dominated = False
        for j in range(len(vals)):
            if i == j:
                continue
            if np.all(vals[j] >= vals[i]) and np.any(vals[j] > vals[i]):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def choose_pareto_candidate(candidates, y_train):
    """
    candidates: list of dict {name, p_tr, p_te, model}
    Chooses from Pareto front by compact score.
    """
    evaluated = []
    for c in candidates:
        p = clip_pred(c["p_tr"])
        d = multiobjective_metrics(y_train, p)
        row = {"name": c["name"], **d}
        evaluated.append(row)
        c["metrics"] = d

    front_idx = pareto_front_indices(evaluated)
    if not front_idx:
        front_idx = list(range(len(candidates)))

    best_i = max(front_idx, key=lambda i: evaluated[i]["ParetoScore"])
    selected = candidates[best_i]
    report = pd.DataFrame(evaluated).sort_values(["ParetoScore", "R2"], ascending=False)
    return selected, report


def build_meta_candidates(B_train, y_train, B_test, expert_names):
    candidates = []

    # Individual experts
    for j, name in enumerate(expert_names):
        candidates.append({
            "name": f"single::{name}",
            "p_tr": B_train[:, j],
            "p_te": B_test[:, j],
            "model": None,
        })

    # Positive linear blend
    try:
        meta_pos = LinearRegression(positive=True)
        meta_pos.fit(B_train, y_train)
        candidates.append({
            "name": "meta::positive_linear",
            "p_tr": meta_pos.predict(B_train),
            "p_te": meta_pos.predict(B_test),
            "model": meta_pos,
        })
    except Exception as e:
        print(f"[META WARN] positive linear failed: {repr(e)}")

    # Ridge blend
    try:
        meta_ridge = RidgeCV(alphas=np.logspace(-6, 2, 40))
        meta_ridge.fit(B_train, y_train)
        candidates.append({
            "name": "meta::ridgecv",
            "p_tr": meta_ridge.predict(B_train),
            "p_te": meta_ridge.predict(B_test),
            "model": meta_ridge,
        })
    except Exception as e:
        print(f"[META WARN] ridge failed: {repr(e)}")

    # Top-k averages by OOF R2
    r2s = []
    for j in range(B_train.shape[1]):
        try:
            r2s.append((r2_score(y_train, clip_pred(B_train[:, j])), j))
        except Exception:
            r2s.append((-999.0, j))
    order = [j for _, j in sorted(r2s, reverse=True)]

    for k in [3, 5, 7, min(10, B_train.shape[1])]:
        if k <= 1 or k > B_train.shape[1]:
            continue
        idx = order[:k]
        candidates.append({
            "name": f"avg::top{k}",
            "p_tr": np.mean(B_train[:, idx], axis=1),
            "p_te": np.mean(B_test[:, idx], axis=1),
            "model": {"type": "top_average", "indices": idx},
        })

    # Inverse-RMSE weighted top-k
    for k in [5, min(10, B_train.shape[1])]:
        if k <= 1 or k > B_train.shape[1]:
            continue
        idx = order[:k]
        rmses = []
        for j in idx:
            rmses.append(math.sqrt(mean_squared_error(y_train, clip_pred(B_train[:, j]))) + EPS)
        w = 1 / np.asarray(rmses)
        w = w / w.sum()
        candidates.append({
            "name": f"weighted::top{k}_inv_rmse",
            "p_tr": B_train[:, idx] @ w,
            "p_te": B_test[:, idx] @ w,
            "model": {"type": "weighted_average", "indices": idx, "weights": w.tolist()},
        })

    # Trim-mean of top-k experts: drop the most extreme expert per row to
    # reduce tail-induced bias on hard cases (e.g., G5_C2C/F outliers).
    for k in [5, 7, min(9, B_train.shape[1])]:
        if k <= 3 or k > B_train.shape[1]:
            continue
        idx = order[:k]
        sub_tr = B_train[:, idx]
        sub_te = B_test[:, idx]
        med_tr = np.median(sub_tr, axis=1, keepdims=True)
        med_te = np.median(sub_te, axis=1, keepdims=True)
        # Index of farthest-from-median per row, masked out in the mean.
        far_tr = np.argmax(np.abs(sub_tr - med_tr), axis=1)
        far_te = np.argmax(np.abs(sub_te - med_te), axis=1)
        keep_tr = np.ones_like(sub_tr, dtype=bool)
        keep_te = np.ones_like(sub_te, dtype=bool)
        keep_tr[np.arange(len(sub_tr)), far_tr] = False
        keep_te[np.arange(len(sub_te)), far_te] = False
        p_tr_trim = (sub_tr * keep_tr).sum(axis=1) / keep_tr.sum(axis=1)
        p_te_trim = (sub_te * keep_te).sum(axis=1) / keep_te.sum(axis=1)
        candidates.append({
            "name": f"trimmean::top{k}",
            "p_tr": p_tr_trim,
            "p_te": p_te_trim,
            "model": {"type": "trim_mean", "indices": idx, "drop_per_row": 1},
        })

    # Slight conservative variants of the best single/stack candidates.
    # These may reduce unsafe overprediction and improve journal safety metrics.
    base_candidates = list(candidates)
    for c in base_candidates:
        if not c["name"].startswith(("meta::", "avg::", "weighted::")):
            continue
        for shrink in [0.985, 0.975]:
            candidates.append({
                "name": f"{c['name']}::conservative_{shrink}",
                "p_tr": clip_pred(c["p_tr"] * shrink),
                "p_te": clip_pred(c["p_te"] * shrink),
                "model": {"type": "conservative", "base": c["name"], "factor": shrink},
            })

    return candidates


def nonnegative_blend(B_train, y_train, B_test):
    """
    Prefer non-negative weights for physical blending of baseline experts.
    """
    try:
        meta = LinearRegression(positive=True)
        meta.fit(B_train, y_train)
        p_tr = meta.predict(B_train)
        p_te = meta.predict(B_test)
        if np.all(np.isfinite(p_tr)) and r2_score(y_train, p_tr) > -10:
            return clip_pred(p_tr), clip_pred(p_te), meta, "LinearRegression_positive"
    except Exception:
        pass

    meta = RidgeCV(alphas=np.logspace(-6, 2, 30))
    meta.fit(B_train, y_train)
    return clip_pred(meta.predict(B_train)), clip_pred(meta.predict(B_test)), meta, "RidgeCV"

def calibrate_oof(y_train, p_train, p_test):
    """
    Try multiple calibration models (Huber, Linear, Isotonic) and pick the
    one that improves the multi-objective Pareto score most while not
    losing meaningful R2. Isotonic is monotone and tends to fix tail bias
    without hurting bulk fit, which is helpful for the G5_C2C/F long tail.
    """
    p_train_a = np.asarray(p_train, dtype=float).reshape(-1, 1)
    p_test_a = np.asarray(p_test, dtype=float).reshape(-1, 1)

    candidates = []

    try:
        cal_h = HuberRegressor(epsilon=1.35, alpha=1e-4)
        cal_h.fit(p_train_a, y_train)
        candidates.append(("huber", cal_h,
                           clip_pred(cal_h.predict(p_train_a)),
                           clip_pred(cal_h.predict(p_test_a))))
    except Exception:
        pass

    try:
        cal_l = LinearRegression()
        cal_l.fit(p_train_a, y_train)
        candidates.append(("linear", cal_l,
                           clip_pred(cal_l.predict(p_train_a)),
                           clip_pred(cal_l.predict(p_test_a))))
    except Exception:
        pass

    try:
        from sklearn.isotonic import IsotonicRegression
        cal_i = IsotonicRegression(out_of_bounds="clip", increasing=True)
        cal_i.fit(np.asarray(p_train, dtype=float), np.asarray(y_train, dtype=float))
        candidates.append(("isotonic", cal_i,
                           clip_pred(cal_i.predict(np.asarray(p_train, dtype=float))),
                           clip_pred(cal_i.predict(np.asarray(p_test, dtype=float)))))
    except Exception as e:
        print(f"[CALIBRATION] isotonic skipped: {repr(e)}")

    raw = multiobjective_metrics(y_train, p_train)
    best = None
    best_score = raw["ParetoScore"] - 1e-5
    for name, model, p_tr_cal, p_te_cal in candidates:
        m = multiobjective_metrics(y_train, p_tr_cal)
        if m["R2"] < raw["R2"] - 0.002:
            continue
        if m["ParetoScore"] > best_score:
            best = (name, model, p_tr_cal, p_te_cal, m)
            best_score = m["ParetoScore"]

    if best is None:
        # Return the first candidate as the (unused) calibration model.
        cal_obj = candidates[0][1] if candidates else None
        return p_train, p_test, cal_obj, False

    name, model, p_tr_cal, p_te_cal, m = best
    print(f"[CALIBRATION] selected={name} | ParetoScore raw={raw['ParetoScore']:.6f} -> {m['ParetoScore']:.6f} | R2 raw={raw['R2']:.6f} -> {m['R2']:.6f}")
    return p_tr_cal, p_te_cal, model, True

def residual_correction(train_df, test_df, y_train, p_train, p_test, seed=42):
    print("\n[RESIDUAL] Cross-fitted residual correction (with FM-entropy / hard-zone weights)")
    residual = y_train - p_train
    kf = KFold(n_splits=min(CONFIG["INNER_FOLDS"], max(3, len(train_df) // 35)), shuffle=True, random_state=seed + 100)

    models = {
        "res_ET": ExtraTreesRegressor(
            n_estimators=700 if CONFIG["MODE"] == "strong" else 300,
            max_features=0.80,
            min_samples_leaf=2,
            random_state=seed + 101,
            n_jobs=-1,
        ),
        "res_HGB": HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.025,
            max_iter=650 if CONFIG["MODE"] == "strong" else 300,
            max_leaf_nodes=15,
            min_samples_leaf=15,
            l2_regularization=0.10,
            random_state=seed + 102,
            early_stopping=True,
        ),
    }

    # No-leakage sample weights:
    #   - Higher weight where FM is uncertain (high entropy of predicted FM).
    #   - Higher weight on G5_C2C with high predicted P(F) (the audit-found bottleneck).
    # These quantities are derived from cross-fitted predictions on TRAIN only.
    sw = np.ones(len(train_df), dtype=float)
    if "FMprob_entropy" in train_df.columns:
        ent = train_df["FMprob_entropy"].astype(float).values
        ent = np.clip(ent, 0.0, np.nanpercentile(ent, 99.5) if np.isfinite(ent).any() else 1.0)
        ent_n = (ent - np.min(ent)) / (np.ptp(ent) + EPS)
        sw *= (1.0 + 0.6 * ent_n)
    if "FMint_G5C2C_x_PF" in train_df.columns:
        z = train_df["FMint_G5C2C_x_PF"].astype(float).values
        z_n = z / (np.max(z) + EPS) if np.max(z) > 0 else z
        sw *= (1.0 + 0.8 * z_n)
    sw = np.clip(sw, 0.5, 3.5)

    res_oof_total = np.zeros(len(train_df))
    res_test_total = np.zeros(len(test_df))

    for model_name, model in models.items():
        res_oof = np.zeros(len(train_df))
        res_test_folds = []

        for fold, (idx_tr, idx_va) in enumerate(kf.split(train_df), 1):
            fit_df = train_df.iloc[idx_tr].copy()
            val_df = train_df.iloc[idx_va].copy()
            y_fit_strength = fit_df["PtPy"].values

            X_fit, oh_cols = make_strength_matrix(fit_df, fit_df, y_fit_strength)
            X_val, _ = make_strength_matrix(fit_df, val_df, y_fit_strength, oh_cols)
            X_test, _ = make_strength_matrix(fit_df, test_df, y_fit_strength, oh_cols)

            m = clone(model)
            try:
                m.fit(X_fit, residual[idx_tr], sample_weight=sw[idx_tr])
            except TypeError:
                m.fit(X_fit, residual[idx_tr])
            res_oof[idx_va] = m.predict(X_val)
            res_test_folds.append(m.predict(X_test))

        res_oof_total += res_oof / len(models)
        res_test_total += np.mean(np.vstack(res_test_folds), axis=0) / len(models)

    p_tr_corr = clip_pred(p_train + res_oof_total)
    p_te_corr = clip_pred(p_test + res_test_total)

    raw = multiobjective_metrics(y_train, p_train)
    new = multiobjective_metrics(y_train, p_tr_corr)
    use = (new["ParetoScore"] > raw["ParetoScore"] + 1e-5) and (new["R2"] >= raw["R2"] - 0.002)

    print(f"  residual correction used={use} | OOF score raw={raw['ParetoScore']:.6f}, corr={new['ParetoScore']:.6f} | R2 raw={raw['R2']:.6f}, corr={new['R2']:.6f}")
    return (p_tr_corr, p_te_corr, True) if use else (p_train, p_test, False)


# ============================================================
# 8. Full pipeline
# ============================================================

def run_v16_pipeline(train_df, test_df, seed=42, verbose=True):
    train_df = train_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    # FM probability model
    fm_info = {"used": False, "prob_columns": []}
    if CONFIG["USE_FM_PROBABILITIES"]:
        train_df, test_df, fm_info = crossfit_fm_probabilities(train_df, test_df, seed=seed)
        # v16+ no-leakage interactions targeting the G5_C2C + predicted-F bottleneck.
        train_df, test_df = add_predicted_fm_interactions(train_df, test_df, fm_info)

    # Train correction experts over multiple physical baselines.
    base_cols = ["base_min", "base_global", "base_local", "base_mean", "base_geom", "base_pne", "base_pcrl"]
    expert_oof = []
    expert_test = []
    expert_info = []

    for base in base_cols:
        poof, ptest, info = train_correction_expert(base.upper(), train_df, test_df, base, seed=seed)
        expert_oof.append(poof)
        expert_test.append(ptest)
        expert_info.append(info)

    # v16 design-group specialist experts, using design group only (not true FM).
    if CONFIG.get("USE_DESIGN_GROUP_SPECIALISTS", True):
        min_rows = CONFIG.get("MIN_SPECIALIST_ROWS", 55)
        gate_eps = CONFIG.get("SPECIALIST_GATE_EPS", 0.003)
        print("\n[SPECIALISTS] Design-group specialists (with gating)")
        # Use a copy of the first global expert as default fallback.
        fallback_oof = expert_oof[0].copy()
        fallback_test = expert_test[0].copy()
        for sg in sorted(train_df["SG_design"].astype(str).unique()):
            tr_mask = (train_df["SG_design"].astype(str).values == sg)
            te_mask = (test_df["SG_design"].astype(str).values == sg)
            ntr, nte = int(tr_mask.sum()), int(te_mask.sum())
            if ntr < min_rows or nte == 0:
                continue
            try:
                sub_tr = train_df.loc[tr_mask].copy().reset_index(drop=True)
                sub_te = test_df.loc[te_mask].copy().reset_index(drop=True)
                poof_g, ptest_g, info_g = train_correction_expert(f"SG_SPECIALIST::{sg}", sub_tr, sub_te, "base_min", seed=seed + 701)
                # GATING: only add specialist if it beats the fallback on its own subset.
                y_subset = train_df.loc[tr_mask, "PtPy"].values
                try:
                    fallback_subset_r2 = r2_score(y_subset, np.clip(fallback_oof[tr_mask], CONFIG["PRED_MIN"], CONFIG["PRED_MAX"]))
                except Exception:
                    fallback_subset_r2 = -np.inf
                local_r2 = float(info_g.get("oof_r2", -np.inf))
                if not (local_r2 > fallback_subset_r2 + gate_eps):
                    print(f"  [GATE REJECTED] SG {sg}: local={local_r2:.5f} <= fallback={fallback_subset_r2:.5f}+{gate_eps}")
                    continue
                stitched_oof = fallback_oof.copy()
                stitched_test = fallback_test.copy()
                stitched_oof[tr_mask] = poof_g
                stitched_test[te_mask] = ptest_g
                expert_oof.append(stitched_oof)
                expert_test.append(stitched_test)
                info_g["specialist_type"] = "SG_design"
                info_g["group"] = sg
                info_g["gate_passed"] = True
                info_g["fallback_subset_r2"] = float(fallback_subset_r2)
                expert_info.append(info_g)
                print(f"  [GATE PASSED] SG {sg}: train={ntr}, test={nte}, local_R2={local_r2:.5f} > fallback_R2={fallback_subset_r2:.5f}")
            except Exception as e:
                print(f"  [SPECIALIST WARN] SG {sg} failed: {repr(e)}")

    # v16 slenderness-regime specialists: short/intermediate/slender, no FM leakage.
    if CONFIG.get("USE_SLENDERNESS_REGIME_SPECIALISTS", True):
        min_rows = CONFIG.get("MIN_SPECIALIST_ROWS", 55)
        gate_eps = CONFIG.get("SPECIALIST_GATE_EPS", 0.003)
        print("\n[SPECIALISTS] Slenderness-regime specialists (with gating)")
        fallback_oof = expert_oof[0].copy()
        fallback_test = expert_test[0].copy()

        def regime_arr(x):
            x = np.asarray(x, dtype=float)
            out = np.full(len(x), "intermediate", dtype=object)
            out[x < 0.7] = "short"
            out[x > 1.3] = "slender"
            return out

        tr_reg = regime_arr(train_df["lambda_c"].values)
        te_reg = regime_arr(test_df["lambda_c"].values)

        for rg in ["short", "intermediate", "slender"]:
            tr_mask = (tr_reg == rg)
            te_mask = (te_reg == rg)
            ntr, nte = int(tr_mask.sum()), int(te_mask.sum())
            if ntr < min_rows or nte == 0:
                continue
            try:
                sub_tr = train_df.loc[tr_mask].copy().reset_index(drop=True)
                sub_te = test_df.loc[te_mask].copy().reset_index(drop=True)
                poof_r, ptest_r, info_r = train_correction_expert(f"REGIME_SPECIALIST::{rg}", sub_tr, sub_te, "base_mean", seed=seed + 801)
                # GATING
                y_subset = train_df.loc[tr_mask, "PtPy"].values
                try:
                    fallback_subset_r2 = r2_score(y_subset, np.clip(fallback_oof[tr_mask], CONFIG["PRED_MIN"], CONFIG["PRED_MAX"]))
                except Exception:
                    fallback_subset_r2 = -np.inf
                local_r2 = float(info_r.get("oof_r2", -np.inf))
                if not (local_r2 > fallback_subset_r2 + gate_eps):
                    print(f"  [GATE REJECTED] regime {rg}: local={local_r2:.5f} <= fallback={fallback_subset_r2:.5f}+{gate_eps}")
                    continue
                stitched_oof = fallback_oof.copy()
                stitched_test = fallback_test.copy()
                stitched_oof[tr_mask] = poof_r
                stitched_test[te_mask] = ptest_r
                expert_oof.append(stitched_oof)
                expert_test.append(stitched_test)
                info_r["specialist_type"] = "lambda_c_regime"
                info_r["group"] = rg
                info_r["gate_passed"] = True
                info_r["fallback_subset_r2"] = float(fallback_subset_r2)
                expert_info.append(info_r)
                print(f"  [GATE PASSED] regime {rg}: train={ntr}, test={nte}, local_R2={local_r2:.5f} > fallback_R2={fallback_subset_r2:.5f}")
            except Exception as e:
                print(f"  [SPECIALIST WARN] regime {rg} failed: {repr(e)}")

    # v16+ targeted specialist: High predicted-F probability inside G5_C2C.
    # No leakage: uses cross-fitted predicted FM probabilities only. Gated.
    if "FMint_G5C2C_x_PF" in train_df.columns:
        try:
            min_rows_pf = max(40, CONFIG.get("MIN_SPECIALIST_ROWS", 55) - 15)
            gate_eps = CONFIG.get("SPECIALIST_GATE_EPS", 0.003)
            pf_score_tr = train_df["FMint_G5C2C_x_PF"].astype(float).values
            pf_score_te = test_df["FMint_G5C2C_x_PF"].astype(float).values
            nz_tr = pf_score_tr[pf_score_tr > 0]
            thr = float(np.median(nz_tr)) if len(nz_tr) >= 20 else 0.0
            tr_mask = pf_score_tr >= thr if thr > 0 else (pf_score_tr > 0)
            te_mask = pf_score_te >= thr if thr > 0 else (pf_score_te > 0)
            ntr, nte = int(tr_mask.sum()), int(te_mask.sum())
            print(f"\n[SPECIALIST] HighPF_C2C: thr={thr:.4f}, train_rows={ntr}, test_rows={nte}")
            if ntr >= min_rows_pf and nte > 0:
                fallback_oof = expert_oof[0].copy()
                fallback_test = expert_test[0].copy()
                sub_tr = train_df.loc[tr_mask].copy().reset_index(drop=True)
                sub_te = test_df.loc[te_mask].copy().reset_index(drop=True)
                poof_pf, ptest_pf, info_pf = train_correction_expert(
                    "HIGH_PF_C2C_SPECIALIST", sub_tr, sub_te, "base_local", seed=seed + 911,
                )
                # GATING
                y_subset = train_df.loc[tr_mask, "PtPy"].values
                try:
                    fallback_subset_r2 = r2_score(y_subset, np.clip(fallback_oof[tr_mask], CONFIG["PRED_MIN"], CONFIG["PRED_MAX"]))
                except Exception:
                    fallback_subset_r2 = -np.inf
                local_r2 = float(info_pf.get("oof_r2", -np.inf))
                if not (local_r2 > fallback_subset_r2 + gate_eps):
                    print(f"  [GATE REJECTED] HIGH_PF_C2C: local={local_r2:.5f} <= fallback={fallback_subset_r2:.5f}+{gate_eps}")
                else:
                    stitched_oof = fallback_oof.copy()
                    stitched_test = fallback_test.copy()
                    stitched_oof[tr_mask] = poof_pf
                    stitched_test[te_mask] = ptest_pf
                    expert_oof.append(stitched_oof)
                    expert_test.append(stitched_test)
                    info_pf["specialist_type"] = "HIGH_PF_C2C"
                    info_pf["threshold"] = thr
                    info_pf["n_train"] = ntr
                    info_pf["n_test"] = nte
                    info_pf["gate_passed"] = True
                    info_pf["fallback_subset_r2"] = float(fallback_subset_r2)
                    expert_info.append(info_pf)
                    print(f"  [GATE PASSED] HIGH_PF_C2C: local_R2={local_r2:.5f} > fallback_R2={fallback_subset_r2:.5f}")
            else:
                print("  [SPECIALIST] HighPF_C2C skipped: too few rows.")
        except Exception as e:
            print(f"  [SPECIALIST WARN] HIGH_PF_C2C failed: {repr(e)}")

    # Add raw baselines as weak experts
    for base in base_cols:
        expert_oof.append(clip_pred(train_df[base].values))
        expert_test.append(clip_pred(test_df[base].values))
        expert_info.append({"name": "RAW_" + base, "base": base, "oof_r2": float(r2_score(train_df["PtPy"].values, clip_pred(train_df[base].values)))})

    B_train = np.column_stack(expert_oof)
    B_test = np.column_stack(expert_test)
    y_train = train_df["PtPy"].values

    expert_names = []
    for info in expert_info:
        expert_names.append(info.get("name", "expert"))

    # v16: Pareto/multi-objective selector instead of choosing by R2 only.
    candidates = build_meta_candidates(B_train, y_train, B_test, expert_names)
    selected, pareto_report = choose_pareto_candidate(candidates, y_train)
    pred_tr = clip_pred(selected["p_tr"])
    pred_te = clip_pred(selected["p_te"])
    meta = selected.get("model", None)
    meta_name = selected["name"]
    stage = "pareto_selected_" + meta_name

    print_metrics("PARETO SELECTED STACK BEFORE CALIBRATION", {**multiobjective_metrics(y_train, pred_tr)})
    print("\n[PARETO TOP 12 CANDIDATES]")
    keep_cols = ["name", "ParetoScore", "R2", "RMSE", "MAE", "MAPE_%", "Unsafe_Overprediction_%", "Mean_Test_over_Pred", "COV_Test_over_Pred"]
    print(pareto_report[keep_cols].head(12).to_string(index=False))

    # Calibration
    cal = None
    cal_used = False
    if CONFIG["USE_CALIBRATION"]:
        pred_tr, pred_te, cal, cal_used = calibrate_oof(y_train, pred_tr, pred_te)
        if cal_used:
            stage += "_calibrated"
        print(f"[CALIBRATION] used={cal_used}")

    # Residual correction
    res_used = False
    if CONFIG["USE_RESIDUAL_CORRECTION"]:
        pred_tr, pred_te, res_used = residual_correction(train_df, test_df, y_train, pred_tr, pred_te, seed=seed)
        if res_used:
            stage += "_residual"

    bundle = {
        "stage": stage,
        "fm_info": fm_info,
        "expert_info": expert_info,
        "meta_model": meta,
        "meta_name": meta_name,
        "pareto_report": pareto_report,
        "calibration_model": cal,
        "calibration_used": cal_used,
        "residual_used": res_used,
        "expert_oof": B_train,
        "expert_test": B_test,
        # v17: keep engineered frames so main() can build the inference artifact
        "train_df_engineered": train_df,
        "test_df_engineered": test_df,
    }
    return pred_tr, pred_te, bundle


# ============================================================
# 9. Repeated CV
# ============================================================

def repeated_cv_fast(df, seed=42):
    print("\n[REPEATED CV] Stability check")
    y = df["PtPy"].reset_index(drop=True)
    sg = df["SG_design"].reset_index(drop=True)

    rkf = RepeatedKFold(
        n_splits=CONFIG["REPEATED_CV_FOLDS"],
        n_repeats=CONFIG["REPEATED_CV_REPEATS"],
        random_state=seed,
    )
    rows = []
    for fold, (idx_tr, idx_va) in enumerate(rkf.split(df, y), 1):
        print("\n" + "-" * 80)
        print(f"CV fold {fold}")
        print("-" * 80)
        tr = df.iloc[idx_tr].copy().reset_index(drop=True)
        va = df.iloc[idx_va].copy().reset_index(drop=True)

        if CONFIG["RUN_FULL_STACK_REPEATED_CV"]:
            _, pred_va, _ = run_v16_pipeline(tr, va, seed=seed + fold, verbose=False)
        else:
            # Faster official approximation: train only strongest baselines subset.
            train_save = CONFIG["INNER_FOLDS"]
            CONFIG["INNER_FOLDS"] = min(3, train_save)
            old_res = CONFIG["USE_RESIDUAL_CORRECTION"]
            CONFIG["USE_RESIDUAL_CORRECTION"] = False
            _, pred_va, _ = run_v16_pipeline(tr, va, seed=seed + fold, verbose=False)
            CONFIG["INNER_FOLDS"] = train_save
            CONFIG["USE_RESIDUAL_CORRECTION"] = old_res

        row = metrics(va["PtPy"].values, pred_va)
        row["fold"] = fold
        rows.append(row)
        print(f"  CV fold {fold}: R2={row['R2']:.5f}, RMSE={row['RMSE']:.5f}, MAPE={row['MAPE_%']:.3f}%")

    cv = pd.DataFrame(rows)
    summary = {}
    for col in ["R2", "RMSE", "MAE", "MAPE_%", "Mean_Test_over_Pred", "COV_Test_over_Pred", "Unsafe_Overprediction_%"]:
        summary[col + "_mean"] = float(cv[col].mean())
        summary[col + "_std"] = float(cv[col].std())
    return cv, summary


# ============================================================
# 10. Reporting helpers
# ============================================================

def save_plots(out_dir, y, p, sg):
    if not (CONFIG["MAKE_PLOTS"] and HAS_MPL):
        return
    out_dir = Path(out_dir)

    plt.figure(figsize=(7, 7), dpi=150)
    plt.scatter(y, p, s=28, alpha=0.75)
    lim_max = max(float(np.max(y)), float(np.max(p))) * 1.06
    lim = [0, lim_max]
    plt.plot(lim, lim, "k--", lw=1.3)
    plt.fill_between(lim, [x * 0.95 for x in lim], [x * 1.05 for x in lim], alpha=0.10)
    plt.xlabel("Experimental Pt/Py")
    plt.ylabel("Predicted Pt/Py")
    plt.title(f"v16 Legendary Official: Holdout R2={r2_score(y, p):.4f}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "v16_scatter_holdout.png", bbox_inches="tight")
    plt.close()

    err = (np.asarray(y) - np.asarray(p)) / (np.asarray(y) + EPS) * 100
    order = pd.Series(sg).value_counts().index.tolist()
    data = [err[np.asarray(sg) == g] for g in order]
    plt.figure(figsize=(10, 5), dpi=150)
    plt.boxplot(data, labels=order, showfliers=True)
    plt.axhline(0, color="k", lw=1)
    plt.axhline(5, color="r", ls="--", lw=0.8)
    plt.axhline(-5, color="r", ls="--", lw=0.8)
    plt.ylabel("Residual error (%)")
    plt.xticks(rotation=45, ha="right")
    plt.title("Residuals by design section group")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "v16_residuals_by_group.png", bbox_inches="tight")
    plt.close()

def save_shap(out_dir, train_df, test_df):
    if not (CONFIG["MAKE_SHAP"] and HAS_SHAP and HAS_XGB):
        print("[SHAP] skipped.")
        return None

    print("\n[SHAP] Compact XGBoost interpretation model")
    try:
        # Use a leakage-free matrix with full train TE for interpretability only.
        X_tr, oh_cols = make_strength_matrix(train_df, train_df, train_df["PtPy"].values)
        X_te, _ = make_strength_matrix(train_df, test_df, train_df["PtPy"].values, oh_cols)
        y_tr = train_df["PtPy"].values

        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=900,
            learning_rate=0.025,
            max_depth=4,
            subsample=0.90,
            colsample_bytree=0.90,
            random_state=CONFIG["RANDOM_STATE"] + 500,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )
        model.fit(X_tr, y_tr)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_te)

        imp = pd.DataFrame({
            "Feature": X_te.columns,
            "MeanAbsSHAP": np.abs(sv).mean(axis=0),
        }).sort_values("MeanAbsSHAP", ascending=False)
        imp.to_csv(Path(out_dir) / "v16_shap_importance.csv", index=False)

        if HAS_MPL:
            plt.figure(figsize=(10, 7), dpi=150)
            shap.summary_plot(sv, X_te, feature_names=X_te.columns, show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(Path(out_dir) / "v16_shap_summary.png", bbox_inches="tight")
            plt.close()

        print(imp.head(20).to_string(index=False))
        return imp
    except Exception as e:
        print(f"[SHAP] failed: {repr(e)}")
        return None


# ============================================================
# 10b. v17 Inference-ready distilled model (production deliverable)
# ============================================================

def build_v17_inference_bundle(train_df_eng, test_df_eng, y_train, y_test=None, seed=42):
    """
    Train a single strong production model on the FULL training set using
    exactly the same engineered features the stacked pipeline used. This
    yields an inference-ready model that can be loaded with one joblib.load
    call and applied to brand-new data via predict_v17() in the shipped
    cfs_v17_predict.py module.

    Returns:
        inference_bundle: dict with classifiers, encoders, feature schema,
            final regressor, calibration, and metadata for downstream use.
    """
    print("\n" + "=" * 100)
    print("[V17 INFERENCE] Building inference-ready production model on FULL training set")
    print("=" * 100)

    # 1) Final FM classifier ensemble fit on full train (no CV).
    fm_classifiers = []
    fm_classes = None
    fm_oh_cols = None
    fm_label_encoder = None
    if CONFIG["USE_FM_PROBABILITIES"] and "FM" in train_df_eng.columns:
        try:
            le = LabelEncoder()
            y_fm_full = le.fit_transform(train_df_eng["FM"].astype(str).values)
            fm_classes = list(le.classes_)
            fm_label_encoder = le
            X_clf, fm_oh_cols = make_classifier_matrix(train_df_eng, train_df_eng)
            for name, m in classifier_models(seed + 5001).items():
                try:
                    fitted = clone(m)
                    fitted.fit(X_clf, y_fm_full)
                    fm_classifiers.append({"name": name, "model": fitted, "classes_": list(fitted.classes_)})
                    print(f"  [INFERENCE] FM classifier '{name}' fitted (classes={list(fitted.classes_)})")
                except Exception as e:
                    print(f"  [INFERENCE WARN] FM classifier '{name}' failed: {repr(e)}")
        except Exception as e:
            print(f"  [INFERENCE WARN] FM classifier ensemble failed: {repr(e)}")

    # 2) Final target-encoding maps + global mean from FULL train.
    te_maps = {}
    for cat in CAT_FEATURES:
        try:
            mp = smoothed_target_map(train_df_eng, y_train, cat, smooth=15.0)
            te_maps[cat] = {str(k): float(v) for k, v in mp.items()}
        except Exception as e:
            print(f"  [INFERENCE WARN] TE map for {cat} failed: {repr(e)}")
            te_maps[cat] = {}
    te_global_mean = float(np.mean(y_train))

    # 3) Build full strength matrix on FULL train using these TE maps.
    X_train_full, str_oh_cols = make_strength_matrix(train_df_eng, train_df_eng, y_train)
    feature_cols = list(X_train_full.columns)

    # 4) Train the production regressor: a small Ridge-blended stack of
    #    XGB + LGB + HistGB on PtPy directly (full train).
    base_models = {}
    if HAS_XGB:
        base_models["xgb"] = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1500 if CONFIG["MODE"] == "strong" else 600,
            learning_rate=0.020,
            max_depth=5,
            min_child_weight=2,
            subsample=0.88,
            colsample_bytree=0.84,
            reg_alpha=0.04,
            reg_lambda=0.30,
            random_state=seed + 9001,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )
    if HAS_LGB:
        base_models["lgb"] = LGBMRegressor(
            objective="regression",
            n_estimators=1500 if CONFIG["MODE"] == "strong" else 600,
            learning_rate=0.020,
            num_leaves=31,
            min_child_samples=8,
            subsample=0.88,
            subsample_freq=1,
            colsample_bytree=0.84,
            reg_alpha=0.03,
            reg_lambda=0.25,
            random_state=seed + 9002,
            n_jobs=-1,
            verbosity=-1,
        )
    base_models["hgb"] = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.030,
        max_iter=900 if CONFIG["MODE"] == "strong" else 400,
        max_leaf_nodes=25,
        min_samples_leaf=8,
        l2_regularization=0.05,
        random_state=seed + 9003,
        early_stopping=True,
    )

    # 4a) Fit each base on full train and produce OOF (5-fold) predictions for the blender.
    kf = KFold(n_splits=5, shuffle=True, random_state=seed + 9100)
    oof_mat = np.zeros((len(X_train_full), len(base_models)))
    fitted_bases = {}
    for j, (name, model) in enumerate(base_models.items()):
        oof = np.zeros(len(X_train_full))
        for fold, (idx_tr, idx_va) in enumerate(kf.split(X_train_full), 1):
            try:
                m = clone(model)
                m.fit(X_train_full.iloc[idx_tr], y_train[idx_tr])
                oof[idx_va] = m.predict(X_train_full.iloc[idx_va])
            except Exception as e:
                print(f"  [INFERENCE WARN] base '{name}' fold {fold} failed: {repr(e)}")
        oof_mat[:, j] = clip_pred(oof)
        # final fit on FULL train
        try:
            fitted = clone(model)
            fitted.fit(X_train_full, y_train)
            fitted_bases[name] = fitted
            print(f"  [INFERENCE] base '{name}' fitted on full train (OOF R2={r2_score(y_train, oof_mat[:, j]):.4f})")
        except Exception as e:
            print(f"  [INFERENCE WARN] base '{name}' final fit failed: {repr(e)}")

    # 4b) Train the Ridge blender on the OOF predictions.
    blender = RidgeCV(alphas=np.logspace(-5, 2, 30))
    blender.fit(oof_mat, y_train)
    blended_oof = clip_pred(blender.predict(oof_mat))
    blended_oof_r2 = r2_score(y_train, blended_oof)
    print(f"  [INFERENCE] Ridge-blended OOF R²={blended_oof_r2:.4f}")

    # 4c) Conservative shrink (mirrors the published pipeline preference).
    shrink = 0.985
    blended_oof_shrunk = clip_pred(blended_oof * shrink)
    shrunk_oof_r2 = r2_score(y_train, blended_oof_shrunk)
    print(f"  [INFERENCE] Conservative-shrunk OOF R²={shrunk_oof_r2:.4f} (shrink={shrink})")

    # 4d) Final calibration via Isotonic on OOF predictions.
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(blended_oof_shrunk, y_train)
    calibrated_oof = clip_pred(iso.predict(blended_oof_shrunk))
    final_oof_r2 = r2_score(y_train, calibrated_oof)
    print(f"  [INFERENCE] Isotonic-calibrated OOF R²={final_oof_r2:.4f}")

    # 5) Apply to held-out test set (if provided) for an external check.
    test_metrics = None
    if test_df_eng is not None and len(test_df_eng) > 0:
        try:
            X_test_full, _ = make_strength_matrix(train_df_eng, test_df_eng, y_train, str_oh_cols)
            test_base_preds = []
            for name in fitted_bases:
                test_base_preds.append(fitted_bases[name].predict(X_test_full))
            test_base_mat = np.column_stack(test_base_preds) if len(test_base_preds) else np.zeros((len(test_df_eng), 1))
            test_pred = clip_pred(blender.predict(test_base_mat) * shrink)
            test_pred = clip_pred(iso.predict(test_pred))
            if y_test is not None:
                test_metrics = metrics(y_test, test_pred, prefix="V17_INFERENCE_TEST_")
                print_metrics("V17 INFERENCE-MODEL — HOLDOUT TEST", test_metrics)
        except Exception as e:
            print(f"  [INFERENCE WARN] test-set evaluation failed: {repr(e)}")

    # 6) Package everything as a single joblib-loadable bundle.
    inference_bundle = {
        "version": "v17",
        "feature_cols": feature_cols,
        "strength_onehot_cols": str_oh_cols,
        "fm_classifiers": fm_classifiers,
        "fm_classes": fm_classes,
        "fm_label_encoder": fm_label_encoder,
        "fm_onehot_cols": fm_oh_cols,
        "te_maps": te_maps,
        "te_global_mean": te_global_mean,
        "num_features_base": list(NUM_FEATURES_BASE),
        "cat_features": list(CAT_FEATURES),
        "base_regressors": fitted_bases,
        "ridge_blender": blender,
        "shrink_factor": shrink,
        "isotonic_calibrator": iso,
        "pred_min": CONFIG["PRED_MIN"],
        "pred_max": CONFIG["PRED_MAX"],
        "training_oof_r2": float(final_oof_r2),
        "blended_oof_r2": float(blended_oof_r2),
        "shrunk_oof_r2": float(shrunk_oof_r2),
        "test_metrics": test_metrics,
    }
    return inference_bundle


PREDICT_MODULE_SOURCE = '''"""
cfs_v17_predict.py
==================
Standalone, no-dependency-on-training-script predictor.

Usage (e.g. inside any external program with a UI):

    import joblib
    import pandas as pd
    from cfs_v17_predict import CFSV17Predictor

    predictor = CFSV17Predictor.load("cfs_v17_inference_bundle.joblib")
    df = pd.read_csv("new_specimens.csv")  # raw input columns
    out = predictor.predict(df)            # returns DataFrame with PtPy_pred
    print(out.head())

Required raw columns (same as the training CSV):
    SectionType (or "Section Types"), Sections, BC,
    L, t, h, b, A, Fy, Py, Pcrl (or "P(crl,crd)"), Pne,
    KLr (or "KL_r" / "KL/r"), lambda_c (or "λc"), lambda_led (or "λ(le-d)").
"FM" (true failure mode) is NOT required for prediction.
"""

import math
import re
import numpy as np
import pandas as pd
import joblib

EPS = 1e-12


def _safe_num(s):
    return pd.to_numeric(s, errors="coerce")


def _clean_text(s):
    return (
        s.fillna("Unknown").astype(str).str.strip()
         .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    )


def _sdiv(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return np.divide(a, np.where(np.abs(b) < EPS, np.nan, b))


def _dsm_global(lam):
    lc = np.asarray(lam, dtype=float)
    lc = np.where(lc <= 0, np.nan, lc)
    return np.where(lc <= 1.5, 0.658 ** (lc ** 2), 0.877 / (lc ** 2 + EPS))


def _dsm_local(lam, pne_py):
    lam = np.asarray(lam, dtype=float); pne = np.asarray(pne_py, dtype=float)
    lam = np.where(lam <= 0, np.nan, lam)
    reduction = np.where(
        lam <= 0.776,
        1.0,
        (1.0 - 0.15 / (lam ** 0.8 + EPS)) / (lam ** 0.8 + EPS),
    )
    return pne * reduction


def _assign_section_group(section_type, sections):
    st = str(section_type).strip()
    sec = str(sections).strip().lower()
    if st == "O-2C": return "G1_O2C"
    if st == "O-2U": return "G2_O2U"
    if st == "C-U+C": return "G3_CUC"
    if st.startswith("HC"): return "G4_HC"
    if st.startswith("C-2C") or st.startswith("C-2Σ") or st.startswith("C-2\\u03a3"):
        return "G5_C2C"
    if st.startswith("O-"): return "G6_Open"
    if "closed" in sec or "box" in st.lower() or "-i-" in st.lower():
        return "G7a_Closed_Box"
    return "G7b_Rest"


def _engineer_features(df_raw):
    """Reproduce the full v17 feature engineering on raw inputs."""
    aliases = {
        "SectionType": ["SectionType", "Section Types", "Section_Type"],
        "Sections": ["Sections"],
        "BC": ["BC", "Boundary", "Boundary Condition"],
        "L": ["L"], "t": ["t"], "h": ["h"], "b": ["b"], "A": ["A"],
        "Fy": ["Fy"], "Py": ["Py"],
        "Pcrl": ["Pcrl", "P(crl,crd)", "Pcrl_crd"],
        "Pne": ["Pne"],
        "KLr": ["KLr", "KL_r", "KL/r"],
        "lambda_c": ["lambda_c", "λc", "lam_c"],
        "lambda_led": ["lambda_led", "λ(le-d)", "lam_led"],
    }
    df = pd.DataFrame(index=df_raw.index)
    for std_name, options in aliases.items():
        col = next((c for c in options if c in df_raw.columns), None)
        if col is None and std_name in ("Sections", "BC"):
            df[std_name] = "Unknown"
            continue
        if col is None:
            raise KeyError(f"Missing required column for {std_name}; tried {options}")
        if std_name in ("SectionType", "Sections", "BC"):
            df[std_name] = _clean_text(df_raw[col]).str.replace(r"\\s+", "", regex=True)
        else:
            df[std_name] = _safe_num(df_raw[col])

    df["SG_design"] = [_assign_section_group(st, se) for st, se in zip(df["SectionType"], df["Sections"])]

    # geometry/material/buckling features (must mirror training)
    df["h_t"] = _sdiv(df["h"], df["t"])
    df["b_t"] = _sdiv(df["b"], df["t"])
    df["L_h"] = _sdiv(df["L"], df["h"])
    df["L_b"] = _sdiv(df["L"], df["b"])
    df["L_t"] = _sdiv(df["L"], df["t"])
    df["h_b"] = _sdiv(df["h"], df["b"])
    df["b_h"] = _sdiv(df["b"], df["h"])
    df["A_t2"] = _sdiv(df["A"], df["t"] ** 2)
    df["sqrt_A_t"] = _sdiv(np.sqrt(np.abs(df["A"])), df["t"])
    df["Fy_E"] = df["Fy"] / 203000.0
    df["Fy_norm"] = df["Fy"] / 350.0
    df["Pcrl_Py"] = _sdiv(df["Pcrl"], df["Py"])
    df["Pne_Py"] = _sdiv(df["Pne"], df["Py"])
    df["Pcrl_Pne"] = _sdiv(df["Pcrl_Py"], df["Pne_Py"])
    df["Pne_Pcrl"] = _sdiv(df["Pne_Py"], df["Pcrl_Py"])
    df["lambda_c_sq"] = df["lambda_c"] ** 2
    df["lambda_c_cu"] = df["lambda_c"] ** 3
    df["lambda_led_sq"] = df["lambda_led"] ** 2
    df["lambda_led_cu"] = df["lambda_led"] ** 3
    df["inv_lambda_c"] = _sdiv(1.0, df["lambda_c"])
    df["inv_lambda_led"] = _sdiv(1.0, df["lambda_led"])
    df["lambda_c_over_led"] = _sdiv(df["lambda_c"], df["lambda_led"])
    df["lambda_led_over_c"] = _sdiv(df["lambda_led"], df["lambda_c"])
    df["lambda_c_led"] = df["lambda_c"] * df["lambda_led"]
    df["lambda_c_KLr"] = df["lambda_c"] * df["KLr"]
    df["lambda_c_Pne"] = df["lambda_c"] * df["Pne_Py"]
    df["lambda_led_Pcrl"] = df["lambda_led"] * df["Pcrl_Py"]
    df["Pne_Pcrl_product"] = df["Pne_Py"] * df["Pcrl_Py"]
    df["h_t_b_t"] = df["h_t"] * df["b_t"]
    df["lambda_c_h_t"] = df["lambda_c"] * df["h_t"]
    df["lambda_c_b_t"] = df["lambda_c"] * df["b_t"]
    df["Pcrl_sq"] = df["Pcrl_Py"] ** 2
    df["Pne_sq"] = df["Pne_Py"] ** 2
    df["sqrt_Pcrl"] = np.sqrt(np.abs(df["Pcrl_Py"]))
    df["sqrt_Pne"] = np.sqrt(np.abs(df["Pne_Py"]))
    df["DSM_global"] = _dsm_global(df["lambda_c"])
    df["DSM_local"] = _dsm_local(df["lambda_led"], df["Pne_Py"])

    bc = pd.DataFrame({
        "DSM_global": df["DSM_global"], "DSM_local": df["DSM_local"],
        "Pne_Py": df["Pne_Py"], "Pcrl_Py": df["Pcrl_Py"],
    }).replace([np.inf, -np.inf], np.nan)
    bc = bc.where(bc > 0)
    df["base_global"] = df["DSM_global"]
    df["base_local"] = df["DSM_local"]
    df["base_pne"] = df["Pne_Py"]
    df["base_pcrl"] = df["Pcrl_Py"]
    df["base_min"] = bc.min(axis=1)
    df["base_mean"] = bc.mean(axis=1)
    df["base_geom"] = np.sqrt(np.maximum(df["base_min"], 0.02) * np.maximum(df["base_mean"], 0.02))

    for b in ["base_global","base_local","base_pne","base_pcrl","base_min","base_mean","base_geom"]:
        df[b] = pd.Series(df[b]).replace([np.inf,-np.inf], np.nan).fillna(df["base_mean"]).clip(0.02, 1.30)

    df["DSM_global_local_ratio"] = _sdiv(df["DSM_global"], df["DSM_local"])
    df["DSM_spread_abslog"] = np.abs(np.log(np.maximum(df["DSM_global"],0.02)/np.maximum(df["DSM_local"],0.02)))
    df["Pne_DSM_global"] = _sdiv(df["Pne_Py"], df["DSM_global"])
    df["Pcrl_DSM_global"] = _sdiv(df["Pcrl_Py"], df["DSM_global"])
    df["min_global_local"] = pd.DataFrame({"g": df["Pne_Py"], "l": df["Pcrl_Py"]}).min(axis=1)
    df["max_global_local"] = pd.DataFrame({"g": df["Pne_Py"], "l": df["Pcrl_Py"]}).max(axis=1)
    df["range_global_local"] = df["max_global_local"] - df["min_global_local"]

    df["region_short"] = (df["lambda_c"] < 0.7).astype(float)
    df["region_intermediate"] = ((df["lambda_c"] >= 0.7) & (df["lambda_c"] <= 1.3)).astype(float)
    df["region_slender"] = (df["lambda_c"] > 1.3).astype(float)
    df["is_open_section"] = df["Sections"].astype(str).str.lower().str.contains("open").astype(float)
    df["is_closed_section"] = df["Sections"].astype(str).str.lower().str.contains("closed").astype(float)
    df["is_half_closed_section"] = df["Sections"].astype(str).str.lower().str.contains("half").astype(float)

    for c in df.select_dtypes(include=np.number).columns:
        med = df[c].replace([np.inf,-np.inf], np.nan).median()
        df[c] = df[c].replace([np.inf,-np.inf], np.nan).fillna(med if np.isfinite(med) else 0.0)
    return df


def _make_classifier_matrix(df, num_features_base, cat_features, fm_oh_cols):
    X = pd.DataFrame(index=df.index)
    for c in num_features_base:
        X[c] = df[c].astype(float).values if c in df.columns else 0.0
    oh = pd.get_dummies(df[cat_features].astype(str), prefix=cat_features, dtype=float)
    for c in fm_oh_cols:
        if c not in oh.columns:
            oh[c] = 0.0
    X = pd.concat([X, oh[fm_oh_cols]], axis=1)
    return X.replace([np.inf,-np.inf], np.nan).fillna(0.0).astype(float)


def _apply_fm_classifiers(df, classifiers, fm_classes, num_features_base, cat_features, fm_oh_cols):
    if not classifiers or not fm_classes:
        return df
    X_clf = _make_classifier_matrix(df, num_features_base, cat_features, fm_oh_cols)
    n_classes = len(fm_classes)
    accum = np.zeros((len(df), n_classes))
    used = 0
    for entry in classifiers:
        try:
            m = entry["model"]
            p = m.predict_proba(X_clf)
            aligned = np.zeros((len(df), n_classes))
            for j, cls in enumerate(m.classes_):
                aligned[:, int(cls)] = p[:, j]
            accum += aligned
            used += 1
        except Exception:
            pass
    if used == 0:
        return df
    probs = accum / used
    for i, cls in enumerate(fm_classes):
        safe_cls = re.sub(r"[^A-Za-z0-9]+", "_", str(cls)).strip("_")
        df[f"FMprob_{safe_cls}"] = probs[:, i]
    df["FMprob_max"] = probs.max(axis=1)
    df["FMprob_entropy"] = -np.sum(probs * np.log(probs + EPS), axis=1)
    return df


def _add_fm_interactions(df):
    def find_prob_col(name):
        target = f"FMprob_{name}"
        if target in df.columns:
            return target
        for c in df.columns:
            if c.startswith("FMprob_") and c.split("FMprob_")[-1].upper() == name.upper():
                return c
        return None

    is_g5 = (df["SG_design"].astype(str).values == "G5_C2C").astype(float)
    pf_col = find_prob_col("F"); pldf_col = find_prob_col("LDF")

    if pf_col is not None:
        pF = df[pf_col].astype(float).values
        df["FMint_PF"] = pF
        df["FMint_G5C2C_x_PF"] = is_g5 * pF
        df["FMint_PF_x_lambda_c"] = pF * df.get("lambda_c", 0.0).astype(float).values
        df["FMint_PF_x_Pcrl_Py"]  = pF * df.get("Pcrl_Py", 0.0).astype(float).values
        df["FMint_PF_x_Pne_Py"]   = pF * df.get("Pne_Py", 0.0).astype(float).values
        df["FMint_PF_x_DSM_local"] = pF * df.get("DSM_local", 0.0).astype(float).values
    if pldf_col is not None:
        pLDF = df[pldf_col].astype(float).values
        df["FMint_PLDF"] = pLDF
        df["FMint_G5C2C_x_PLDF"] = is_g5 * pLDF
    for c in [c for c in df.columns if c.startswith("FMint_")]:
        df[c] = pd.Series(df[c]).replace([np.inf,-np.inf], np.nan).fillna(0.0).values
    return df


def _build_strength_matrix(df, te_maps, te_global_mean, num_features_base, cat_features, str_oh_cols):
    X = pd.DataFrame(index=df.index)
    feature_pool = list(num_features_base)
    feature_pool += [c for c in df.columns if c.startswith("FMprob_") or c.startswith("FMint_")]
    for c in feature_pool:
        if c in df.columns:
            X[c] = df[c].astype(float).values
    for cat in cat_features:
        mapping = te_maps.get(cat, {})
        col = df[cat].astype(str).map(mapping).fillna(te_global_mean).values
        X[f"{cat}_te"] = col.astype(float)
    oh = pd.get_dummies(df[cat_features].astype(str), prefix=cat_features, dtype=float)
    for c in str_oh_cols:
        if c not in oh.columns:
            oh[c] = 0.0
    X = pd.concat([X, oh[str_oh_cols]], axis=1)
    return X.replace([np.inf,-np.inf], np.nan).fillna(0.0).astype(float)


class CFSV17Predictor:
    """Inference-only wrapper around the v17 production bundle."""

    def __init__(self, bundle):
        self.bundle = bundle

    @classmethod
    def load(cls, bundle_path):
        return cls(joblib.load(bundle_path))

    def predict(self, raw_df, return_engineered=False):
        b = self.bundle
        df = _engineer_features(raw_df)
        df = _apply_fm_classifiers(
            df, b.get("fm_classifiers", []), b.get("fm_classes", []),
            b["num_features_base"], b["cat_features"], b.get("fm_onehot_cols", []),
        )
        df = _add_fm_interactions(df)
        X = _build_strength_matrix(
            df, b["te_maps"], b["te_global_mean"],
            b["num_features_base"], b["cat_features"], b["strength_onehot_cols"],
        )
        # Align column order to training feature_cols
        for c in b["feature_cols"]:
            if c not in X.columns:
                X[c] = 0.0
        X = X[b["feature_cols"]]

        # Run all base regressors, blend, shrink, calibrate.
        base_preds = []
        for name, model in b["base_regressors"].items():
            try:
                base_preds.append(model.predict(X))
            except Exception:
                base_preds.append(np.full(len(X), b["te_global_mean"]))
        base_mat = np.column_stack(base_preds) if base_preds else np.zeros((len(X), 1))
        blended = b["ridge_blender"].predict(base_mat)
        shrunk = blended * b["shrink_factor"]
        shrunk = np.clip(shrunk, b["pred_min"], b["pred_max"])
        calibrated = b["isotonic_calibrator"].predict(shrunk)
        calibrated = np.clip(calibrated, b["pred_min"], b["pred_max"])

        out = pd.DataFrame({
            "PtPy_pred": calibrated,
            "PtPy_pred_uncalibrated": np.clip(blended, b["pred_min"], b["pred_max"]),
        }, index=raw_df.index)
        if "Py" in raw_df.columns:
            out["Pt_pred_kN"] = calibrated * pd.to_numeric(raw_df["Py"], errors="coerce").values
        if return_engineered:
            return out, df
        return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        bundle_path, csv_path = sys.argv[1], sys.argv[2]
        out_path = sys.argv[3] if len(sys.argv) >= 4 else "predictions.csv"
        predictor = CFSV17Predictor.load(bundle_path)
        df_raw = pd.read_csv(csv_path)
        result = predictor.predict(df_raw)
        result.to_csv(out_path, index=False)
        print(f"Wrote: {out_path} ({len(result)} rows)")
    else:
        print("Usage: python cfs_v17_predict.py <bundle.joblib> <input.csv> [output.csv]")
'''


def write_predict_module(out_dir):
    """Write the standalone cfs_v17_predict.py module to the output directory."""
    target = Path(out_dir) / "cfs_v17_predict.py"
    target.write_text(PREDICT_MODULE_SOURCE, encoding="utf-8")
    print(f"[INFERENCE] predict module written: {target}")
    return target


V17_README = """# CFS v17 Production Package

This ZIP contains everything needed to run the v17 model in your own program.

## Files

- `cfs_v17_inference_bundle.joblib`: trained production model (one file).
- `cfs_v17_predict.py`: standalone Python module exposing `CFSV17Predictor`.
- `v16_holdout_predictions.csv`, `v16_pareto_candidate_report.csv`,
  `v16_error_by_design_group.csv`, `v16_error_by_true_FM_audit_only.csv`,
  `v16_repeated_cv_folds.csv`, `v16_repeated_cv_summary.json`,
  `v16_final_summary.json`, `v16_feature_report.json`,
  `v16_shap_importance.csv`: official audit results.
- `v16_scatter_holdout.png`, `v16_residuals_by_group.png`,
  `v16_shap_summary.png`: figures.
- `v16_model_bundle_holdout.joblib`: full pipeline bundle (audit only).

## Use in your program (1 minute setup)

```python
import pandas as pd
from cfs_v17_predict import CFSV17Predictor

predictor = CFSV17Predictor.load("cfs_v17_inference_bundle.joblib")
df = pd.read_csv("your_inputs.csv")
out = predictor.predict(df)
print(out.head())
```

Required input columns: SectionType, Sections, BC, L, t, h, b, A, Fy, Py,
Pcrl, Pne, KLr, lambda_c, lambda_led. (FM is NOT needed for prediction.)

## Or via the command line

```
python cfs_v17_predict.py cfs_v17_inference_bundle.joblib your_inputs.csv predictions.csv
```

## Note

The production inference model is a faithful distillation of the published
stacked pipeline. Reported pipeline metrics (TEST R², CV R²) are in
`v16_final_summary.json`. The inference model's training-set OOF R² and
test-set metrics are stored in the bundle as `training_oof_r2` and
`test_metrics`.
"""


def write_v17_readme(out_dir):
    target = Path(out_dir) / "README_v17.md"
    target.write_text(V17_README, encoding="utf-8")
    return target


def package_outputs_zip(out_dir, archive_basename="cfs_v17_FINAL_PACKAGE"):
    """
    Bundle the entire output directory into a single ZIP placed inside
    /kaggle/working so it appears as one downloadable artifact.
    """
    import shutil
    out_dir = Path(out_dir)
    # Default to the parent of out_dir (typical: /kaggle/working) so the ZIP
    # sits at the same level Kaggle exposes for easy download.
    parent = out_dir.parent if out_dir.parent.exists() else out_dir
    archive_root = str(parent / archive_basename)
    try:
        archive_path = shutil.make_archive(archive_root, "zip", root_dir=str(out_dir))
        print(f"\n[PACKAGE] Created: {archive_path}")
        return archive_path
    except Exception as e:
        print(f"[PACKAGE WARN] make_archive failed: {repr(e)}")
        # Fallback: use zipfile directly
        try:
            import zipfile
            zip_path = archive_root + ".zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                for p in out_dir.rglob("*"):
                    if p.is_file():
                        z.write(p, arcname=p.relative_to(out_dir))
            print(f"[PACKAGE] Created via zipfile: {zip_path}")
            return zip_path
        except Exception as e2:
            print(f"[PACKAGE ERR] zipfile fallback failed: {repr(e2)}")
            return None


# ============================================================
# 11. Main
# ============================================================

def main():
    out_dir = Path(CONFIG["OUTPUT_DIR"])
    mkdir(out_dir)

    print("=" * 100)
    print("CFS v16 Pareto-Legendary Official No-Fastener Framework")
    print("=" * 100)
    print(json.dumps(CONFIG, indent=2, ensure_ascii=False))
    print("\n[LIBS]")
    print(f"  XGBoost : {HAS_XGB}")
    print(f"  LightGBM: {HAS_LGB}")
    print(f"  CatBoost: {HAS_CAT}")
    print(f"  SHAP    : {HAS_SHAP}")

    raw_df, data_path = load_and_standardize()
    df = engineer_design_features(raw_df)

    print("\n[DATA SUMMARY]")
    print(f"Rows: {len(df)}")
    print(f"Target Pt/Py:\n{df['PtPy'].describe().to_string()}")
    print("\nDesign groups:")
    print(df["SG_design"].value_counts().to_string())
    print("\nTrue FM labels are used only to train cross-fitted FM-probability model, not as direct strength features.")
    print(df["FM"].value_counts().to_string())

    # Holdout split, stratified by target bins + design groups
    strat = stratify_key(df["PtPy"], df["SG_design"])
    idx = np.arange(len(df))
    try:
        tr_idx, te_idx = train_test_split(
            idx,
            test_size=CONFIG["TEST_SIZE"],
            random_state=CONFIG["RANDOM_STATE"],
            stratify=strat,
        )
    except Exception:
        tr_idx, te_idx = train_test_split(
            idx,
            test_size=CONFIG["TEST_SIZE"],
            random_state=CONFIG["RANDOM_STATE"],
        )

    tr = df.iloc[tr_idx].copy().reset_index(drop=True)
    te = df.iloc[te_idx].copy().reset_index(drop=True)

    print("\n[SPLIT]")
    print(f"Train: {len(tr)} | Test: {len(te)}")
    print("Test design group distribution:")
    print(te["SG_design"].value_counts().to_string())

    # Run holdout pipeline
    print("\n" + "=" * 100)
    print("HOLDOUT TRAINING")
    print("=" * 100)
    pred_tr, pred_te, bundle = run_v16_pipeline(tr, te, seed=CONFIG["RANDOM_STATE"], verbose=True)

    train_metrics = metrics(tr["PtPy"].values, pred_tr, prefix="OOF_")
    test_metrics = metrics(te["PtPy"].values, pred_te, prefix="TEST_")
    print_metrics("FINAL HOLDOUT RESULT", {"stage": bundle["stage"], **train_metrics, **test_metrics})

    # v16 split-conformal interval from OOF absolute residuals.
    alpha = CONFIG.get("CONFORMAL_ALPHA", 0.10)
    q_abs = float(np.quantile(np.abs(tr["PtPy"].values - pred_tr), 1.0 - alpha))
    lo_te = clip_pred(pred_te - q_abs)
    hi_te = clip_pred(pred_te + q_abs)
    coverage = float(np.mean((te["PtPy"].values >= lo_te) & (te["PtPy"].values <= hi_te)) * 100.0)
    print(f"\n[CONFORMAL] alpha={alpha}, absolute q={q_abs:.6f}, holdout coverage={coverage:.2f}%")

    # Save predictions
    pred_df = pd.DataFrame({
        "row_index_original": te_idx,
        "PtPy_actual": te["PtPy"].values,
        "PtPy_pred": pred_te,
        "PtPy_PI_low": lo_te,
        "PtPy_PI_high": hi_te,
        "PI_coverage_holdout_%": coverage,
        "AbsErr_%": np.abs((te["PtPy"].values - pred_te) / (np.abs(te["PtPy"].values) + EPS)) * 100,
        "SectionType": te["SectionType"].values,
        "Sections": te["Sections"].values,
        "BC": te["BC"].values,
        "SG_design": te["SG_design"].values,
        "FM_true_not_directly_used": te["FM"].values,
        "Py": te["Py"].values,
        "Pt_actual_kN": te["PtPy"].values * te["Py"].values,
        "Pt_pred_kN": pred_te * te["Py"].values,
    })
    pred_df.to_csv(out_dir / "v16_holdout_predictions.csv", index=False)
    try:
        bundle.get("pareto_report").to_csv(out_dir / "v16_pareto_candidate_report.csv", index=False)
    except Exception:
        pass

    group_err = pred_df.groupby("SG_design")["AbsErr_%"].agg(["mean", "median", "max", "count"]).sort_values("mean", ascending=False)
    group_err.to_csv(out_dir / "v16_error_by_design_group.csv")
    print("\n[ERROR BY DESIGN GROUP]")
    print(group_err.round(3).to_string())

    fm_err = pred_df.groupby("FM_true_not_directly_used")["AbsErr_%"].agg(["mean", "median", "max", "count"]).sort_values("mean", ascending=False)
    fm_err.to_csv(out_dir / "v16_error_by_true_FM_audit_only.csv")
    print("\n[ERROR BY TRUE FM — AUDIT ONLY, NOT DIRECT INPUT]")
    print(fm_err.round(3).to_string())

    # Repeated CV
    cv_df, cv_summary = None, None
    if CONFIG["RUN_REPEATED_CV"]:
        cv_df, cv_summary = repeated_cv_fast(df, seed=CONFIG["RANDOM_STATE"])
        cv_df.to_csv(out_dir / "v16_repeated_cv_folds.csv", index=False)
        with open(out_dir / "v16_repeated_cv_summary.json", "w", encoding="utf-8") as f:
            json.dump(to_json_safe(cv_summary), f, indent=2, ensure_ascii=False)
        print_metrics("REPEATED CV SUMMARY", cv_summary)

    # Plots and SHAP
    save_plots(out_dir, te["PtPy"].values, pred_te, te["SG_design"].values)
    save_shap(out_dir, tr, te)

    # Save feature list and summary
    feature_report = {
        "NUM_FEATURES_BASE": NUM_FEATURES_BASE,
        "CAT_FEATURES": CAT_FEATURES,
        "important_no_leakage_rules": [
            "No fastener/spacing/screw/bolt input columns are used.",
            "No TestLabel/Family/StudyID is used as a strength feature.",
            "True FM is not used directly in the strength model.",
            "FM probabilities are cross-fitted predictions from design inputs only.",
            "No target-derived features such as Pt_DSM_ratio or PtPy residual are used as inputs.",
            "Target encoding is recomputed inside each training fold.",
        ],
    }
    with open(out_dir / "v16_feature_report.json", "w", encoding="utf-8") as f:
        json.dump(feature_report, f, indent=2, ensure_ascii=False)

    # Build expert_info copies that contain only JSON-friendly fields
    # (drop the trained blender objects that previously broke json.dump).
    expert_info_public = []
    for info in (bundle.get("expert_info") or []):
        ei = {}
        for k, v in info.items():
            if k in ("blend",):
                continue
            ei[k] = v
        expert_info_public.append(ei)

    summary = {
        "data_path": data_path,
        "n_rows": int(len(df)),
        "stage": bundle["stage"],
        "holdout_train_metrics": train_metrics,
        "holdout_test_metrics": test_metrics,
        "repeated_cv_summary": cv_summary,
        "config": CONFIG,
        "bundle_public_info": {
            "fm_info": bundle.get("fm_info"),
            "expert_info": expert_info_public,
            "meta_name": bundle.get("meta_name"),
            "calibration_used": bundle.get("calibration_used"),
            "residual_used": bundle.get("residual_used"),
            "pareto_selected": bundle.get("meta_name"),
        },
    }
    try:
        with open(out_dir / "v16_final_summary.json", "w", encoding="utf-8") as f:
            json.dump(to_json_safe(summary), f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] final summary JSON serialization fallback engaged: {repr(e)}")
        with open(out_dir / "v16_final_summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "data_path": data_path,
                "n_rows": int(len(df)),
                "stage": str(bundle.get("stage")),
                "holdout_train_metrics": to_json_safe(train_metrics),
                "holdout_test_metrics": to_json_safe(test_metrics),
                "repeated_cv_summary": to_json_safe(cv_summary),
                "meta_name": str(bundle.get("meta_name")),
                "calibration_used": bool(bundle.get("calibration_used")),
                "residual_used": bool(bundle.get("residual_used")),
            }, f, indent=2, ensure_ascii=False)

    if CONFIG["SAVE_MODELS"] and HAS_JOBLIB:
        try:
            joblib.dump(bundle, out_dir / "v16_model_bundle_holdout.joblib")
        except Exception as e:
            print(f"[WARN] Could not save model bundle: {repr(e)}")

    # ------------------------------------------------------------------
    # v17 production deliverable: inference bundle + predict module + ZIP
    # ------------------------------------------------------------------
    inference_bundle = None
    try:
        train_eng = bundle.get("train_df_engineered")
        test_eng = bundle.get("test_df_engineered")
        y_train_full = tr["PtPy"].values
        y_test_full = te["PtPy"].values
        if train_eng is not None and HAS_JOBLIB:
            inference_bundle = build_v17_inference_bundle(
                train_eng, test_eng, y_train_full, y_test=y_test_full,
                seed=CONFIG["RANDOM_STATE"],
            )
            joblib.dump(inference_bundle, out_dir / "cfs_v17_inference_bundle.joblib")
            print(f"[INFERENCE] cfs_v17_inference_bundle.joblib saved ({(out_dir/'cfs_v17_inference_bundle.joblib').stat().st_size/1024:.1f} KB)")
    except Exception as e:
        print(f"[INFERENCE WARN] could not build inference bundle: {repr(e)}")

    try:
        write_predict_module(out_dir)
        write_v17_readme(out_dir)
    except Exception as e:
        print(f"[INFERENCE WARN] could not write predict module / readme: {repr(e)}")

    # Final ZIP packaging so the user has a single download link in Kaggle.
    try:
        archive_path = package_outputs_zip(out_dir, archive_basename="cfs_v17_FINAL_PACKAGE")
    except Exception as e:
        print(f"[PACKAGE WARN] failed to package outputs: {repr(e)}")
        archive_path = None

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Outputs saved to: {out_dir}")
    print(f"Holdout TEST R2 = {test_metrics.get('TEST_R2', np.nan):.6f}")
    if cv_summary:
        print(f"Repeated CV R2 mean = {cv_summary.get('R2_mean', np.nan):.6f} ± {cv_summary.get('R2_std', np.nan):.6f}")
    if inference_bundle is not None:
        print(f"V17 Inference OOF R2 = {inference_bundle.get('training_oof_r2', np.nan):.6f}")
        if inference_bundle.get("test_metrics"):
            tm = inference_bundle["test_metrics"]
            print(f"V17 Inference TEST R2 = {tm.get('V17_INFERENCE_TEST_R2', np.nan):.6f}")
    if archive_path:
        print(f"\n>>> DOWNLOAD: {archive_path}")
        print(">>> In Kaggle, click this file in the Output panel to download the full package.")

    print("\nScientific interpretation:")
    print("- This is an official no-fastener, no-direct-FM, no-target-leakage framework.")
    print("- If it scores lower than leakage-heavy v11/v13, that is scientifically correct.")
    print("- For a paper, report repeated CV and the leakage rules, not only one holdout split.")


if __name__ == "__main__":
    main()
