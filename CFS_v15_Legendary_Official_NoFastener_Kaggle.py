#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CFS Built-up Columns — v15 Legendary Official No-Fastener Framework
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
    "OUTPUT_DIR": "/kaggle/working/cfs_v15_legendary_official",

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
    # True FM is intentionally not added.
    return [c for c in cols if c in df.columns]

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
    try:
        cal = HuberRegressor(epsilon=1.35, alpha=1e-4)
        cal.fit(np.asarray(p_train).reshape(-1, 1), y_train)
    except Exception:
        cal = LinearRegression()
        cal.fit(np.asarray(p_train).reshape(-1, 1), y_train)

    p_tr_cal = clip_pred(cal.predict(np.asarray(p_train).reshape(-1, 1)))
    p_te_cal = clip_pred(cal.predict(np.asarray(p_test).reshape(-1, 1)))

    raw = metrics(y_train, p_train)
    new = metrics(y_train, p_tr_cal)

    use = (new["R2"] >= raw["R2"] - 1e-5) and (new["RMSE"] <= raw["RMSE"] * 1.002)
    return (p_tr_cal, p_te_cal, cal, True) if use else (p_train, p_test, cal, False)

def residual_correction(train_df, test_df, y_train, p_train, p_test, seed=42):
    print("\n[RESIDUAL] Cross-fitted residual correction")
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
            m.fit(X_fit, residual[idx_tr])
            res_oof[idx_va] = m.predict(X_val)
            res_test_folds.append(m.predict(X_test))

        res_oof_total += res_oof / len(models)
        res_test_total += np.mean(np.vstack(res_test_folds), axis=0) / len(models)

    p_tr_corr = clip_pred(p_train + res_oof_total)
    p_te_corr = clip_pred(p_test + res_test_total)

    raw = metrics(y_train, p_train)
    new = metrics(y_train, p_tr_corr)
    use = (new["R2"] > raw["R2"] + 1e-5) and (new["RMSE"] <= raw["RMSE"])

    print(f"  residual correction used={use} | OOF R2 raw={raw['R2']:.6f}, corr={new['R2']:.6f}")
    return (p_tr_corr, p_te_corr, True) if use else (p_train, p_test, False)


# ============================================================
# 8. Full pipeline
# ============================================================

def run_v15_pipeline(train_df, test_df, seed=42, verbose=True):
    train_df = train_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    # FM probability model
    fm_info = {"used": False, "prob_columns": []}
    if CONFIG["USE_FM_PROBABILITIES"]:
        train_df, test_df, fm_info = crossfit_fm_probabilities(train_df, test_df, seed=seed)

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

    # Add raw baselines as weak experts
    for base in base_cols:
        expert_oof.append(clip_pred(train_df[base].values))
        expert_test.append(clip_pred(test_df[base].values))
        expert_info.append({"name": "RAW_" + base, "base": base, "oof_r2": float(r2_score(train_df["PtPy"].values, clip_pred(train_df[base].values)))})

    B_train = np.column_stack(expert_oof)
    B_test = np.column_stack(expert_test)
    y_train = train_df["PtPy"].values

    pred_tr, pred_te, meta, meta_name = nonnegative_blend(B_train, y_train, B_test)
    stage = "expert_stack_" + meta_name

    print_metrics("STACK BEFORE CALIBRATION", {**metrics(y_train, pred_tr, prefix="OOF_")})

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
        "calibration_model": cal,
        "calibration_used": cal_used,
        "residual_used": res_used,
        "expert_oof": B_train,
        "expert_test": B_test,
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
            _, pred_va, _ = run_v15_pipeline(tr, va, seed=seed + fold, verbose=False)
        else:
            # Faster official approximation: train only strongest baselines subset.
            train_save = CONFIG["INNER_FOLDS"]
            CONFIG["INNER_FOLDS"] = min(3, train_save)
            old_res = CONFIG["USE_RESIDUAL_CORRECTION"]
            CONFIG["USE_RESIDUAL_CORRECTION"] = False
            _, pred_va, _ = run_v15_pipeline(tr, va, seed=seed + fold, verbose=False)
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
    plt.title(f"v15 Legendary Official: Holdout R2={r2_score(y, p):.4f}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "v15_scatter_holdout.png", bbox_inches="tight")
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
    plt.savefig(out_dir / "v15_residuals_by_group.png", bbox_inches="tight")
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
        imp.to_csv(Path(out_dir) / "v15_shap_importance.csv", index=False)

        if HAS_MPL:
            plt.figure(figsize=(10, 7), dpi=150)
            shap.summary_plot(sv, X_te, feature_names=X_te.columns, show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(Path(out_dir) / "v15_shap_summary.png", bbox_inches="tight")
            plt.close()

        print(imp.head(20).to_string(index=False))
        return imp
    except Exception as e:
        print(f"[SHAP] failed: {repr(e)}")
        return None


# ============================================================
# 11. Main
# ============================================================

def main():
    out_dir = Path(CONFIG["OUTPUT_DIR"])
    mkdir(out_dir)

    print("=" * 100)
    print("CFS v15 Legendary Official No-Fastener Framework")
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
    pred_tr, pred_te, bundle = run_v15_pipeline(tr, te, seed=CONFIG["RANDOM_STATE"], verbose=True)

    train_metrics = metrics(tr["PtPy"].values, pred_tr, prefix="OOF_")
    test_metrics = metrics(te["PtPy"].values, pred_te, prefix="TEST_")
    print_metrics("FINAL HOLDOUT RESULT", {"stage": bundle["stage"], **train_metrics, **test_metrics})

    # Save predictions
    pred_df = pd.DataFrame({
        "row_index_original": te_idx,
        "PtPy_actual": te["PtPy"].values,
        "PtPy_pred": pred_te,
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
    pred_df.to_csv(out_dir / "v15_holdout_predictions.csv", index=False)

    group_err = pred_df.groupby("SG_design")["AbsErr_%"].agg(["mean", "median", "max", "count"]).sort_values("mean", ascending=False)
    group_err.to_csv(out_dir / "v15_error_by_design_group.csv")
    print("\n[ERROR BY DESIGN GROUP]")
    print(group_err.round(3).to_string())

    fm_err = pred_df.groupby("FM_true_not_directly_used")["AbsErr_%"].agg(["mean", "median", "max", "count"]).sort_values("mean", ascending=False)
    fm_err.to_csv(out_dir / "v15_error_by_true_FM_audit_only.csv")
    print("\n[ERROR BY TRUE FM — AUDIT ONLY, NOT DIRECT INPUT]")
    print(fm_err.round(3).to_string())

    # Repeated CV
    cv_df, cv_summary = None, None
    if CONFIG["RUN_REPEATED_CV"]:
        cv_df, cv_summary = repeated_cv_fast(df, seed=CONFIG["RANDOM_STATE"])
        cv_df.to_csv(out_dir / "v15_repeated_cv_folds.csv", index=False)
        with open(out_dir / "v15_repeated_cv_summary.json", "w", encoding="utf-8") as f:
            json.dump(cv_summary, f, indent=2, ensure_ascii=False)
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
    with open(out_dir / "v15_feature_report.json", "w", encoding="utf-8") as f:
        json.dump(feature_report, f, indent=2, ensure_ascii=False)

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
            "expert_info": bundle.get("expert_info"),
            "meta_name": bundle.get("meta_name"),
            "calibration_used": bundle.get("calibration_used"),
            "residual_used": bundle.get("residual_used"),
        },
    }
    with open(out_dir / "v15_final_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if CONFIG["SAVE_MODELS"] and HAS_JOBLIB:
        try:
            joblib.dump(bundle, out_dir / "v15_model_bundle_holdout.joblib")
        except Exception as e:
            print(f"[WARN] Could not save model bundle: {repr(e)}")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Outputs saved to: {out_dir}")
    print(f"Holdout TEST R2 = {test_metrics.get('TEST_R2', np.nan):.6f}")
    if cv_summary:
        print(f"Repeated CV R2 mean = {cv_summary.get('R2_mean', np.nan):.6f} ± {cv_summary.get('R2_std', np.nan):.6f}")

    print("\nScientific interpretation:")
    print("- This is an official no-fastener, no-direct-FM, no-target-leakage framework.")
    print("- If it scores lower than leakage-heavy v11/v13, that is scientifically correct.")
    print("- For a paper, report repeated CV and the leakage rules, not only one holdout split.")


if __name__ == "__main__":
    main()
