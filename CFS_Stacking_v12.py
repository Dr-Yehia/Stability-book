#!/usr/bin/env python3
"""
CFS Built-up Columns - v12 Physics-Correction Ensemble
=====================================================

Purpose
-------
This script is a stronger successor to v11.  It is built for Kaggle and for
research reporting on the Bristol CFS built-up columns dataset.

The main change is conceptual:
    v11 predicts Pt/Py directly.
    v12 predicts a physics correction over design-equation baselines.

Scientific safeguards
---------------------
1. No target-derived feature is used.  In particular, v11's Pt_DSM_ratio is
   removed because it contains Pt/Py and therefore leaks the answer.
2. Target encoding is computed inside every training fold only from the fold's
   training rows.
3. The final local residual KNN is cross-fitted on OOF residuals before it is
   allowed to correct the test set.
4. A random split is used by default to match prior notebooks.  Set
   VALIDATION_MODE = "group" for the stricter publication audit by specimen
   family.

Expected behavior
-----------------
On the same random split style used by v9-v11, this pipeline is designed to
push the result as high as the data allows, with an ambitious target near
R2(Pt/Py) = 0.985.  No script can honestly guarantee that value before running
on the actual environment and split, but this is the strongest no-leak version
in this repository.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path


def ensure_package(package: str, import_name: str | None = None) -> None:
    """Install a package only when it is missing."""
    name = import_name or package
    if importlib.util.find_spec(name) is None:
        print(f"Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


ensure_package("xgboost")
ensure_package("catboost")
ensure_package("lightgbm")
ensure_package("shap")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["figure.dpi"] = 150
warnings.filterwarnings("ignore")

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, KFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb_lib
import shap
import xgboost as _xgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/Dr-Yehia/Stability-book/main/"
    "CFS_Built-up_Columns_ML_Dataset.csv"
)

VALIDATION_MODE = "random"  # "random" for strongest benchmark, "group" for publication audit
TEST_SIZE = 0.20
RANDOM_STATE = 42
EARLY_STOPPING = 300
TE_SMOOTH = 12.0
MIN_SPECIALIST_ROWS = 45
PRED_MIN = 0.015
PRED_MAX = 1.350
RUN_SHAP = True

np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------


def find_dataset() -> str:
    candidates = [
        "CFS_Built-up_Columns_ML_Dataset.csv",
        "/kaggle/working/CFS_Built-up_Columns_ML_Dataset.csv",
    ]
    candidates.extend(glob.glob("/kaggle/input/**/CFS_Built-up_Columns_ML_Dataset.csv", recursive=True))
    for path in candidates:
        if Path(path).exists():
            return path
    return DATA_URL


def first_col(df: pd.DataFrame, names: list[str], required: bool = True) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    if required:
        raise KeyError(f"Missing required column. Tried: {names}")
    return None


def clean_text(s: pd.Series) -> pd.Series:
    return (
        s.fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    )


def make_family(label: str) -> str:
    label = str(label).strip()
    label = re.sub(r"[-_]\d+$", "", label)
    return label if label else "Unknown"


source = find_dataset()
print(f"Loading data from: {source}")
raw = pd.read_csv(source)
raw = raw.loc[:, [c for c in raw.columns if str(c).strip() and not str(c).startswith("Unnamed")]]
raw = raw.dropna(axis=1, how="all")

colmap = {
    "No": ["No"],
    "TestID": ["Test_ID", "TestID"],
    "TestLabel": ["Test_Label", "TestLabel"],
    "SectionType": ["Section Types", "SectionType"],
    "Sections": ["Sections"],
    "L": ["L"],
    "t": ["t"],
    "h": ["h"],
    "b": ["b"],
    "A": ["A"],
    "Pt": ["Pt"],
    "BC": ["BC"],
    "FailureModeRaw": ["Failure Mode", "FailureMode"],
    "FM": ["FM"],
    "Py": ["Py"],
    "Fy": ["Fy"],
    "PtPy": ["Pt/Py", "PtPy"],
    "Pcrl_crd": ["P(crl,crd)", "Pcrl_crd", "Pcrl"],
    "KL_r": ["KL_r", "KL/r"],
    "lam_c": ["lambda_c", "lam_c", "λc", "位c"],
    "Pne": ["Pne"],
    "lam_led": ["lambda_led", "lam_led", "λ(le-d)", "位(le-d)"],
}

df = pd.DataFrame(index=raw.index)

for new_name in ["TestLabel", "SectionType", "Sections", "BC", "FailureModeRaw", "FM"]:
    c = first_col(raw, colmap[new_name])
    df[new_name] = clean_text(raw[c])

for new_name in [
    "No",
    "L",
    "t",
    "h",
    "b",
    "A",
    "Pt",
    "Py",
    "Fy",
    "PtPy",
    "Pcrl_crd",
    "KL_r",
    "lam_c",
    "Pne",
    "lam_led",
]:
    c = first_col(raw, colmap[new_name])
    df[new_name] = pd.to_numeric(raw[c], errors="coerce")

df["FM"] = df["FM"].str.replace(r"\s+", "", regex=True)
df["BC"] = df["BC"].str.replace(r"\s+", "", regex=True)
df["Family"] = df["TestLabel"].apply(make_family)

df = df[df["PtPy"].notna()].copy()
df = df[(df["PtPy"] > 0) & (df["Py"] > 0) & (df["t"] > 0) & (df["h"] > 0)].copy()
df = df.reset_index(drop=True)

print(f"Clean rows: {len(df)}")
print(f"Section types: {df['SectionType'].nunique()} | FM: {df['FM'].nunique()} | Families: {df['Family'].nunique()}")


# ---------------------------------------------------------------------------
# Physics features and baselines
# ---------------------------------------------------------------------------


EPS = 1e-9


def sdiv(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.divide(a, np.where(np.abs(b) < EPS, np.nan, b))


def dsm_global(lam_c: pd.Series | np.ndarray) -> np.ndarray:
    lc = np.asarray(lam_c, dtype=float)
    lc = np.where(lc <= 0, np.nan, lc)
    return np.where(lc <= 1.5, 0.658 ** (lc**2), 0.877 / (lc**2))


def dsm_local_from_lam(lam_led: pd.Series | np.ndarray, pne_py: pd.Series | np.ndarray) -> np.ndarray:
    lam = np.asarray(lam_led, dtype=float)
    pne = np.asarray(pne_py, dtype=float)
    lam = np.where(lam <= 0, np.nan, lam)
    reduction = np.where(lam <= 0.776, 1.0, (1.0 - 0.15 / (lam**0.8)) / (lam**0.8))
    return pne * reduction


def assign_section_group(row: pd.Series) -> str:
    st = str(row["SectionType"]).strip()
    sec = str(row["Sections"]).strip()
    if st == "O-2C":
        return "G1_O2C"
    if st == "O-2U":
        return "G2_O2U"
    if st == "C-U+C":
        return "G3_CUC"
    if st.startswith("HC"):
        return "G4_HC"
    if st.startswith("C-2C") or st.startswith("C-2" + "\u03a3"):
        return "G5_C2C"
    if st.startswith("O-"):
        return "G6_Open"
    if st.lower().startswith("box") or sec == "Closed":
        return "G7a_Closed"
    return "G7b_Rest"


df["SG"] = df.apply(assign_section_group, axis=1)
df.loc[(df["SG"] == "G5_C2C") & (df["FM"] == "F"), "SG"] = "G5a_C2C_F"
df.loc[(df["SG"] == "G5_C2C") & (df["FM"] == "LF"), "SG"] = "G5c_C2C_LF"
df.loc[df["SG"] == "G5_C2C", "SG"] = "G5b_C2C_Other"

df["h_t"] = sdiv(df["h"], df["t"])
df["b_t"] = sdiv(df["b"], df["t"])
df["L_h"] = sdiv(df["L"], df["h"])
df["L_t"] = sdiv(df["L"], df["t"])
df["h_b"] = sdiv(df["h"], df["b"])
df["b_h"] = sdiv(df["b"], df["h"])
df["A_t2"] = sdiv(df["A"], df["t"] ** 2)
df["sqrt_A_t"] = sdiv(np.sqrt(np.abs(df["A"])), df["t"])
df["Fy_norm"] = df["Fy"] / 350.0
df["log_Py"] = np.log1p(df["Py"].clip(lower=0))

df["Pcrl_Py"] = sdiv(df["Pcrl_crd"], df["Py"])
df["Pne_Py"] = sdiv(df["Pne"], df["Py"])
df["Pcrl_Pne"] = sdiv(df["Pcrl_Py"], df["Pne_Py"])
df["Pne_Pcrl"] = sdiv(df["Pne_Py"], df["Pcrl_Py"])

df["lam_c_sq"] = df["lam_c"] ** 2
df["lam_led_sq"] = df["lam_led"] ** 2
df["lam_c_lam_led"] = df["lam_c"] * df["lam_led"]
df["lam_c3"] = df["lam_c"] ** 3
df["lam_led3"] = df["lam_led"] ** 3
df["inv_lam_c"] = sdiv(1.0, df["lam_c"])
df["inv_lam_led"] = sdiv(1.0, df["lam_led"])
df["lam_ratio"] = sdiv(df["lam_led"], df["lam_c"])
df["lam_c_over_led"] = sdiv(df["lam_c"], df["lam_led"])

df["Pcrl_sq"] = df["Pcrl_Py"] ** 2
df["Pne_sq"] = df["Pne_Py"] ** 2
df["sqrt_Pcrl"] = np.sqrt(np.abs(df["Pcrl_Py"]))
df["sqrt_Pne"] = np.sqrt(np.abs(df["Pne_Py"]))
df["h_t_b_t"] = df["h_t"] * df["b_t"]
df["Pcrl_lam_led"] = df["Pcrl_Py"] * df["lam_led"]
df["Pne_lam_c"] = df["Pne_Py"] * df["lam_c"]
df["Pne_lam_led"] = df["Pne_Py"] * df["lam_led"]

df["DSM_global"] = dsm_global(df["lam_c"])
df["DSM_local"] = dsm_local_from_lam(df["lam_led"], df["Pne_Py"])
df["DSM_min"] = pd.DataFrame(
    {"g": df["DSM_global"], "l": df["DSM_local"], "pne": df["Pne_Py"], "pcrl": df["Pcrl_Py"]}
).where(lambda x: x > 0).min(axis=1)
df["DSM_mean"] = pd.DataFrame(
    {"g": df["DSM_global"], "l": df["DSM_local"], "pne": df["Pne_Py"], "pcrl": df["Pcrl_Py"]}
).where(lambda x: x > 0).mean(axis=1)
df["DSM_gap"] = np.abs(np.log(sdiv(df["DSM_global"], df["DSM_local"])))
df["Pne_DSM_ratio"] = sdiv(df["Pne_Py"], df["DSM_global"])
df["Pcrl_DSM_ratio"] = sdiv(df["Pcrl_Py"], df["DSM_global"])
df["local_global_ratio"] = sdiv(df["DSM_local"], df["DSM_global"])

df["base_design"] = df["DSM_min"].fillna(df["DSM_global"]).clip(PRED_MIN, PRED_MAX)
df["base_global"] = df["DSM_global"].clip(PRED_MIN, PRED_MAX)
df["base_local"] = df["DSM_local"].fillna(df["base_design"]).clip(PRED_MIN, PRED_MAX)
df["base_mean"] = df["DSM_mean"].fillna(df["base_design"]).clip(PRED_MIN, PRED_MAX)

df["is_F"] = (df["FM"] == "F").astype(float)
df["is_L"] = (df["FM"] == "L").astype(float)
df["is_LF"] = (df["FM"] == "LF").astype(float)
df["is_D"] = (df["FM"] == "D").astype(float)
df["is_closed"] = (df["Sections"].str.lower() == "closed").astype(float)
df["is_open"] = (df["Sections"].str.lower() == "open").astype(float)

for c in df.select_dtypes(include=np.number).columns:
    med = df[c].replace([np.inf, -np.inf], np.nan).median()
    df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(med if np.isfinite(med) else 0.0)

df["corr_base"] = np.log(df["PtPy"].clip(PRED_MIN, PRED_MAX) / df["base_design"].clip(PRED_MIN, PRED_MAX))
df["corr_global"] = np.log(df["PtPy"].clip(PRED_MIN, PRED_MAX) / df["base_global"].clip(PRED_MIN, PRED_MAX))
df["corr_local"] = np.log(df["PtPy"].clip(PRED_MIN, PRED_MAX) / df["base_local"].clip(PRED_MIN, PRED_MAX))
df["corr_base"] = df["corr_base"].clip(-1.5, 1.5)
df["corr_global"] = df["corr_global"].clip(-1.5, 1.5)
df["corr_local"] = df["corr_local"].clip(-1.5, 1.5)

print("\nSection groups:")
print(df["SG"].value_counts().to_string())
print("\nFM distribution:")
print(df["FM"].value_counts().to_string())


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def make_strata(data: pd.DataFrame) -> pd.Series:
    s = data["SG"].astype(str) + "_" + data["FM"].astype(str)
    vc = s.value_counts()
    return s.where(s.map(vc) >= 4, data["SG"].astype(str))


if VALIDATION_MODE == "group":
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(splitter.split(df, groups=df["Family"]))
else:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(splitter.split(df, make_strata(df)))

tr = df.iloc[tr_idx].copy().reset_index(drop=True)
te = df.iloc[te_idx].copy().reset_index(drop=True)

family_overlap = len(set(tr["Family"]) & set(te["Family"]))
print(f"\nValidation mode: {VALIDATION_MODE}")
print(f"Train: {len(tr)} | Test: {len(te)} | family overlap: {family_overlap}")
print("Test SG distribution:")
print(te["SG"].value_counts().to_string())


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------


NUM_FEATURES = [
    "L",
    "t",
    "h",
    "b",
    "A",
    "Fy",
    "h_t",
    "b_t",
    "L_h",
    "L_t",
    "h_b",
    "b_h",
    "A_t2",
    "sqrt_A_t",
    "Fy_norm",
    "log_Py",
    "Pcrl_Py",
    "Pne_Py",
    "Pcrl_Pne",
    "Pne_Pcrl",
    "lam_c",
    "lam_led",
    "KL_r",
    "lam_c_sq",
    "lam_led_sq",
    "lam_c_lam_led",
    "lam_c3",
    "lam_led3",
    "inv_lam_c",
    "inv_lam_led",
    "lam_ratio",
    "lam_c_over_led",
    "Pcrl_sq",
    "Pne_sq",
    "sqrt_Pcrl",
    "sqrt_Pne",
    "h_t_b_t",
    "Pcrl_lam_led",
    "Pne_lam_c",
    "Pne_lam_led",
    "DSM_global",
    "DSM_local",
    "DSM_min",
    "DSM_mean",
    "DSM_gap",
    "Pne_DSM_ratio",
    "Pcrl_DSM_ratio",
    "local_global_ratio",
    "base_design",
    "base_global",
    "base_local",
    "base_mean",
    "is_F",
    "is_L",
    "is_LF",
    "is_D",
    "is_closed",
    "is_open",
]

TE_CATS = ["SectionType", "Sections", "BC", "FM", "SG"]
ONEHOT_CATS = ["SectionType", "Sections", "BC", "FM", "SG"]


def smoothed_means(fit_df: pd.DataFrame, y_fit: np.ndarray, cat: str) -> pd.Series:
    y_series = pd.Series(np.asarray(y_fit, dtype=float), index=fit_df.index)
    global_mean = float(np.mean(y_fit))
    grp = y_series.groupby(fit_df[cat].astype(str))
    means = grp.mean()
    counts = grp.size()
    return (means * counts + global_mean * TE_SMOOTH) / (counts + TE_SMOOTH)


def onehot_columns_from(data: pd.DataFrame) -> list[str]:
    oh = pd.get_dummies(data[ONEHOT_CATS].astype(str), prefix=ONEHOT_CATS, dtype=float)
    return list(oh.columns)


def make_matrix(
    fit_df: pd.DataFrame,
    apply_df: pd.DataFrame,
    y_fit: np.ndarray,
    onehot_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    X = pd.DataFrame(index=apply_df.index)
    for col in NUM_FEATURES:
        X[col] = apply_df[col].astype(float).values

    global_mean = float(np.mean(y_fit))
    for cat in TE_CATS:
        enc = smoothed_means(fit_df, y_fit, cat)
        X[f"{cat}_te"] = apply_df[cat].astype(str).map(enc).fillna(global_mean).values

    oh_apply = pd.get_dummies(apply_df[ONEHOT_CATS].astype(str), prefix=ONEHOT_CATS, dtype=float)
    if onehot_cols is None:
        onehot_cols = onehot_columns_from(fit_df)
    for col in onehot_cols:
        if col not in oh_apply.columns:
            oh_apply[col] = 0.0
    X = pd.concat([X, oh_apply[onehot_cols]], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    return X, onehot_cols


def clip_pred(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), PRED_MIN, PRED_MAX)


def correction_target(data: pd.DataFrame, base_col: str) -> np.ndarray:
    y = np.log(data["PtPy"].values.clip(PRED_MIN, PRED_MAX) / data[base_col].values.clip(PRED_MIN, PRED_MAX))
    return np.clip(y, -1.5, 1.5)


_xgb_major = int(str(_xgb.__version__).split(".")[0])


def fit_xgb(params: dict, Xf, yf, Xv, yv):
    fit_kwargs = {"eval_set": [(Xv, yv)], "verbose": False}
    if _xgb_major < 2:
        fit_kwargs["early_stopping_rounds"] = EARLY_STOPPING
    model = XGBRegressor(**params)
    model.fit(Xf, yf, **fit_kwargs)
    return model


def model_pack(n_rows: int, seed: int) -> list[tuple[str, object, dict]]:
    depth = 4 if n_rows < 90 else 5 if n_rows < 220 else 6
    leaves = min(95, max(15, n_rows // 5))
    min_child = max(2, n_rows // 120)

    xgb_es = {"early_stopping_rounds": EARLY_STOPPING} if _xgb_major >= 2 else {}

    xgb_huber = dict(
        n_estimators=7000,
        learning_rate=0.009,
        max_depth=depth,
        min_child_weight=min_child,
        subsample=0.86,
        colsample_bytree=0.76,
        reg_alpha=0.02,
        reg_lambda=0.80,
        objective="reg:pseudohubererror",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
        **xgb_es,
    )
    xgb_square = dict(
        n_estimators=4500,
        learning_rate=0.014,
        max_depth=max(3, depth - 1),
        min_child_weight=max(1, min_child),
        subsample=0.90,
        colsample_bytree=0.82,
        reg_alpha=0.01,
        reg_lambda=1.15,
        objective="reg:squarederror",
        random_state=seed + 11,
        n_jobs=-1,
        verbosity=0,
        **xgb_es,
    )
    lgb = dict(
        n_estimators=7000,
        learning_rate=0.009,
        num_leaves=leaves,
        min_child_samples=max(4, n_rows // 70),
        subsample=0.86,
        colsample_bytree=0.74,
        reg_alpha=0.015,
        reg_lambda=0.70,
        objective="huber",
        alpha=0.88,
        random_state=seed + 23,
        n_jobs=-1,
        verbose=-1,
    )
    cat = dict(
        iterations=7000,
        learning_rate=0.009,
        depth=min(8, max(4, depth + 1)),
        l2_leaf_reg=1.2,
        loss_function="Huber:delta=0.45",
        bootstrap_type="Bernoulli",
        subsample=0.86,
        early_stopping_rounds=EARLY_STOPPING,
        random_seed=seed + 31,
        verbose=0,
    )
    extra = ExtraTreesRegressor(
        n_estimators=900,
        max_features=0.72,
        min_samples_leaf=1 if n_rows >= 90 else 2,
        random_state=seed + 41,
        n_jobs=-1,
    )
    return [
        ("xgb_huber", "xgb", xgb_huber),
        ("xgb_square", "xgb", xgb_square),
        ("lgb_huber", "lgb", lgb),
        ("cat_huber", "cat", cat),
        ("extra_trees", extra, {}),
    ]


def train_correction_expert(
    name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_col: str,
    n_folds: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    y_corr = correction_target(train_df, base_col)
    y_ptpy = train_df["PtPy"].values
    base_train = train_df[base_col].values.clip(PRED_MIN, PRED_MAX)
    base_test = test_df[base_col].values.clip(PRED_MIN, PRED_MAX)

    n = len(train_df)
    if n_folds is None:
        n_folds = max(3, min(10, n // 18))
    n_folds = min(n_folds, n)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    onehot_cols = onehot_columns_from(train_df)
    learner_names = [x[0] for x in model_pack(n, RANDOM_STATE)]
    oof_corr = {m: np.zeros(n) for m in learner_names}
    test_corr = {m: np.zeros(len(test_df)) for m in learner_names}

    for fold, (idx_fit, idx_val) in enumerate(kf.split(train_df), 1):
        fit_df = train_df.iloc[idx_fit].copy()
        val_df = train_df.iloc[idx_val].copy()
        y_fit = y_corr[idx_fit]
        y_val = y_corr[idx_val]

        X_fit, _ = make_matrix(fit_df, fit_df, y_fit, onehot_cols)
        X_val, _ = make_matrix(fit_df, val_df, y_fit, onehot_cols)
        X_test, _ = make_matrix(fit_df, test_df, y_fit, onehot_cols)

        Xf = X_fit.values
        Xv = X_val.values
        Xt = X_test.values

        for learner_name, kind, params in model_pack(n, RANDOM_STATE + fold * 101):
            if kind == "xgb":
                model = fit_xgb(params, Xf, y_fit, Xv, y_val)
            elif kind == "lgb":
                model = LGBMRegressor(**params)
                callbacks = [
                    lgb_lib.early_stopping(EARLY_STOPPING, verbose=False),
                    lgb_lib.log_evaluation(-1),
                ]
                model.fit(Xf, y_fit, eval_set=[(Xv, y_val)], callbacks=callbacks)
            elif kind == "cat":
                model = CatBoostRegressor(**params)
                model.fit(Xf, y_fit, eval_set=(Xv, y_val), verbose=False)
            else:
                model = clone(kind)
                model.fit(Xf, y_fit)

            oof_corr[learner_name][idx_val] = model.predict(Xv)
            test_corr[learner_name] += model.predict(Xt) / n_folds

    B_train = np.column_stack([oof_corr[m] for m in learner_names])
    B_test = np.column_stack([test_corr[m] for m in learner_names])
    blend = Ridge(alpha=0.025)
    blend.fit(B_train, y_corr)
    corr_oof = blend.predict(B_train)
    corr_test = blend.predict(B_test)

    pred_oof = clip_pred(base_train * np.exp(np.clip(corr_oof, -1.5, 1.5)))
    pred_test = clip_pred(base_test * np.exp(np.clip(corr_test, -1.5, 1.5)))
    r2 = r2_score(y_ptpy, pred_oof)

    print(f"{name:18s} rows={n:4d} folds={n_folds} OOF_R2={r2:.5f} blend={np.round(blend.coef_, 3)}")
    return pred_oof, pred_test, r2


# ---------------------------------------------------------------------------
# Train global, SG specialists, and FM specialists
# ---------------------------------------------------------------------------


y_tr = tr["PtPy"].values
y_te = te["PtPy"].values

print("\nTraining experts:")
global_oof, global_test, global_r2 = train_correction_expert("GLOBAL/base", tr, te, "base_design", n_folds=10)

sg_oof = global_oof.copy()
sg_test = global_test.copy()
for sg in sorted(tr["SG"].unique()):
    tr_mask = tr["SG"].values == sg
    te_mask = te["SG"].values == sg
    if tr_mask.sum() < MIN_SPECIALIST_ROWS or te_mask.sum() == 0:
        continue
    base_col = "base_global" if sg == "G5a_C2C_F" else "base_design"
    oof_g, test_g, r2_g = train_correction_expert(f"SG/{sg}", tr.loc[tr_mask].copy(), te.loc[te_mask].copy(), base_col)
    sg_oof[tr_mask] = oof_g
    sg_test[te_mask] = test_g

fm_oof = global_oof.copy()
fm_test = global_test.copy()
for fm in sorted(tr["FM"].unique()):
    tr_mask = tr["FM"].values == fm
    te_mask = te["FM"].values == fm
    if tr_mask.sum() < MIN_SPECIALIST_ROWS or te_mask.sum() == 0:
        continue
    base_col = "base_global" if fm == "F" else "base_design"
    oof_f, test_f, r2_f = train_correction_expert(f"FM/{fm}", tr.loc[tr_mask].copy(), te.loc[te_mask].copy(), base_col)
    fm_oof[tr_mask] = oof_f
    fm_test[te_mask] = test_f


# ---------------------------------------------------------------------------
# Meta blend, calibration, and local residual correction
# ---------------------------------------------------------------------------


def positive_meta_blend(B_train: np.ndarray, y: np.ndarray, B_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, object]:
    try:
        meta = LinearRegression(positive=True)
        meta.fit(B_train, y)
    except TypeError:
        meta = Ridge(alpha=0.001)
        meta.fit(B_train, y)
    pred_train = clip_pred(meta.predict(B_train))
    pred_test = clip_pred(meta.predict(B_test))
    return pred_train, pred_test, meta


B_tr = np.column_stack(
    [
        tr["base_design"].values,
        tr["base_global"].values,
        tr["base_local"].values,
        tr["base_mean"].values,
        global_oof,
        sg_oof,
        fm_oof,
    ]
)
B_te = np.column_stack(
    [
        te["base_design"].values,
        te["base_global"].values,
        te["base_local"].values,
        te["base_mean"].values,
        global_test,
        sg_test,
        fm_test,
    ]
)

meta_oof, meta_test, meta = positive_meta_blend(B_tr, y_tr, B_te)
print(f"\nMeta OOF R2={r2_score(y_tr, meta_oof):.5f}")
if hasattr(meta, "coef_"):
    print(f"Meta weights: {np.round(meta.coef_, 4)}")


def calibration_features(pred: np.ndarray, base: np.ndarray) -> np.ndarray:
    pred = clip_pred(pred)
    base = clip_pred(base)
    return np.column_stack(
        [
            np.log(pred),
            np.log(base),
            np.log(pred / base),
            pred,
            base,
        ]
    )


def crossfit_calibration(
    train_pred: np.ndarray,
    test_pred: np.ndarray,
    train_base: np.ndarray,
    test_base: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    kf = KFold(n_splits=8, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros_like(y, dtype=float)
    X_all = calibration_features(train_pred, train_base)
    X_test = calibration_features(test_pred, test_base)
    log_y = np.log(y.clip(PRED_MIN, PRED_MAX))
    for idx_fit, idx_val in kf.split(X_all):
        cal = HuberRegressor(epsilon=1.35, alpha=0.0005, max_iter=1000)
        cal.fit(X_all[idx_fit], log_y[idx_fit])
        oof[idx_val] = np.exp(cal.predict(X_all[idx_val]))
    cal_full = HuberRegressor(epsilon=1.35, alpha=0.0005, max_iter=1000)
    cal_full.fit(X_all, log_y)
    test_cal = np.exp(cal_full.predict(X_test))
    return clip_pred(oof), clip_pred(test_cal)


cal_oof, cal_test = crossfit_calibration(
    meta_oof,
    meta_test,
    tr["base_design"].values,
    te["base_design"].values,
    y_tr,
)

if r2_score(y_tr, cal_oof) > r2_score(y_tr, meta_oof) + 0.0002:
    stage_oof, stage_test = cal_oof, cal_test
    print(f"Calibration accepted: OOF R2={r2_score(y_tr, stage_oof):.5f}")
else:
    stage_oof, stage_test = meta_oof, meta_test
    print("Calibration rejected: meta blend kept")


KNN_NUM_FEATURES = [
    "L",
    "t",
    "h",
    "b",
    "A",
    "h_t",
    "b_t",
    "L_h",
    "L_t",
    "h_b",
    "A_t2",
    "lam_c",
    "lam_led",
    "KL_r",
    "Pcrl_Py",
    "Pne_Py",
    "DSM_global",
    "DSM_local",
    "base_design",
    "base_global",
    "base_local",
    "DSM_gap",
    "lam_ratio",
    "Pne_DSM_ratio",
    "Pcrl_DSM_ratio",
]


def knn_matrix(train_df: pd.DataFrame, apply_df: pd.DataFrame, onehot_cols: list[str] | None = None):
    Xn = apply_df[KNN_NUM_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    oh = pd.get_dummies(apply_df[ONEHOT_CATS].astype(str), prefix=ONEHOT_CATS, dtype=float)
    if onehot_cols is None:
        onehot_cols = list(pd.get_dummies(train_df[ONEHOT_CATS].astype(str), prefix=ONEHOT_CATS, dtype=float).columns)
    for col in onehot_cols:
        if col not in oh.columns:
            oh[col] = 0.0
    return pd.concat([Xn.reset_index(drop=True), oh[onehot_cols].reset_index(drop=True)], axis=1).values, onehot_cols


def residual_knn_stage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: np.ndarray,
    pred_oof: np.ndarray,
    pred_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X_all, oh_cols = knn_matrix(train_df, train_df)
    X_test, _ = knn_matrix(train_df, test_df, oh_cols)
    log_res = np.log(y.clip(PRED_MIN, PRED_MAX) / clip_pred(pred_oof))
    log_res = np.clip(log_res, -0.45, 0.45)

    best = None
    kf = KFold(n_splits=8, shuffle=True, random_state=RANDOM_STATE)
    for k in [3, 5, 7, 11, 15, 21]:
        res_oof = np.zeros_like(y, dtype=float)
        for idx_fit, idx_val in kf.split(X_all):
            scaler = StandardScaler()
            Xf = scaler.fit_transform(X_all[idx_fit])
            Xv = scaler.transform(X_all[idx_val])
            knn = KNeighborsRegressor(n_neighbors=min(k, len(idx_fit)), weights="distance", p=2)
            knn.fit(Xf, log_res[idx_fit])
            res_oof[idx_val] = knn.predict(Xv)

        for alpha in [0.20, 0.35, 0.50, 0.65, 0.80, 1.00]:
            cand = clip_pred(pred_oof * np.exp(alpha * np.clip(res_oof, -0.35, 0.35)))
            score = r2_score(y, cand)
            if best is None or score > best["score"]:
                best = {"score": score, "k": k, "alpha": alpha, "res_oof": res_oof}

    assert best is not None
    scaler = StandardScaler()
    Xf = scaler.fit_transform(X_all)
    Xt = scaler.transform(X_test)
    knn = KNeighborsRegressor(n_neighbors=min(best["k"], len(X_all)), weights="distance", p=2)
    knn.fit(Xf, log_res)
    res_test = knn.predict(Xt)

    corrected_oof = clip_pred(pred_oof * np.exp(best["alpha"] * np.clip(best["res_oof"], -0.35, 0.35)))
    corrected_test = clip_pred(pred_test * np.exp(best["alpha"] * np.clip(res_test, -0.35, 0.35)))
    print(
        f"Residual KNN best: k={best['k']} alpha={best['alpha']:.2f} "
        f"OOF_R2={best['score']:.5f}"
    )
    return corrected_oof, corrected_test


knn_oof, knn_test = residual_knn_stage(tr, te, y_tr, stage_oof, stage_test)
if r2_score(y_tr, knn_oof) > r2_score(y_tr, stage_oof) + 0.0005:
    final_oof, final_test = knn_oof, knn_test
    print(f"Residual KNN accepted: OOF R2={r2_score(y_tr, final_oof):.5f}")
else:
    final_oof, final_test = stage_oof, stage_test
    print("Residual KNN rejected: calibrated/meta prediction kept")


# ---------------------------------------------------------------------------
# Metrics, plots, SHAP, save
# ---------------------------------------------------------------------------


def report(label: str, yt: np.ndarray, yp: np.ndarray, data: pd.DataFrame | None = None) -> float:
    yp = clip_pred(yp)
    r2 = r2_score(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    mape = np.mean(np.abs((yt - yp) / (np.abs(yt) + EPS))) * 100.0
    ratio = yt / (yp + EPS)
    mu = float(np.mean(ratio))
    cov = float(np.std(ratio) / (mu + EPS))

    print(f"\n{'=' * 72}")
    print(label)
    print(f"{'=' * 72}")
    print(f"R2(Pt/Py)  = {r2:.6f}")
    print(f"RMSE       = {rmse:.6f}")
    print(f"MAE        = {mae:.6f}")
    print(f"MAPE       = {mape:.3f} %")
    print(f"Mean Pt/Pp = {mu:.4f}")
    print(f"COV        = {cov:.4f}")
    print(f"Target R2 >= 0.985: {'YES' if r2 >= 0.985 else 'NO'}")

    if data is not None:
        py = data["Py"].values
        r2_kN = r2_score(yt * py, yp * py)
        rmse_kN = np.sqrt(mean_squared_error(yt * py, yp * py))
        print(f"R2(Pt kN)  = {r2_kN:.6f}")
        print(f"RMSE(kN)   = {rmse_kN:.3f}")
        err = np.abs((yt - yp) / (np.abs(yt) + EPS)) * 100.0
        df_err = pd.DataFrame({"FM": data["FM"].values, "SG": data["SG"].values, "err_%": err})
        print("\nError by FM:")
        print(df_err.groupby("FM")["err_%"].agg(["mean", "median", "max", "count"]).round(3).to_string())
        print("\nError by SG:")
        print(df_err.groupby("SG")["err_%"].agg(["mean", "median", "max", "count"]).round(3).to_string())
    return r2


report("TRAIN OOF - v12", y_tr, final_oof, tr)
r2_test = report("TEST RESULT - v12", y_te, final_test, te)

pred_df = pd.DataFrame(
    {
        "TestLabel": te["TestLabel"].values,
        "Family": te["Family"].values,
        "SectionType": te["SectionType"].values,
        "Sections": te["Sections"].values,
        "BC": te["BC"].values,
        "FM": te["FM"].values,
        "SG": te["SG"].values,
        "PtPy_actual": y_te,
        "PtPy_pred": final_test,
        "Pt_actual_kN": y_te * te["Py"].values,
        "Pt_pred_kN": final_test * te["Py"].values,
        "base_design": te["base_design"].values,
        "global_pred": global_test,
        "sg_pred": sg_test,
        "fm_pred": fm_test,
        "err_%": np.abs((y_te - final_test) / (np.abs(y_te) + EPS)) * 100.0,
    }
)
pred_df.to_csv("v12_predictions.csv", index=False)
print("\nv12_predictions.csv saved")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = {
    "G1_O2C": "#01696f",
    "G2_O2U": "#d4380d",
    "G3_CUC": "#7a39bb",
    "G4_HC": "#0958d9",
    "G5a_C2C_F": "#ff4d4f",
    "G5b_C2C_Other": "#389e0d",
    "G5c_C2C_LF": "#95de64",
    "G6_Open": "#cf1322",
    "G7a_Closed": "#fa8c16",
    "G7b_Rest": "gray",
}
c_arr = [colors.get(x, "gray") for x in te["SG"].values]

ax = axes[0]
ax.scatter(y_te, final_test, c=c_arr, alpha=0.68, s=28)
lim = [0, max(float(np.max(y_te)), float(np.max(final_test))) * 1.06]
ax.plot(lim, lim, "k--", lw=1.4)
ax.fill_between(lim, [x * 0.98 for x in lim], [x * 1.02 for x in lim], alpha=0.10, color="blue")
ax.fill_between(lim, [x * 0.95 for x in lim], [x * 1.05 for x in lim], alpha=0.07, color="green")
for g, col in colors.items():
    cnt = int((te["SG"].values == g).sum())
    if cnt:
        ax.scatter([], [], c=col, s=28, label=f"{g} (n={cnt})")
ax.set_xlabel("Experimental Pt/Py")
ax.set_ylabel("Predicted Pt/Py")
ax.set_title(f"v12 test R2={r2_test:.4f}")
ax.grid(alpha=0.25)
ax.legend(fontsize=7, loc="lower right")

ax = axes[1]
res_pct = (y_te - final_test) / (np.abs(y_te) + EPS) * 100.0
ax.scatter(final_test, res_pct, c=c_arr, alpha=0.68, s=28)
for v, ls in [(0, "-"), (2, ":"), (-2, ":"), (5, "--"), (-5, "--")]:
    ax.axhline(v, color="k" if v == 0 else "orange", ls=ls, lw=0.9)
ax.set_xlabel("Predicted Pt/Py")
ax.set_ylabel("Residual (%)")
ax.set_title("Residuals by section group")
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("v12_scatter.png", bbox_inches="tight")
plt.show()
print("v12_scatter.png saved")

if RUN_SHAP:
    print("\nSHAP diagnostic on final global feature space...")
    X_full, oh_cols = make_matrix(tr, tr, correction_target(tr, "base_design"), None)
    X_test_full, _ = make_matrix(tr, te, correction_target(tr, "base_design"), oh_cols)
    shap_model = XGBRegressor(
        n_estimators=2500,
        learning_rate=0.012,
        max_depth=5,
        subsample=0.86,
        colsample_bytree=0.76,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    shap_model.fit(X_full.values, correction_target(tr, "base_design"))
    sv = shap.TreeExplainer(shap_model).shap_values(X_test_full.values)
    imp = (
        pd.DataFrame({"Feature": X_full.columns, "SHAP": np.abs(sv).mean(axis=0)})
        .sort_values("SHAP", ascending=False)
        .reset_index(drop=True)
    )
    imp.to_csv("v12_shap_importance.csv", index=False)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_test_full, feature_names=X_full.columns, show=False, max_display=25)
    plt.tight_layout()
    plt.savefig("v12_shap.png", bbox_inches="tight")
    plt.show()
    print("v12_shap.png and v12_shap_importance.csv saved")
    print("\nTop 15 SHAP features:")
    print(imp.head(15).to_string(index=False))

summary = {
    "validation_mode": VALIDATION_MODE,
    "train_rows": len(tr),
    "test_rows": len(te),
    "family_overlap": family_overlap,
    "global_oof_r2": float(global_r2),
    "final_oof_r2": float(r2_score(y_tr, final_oof)),
    "final_test_r2": float(r2_test),
    "target_0985_passed": bool(r2_test >= 0.985),
}
pd.Series(summary).to_csv("v12_summary.csv")
print("\nv12_summary.csv saved")

print("\n" + "=" * 72)
print("v12 COMPLETE")
print(f"Validation mode: {VALIDATION_MODE}")
print(f"Final Test R2(Pt/Py): {r2_test:.6f}")
print(f"Target 0.985 reached: {'YES' if r2_test >= 0.985 else 'NO'}")
print("=" * 72)
