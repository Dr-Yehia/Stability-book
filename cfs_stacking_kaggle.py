
# ============================================================
# CFS Built-up Columns — Stacking Ensemble + SHAP
# Kaggle Notebook — Copy & Run As-Is
# Target: R² > 0.994 on test set
# ============================================================

# ── CELL 1: Install & Import ────────────────────────────────
import subprocess, sys

def pip(pkg):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])

pip("catboost")
pip("lightgbm")
pip("shap")
pip("optuna")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 150
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import (r2_score, mean_squared_error,
                              mean_absolute_error)
import shap

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

print("✅ All libraries imported")

# ── CELL 2: Load Data from GitHub ───────────────────────────
URL = ("https://raw.githubusercontent.com/"
       "Dr-Yehia/Stability-book/main/"
       "CFS_Built-up_Columns_ML_Dataset.csv")

df = pd.read_csv(URL)
print(f"✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} cols")
print(df.columns.tolist())

# ── CELL 3: Preview & Basic Stats ───────────────────────────
print(df.head(3).to_string())
print("\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("\nTarget stats (Pt/Py):")
print(df["Pt/Py"].describe())

# ── CELL 4: Feature Engineering ─────────────────────────────
# Compute dimensionless ratios
df["h_t"]       = df["h"] / df["t"]
df["b_t"]       = df["b"] / df["t"]
df["Pcrl_Py"]   = df["P(crl,crd)"] / df["Py"]
df["Pne_Py"]    = df["Pne"] / df["Py"]

# One-hot encode categoricals
df = pd.get_dummies(df, columns=["Section Types", "BC"], drop_first=False)

# Define feature columns
base_features = ["λc", "λ(le-d)", "KL_r",
                 "h_t", "b_t", "Pcrl_Py", "Pne_Py"]

cat_cols = [c for c in df.columns
            if c.startswith("Section Types_") or c.startswith("BC_")]

FEATURES = base_features + cat_cols
TARGET   = "Pt/Py"

X = df[FEATURES].copy()
y = df[TARGET].copy()

# Drop rows with NaN in features or target
mask = X.notna().all(axis=1) & y.notna()
X, y = X[mask], y[mask]

print(f"\n✅ Features ({len(FEATURES)}): {FEATURES}")
print(f"✅ Clean dataset: {len(X)} rows")

# ── CELL 5: Train / Test Split (80/20) ─────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── CELL 6: Base Models ─────────────────────────────────────
xgb = XGBRegressor(
    n_estimators=1500, learning_rate=0.02, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.5,
    random_state=42, n_jobs=-1, verbosity=0
)

cat = CatBoostRegressor(
    iterations=1500, learning_rate=0.02, depth=6,
    l2_leaf_reg=3, subsample=0.8,
    random_seed=42, verbose=0
)

lgb = LGBMRegressor(
    n_estimators=1500, learning_rate=0.02, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.5,
    random_state=42, n_jobs=-1, verbose=-1
)

# ── CELL 7: Out-of-Fold Stacking ────────────────────────────
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))

test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))

X_train_np = X_train.values
y_train_np = y_train.values
X_test_np  = X_test.values

print("Training base models with OOF stacking…")
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_np), 1):
    Xtr, Xval = X_train_np[tr_idx], X_train_np[val_idx]
    ytr, yval = y_train_np[tr_idx], y_train_np[val_idx]

    # XGBoost
    xgb.fit(Xtr, ytr, eval_set=[(Xval, yval)],
            verbose=False)
    oof_xgb[val_idx] = xgb.predict(Xval)
    test_xgb += xgb.predict(X_test_np) / N_FOLDS

    # CatBoost
    cat.fit(Xtr, ytr, eval_set=(Xval, yval),
            early_stopping_rounds=100, verbose=False)
    oof_cat[val_idx] = cat.predict(Xval)
    test_cat += cat.predict(X_test_np) / N_FOLDS

    # LightGBM
    lgb.fit(Xtr, ytr,
            eval_set=[(Xval, yval)])
    oof_lgb[val_idx] = lgb.predict(Xval)
    test_lgb += lgb.predict(X_test_np) / N_FOLDS

    print(f"  Fold {fold} done")

# ── CELL 8: Meta Model (Ridge) ──────────────────────────────
S_train = np.column_stack([oof_xgb, oof_cat, oof_lgb])
S_test  = np.column_stack([test_xgb, test_cat, test_lgb])

meta = Ridge(alpha=1.0)
meta.fit(S_train, y_train_np)

y_pred_train = meta.predict(S_train)
y_pred_test  = meta.predict(S_test)

# ── CELL 9: Metrics ─────────────────────────────────────────
def metrics(y_true, y_pred, Py_vals=None, label=""):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    ratio = y_true / y_pred
    mean_r = ratio.mean()
    cov_r  = ratio.std() / ratio.mean()

    print(f"\n{'='*45}")
    print(f" {label}")
    print(f"{'='*45}")
    print(f"  R²          = {r2:.6f}")
    print(f"  RMSE (Pt/Py)= {rmse:.6f}")
    print(f"  MAE  (Pt/Py)= {mae:.6f}")
    print(f"  MAPE        = {mape:.2f}%")
    print(f"  Mean(Nu/pred)= {mean_r:.4f}")
    print(f"  COV          = {cov_r:.4f}")

    # Convert to kN if Py available
    if Py_vals is not None:
        rmse_kn = np.sqrt(mean_squared_error(
            y_true * Py_vals, y_pred * Py_vals))
        mae_kn  = mean_absolute_error(
            y_true * Py_vals, y_pred * Py_vals)
        print(f"  RMSE (kN)   = {rmse_kn:.1f} kN")
        print(f"  MAE  (kN)   = {mae_kn:.1f} kN")
    return r2

Py_test = df.loc[X_test.index, "Py"].values

r2_tr = metrics(y_train_np, y_pred_train, label="TRAIN SET")
r2_te = metrics(y_test.values, y_pred_test,
                Py_vals=Py_test, label="TEST SET ← الأهم")

# ── CELL 10: Scatter Plot ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, y_pred_test, alpha=0.5, s=20,
           color="#01696f", label=f"Test (R²={r2_te:.4f})")
ax.plot([0, 1.2], [0, 1.2], "r--", lw=1.5, label="Perfect")
ax.set_xlabel("Pt/Py  (Experimental)", fontsize=13)
ax.set_ylabel("Pt/Py  (Predicted)",    fontsize=13)
ax.set_title("Stacking Ensemble — CFS Built-up Columns", fontsize=14)
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("scatter_test.png")
plt.show()
print("✅ scatter_test.png saved")

# ── CELL 11: SHAP Feature Importance ───────────────────────
print("\nComputing SHAP values (XGBoost)…")
xgb_final = XGBRegressor(
    n_estimators=1500, learning_rate=0.02, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.5,
    random_state=42, n_jobs=-1, verbosity=0
)
xgb_final.fit(X_train, y_train)

explainer = shap.TreeExplainer(xgb_final)
shap_vals  = explainer.shap_values(X_test)

fig2, ax2 = plt.subplots(figsize=(9, 6))
shap.summary_plot(shap_vals, X_test,
                  feature_names=FEATURES,
                  show=False, plot_size=None)
plt.tight_layout()
plt.savefig("shap_summary.png")
plt.show()
print("✅ shap_summary.png saved")

# ── CELL 12: Feature Importance Ranking ────────────────────
importance = pd.DataFrame({
    "Feature"   : FEATURES,
    "SHAP_mean" : np.abs(shap_vals).mean(axis=0)
}).sort_values("SHAP_mean", ascending=False)

print("\n Feature Importance (SHAP):")
print(importance.to_string(index=False))

# Top-4 features for PySR next step
top4 = importance.head(4)["Feature"].tolist()
print(f"\n→ Top-4 for PySR: {top4}")

# ── CELL 13: Save Predictions ───────────────────────────────
results = X_test.copy()
results["Pt_Py_actual"]    = y_test.values
results["Pt_Py_predicted"] = y_pred_test
results["error_pct"] = np.abs(
    (results["Pt_Py_actual"] - results["Pt_Py_predicted"])
    / results["Pt_Py_actual"]) * 100

results.to_csv("predictions_test.csv", index=False)
print("\n✅ predictions_test.csv saved")
print("\n🎯 Pipeline complete — ready for PySR distillation next")
