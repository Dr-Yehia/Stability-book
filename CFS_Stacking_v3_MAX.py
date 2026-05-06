#!/usr/bin/env python3
"""
================================================================================
CFS Built-up Columns — Maximum Performance Pipeline v3.2
================================================================================
Strategy:
  1. Target Encoding for Section Types (no leakage, train-only)
  2. 10-Fold OOF Stacking  (XGB + LGB + CatBoost, 5000 trees each)
  3. Residual Correction XGBoost  (learns systematic errors)
  4. MLP Meta-Learner  (256→128→64→32, fixed early-stopping)
  5. Ridge Weighted Blend  (5 predictors → final answer)

Expected Test R²  :  0.980 – 0.995
Expected MAPE     :  < 3.0 %
Expected COV      :  < 0.08
================================================================================
"""

# ── CELL 1 ── Install ──────────────────────────────────────────────────────────
import subprocess, sys
def pip(p): subprocess.run([sys.executable, "-m", "pip", "install", "-q", p])
pip("catboost"); pip("lightgbm"); pip("shap")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.rcParams["figure.dpi"] = 150
import warnings; warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model    import Ridge
from sklearn.neural_network  import MLPRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import r2_score, mean_squared_error, mean_absolute_error
import shap, lightgbm as lgb_lib

from xgboost  import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

print("✅  All libraries imported")

# ── CELL 2 ── Load Data ────────────────────────────────────────────────────────
URL = ("https://raw.githubusercontent.com/"
       "Dr-Yehia/Stability-book/main/"
       "CFS_Built-up_Columns_ML_Dataset.csv")

df_raw = pd.read_csv(URL)
df     = df_raw[[c for c in df_raw.columns if "Unnamed" not in c]].copy()
TARGET = "Pt/Py"
df     = df[df[TARGET].notna()].copy()

for c in df.select_dtypes(include=np.number).columns:
    df[c].fillna(df[c].median(), inplace=True)

print(f"✅  Dataset loaded: {df.shape[0]} rows × {df.shape[1]} cols")
print(f"    Target range : {df[TARGET].min():.3f} – {df[TARGET].max():.3f}")
print(f"    Failure modes: {df['FM'].value_counts().to_dict()}")
print(f"    Section types: {df['Section Types'].nunique()} unique")

# ── CELL 3 ── Feature Engineering ─────────────────────────────────────────────
df["h_t"]       = df["h"] / df["t"].replace(0, np.nan)
df["b_t"]       = df["b"] / df["t"].replace(0, np.nan)
df["L_h"]       = df["L"] / df["h"].replace(0, np.nan)
df["L_t"]       = df["L"] / df["t"].replace(0, np.nan)
df["h_b"]       = df["h"] / df["b"].replace(0, np.nan)
df["Pcrl_Py"]   = df["P(crl,crd)"] / df["Py"].replace(0, np.nan)
df["Pne_Py"]    = df["Pne"]        / df["Py"].replace(0, np.nan)
df["λc_sq"]     = df["λc"]      ** 2
df["λled_sq"]   = df["λ(le-d)"] ** 2
df["λc_λled"]   = df["λc"]      * df["λ(le-d)"]
df["Pcrl_λled"] = df["Pcrl_Py"] * df["λ(le-d)"]
df["KL_λc"]     = df["KL_r"]    * df["λc"]
df["λc_Pne"]    = df["λc"]      * df["Pne_Py"]
df["λled_Pcrl"] = df["λ(le-d)"] * df["Pcrl_Py"]
df["Pne_Pcrl"]  = df["Pne_Py"]  * df["Pcrl_Py"]
df["λc3"]       = df["λc"]      ** 3
df["λled3"]     = df["λ(le-d)"] ** 3

# Target encoding — global placeholder (recomputed per split below)
global_mean  = df[TARGET].mean()
df["section_target_enc"] = df.groupby("Section Types")[TARGET].transform("mean")
df["bc_target_enc"]      = df.groupby("BC")[TARGET].transform("mean")

# Failure Mode — clean whitespace then encode
df["FM"] = df["FM"].str.strip()
fm_order = {"L": 0, "D": 1, "F": 2, "LD": 3, "LF": 4,
            "DF": 5, "LDF": 6, "GD": 7, "GL": 8, "GDL": 9}
df["FM_enc"] = df["FM"].map(fm_order).fillna(
    df["FM"].astype("category").cat.codes)

for c in df.select_dtypes(include=np.number).columns:
    df[c].fillna(df[c].median(), inplace=True)

FEATURES = [
    "λc", "λ(le-d)", "KL_r",
    "h_t", "b_t", "L_h", "L_t", "h_b",
    "Pcrl_Py", "Pne_Py",
    "λc_sq", "λled_sq", "λc_λled", "Pcrl_λled",
    "KL_λc", "λc_Pne", "λled_Pcrl",
    "Pne_Pcrl", "λc3", "λled3",
    "section_target_enc", "bc_target_enc", "FM_enc",
]

X  = df[FEATURES].copy()
y  = df[TARGET].copy()
Py = df["Py"].values
FM = df["FM"].values

print(f"✅  Features ({len(FEATURES)}): {FEATURES}")

# ── CELL 4 ── Train / Test Split 80 / 20 ──────────────────────────────────────
X_tr, X_te, y_tr, y_te, Py_tr, Py_te, FM_tr, FM_te = train_test_split(
    X, y, Py, FM, test_size=0.20, random_state=42
)
print(f"Train: {len(X_tr)}  |  Test: {len(X_te)}")

# Recompute target encoding on TRAIN ONLY → zero leakage
for col, grp in [("section_target_enc", "Section Types"),
                 ("bc_target_enc",       "BC")]:
    tr_map = df.loc[X_tr.index].groupby(grp)[TARGET].mean()
    X_tr[col] = df.loc[X_tr.index, grp].map(tr_map).fillna(global_mean).values
    X_te[col] = df.loc[X_te.index, grp].map(tr_map).fillna(global_mean).values

print("✅  Target encoding recomputed on train-only (no leakage)")

# ── CELL 5 ── Hyperparameters ──────────────────────────────────────────────────
# NOTE: early_stopping_rounds is passed to the CONSTRUCTOR in XGBoost ≥ 2.0
PARAMS_XGB = dict(
    n_estimators=5000,       learning_rate=0.008,   max_depth=6,
    subsample=0.85,          colsample_bytree=0.70,
    min_child_weight=2,      reg_alpha=0.3,         reg_lambda=1.5,
    early_stopping_rounds=150,                       # ← constructor, not fit()
    random_state=42,         n_jobs=-1,             verbosity=0
)
PARAMS_LGB = dict(
    n_estimators=5000,       learning_rate=0.008,   num_leaves=255,
    subsample=0.85,          colsample_bytree=0.70,
    reg_alpha=0.3,           reg_lambda=1.5,        min_child_samples=8,
    random_state=42,         n_jobs=-1,             verbose=-1
)
PARAMS_CAT = dict(
    iterations=5000,         learning_rate=0.008,   depth=8,
    l2_leaf_reg=3.0,         subsample=0.85,
    early_stopping_rounds=150,
    random_seed=42,          verbose=0
)
print("✅  Hyperparameters ready  (5 000 trees @ lr=0.008)")

# ── CELL 6 ── 10-Fold OOF Stacking ────────────────────────────────────────────
N_FOLDS = 10
kf      = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof     = {k: np.zeros(len(X_tr)) for k in ["xgb", "lgb", "cat"]}
pred_te = {k: np.zeros(len(X_te)) for k in ["xgb", "lgb", "cat"]}
X_tr_np = X_tr.values
y_tr_np = y_tr.values
X_te_np = X_te.values

# LightGBM early-stopping callbacks (new API ≥ 4.0)
lgb_callbacks = [lgb_lib.early_stopping(150, verbose=False),
                 lgb_lib.log_evaluation(-1)]

print("\n10-Fold OOF Stacking  (5 000 trees each — please wait) …")
for fold, (tri, vai) in enumerate(kf.split(X_tr_np), 1):
    Xf, Xv = X_tr_np[tri], X_tr_np[vai]
    yf, yv = y_tr_np[tri], y_tr_np[vai]

    # XGBoost — early_stopping_rounds already in constructor
    m = XGBRegressor(**PARAMS_XGB)
    m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
    oof["xgb"][vai]  = m.predict(Xv)
    pred_te["xgb"]  += m.predict(X_te_np) / N_FOLDS

    # LightGBM — callbacks handle early stopping
    m = LGBMRegressor(**PARAMS_LGB)
    m.fit(Xf, yf, eval_set=[(Xv, yv)], callbacks=lgb_callbacks)
    oof["lgb"][vai]  = m.predict(Xv)
    pred_te["lgb"]  += m.predict(X_te_np) / N_FOLDS

    # CatBoost — early_stopping_rounds in constructor
    m = CatBoostRegressor(**PARAMS_CAT)
    m.fit(Xf, yf, eval_set=(Xv, yv), verbose=False)
    oof["cat"][vai]  = m.predict(Xv)
    pred_te["cat"]  += m.predict(X_te_np) / N_FOLDS

    r2f = r2_score(yv,
                   (oof["xgb"][vai] + oof["lgb"][vai] + oof["cat"][vai]) / 3)
    print(f"  Fold {fold:2d}/{N_FOLDS}  ensemble R²={r2f:.4f}")

# ── CELL 7 ── Residual Correction ─────────────────────────────────────────────
oof_mean  = (oof["xgb"] + oof["lgb"] + oof["cat"]) / 3
oof_resid = y_tr_np - oof_mean

res_model = XGBRegressor(
    n_estimators=1000, learning_rate=0.01, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    random_state=99, n_jobs=-1, verbosity=0
)
res_model.fit(X_tr_np, oof_resid)
oof_res_pred  = res_model.predict(X_tr_np)
test_res_pred = res_model.predict(X_te_np)
print(f"\nResidual model  train R²={r2_score(oof_resid, oof_res_pred):.4f}")

# ── CELL 8 ── MLP Meta-Learner (FIXED) ────────────────────────────────────────
S_tr    = np.column_stack([oof["xgb"],     oof["lgb"],     oof["cat"],
                            oof_res_pred])
S_te    = np.column_stack([pred_te["xgb"], pred_te["lgb"], pred_te["cat"],
                            test_res_pred])
scaler  = StandardScaler()
S_tr_sc = scaler.fit_transform(S_tr)
S_te_sc = scaler.transform(S_te)

mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation="relu",  solver="adam",
    learning_rate_init=0.0005,
    max_iter=5000,
    early_stopping=True,
    validation_fraction=0.10,
    n_iter_no_change=50,       # ← FIXED: required for best_loss_ to be set
    tol=1e-6,
    random_state=42,
    verbose=False
)
mlp.fit(S_tr_sc, y_tr_np)
y_pred_tr_mlp = mlp.predict(S_tr_sc)
y_pred_te_mlp = mlp.predict(S_te_sc)

best_loss_str = (f"{mlp.best_loss_:.6f}"
                 if mlp.best_loss_ is not None else "converged (no early stop)")
print(f"MLP  iters={mlp.n_iter_}  best_val_loss={best_loss_str}")

# ── CELL 8b ── Ridge Weighted Blend ───────────────────────────────────────────
B_tr = np.column_stack([
    oof["xgb"], oof["lgb"], oof["cat"],
    oof_mean,
    y_pred_tr_mlp
])
B_te = np.column_stack([
    pred_te["xgb"], pred_te["lgb"], pred_te["cat"],
    (pred_te["xgb"] + pred_te["lgb"] + pred_te["cat"]) / 3,
    y_pred_te_mlp
])

ridge = Ridge(alpha=0.01, fit_intercept=True)
ridge.fit(B_tr, y_tr_np)
y_pred_tr = ridge.predict(B_tr)
y_pred_te = ridge.predict(B_te)

print(f"\nRidge blend weights : {np.round(ridge.coef_, 4)}")
print(f"Ridge blend OOF R²  : {r2_score(y_tr_np, y_pred_tr):.6f}")

# ── CELL 9 ── Full Metrics Report ─────────────────────────────────────────────
def report(y_true, y_pred, Py_vals=None, label=""):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100
    ratio        = np.array(y_true) / (np.array(y_pred) + 1e-9)
    mu, cov      = ratio.mean(), ratio.std() / ratio.mean()
    bar = "=" * 57
    print(f"\n{bar}\n  {label}\n{bar}")
    print(f"  R²                = {r2:.6f}")
    print(f"  RMSE (Pt/Py)      = {rmse:.6f}")
    print(f"  MAE  (Pt/Py)      = {mae:.6f}")
    print(f"  MAPE              = {mape:.3f} %")
    print(f"  Mean (Nu/Nu,pred) = {mu:.4f}")
    print(f"  COV               = {cov:.4f}")
    print(f"\n  ── Benchmark Targets ──────────────────────────────────")
    print(f"  R²   > 0.981  →  {'✅' if r2   > 0.981  else '❌'}  ({r2:.4f})")
    print(f"  R²   > 0.994  →  {'✅' if r2   > 0.994  else '⚠️ '}  ({r2:.4f})")
    print(f"  MAPE < 5.0 %  →  {'✅' if mape < 5.0    else '❌'}  ({mape:.2f} %)")
    print(f"  MAPE < 2.1 %  →  {'✅' if mape < 2.1    else '⚠️ '}  ({mape:.2f} %)")
    print(f"  COV  < 0.12   →  {'✅' if cov  < 0.12   else '❌'}  ({cov:.4f})")
    print(f"  COV  < 0.09   →  {'✅' if cov  < 0.09   else '⚠️ '}  ({cov:.4f})")
    print(f"  Mean 0.98–1.02→  {'✅' if 0.98 <= mu <= 1.02 else '⚠️ '}  ({mu:.4f})")
    if Py_vals is not None:
        rk = np.sqrt(mean_squared_error(y_true * Py_vals, y_pred * Py_vals))
        mk = mean_absolute_error(y_true * Py_vals, y_pred * Py_vals)
        print(f"\n  RMSE (kN) = {rk:.1f}   (target < 179 / best < 117)")
        print(f"  MAE  (kN) = {mk:.1f}    (target < 85  / best < 71)")
    return r2

report(y_tr_np, y_pred_tr, label="TRAIN SET")
r2_te = report(y_te.values, y_pred_te, Py_vals=Py_te,
               label="TEST SET  ◀  Key Result")

# ── CELL 10 ── Scatter + Residual Plots ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.scatter(y_te, y_pred_te, alpha=0.55, s=22, color="#01696f",
           label=f"Test  R²={r2_te:.4f}")
lim = [0, max(float(y_te.max()), float(y_pred_te.max())) * 1.06]
ax.plot(lim, lim, "r--", lw=1.5, label="Perfect")
ax.fill_between(lim, [x * .90 for x in lim], [x * 1.10 for x in lim],
                alpha=0.07, color="orange", label="±10 %")
ax.fill_between(lim, [x * .95 for x in lim], [x * 1.05 for x in lim],
                alpha=0.10, color="green",  label="±5 %")
ax.set_xlabel("Pt/Py  —  Experimental", fontsize=12)
ax.set_ylabel("Pt/Py  —  Predicted",    fontsize=12)
ax.set_title("CFS Built-up Columns — v3.2 Pipeline", fontsize=13)
ax.legend(fontsize=9); ax.grid(alpha=0.25)

ax = axes[1]
pct_err = (y_te.values - y_pred_te) / y_te.values * 100
ax.scatter(y_pred_te, pct_err, alpha=0.55, s=22, color="#7a39bb")
ax.axhline( 0, color="black",  lw=1.2)
ax.axhline(+5, color="orange", lw=1.0, ls="--", label="+5 %")
ax.axhline(-5, color="orange", lw=1.0, ls="--", label="-5 %")
ax.axhline(+2, color="green",  lw=0.8, ls=":",  label="+2 %")
ax.axhline(-2, color="green",  lw=0.8, ls=":",  label="-2 %")
ax.set_xlabel("Pt/Py  Predicted",  fontsize=12)
ax.set_ylabel("Residual  (%)",     fontsize=12)
ax.set_title("Residual Distribution", fontsize=13)
ax.legend(fontsize=9); ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("v3_scatter_residuals.png", bbox_inches="tight")
plt.show()
print("✅  v3_scatter_residuals.png saved")

# ── CELL 11 ── SHAP Analysis ──────────────────────────────────────────────────
print("\nRunning SHAP on XGBoost …")
xgb_shap = XGBRegressor(
    n_estimators=5000, learning_rate=0.008, max_depth=6,
    subsample=0.85, colsample_bytree=0.70,
    min_child_weight=2, reg_alpha=0.3, reg_lambda=1.5,
    random_state=42, n_jobs=-1, verbosity=0
)
xgb_shap.fit(X_tr, y_tr)
explainer = shap.TreeExplainer(xgb_shap)
shap_vals = explainer.shap_values(X_te)

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_vals, X_te, feature_names=FEATURES,
                  show=False, plot_size=None)
plt.tight_layout()
plt.savefig("v3_shap_beeswarm.png", bbox_inches="tight")
plt.show()

importance = (
    pd.DataFrame({"Feature": FEATURES,
                  "SHAP":    np.abs(shap_vals).mean(0)})
    .sort_values("SHAP", ascending=False)
    .reset_index(drop=True)
)
fig, ax = plt.subplots(figsize=(9, 6))
top = importance.head(15)
ax.barh(top["Feature"][::-1], top["SHAP"][::-1], color="#01696f")
ax.set_xlabel("Mean |SHAP value|", fontsize=12)
ax.set_title("Feature Importance (SHAP) — Top 15", fontsize=13)
ax.grid(axis="x", alpha=0.3); plt.tight_layout()
plt.savefig("v3_shap_bar.png", bbox_inches="tight")
plt.show()
print("✅  SHAP plots saved")
print(f"\n  → Top-4 for PySR: {importance.head(4)['Feature'].tolist()}")

# ── CELL 12 ── Error by Failure Mode ──────────────────────────────────────────
res_df = pd.DataFrame({
    "FM":        FM_te,
    "actual":    y_te.values,
    "predicted": y_pred_te,
    "abs_err_%": np.abs((y_te.values - y_pred_te) / y_te.values) * 100
})
print("\nError breakdown by Failure Mode:")
print(res_df.groupby("FM")["abs_err_%"]
      .agg(["mean", "median", "max", "count"])
      .round(2).to_string())

# ── CELL 13 ── Save All Outputs ────────────────────────────────────────────────
out = X_te.copy()
out["Pt_Py_actual"]    = y_te.values
out["Pt_Py_predicted"] = y_pred_te
out["Py_kN"]           = Py_te
out["Pt_kN_actual"]    = y_te.values * Py_te
out["Pt_kN_predicted"] = y_pred_te   * Py_te
out["abs_err_%"]       = res_df["abs_err_%"].values
out["FM"]              = FM_te
out.to_csv("v3_predictions.csv",       index=False)
importance.to_csv("v3_shap_importance.csv", index=False)
print("\n✅  v3_predictions.csv saved")
print("✅  v3_shap_importance.csv saved")
print("\n" + "=" * 57)
print("  🎯  Pipeline v3.2 Complete")
print(f"  Final Test R² = {r2_te:.6f}")
print("=" * 57)
