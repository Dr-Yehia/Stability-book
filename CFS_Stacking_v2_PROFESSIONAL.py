# ============================================================
# CFS Built-up Columns — PROFESSIONAL VERSION v2
# Stacking + Optuna Tuning + Full Feature Engineering
# Target: R² > 0.994 | MAPE < 2.1% | COV < 0.09
# ============================================================

# ── CELL 1: Install ─────────────────────────────────────────
import subprocess, sys
def pip(p): subprocess.run([sys.executable,"-m","pip","install","-q",p])
pip("catboost"); pip("lightgbm"); pip("shap"); pip("optuna")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.rcParams["figure.dpi"]=150
import warnings; warnings.filterwarnings("ignore")
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model    import Ridge
from sklearn.metrics         import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing   import LabelEncoder
import shap

from xgboost  import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

print("✅ Libraries ready")

# ── CELL 2: Load Data ───────────────────────────────────────
URL = ("https://raw.githubusercontent.com/"
       "Dr-Yehia/Stability-book/main/"
       "CFS_Built-up_Columns_ML_Dataset.csv")

df_raw = pd.read_csv(URL)

# Keep only named columns (drop Unnamed)
cols_keep = [c for c in df_raw.columns if "Unnamed" not in c]
df = df_raw[cols_keep].copy()
print(f"✅ Loaded: {df.shape}  →  columns: {df.columns.tolist()}")

# ── CELL 3: Clean ───────────────────────────────────────────
TARGET = "Pt/Py"

# Drop rows where target is NaN
df = df[df[TARGET].notna()].copy()
print(f"After removing NaN target: {len(df)} rows")

# Fill numeric NaN with median
num_cols = df.select_dtypes(include=np.number).columns.tolist()
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

# Encode categoricals
for c in ["Section Types","Sections","BC","Failure Mode","FM"]:
    if c in df.columns:
        le = LabelEncoder()
        df[c+"_enc"] = le.fit_transform(df[c].astype(str))

print("✅ Cleaning done")

# ── CELL 4: Feature Engineering ─────────────────────────────
df["h_t"]       = df["h"]   / df["t"].replace(0, np.nan)
df["b_t"]       = df["b"]   / df["t"].replace(0, np.nan)
df["L_h"]       = df["L"]   / df["h"].replace(0, np.nan)
df["L_t"]       = df["L"]   / df["t"].replace(0, np.nan)
df["Pcrl_Py"]   = df["P(crl,crd)"] / df["Py"].replace(0, np.nan)
df["Pne_Py"]    = df["Pne"] / df["Py"].replace(0, np.nan)
df["Pt_kN"]     = df["Pt"]
df["λc_sq"]     = df["λc"]  ** 2
df["λled_sq"]   = df["λ(le-d)"] ** 2
df["λc_λled"]   = df["λc"]  * df["λ(le-d)"]
df["Pcrl_λled"] = df["Pcrl_Py"] * df["λ(le-d)"]
df["KL_λc"]     = df["KL_r"] * df["λc"]

for c in df.select_dtypes(include=np.number).columns:
    df[c] = df[c].fillna(df[c].median())

FEATURES = [
    "λc", "λ(le-d)", "KL_r",
    "h_t", "b_t", "L_h", "L_t",
    "Pcrl_Py", "Pne_Py",
    "λc_sq", "λled_sq", "λc_λled", "Pcrl_λled", "KL_λc",
    "Section Types_enc", "BC_enc", "FM_enc",
]

X = df[FEATURES].copy()
y = df[TARGET].copy()
Py_all = df["Py"].values

print(f"✅ Features ({len(FEATURES)}): {FEATURES}")
print(f"✅ Dataset: {len(X)} rows")

# ── CELL 5: Split 80/20 ─────────────────────────────────────
X_train, X_test, y_train, y_test, Py_train, Py_test = train_test_split(
    X, y, Py_all, test_size=0.20, random_state=42
)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── CELL 6: Optuna — Tune XGBoost ───────────────────────────
print("\nOptimizing XGBoost (150 trials)…")

def objective_xgb(trial):
    p = dict(
        n_estimators      = trial.suggest_int("n_est",    800, 3000),
        learning_rate     = trial.suggest_float("lr",     0.005, 0.05,  log=True),
        max_depth         = trial.suggest_int("depth",    4, 9),
        subsample         = trial.suggest_float("sub",    0.6, 1.0),
        colsample_bytree  = trial.suggest_float("col",    0.5, 1.0),
        min_child_weight  = trial.suggest_int("mcw",      1, 10),
        reg_alpha         = trial.suggest_float("alpha",  0.0, 2.0),
        reg_lambda        = trial.suggest_float("lambda", 0.5, 5.0),
        random_state=42, n_jobs=-1, verbosity=0
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_i, va_i in kf.split(X_train):
        m = XGBRegressor(**p)
        m.fit(X_train.iloc[tr_i], y_train.iloc[tr_i], verbose=False)
        scores.append(r2_score(y_train.iloc[va_i], m.predict(X_train.iloc[va_i])))
    return np.mean(scores)

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=150, show_progress_bar=False)
best_xgb = study_xgb.best_params
print(f"  XGB best R²={study_xgb.best_value:.5f}  params={best_xgb}")

# ── CELL 7: Optuna — Tune LightGBM ──────────────────────────
print("Optimizing LightGBM (150 trials)…")

def objective_lgb(trial):
    p = dict(
        n_estimators  = trial.suggest_int("n_est",      800, 3000),
        learning_rate = trial.suggest_float("lr",        0.005, 0.05, log=True),
        num_leaves    = trial.suggest_int("leaves",      31, 255),
        subsample     = trial.suggest_float("sub",       0.6, 1.0),
        colsample_bytree = trial.suggest_float("col",    0.5, 1.0),
        reg_alpha     = trial.suggest_float("alpha",     0.0, 2.0),
        reg_lambda    = trial.suggest_float("lambda",    0.5, 5.0),
        min_child_samples = trial.suggest_int("mcs",     5, 50),
        random_state=42, n_jobs=-1, verbose=-1
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_i, va_i in kf.split(X_train):
        m = LGBMRegressor(**p)
        m.fit(X_train.iloc[tr_i], y_train.iloc[tr_i])
        scores.append(r2_score(y_train.iloc[va_i], m.predict(X_train.iloc[va_i])))
    return np.mean(scores)

study_lgb = optuna.create_study(direction="maximize")
study_lgb.optimize(objective_lgb, n_trials=150, show_progress_bar=False)
best_lgb = study_lgb.best_params
print(f"  LGB best R²={study_lgb.best_value:.5f}  params={best_lgb}")

# ── CELL 8: Optuna — Tune CatBoost ──────────────────────────
print("Optimizing CatBoost (100 trials)…")

def objective_cat(trial):
    p = dict(
        iterations    = trial.suggest_int("iters",   800, 3000),
        learning_rate = trial.suggest_float("lr",     0.005, 0.05, log=True),
        depth         = trial.suggest_int("depth",    4, 10),
        l2_leaf_reg   = trial.suggest_float("l2",     1.0, 10.0),
        subsample     = trial.suggest_float("sub",    0.6, 1.0),
        random_seed=42, verbose=0
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_i, va_i in kf.split(X_train):
        m = CatBoostRegressor(**p)
        m.fit(X_train.iloc[tr_i], y_train.iloc[tr_i],
              eval_set=(X_train.iloc[va_i], y_train.iloc[va_i]),
              early_stopping_rounds=50, verbose=False)
        scores.append(r2_score(y_train.iloc[va_i], m.predict(X_train.iloc[va_i])))
    return np.mean(scores)

study_cat = optuna.create_study(direction="maximize")
study_cat.optimize(objective_cat, n_trials=100, show_progress_bar=False)
best_cat = study_cat.best_params
print(f"  CAT best R²={study_cat.best_value:.5f}  params={best_cat}")

# ── CELL 9: OOF Stacking with Best Params ───────────────────
N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))
test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))

X_tr_np = X_train.values
y_tr_np = y_train.values
X_te_np = X_test.values

print("\n10-Fold OOF Stacking with tuned params…")
for fold, (tri, vai) in enumerate(kf.split(X_tr_np), 1):
    Xtr, Xval = X_tr_np[tri], X_tr_np[vai]
    ytr, yval = y_tr_np[tri], y_tr_np[vai]

    xm = XGBRegressor(**best_xgb, random_state=42, n_jobs=-1, verbosity=0)
    xm.fit(Xtr, ytr, verbose=False)
    oof_xgb[vai] = xm.predict(Xval)
    test_xgb += xm.predict(X_te_np) / N_FOLDS

    cm = CatBoostRegressor(**best_cat, random_seed=42, verbose=0)
    cm.fit(Xtr, ytr, eval_set=(Xval,yval), early_stopping_rounds=50, verbose=False)
    oof_cat[vai] = cm.predict(Xval)
    test_cat += cm.predict(X_te_np) / N_FOLDS

    lm = LGBMRegressor(**best_lgb, random_state=42, n_jobs=-1, verbose=-1)
    lm.fit(Xtr, ytr)
    oof_lgb[vai] = lm.predict(Xval)
    test_lgb += lm.predict(X_te_np) / N_FOLDS

    print(f"  Fold {fold:2d}/{N_FOLDS} — XGB:{r2_score(yval,oof_xgb[vai]):.4f}  CAT:{r2_score(yval,oof_cat[vai]):.4f}  LGB:{r2_score(yval,oof_lgb[vai]):.4f}")

# ── CELL 10: Meta Model ─────────────────────────────────────
S_train = np.column_stack([oof_xgb, oof_cat, oof_lgb])
S_test  = np.column_stack([test_xgb, test_cat, test_lgb])

meta = Ridge(alpha=0.5)
meta.fit(S_train, y_tr_np)
y_pred_tr = meta.predict(S_train)
y_pred_te = meta.predict(S_test)
print(f"\nMeta weights: XGB={meta.coef_[0]:.3f}  CAT={meta.coef_[1]:.3f}  LGB={meta.coef_[2]:.3f}")

# ── CELL 11: Full Metrics ───────────────────────────────────
def full_metrics(y_true, y_pred, Py_vals=None, label=""):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    ratio = np.array(y_true) / (np.array(y_pred) + 1e-9)
    mean_r = ratio.mean();  cov_r = ratio.std() / ratio.mean()

    sep = "="*50
    print(f"\n{sep}\n {label}\n{sep}")
    print(f"  R²              = {r2:.6f}")
    print(f"  RMSE (Pt/Py)    = {rmse:.6f}")
    print(f"  MAE  (Pt/Py)    = {mae:.6f}")
    print(f"  MAPE            = {mape:.3f}%")
    print(f"  Mean(Nu/Nu,pred)= {mean_r:.4f}")
    print(f"  COV             = {cov_r:.4f}")

    print(f"\n  Benchmarks vs image targets:")
    print(f"  R²   > 0.994 → {'✅' if r2>0.994 else '❌'} ({r2:.4f})")
    print(f"  MAPE < 2.1%  → {'✅' if mape<2.1  else '❌'} ({mape:.2f}%)")
    print(f"  COV  < 0.09  → {'✅' if cov_r<0.09 else '❌'} ({cov_r:.4f})")
    print(f"  Mean 0.98-1.02→ {'✅' if 0.98<=mean_r<=1.02 else '❌'} ({mean_r:.4f})")

    if Py_vals is not None:
        rmse_kn = np.sqrt(mean_squared_error(y_true*Py_vals, y_pred*Py_vals))
        mae_kn  = mean_absolute_error(y_true*Py_vals, y_pred*Py_vals)
        print(f"\n  RMSE (kN)       = {rmse_kn:.2f} kN  (target <117)")
        print(f"  MAE  (kN)       = {mae_kn:.2f} kN   (target <71)")
    return r2

full_metrics(y_tr_np, y_pred_tr, label="TRAIN SET")
r2_te = full_metrics(y_test.values, y_pred_te, Py_vals=Py_test, label="TEST SET ← النتيجة الحقيقية")

# ── CELL 12: Scatter + Residuals ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
ax1.scatter(y_test, y_pred_te, alpha=0.5, s=20, color="#01696f", label=f"Test  R²={r2_te:.4f}")
lim = [0, max(y_test.max(), y_pred_te.max())*1.05]
ax1.plot(lim, lim, "r--", lw=1.5, label="Perfect")
ax1.fill_between(lim, [x*0.95 for x in lim], [x*1.05 for x in lim], alpha=0.08, color="orange", label="±5%")
ax1.set_xlabel("Pt/Py  Experimental", fontsize=12)
ax1.set_ylabel("Pt/Py  Predicted", fontsize=12)
ax1.set_title("Stacking Ensemble — Test Set", fontsize=13)
ax1.legend(); ax1.grid(alpha=0.3)

ax2 = axes[1]
resid = (y_test.values - y_pred_te) / y_test.values * 100
ax2.scatter(y_pred_te, resid, alpha=0.5, s=20, color="#a12c7b")
ax2.axhline(0, color="black", lw=1)
ax2.axhline(+5, color="orange", lw=1, ls="--", label="+5%")
ax2.axhline(-5, color="orange", lw=1, ls="--", label="-5%")
ax2.set_xlabel("Pt/Py Predicted", fontsize=12)
ax2.set_ylabel("Residual %", fontsize=12)
ax2.set_title("Residual Plot", fontsize=13)
ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("scatter_residuals.png", bbox_inches="tight")
plt.show()
print("✅ scatter_residuals.png saved")

# ── CELL 13: SHAP ───────────────────────────────────────────
print("\nSHAP analysis…")
xgb_full = XGBRegressor(**best_xgb, random_state=42, n_jobs=-1, verbosity=0)
xgb_full.fit(X_train, y_train)

explainer = shap.TreeExplainer(xgb_full)
shap_vals = explainer.shap_values(X_test)

plt.figure(figsize=(10,7))
shap.summary_plot(shap_vals, X_test, feature_names=FEATURES, show=False, plot_size=None)
plt.tight_layout()
plt.savefig("shap_summary.png", bbox_inches="tight")
plt.show()
print("✅ shap_summary.png saved")

# ── CELL 14: SHAP Bar ───────────────────────────────────────
importance = pd.DataFrame({"Feature": FEATURES, "SHAP_mean": np.abs(shap_vals).mean(axis=0)}).sort_values("SHAP_mean", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(9, 6))
top = importance.head(12)
ax.barh(top["Feature"][::-1], top["SHAP_mean"][::-1], color="#01696f")
ax.set_xlabel("Mean |SHAP value|", fontsize=12)
ax.set_title("Top-12 Feature Importance (SHAP)", fontsize=13)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("shap_bar.png", bbox_inches="tight")
plt.show()

top4 = importance.head(4)["Feature"].tolist()
print(f"\n→ Top-4 for PySR Distillation: {top4}")
print("\n🎯 v2 Pipeline Complete")

# ── CELL 15: Save ───────────────────────────────────────────
out = X_test.copy()
out["Pt_Py_actual"] = y_test.values
out["Pt_Py_predicted"] = y_pred_te
out["error_pct"] = np.abs((out["Pt_Py_actual"]-out["Pt_Py_predicted"]) / out["Pt_Py_actual"]) * 100
out.to_csv("predictions_v2.csv", index=False)
print("✅ predictions_v2.csv saved")
