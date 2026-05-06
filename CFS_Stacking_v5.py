#!/usr/bin/env python3
"""
CFS Built-up Columns — v5.0 — ROOT-CAUSE FIX
Key changes:
  1. Target: log1p(Pt) directly — Py is a feature
  2. StratifiedKFold by Failure Mode
  3. 5-Fold for more training data per fold
  4. Simpler blend (no MLP overhead)
Expected: R²(Pt) > 0.99, R²(Pt/Py) > 0.98
"""
import subprocess, sys
def pip(p): subprocess.run([sys.executable, "-m", "pip", "install", "-q", p])
pip("catboost"); pip("lightgbm"); pip("shap")

import numpy as np, pandas as pd, matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["figure.dpi"] = 150
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap, lightgbm as lgb_lib
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import xgboost as _xgb
_xgb_major = int(_xgb.__version__.split('.')[0])
print("✅  All libraries imported")

# ── DATA ──────────────────────────────────────────────────────────────────────
URL = ("https://raw.githubusercontent.com/"
       "Dr-Yehia/Stability-book/main/"
       "CFS_Built-up_Columns_ML_Dataset.csv")
df = pd.read_csv(URL)
df = df[[c for c in df.columns if "Unnamed" not in c]].copy()
df = df[df["Pt/Py"].notna()].copy()
for c in df.select_dtypes(include=np.number).columns:
    df[c].fillna(df[c].median(), inplace=True)
df["FM"] = df["FM"].str.strip()
print(f"✅  Loaded: {len(df)} rows, {df['Section Types'].nunique()} section types")

# ── TARGET: log1p(Pt) ────────────────────────────────────────────────────────
# Pt in kN — predict this directly, derive Pt/Py after
df["Pt"] = df["Pt/Py"] * df["Py"]  # reconstruct if needed
TARGET_RAW = "Pt"
df["log_Pt"] = np.log1p(df[TARGET_RAW])
TARGET = "log_Pt"

# ── FEATURES ──────────────────────────────────────────────────────────────────
df["h_t"] = df["h"] / df["t"].replace(0, np.nan)
df["b_t"] = df["b"] / df["t"].replace(0, np.nan)
df["L_h"] = df["L"] / df["h"].replace(0, np.nan)
df["L_t"] = df["L"] / df["t"].replace(0, np.nan)
df["h_b"] = df["h"] / df["b"].replace(0, np.nan)
df["Pcrl_Py"] = df["P(crl,crd)"] / df["Py"].replace(0, np.nan)
df["Pne_Py"]  = df["Pne"] / df["Py"].replace(0, np.nan)
df["λc_sq"]    = df["λc"] ** 2
df["λled_sq"]  = df["λ(le-d)"] ** 2
df["λc_λled"]  = df["λc"] * df["λ(le-d)"]
df["Pcrl_λled"]= df["Pcrl_Py"] * df["λ(le-d)"]
df["KL_λc"]    = df["KL_r"] * df["λc"]
df["λc_Pne"]   = df["λc"] * df["Pne_Py"]
df["λled_Pcrl"]= df["λ(le-d)"] * df["Pcrl_Py"]
df["Pne_Pcrl"] = df["Pne_Py"] * df["Pcrl_Py"]
df["λc3"]      = df["λc"] ** 3
df["λled3"]    = df["λ(le-d)"] ** 3
df["inv_λc"]   = 1.0 / df["λc"].replace(0, np.nan)
df["λc_inv_led"] = df["λc"] / df["λ(le-d)"].replace(0, np.nan)
df["log_Py"]   = np.log1p(df["Py"])
df["log_Pne"]  = np.log1p(df["Pne"])
df["log_Pcrl"] = np.log1p(df["P(crl,crd)"])

# Encode categoricals
le_fm = LabelEncoder(); df["FM_enc"] = le_fm.fit_transform(df["FM"])
le_bc = LabelEncoder(); df["BC_enc"] = le_bc.fit_transform(df["BC"].astype(str))
le_st = LabelEncoder(); df["ST_enc"] = le_st.fit_transform(df["Section Types"].astype(str))

# Target encoding (will recompute per split)
global_mean = df[TARGET].mean()
df["section_te"] = df.groupby("Section Types")[TARGET].transform("mean")
df["bc_te"]      = df.groupby("BC")[TARGET].transform("mean")
df["fm_te"]      = df.groupby("FM")[TARGET].transform("mean")

for c in df.select_dtypes(include=np.number).columns:
    df[c].fillna(df[c].median(), inplace=True)

FEATURES = [
    # Raw strength features (KEY — Py as feature!)
    "log_Py", "log_Pne", "log_Pcrl",
    # Slenderness
    "λc", "λ(le-d)", "KL_r",
    # Geometric ratios
    "h_t", "b_t", "L_h", "L_t", "h_b",
    # Strength ratios
    "Pcrl_Py", "Pne_Py",
    # Interactions
    "λc_sq", "λled_sq", "λc_λled", "Pcrl_λled",
    "KL_λc", "λc_Pne", "λled_Pcrl", "Pne_Pcrl",
    "λc3", "λled3", "inv_λc", "λc_inv_led",
    # Categoricals
    "FM_enc", "BC_enc", "ST_enc",
    "section_te", "bc_te", "fm_te",
]
print(f"✅  Features: {len(FEATURES)}")

# ── SPLIT ─────────────────────────────────────────────────────────────────────
X = df[FEATURES].copy()
y = df[TARGET].copy()
Py_all = df["Py"].values
PtPy_all = df["Pt/Py"].values
FM_all = df["FM"].values

X_tr, X_te, y_tr, y_te, Py_tr, Py_te, PtPy_tr, PtPy_te, FM_tr, FM_te = \
    train_test_split(X, y, Py_all, PtPy_all, FM_all,
                     test_size=0.20, random_state=42)
print(f"Train: {len(X_tr)} | Test: {len(X_te)}")

# Recompute target encoding on TRAIN only
for col, grp in [("section_te","Section Types"), ("bc_te","BC"), ("fm_te","FM")]:
    tr_map = df.loc[X_tr.index].groupby(grp)[TARGET].mean()
    X_tr[col] = df.loc[X_tr.index, grp].map(tr_map).fillna(global_mean).values
    X_te[col] = df.loc[X_te.index, grp].map(tr_map).fillna(global_mean).values
print("✅  Target encoding — no leakage")

# ── HYPERPARAMETERS ───────────────────────────────────────────────────────────
PARAMS_XGB = dict(
    n_estimators=8000, learning_rate=0.01, max_depth=9,
    subsample=0.80, colsample_bytree=0.65,
    min_child_weight=1, reg_alpha=0.05, reg_lambda=0.5,
    random_state=42, n_jobs=-1, verbosity=0
)
PARAMS_LGB = dict(
    n_estimators=8000, learning_rate=0.01, num_leaves=512,
    max_depth=-1, subsample=0.80, colsample_bytree=0.65,
    reg_alpha=0.05, reg_lambda=0.5, min_child_samples=3,
    random_state=42, n_jobs=-1, verbose=-1
)
PARAMS_CAT = dict(
    iterations=8000, learning_rate=0.01, depth=10,
    l2_leaf_reg=1.0, subsample=0.80,
    early_stopping_rounds=200,
    random_seed=42, verbose=0
)
if _xgb_major >= 2:
    PARAMS_XGB['early_stopping_rounds'] = 200
    _XGB_FIT_ES = {}
else:
    _XGB_FIT_ES = {'early_stopping_rounds': 200}
lgb_cbs = [lgb_lib.early_stopping(200, verbose=False), lgb_lib.log_evaluation(-1)]

# ── STRATIFIED 5-FOLD OOF ────────────────────────────────────────────────────
N_FOLDS = 5
# Stratify by FM
fm_labels = LabelEncoder().fit_transform(
    df.loc[X_tr.index, "FM"].values)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof     = {k: np.zeros(len(X_tr)) for k in ["xgb","lgb","cat"]}
pred_te = {k: np.zeros(len(X_te)) for k in ["xgb","lgb","cat"]}
X_tr_np, y_tr_np, X_te_np = X_tr.values, y_tr.values, X_te.values

print(f"\n{N_FOLDS}-Fold Stratified OOF (8000 trees) …")
for fold, (tri, vai) in enumerate(skf.split(X_tr_np, fm_labels), 1):
    Xf, Xv = X_tr_np[tri], X_tr_np[vai]
    yf, yv = y_tr_np[tri], y_tr_np[vai]

    m = XGBRegressor(**PARAMS_XGB)
    m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False, **_XGB_FIT_ES)
    oof["xgb"][vai]  = m.predict(Xv)
    pred_te["xgb"]  += m.predict(X_te_np) / N_FOLDS

    m = LGBMRegressor(**PARAMS_LGB)
    m.fit(Xf, yf, eval_set=[(Xv, yv)], callbacks=lgb_cbs)
    oof["lgb"][vai]  = m.predict(Xv)
    pred_te["lgb"]  += m.predict(X_te_np) / N_FOLDS

    m = CatBoostRegressor(**PARAMS_CAT)
    m.fit(Xf, yf, eval_set=(Xv, yv), verbose=False)
    oof["cat"][vai]  = m.predict(Xv)
    pred_te["cat"]  += m.predict(X_te_np) / N_FOLDS

    avg = (oof["xgb"][vai]+oof["lgb"][vai]+oof["cat"][vai])/3
    r2f = r2_score(yv, avg)
    # Also show R² on actual Pt/Py
    pred_ptpy = np.expm1(avg) / Py_tr[vai]
    r2_ratio = r2_score(PtPy_tr[vai], pred_ptpy)
    print(f"  Fold {fold}/{N_FOLDS}  log_Pt R²={r2f:.4f}  Pt/Py R²={r2_ratio:.4f}")

# ── RIDGE BLEND ───────────────────────────────────────────────────────────────
B_tr = np.column_stack([oof["xgb"], oof["lgb"], oof["cat"]])
B_te = np.column_stack([pred_te["xgb"], pred_te["lgb"], pred_te["cat"]])

ridge = Ridge(alpha=0.01, fit_intercept=True)
ridge.fit(B_tr, y_tr_np)
y_pred_tr_log = ridge.predict(B_tr)
y_pred_te_log = ridge.predict(B_te)
print(f"\nRidge OOF R² (log_Pt): {r2_score(y_tr_np, y_pred_tr_log):.6f}")

# ── CONVERT BACK ──────────────────────────────────────────────────────────────
# log_Pt → Pt → Pt/Py
Pt_pred_tr = np.expm1(y_pred_tr_log)
Pt_pred_te = np.expm1(y_pred_te_log)
Pt_actual_tr = np.expm1(y_tr_np)
Pt_actual_te = np.expm1(y_te.values)

PtPy_pred_tr = Pt_pred_tr / Py_tr
PtPy_pred_te = Pt_pred_te / Py_te

# ── METRICS ───────────────────────────────────────────────────────────────────
def report(y_true, y_pred, label, Py_v=None):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true-y_pred)/(np.abs(y_true)+1e-9)))*100
    ratio = y_true / (y_pred+1e-9)
    mu, cov = ratio.mean(), ratio.std()/ratio.mean()
    bar = "="*57
    print(f"\n{bar}\n  {label}\n{bar}")
    print(f"  R²  (Pt/Py)       = {r2:.6f}")
    print(f"  RMSE (Pt/Py)      = {rmse:.6f}")
    print(f"  MAE  (Pt/Py)      = {mae:.6f}")
    print(f"  MAPE              = {mape:.3f} %")
    print(f"  Mean (Nu/Nu,pred) = {mu:.4f}")
    print(f"  COV               = {cov:.4f}")
    print(f"  ── Benchmarks ─────────────────────")
    print(f"  R²   > 0.981 → {'✅' if r2>0.981 else '❌'} ({r2:.4f})")
    print(f"  R²   > 0.994 → {'✅' if r2>0.994 else '⚠️ '} ({r2:.4f})")
    print(f"  MAPE < 5.0%  → {'✅' if mape<5 else '❌'} ({mape:.2f}%)")
    print(f"  MAPE < 2.1%  → {'✅' if mape<2.1 else '⚠️ '} ({mape:.2f}%)")
    print(f"  COV  < 0.09  → {'✅' if cov<0.09 else '⚠️ '} ({cov:.4f})")
    print(f"  Mean ∈[0.98,1.02] → {'✅' if 0.98<=mu<=1.02 else '⚠️ '} ({mu:.4f})")
    if Py_v is not None:
        rk = np.sqrt(mean_squared_error(y_true*Py_v, y_pred*Py_v))
        mk = mean_absolute_error(y_true*Py_v, y_pred*Py_v)
        print(f"  RMSE (kN) = {rk:.1f}")
        print(f"  MAE  (kN) = {mk:.1f}")
    return r2

# R² on Pt directly
r2_pt_tr = r2_score(Pt_actual_tr, Pt_pred_tr)
r2_pt_te = r2_score(Pt_actual_te, Pt_pred_te)
print(f"\n{'='*57}")
print(f"  R² on Pt (kN):  Train={r2_pt_tr:.6f}  Test={r2_pt_te:.6f}")
print(f"{'='*57}")

# R² on Pt/Py ratio
report(PtPy_tr, PtPy_pred_tr, "TRAIN — Pt/Py Ratio")
r2_te = report(PtPy_te, PtPy_pred_te, "TEST — Pt/Py Ratio  ◀ Key", Py_te)

# ── ERROR BY FM ───────────────────────────────────────────────────────────────
res_df = pd.DataFrame({
    "FM": FM_te,
    "actual": PtPy_te,
    "predicted": PtPy_pred_te,
    "abs_err_%": np.abs((PtPy_te-PtPy_pred_te)/PtPy_te)*100
})
print("\nError by Failure Mode:")
print(res_df.groupby("FM")["abs_err_%"]
      .agg(["mean","median","max","count"]).round(2).to_string())

# ── PLOTS ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
ax.scatter(PtPy_te, PtPy_pred_te, alpha=0.55, s=22, c="#01696f",
           label=f"Test R²={r2_te:.4f}")
lim = [0, max(PtPy_te.max(), PtPy_pred_te.max())*1.06]
ax.plot(lim, lim, "r--", lw=1.5)
ax.set_xlabel("Pt/Py Experimental"); ax.set_ylabel("Pt/Py Predicted")
ax.set_title("v5.0 — Predict log(Pt)"); ax.legend(); ax.grid(alpha=0.25)

ax = axes[1]
pct = (PtPy_te - PtPy_pred_te)/PtPy_te*100
ax.scatter(PtPy_pred_te, pct, alpha=0.55, s=22, c="#7a39bb")
ax.axhline(0, c="k", lw=1.2)
ax.axhline(5, c="orange", ls="--"); ax.axhline(-5, c="orange", ls="--")
ax.set_xlabel("Predicted"); ax.set_ylabel("Residual (%)"); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig("v5_scatter.png", bbox_inches="tight"); plt.show()

# ── SHAP ──────────────────────────────────────────────────────────────────────
print("\nSHAP …")
xgb_full = XGBRegressor(**{k:v for k,v in PARAMS_XGB.items()
                           if k != 'early_stopping_rounds'})
xgb_full.set_params(n_estimators=3000)
xgb_full.fit(X_tr, y_tr)
exp = shap.TreeExplainer(xgb_full)
sv = exp.shap_values(X_te)
plt.figure(figsize=(10,7))
shap.summary_plot(sv, X_te, feature_names=FEATURES, show=False)
plt.tight_layout(); plt.savefig("v5_shap.png", bbox_inches="tight"); plt.show()
imp = pd.DataFrame({"Feature":FEATURES,"SHAP":np.abs(sv).mean(0)}).sort_values("SHAP",ascending=False)
print(f"\nTop-5: {imp.head(5)['Feature'].tolist()}")
imp.to_csv("v5_shap_importance.csv", index=False)

# ── SAVE ──────────────────────────────────────────────────────────────────────
out = pd.DataFrame({
    "Pt_Py_actual": PtPy_te, "Pt_Py_pred": PtPy_pred_te,
    "Pt_actual_kN": Pt_actual_te, "Pt_pred_kN": Pt_pred_te,
    "Py_kN": Py_te, "FM": FM_te
})
out.to_csv("v5_predictions.csv", index=False)
print(f"\n{'='*57}")
print(f"  🎯 v5.0 Complete — Test R²(Pt/Py) = {r2_te:.6f}")
print(f"  🎯               — Test R²(Pt kN) = {r2_pt_te:.6f}")
print(f"{'='*57}")
