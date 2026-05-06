#!/usr/bin/env python3
"""
CFS Built-up Columns — v7.0 — Physics-Informed Residual Learning
═══════════════════════════════════════════════════════════════════
INNOVATION:  Pt/Py = Pne/Py + ML_correction(features)
             DSM does 80% → ML learns only the 20% residual
═══════════════════════════════════════════════════════════════════
"""
import subprocess, sys
def pip(p): subprocess.run([sys.executable,"-m","pip","install","-q",p])
pip("catboost"); pip("lightgbm"); pip("shap")

import numpy as np, pandas as pd, matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["figure.dpi"] = 150
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap, lightgbm as lgb_lib
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import xgboost as _xgb
_xgb_v = int(_xgb.__version__.split('.')[0])
print("✅ Libraries imported")

# ══════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════
URL = ("https://raw.githubusercontent.com/Dr-Yehia/Stability-book/main/"
       "CFS_Built-up_Columns_ML_Dataset.csv")
df = pd.read_csv(URL)
df = df[[c for c in df.columns if "Unnamed" not in c]].copy()
df = df[df["Pt/Py"].notna()].copy()
df["FM"] = df["FM"].str.strip()
for c in df.select_dtypes(include=np.number).columns:
    df[c].fillna(df[c].median(), inplace=True)

# Reconstruct Pt (kN)
df["Pt_kN"] = df["Pt/Py"] * df["Py"]

# DSM baseline: Pne/Py
df["Pne_Py"] = df["Pne"] / df["Py"].replace(0, np.nan)
df["Pne_Py"].fillna(df["Pne_Py"].median(), inplace=True)

# ══════════════════════════════════════════════════════════════════
# PHYSICS-INFORMED RESIDUAL (the innovation)
# ══════════════════════════════════════════════════════════════════
df["residual"] = df["Pt/Py"] - df["Pne_Py"]
TARGET = "residual"  # ML predicts ONLY this correction

print(f"✅ Dataset: {len(df)} rows")
print(f"   Pne/Py range: {df['Pne_Py'].min():.3f} – {df['Pne_Py'].max():.3f}")
print(f"   Residual range: {df[TARGET].min():.3f} – {df[TARGET].max():.3f}")
print(f"   Residual std: {df[TARGET].std():.4f}  (vs Pt/Py std: {df['Pt/Py'].std():.4f})")

# ══════════════════════════════════════════════════════════════════
# FEATURES
# ══════════════════════════════════════════════════════════════════
df["h_t"]       = df["h"] / df["t"].replace(0, np.nan)
df["b_t"]       = df["b"] / df["t"].replace(0, np.nan)
df["L_h"]       = df["L"] / df["h"].replace(0, np.nan)
df["L_t"]       = df["L"] / df["t"].replace(0, np.nan)
df["h_b"]       = df["h"] / df["b"].replace(0, np.nan)
df["Pcrl_Py"]   = df["P(crl,crd)"] / df["Py"].replace(0, np.nan)
df["λc_sq"]     = df["λc"] ** 2
df["λled_sq"]   = df["λ(le-d)"] ** 2
df["λc_λled"]   = df["λc"] * df["λ(le-d)"]
df["Pcrl_λled"] = df["Pcrl_Py"] * df["λ(le-d)"]
df["KL_λc"]     = df["KL_r"] * df["λc"]
df["λc_Pne"]    = df["λc"] * df["Pne_Py"]
df["λled_Pcrl"] = df["λ(le-d)"] * df["Pcrl_Py"]
df["Pne_Pcrl"]  = df["Pne_Py"] * df["Pcrl_Py"]
df["λc3"]       = df["λc"] ** 3
df["λled3"]     = df["λ(le-d)"] ** 3
df["inv_λc"]    = 1.0 / df["λc"].replace(0, np.nan)
df["λc_inv_led"]= df["λc"] / df["λ(le-d)"].replace(0, np.nan)
df["Pcrl_sq"]   = df["Pcrl_Py"] ** 2
df["Pne_sq"]    = df["Pne_Py"] ** 2
df["h_t_b_t"]   = df["h_t"] * df["b_t"]
df["Fy_norm"]   = df["Fy"] / 350.0
df["A_t2"]      = df["A"] / (df["t"]**2).replace(0, np.nan)

# Pne/Py itself as feature (the model learns to correct it)
# + additional physics ratios
df["Pne_Py_sq"]    = df["Pne_Py"] ** 2
df["Pcrl_Pne"]     = df["Pcrl_Py"] / df["Pne_Py"].replace(0, np.nan)
df["λc_x_Pne_Py"]  = df["λc"] * df["Pne_Py"]

le_fm = LabelEncoder(); df["FM_enc"] = le_fm.fit_transform(df["FM"])
le_bc = LabelEncoder(); df["BC_enc"] = le_bc.fit_transform(df["BC"].astype(str))
le_st = LabelEncoder(); df["ST_enc"] = le_st.fit_transform(df["Section Types"].astype(str))

gm = df[TARGET].mean()
df["section_te"] = df.groupby("Section Types")[TARGET].transform("mean")
df["bc_te"]      = df.groupby("BC")[TARGET].transform("mean")
df["fm_te"]      = df.groupby("FM")[TARGET].transform("mean")

for c in df.select_dtypes(include=np.number).columns:
    df[c].fillna(df[c].median(), inplace=True)

FEATURES = [
    "Pne_Py",  # ← KEY: the physics baseline itself
    "λc", "λ(le-d)", "KL_r",
    "h_t", "b_t", "L_h", "L_t", "h_b",
    "Pcrl_Py", "Pne_sq", "Pne_Py_sq", "Pcrl_Pne", "λc_x_Pne_Py",
    "λc_sq", "λled_sq", "λc_λled", "Pcrl_λled",
    "KL_λc", "λc_Pne", "λled_Pcrl", "Pne_Pcrl",
    "λc3", "λled3", "inv_λc", "λc_inv_led",
    "Pcrl_sq", "h_t_b_t", "Fy_norm", "A_t2",
    "FM_enc", "BC_enc", "ST_enc",
    "section_te", "bc_te", "fm_te",
]
print(f"✅ Features: {len(FEATURES)}")

# ══════════════════════════════════════════════════════════════════
# SPLIT
# ══════════════════════════════════════════════════════════════════
X = df[FEATURES].copy()
y = df[TARGET].copy()  # residual = Pt/Py - Pne/Py
Py_all    = df["Py"].values
PtPy_all  = df["Pt/Py"].values
PnePy_all = df["Pne_Py"].values
Pt_all    = df["Pt_kN"].values
FM_all    = df["FM"].values

X_tr, X_te, y_tr, y_te, Py_tr, Py_te, PtPy_tr, PtPy_te, \
PnePy_tr, PnePy_te, Pt_tr, Pt_te, FM_tr, FM_te = \
    train_test_split(X, y, Py_all, PtPy_all, PnePy_all, Pt_all, FM_all,
                     test_size=0.20, random_state=42)
print(f"Train: {len(X_tr)} | Test: {len(X_te)}")

# Recompute target encoding
for col, grp in [("section_te","Section Types"),("bc_te","BC"),("fm_te","FM")]:
    tr_map = df.loc[X_tr.index].groupby(grp)[TARGET].mean()
    X_tr[col] = df.loc[X_tr.index, grp].map(tr_map).fillna(gm).values
    X_te[col] = df.loc[X_te.index, grp].map(tr_map).fillna(gm).values
print("✅ No leakage")

# ══════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════
PARAMS_XGB = dict(
    n_estimators=8000, learning_rate=0.01, max_depth=9,
    subsample=0.80, colsample_bytree=0.65,
    min_child_weight=1, reg_alpha=0.05, reg_lambda=0.5,
    random_state=42, n_jobs=-1, verbosity=0)
PARAMS_LGB = dict(
    n_estimators=8000, learning_rate=0.01, num_leaves=512,
    max_depth=-1, subsample=0.80, colsample_bytree=0.65,
    reg_alpha=0.05, reg_lambda=0.5, min_child_samples=3,
    random_state=42, n_jobs=-1, verbose=-1)
PARAMS_CAT = dict(
    iterations=8000, learning_rate=0.01, depth=10,
    l2_leaf_reg=1.0, subsample=0.80,
    early_stopping_rounds=200, random_seed=42, verbose=0)

if _xgb_v >= 2:
    PARAMS_XGB['early_stopping_rounds'] = 200; _ES = {}
else:
    _ES = {'early_stopping_rounds': 200}
lgb_cbs = [lgb_lib.early_stopping(200, verbose=False),
           lgb_lib.log_evaluation(-1)]

# ══════════════════════════════════════════════════════════════════
# 10-FOLD STRATIFIED OOF STACKING
# ══════════════════════════════════════════════════════════════════
N_FOLDS = 10
fm_lab = LabelEncoder().fit_transform(df.loc[X_tr.index, "FM"].values)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof  = {k: np.zeros(len(X_tr)) for k in ["xgb","lgb","cat"]}
pte  = {k: np.zeros(len(X_te)) for k in ["xgb","lgb","cat"]}
X_tr_np, y_tr_np, X_te_np = X_tr.values, y_tr.values, X_te.values

print(f"\n10-Fold Stratified OOF — predicting RESIDUAL …")
for fold, (tri, vai) in enumerate(skf.split(X_tr_np, fm_lab), 1):
    Xf, Xv = X_tr_np[tri], X_tr_np[vai]
    yf, yv = y_tr_np[tri], y_tr_np[vai]

    m = XGBRegressor(**PARAMS_XGB)
    m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False, **_ES)
    oof["xgb"][vai] = m.predict(Xv); pte["xgb"] += m.predict(X_te_np)/N_FOLDS

    m = LGBMRegressor(**PARAMS_LGB)
    m.fit(Xf, yf, eval_set=[(Xv, yv)], callbacks=lgb_cbs)
    oof["lgb"][vai] = m.predict(Xv); pte["lgb"] += m.predict(X_te_np)/N_FOLDS

    m = CatBoostRegressor(**PARAMS_CAT)
    m.fit(Xf, yf, eval_set=(Xv, yv), verbose=False)
    oof["cat"][vai] = m.predict(Xv); pte["cat"] += m.predict(X_te_np)/N_FOLDS

    # Reconstruct Pt/Py = Pne/Py + predicted_residual
    res_pred = (oof["xgb"][vai] + oof["lgb"][vai] + oof["cat"][vai]) / 3
    ptpy_pred = PnePy_tr[vai] + res_pred
    r2_ratio = r2_score(PtPy_tr[vai], ptpy_pred)
    print(f"  Fold {fold}/{N_FOLDS}  R²(Pt/Py)={r2_ratio:.4f}")

# ══════════════════════════════════════════════════════════════════
# RIDGE BLEND
# ══════════════════════════════════════════════════════════════════
B_tr = np.column_stack([oof["xgb"], oof["lgb"], oof["cat"]])
B_te = np.column_stack([pte["xgb"], pte["lgb"], pte["cat"]])

ridge = Ridge(alpha=0.01, fit_intercept=True)
ridge.fit(B_tr, y_tr_np)
resid_pred_tr = ridge.predict(B_tr)
resid_pred_te = ridge.predict(B_te)

# ══════════════════════════════════════════════════════════════════
# RECONSTRUCT FINAL PREDICTIONS
# Pt/Py = Pne/Py + ML_correction
# ══════════════════════════════════════════════════════════════════
PtPy_final_tr = PnePy_tr + resid_pred_tr
PtPy_final_te = PnePy_te + resid_pred_te

# Pt (kN) = Pt/Py × Py
Pt_final_tr = PtPy_final_tr * Py_tr
Pt_final_te = PtPy_final_te * Py_te

# ══════════════════════════════════════════════════════════════════
# COMPREHENSIVE METRICS
# ══════════════════════════════════════════════════════════════════
def full_report(label, ptpy_true, ptpy_pred, pt_true, pt_pred, Py_v, fm_v):
    # R² on Pt/Py
    r2_ratio = r2_score(ptpy_true, ptpy_pred)
    rmse_ratio = np.sqrt(mean_squared_error(ptpy_true, ptpy_pred))
    mae_ratio = mean_absolute_error(ptpy_true, ptpy_pred)
    mape_ratio = np.mean(np.abs((ptpy_true-ptpy_pred)/(np.abs(ptpy_true)+1e-9)))*100

    # R² on Pt (kN)
    r2_pt = r2_score(pt_true, pt_pred)
    rmse_pt = np.sqrt(mean_squared_error(pt_true, pt_pred))
    mae_pt = mean_absolute_error(pt_true, pt_pred)

    # Structural engineering metrics
    ratio = pt_true / (pt_pred + 1e-9)
    mu = ratio.mean()
    cov = ratio.std() / ratio.mean()

    bar = "═" * 60
    print(f"\n{bar}")
    print(f"  {label}")
    print(f"{bar}")
    print(f"  ── Pt/Py (ratio) ──")
    print(f"  R²       = {r2_ratio:.6f}")
    print(f"  RMSE     = {rmse_ratio:.6f}")
    print(f"  MAE      = {mae_ratio:.6f}")
    print(f"  MAPE     = {mape_ratio:.3f} %")
    print(f"  ── Pt (kN) ──")
    print(f"  R²       = {r2_pt:.6f}")
    print(f"  RMSE     = {rmse_pt:.2f} kN")
    print(f"  MAE      = {mae_pt:.2f} kN")
    print(f"  ── Engineering Metrics ──")
    print(f"  Mean(Pt_exp/Pt_pred) = {mu:.4f}")
    print(f"  COV                  = {cov:.4f}")
    print(f"  ── Benchmarks ──")
    print(f"  R²(Pt)>0.985    → {'✅' if r2_pt>0.985 else '❌'} ({r2_pt:.4f})")
    print(f"  R²(Pt/Py)>0.981 → {'✅' if r2_ratio>0.981 else '❌'} ({r2_ratio:.4f})")
    print(f"  MAPE<5%          → {'✅' if mape_ratio<5 else '❌'} ({mape_ratio:.2f}%)")
    print(f"  COV<0.09         → {'✅' if cov<0.09 else '⚠️ '} ({cov:.4f})")
    print(f"  Mean∈[0.98,1.02] → {'✅' if 0.98<=mu<=1.02 else '⚠️ '} ({mu:.4f})")
    return r2_ratio, r2_pt

r2_ratio_tr, r2_pt_tr = full_report("TRAIN (OOF)",
    PtPy_tr, PtPy_final_tr, Pt_tr, Pt_final_tr, Py_tr, FM_tr)
r2_ratio_te, r2_pt_te = full_report("TEST ◀ KEY RESULT",
    PtPy_te, PtPy_final_te, Pt_te, Pt_final_te, Py_te, FM_te)

# DSM-only baseline (no ML)
print(f"\n{'─'*60}")
print(f"  DSM-only baseline (Pne/Py without ML correction):")
r2_dsm = r2_score(PtPy_te, PnePy_te)
print(f"  R²(Pt/Py) = {r2_dsm:.4f}")
print(f"  → ML improvement: +{(r2_ratio_te - r2_dsm)*100:.2f} percentage points")
print(f"{'─'*60}")

# Error by FM
res = pd.DataFrame({
    "FM": FM_te, "PtPy_actual": PtPy_te, "PtPy_pred": PtPy_final_te,
    "err_%": np.abs((PtPy_te - PtPy_final_te) / PtPy_te) * 100
})
print("\nError by Failure Mode:")
print(res.groupby("FM")["err_%"].agg(["mean","median","max","count"]).round(2).to_string())

# ══════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Pt/Py scatter
ax = axes[0, 0]
ax.scatter(PtPy_te, PtPy_final_te, alpha=0.55, s=22, c="#01696f",
           label=f"R²(Pt/Py)={r2_ratio_te:.4f}")
lim = [0, max(PtPy_te.max(), PtPy_final_te.max()) * 1.06]
ax.plot(lim, lim, "r--", lw=1.5)
ax.set_xlabel("Pt/Py Experimental"); ax.set_ylabel("Pt/Py Predicted")
ax.set_title("v7 — Physics-Informed Residual"); ax.legend(); ax.grid(alpha=0.25)

# 2. Pt (kN) scatter
ax = axes[0, 1]
ax.scatter(Pt_te, Pt_final_te, alpha=0.55, s=22, c="#d4380d",
           label=f"R²(Pt)={r2_pt_te:.4f}")
lim2 = [0, max(Pt_te.max(), Pt_final_te.max()) * 1.06]
ax.plot(lim2, lim2, "r--", lw=1.5)
ax.set_xlabel("Pt Experimental (kN)"); ax.set_ylabel("Pt Predicted (kN)")
ax.set_title("Load Capacity Prediction"); ax.legend(); ax.grid(alpha=0.25)

# 3. Residual distribution
ax = axes[1, 0]
pct = (PtPy_te - PtPy_final_te) / PtPy_te * 100
ax.scatter(PtPy_final_te, pct, alpha=0.55, s=22, c="#7a39bb")
ax.axhline(0, c="k", lw=1.2)
ax.axhline(5, c="orange", ls="--"); ax.axhline(-5, c="orange", ls="--")
ax.set_xlabel("Predicted Pt/Py"); ax.set_ylabel("Residual (%)")
ax.set_title("Residual Distribution"); ax.grid(alpha=0.25)

# 4. DSM vs ML improvement
ax = axes[1, 1]
ax.scatter(PtPy_te, PnePy_te, alpha=0.4, s=18, c="gray", label=f"DSM only R²={r2_dsm:.4f}")
ax.scatter(PtPy_te, PtPy_final_te, alpha=0.55, s=22, c="#01696f",
           label=f"DSM+ML R²={r2_ratio_te:.4f}")
ax.plot(lim, lim, "r--", lw=1.5)
ax.set_xlabel("Pt/Py Experimental"); ax.set_ylabel("Pt/Py Predicted")
ax.set_title("DSM vs DSM+ML Correction"); ax.legend(); ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("v7_results.png", bbox_inches="tight")
plt.show()
print("✅ v7_results.png saved")

# ══════════════════════════════════════════════════════════════════
# SHAP
# ══════════════════════════════════════════════════════════════════
print("\nSHAP analysis …")
xf = XGBRegressor(n_estimators=3000, learning_rate=0.01, max_depth=9,
    subsample=0.8, colsample_bytree=0.65, random_state=42, n_jobs=-1, verbosity=0)
xf.fit(X_tr, y_tr)
sv = shap.TreeExplainer(xf).shap_values(X_te)

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(sv, X_te, feature_names=FEATURES, show=False)
plt.tight_layout(); plt.savefig("v7_shap.png", bbox_inches="tight"); plt.show()

imp = (pd.DataFrame({"Feature": FEATURES, "SHAP": np.abs(sv).mean(0)})
       .sort_values("SHAP", ascending=False).reset_index(drop=True))
print(f"\nTop-5 for PySR: {imp.head(5)['Feature'].tolist()}")
imp.to_csv("v7_shap_importance.csv", index=False)

# ══════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════
out = pd.DataFrame({
    "PtPy_actual": PtPy_te, "PtPy_pred": PtPy_final_te,
    "PnePy_DSM": PnePy_te, "ML_correction": resid_pred_te,
    "Pt_actual_kN": Pt_te, "Pt_pred_kN": Pt_final_te,
    "Py_kN": Py_te, "FM": FM_te
})
out.to_csv("v7_predictions.csv", index=False)

print(f"\n{'═'*60}")
print(f"  🎯 v7.0 Physics-Informed Residual Learning — COMPLETE")
print(f"  ─────────────────────────────────────────────────────")
print(f"  Test R²(Pt kN)   = {r2_pt_te:.6f}")
print(f"  Test R²(Pt/Py)   = {r2_ratio_te:.6f}")
print(f"  DSM baseline     = {r2_dsm:.6f}")
print(f"  ML improvement   = +{(r2_ratio_te - r2_dsm)*100:.2f} pp")
print(f"{'═'*60}")
