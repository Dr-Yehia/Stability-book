#!/usr/bin/env python3
"""
CFS v9 — Section-Type Specialist Models
════════════════════════════════════════
Strategy: separate model per major section group
  Group 1: O-2C              (219 samples) ← largest
  Group 2: O-2U              (104 samples)
  Group 3: C-U+C             (89 samples)
  Group 4: HC-* (half-closed) (110 samples combined)
  Group 5: C-2C-* (closed)   (211 samples combined)
  Group 6: Remainder          (~304 samples)

Each group: XGB + LGB + CatBoost → Ridge blend → OOF prediction
Final: combine all group predictions into one vector
"""
import subprocess, sys
def pip(p): subprocess.run([sys.executable,"-m","pip","install","-q",p])
pip("catboost"); pip("lightgbm"); pip("shap")

import numpy as np, pandas as pd, matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["figure.dpi"] = 150
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, KFold
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

# ── DATA ──
URL = ("https://raw.githubusercontent.com/Dr-Yehia/Stability-book/main/"
       "CFS_Built-up_Columns_ML_Dataset.csv")
df = pd.read_csv(URL)
df = df[[c for c in df.columns if "Unnamed" not in c]].copy()
df = df[df["Pt/Py"].notna()].copy()
df["FM"] = df["FM"].str.strip()
for c in df.select_dtypes(include=np.number).columns:
    df[c].fillna(df[c].median(), inplace=True)

TARGET = "Pt/Py"
print(f"✅ {len(df)} rows | {df['Section Types'].nunique()} section types")

# ── SECTION GROUPS ──
def assign_group(st):
    if st == "O-2C":       return "G1_O2C"
    if st == "O-2U":       return "G2_O2U"
    if st == "C-U+C":      return "G3_CUC"
    if st.startswith("HC"):return "G4_HC"
    if st.startswith("C-2C") or st.startswith("C-2Σ") or st.startswith("C-2\u03a3"):
        return "G5_C2C"
    if st.startswith("O-"):return "G6_Open"
    return "G7_Other"

df["SG"] = df["Section Types"].apply(assign_group)
print("\nSection Groups:")
print(df["SG"].value_counts().to_string())

# ── FEATURES ──
df["h_t"] = df["h"]/df["t"].replace(0,np.nan)
df["b_t"] = df["b"]/df["t"].replace(0,np.nan)
df["L_h"] = df["L"]/df["h"].replace(0,np.nan)
df["L_t"] = df["L"]/df["t"].replace(0,np.nan)
df["h_b"] = df["h"]/df["b"].replace(0,np.nan)
df["Pcrl_Py"] = df["P(crl,crd)"]/df["Py"].replace(0,np.nan)
df["Pne_Py"] = df["Pne"]/df["Py"].replace(0,np.nan)
df["λc_sq"] = df["λc"]**2
df["λled_sq"] = df["λ(le-d)"]**2
df["λc_λled"] = df["λc"]*df["λ(le-d)"]
df["Pcrl_λled"] = df["Pcrl_Py"]*df["λ(le-d)"]
df["KL_λc"] = df["KL_r"]*df["λc"]
df["λc_Pne"] = df["λc"]*df["Pne_Py"]
df["λled_Pcrl"] = df["λ(le-d)"]*df["Pcrl_Py"]
df["Pne_Pcrl"] = df["Pne_Py"]*df["Pcrl_Py"]
df["λc3"] = df["λc"]**3
df["λled3"] = df["λ(le-d)"]**3
df["inv_λc"] = 1.0/df["λc"].replace(0,np.nan)
df["λc_inv_led"] = df["λc"]/df["λ(le-d)"].replace(0,np.nan)
df["Pcrl_sq"] = df["Pcrl_Py"]**2
df["Pne_sq"] = df["Pne_Py"]**2
df["h_t_b_t"] = df["h_t"]*df["b_t"]
df["Fy_norm"] = df["Fy"]/350.0
df["A_t2"] = df["A"]/(df["t"]**2).replace(0,np.nan)
df["Pcrl_Pne"] = df["Pcrl_Py"]/df["Pne_Py"].replace(0,np.nan)
df["sqrt_Pcrl"] = np.sqrt(np.abs(df["Pcrl_Py"]))
df["sqrt_Pne"] = np.sqrt(np.abs(df["Pne_Py"]))

le_fm = LabelEncoder(); df["FM_enc"] = le_fm.fit_transform(df["FM"])
le_bc = LabelEncoder(); df["BC_enc"] = le_bc.fit_transform(df["BC"].astype(str))
le_st = LabelEncoder(); df["ST_enc"] = le_st.fit_transform(df["Section Types"].astype(str))

gm = df[TARGET].mean()
df["section_te"] = df.groupby("Section Types")[TARGET].transform("mean")
df["bc_te"] = df.groupby("BC")[TARGET].transform("mean")
df["fm_te"] = df.groupby("FM")[TARGET].transform("mean")
df["sg_te"] = df.groupby("SG")[TARGET].transform("mean")

for c in df.select_dtypes(include=np.number).columns:
    df[c].fillna(df[c].median(), inplace=True)

FEATURES = [
    "λc","λ(le-d)","KL_r","h_t","b_t","L_h","L_t","h_b",
    "Pcrl_Py","Pne_Py","λc_sq","λled_sq","λc_λled","Pcrl_λled",
    "KL_λc","λc_Pne","λled_Pcrl","Pne_Pcrl","λc3","λled3",
    "inv_λc","λc_inv_led","Pcrl_sq","Pne_sq","h_t_b_t",
    "Fy_norm","A_t2","Pcrl_Pne","sqrt_Pcrl","sqrt_Pne",
    "FM_enc","BC_enc","ST_enc","section_te","bc_te","fm_te","sg_te",
]
print(f"✅ Features: {len(FEATURES)}")

# ── SPLIT (STRATIFIED by SG) ──
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
tr_idx, te_idx = next(sss.split(df, df["SG"]))

X_tr = df.iloc[tr_idx][FEATURES].copy()
X_te = df.iloc[te_idx][FEATURES].copy()
y_tr = df.iloc[tr_idx][TARGET].copy()
y_te = df.iloc[te_idx][TARGET].copy()
Py_tr = df.iloc[tr_idx]["Py"].values
Py_te = df.iloc[te_idx]["Py"].values
FM_tr = df.iloc[tr_idx]["FM"].values
FM_te = df.iloc[te_idx]["FM"].values
SG_tr = df.iloc[tr_idx]["SG"].values
SG_te = df.iloc[te_idx]["SG"].values

# Recompute target encoding
for col,grp in [("section_te","Section Types"),("bc_te","BC"),
                ("fm_te","FM"),("sg_te","SG")]:
    tr_map = df.iloc[tr_idx].groupby(grp)[TARGET].mean()
    X_tr[col] = df.iloc[tr_idx][grp].map(tr_map).fillna(gm).values
    X_te[col] = df.iloc[te_idx][grp].map(tr_map).fillna(gm).values

print(f"Train:{len(X_tr)} Test:{len(X_te)}")
print(f"Test SG distribution:\n{pd.Series(SG_te).value_counts().to_string()}")

# ── MODEL TRAINING PER GROUP ──
X_tr_np = X_tr.values; y_tr_np = y_tr.values; X_te_np = X_te.values

if _xgb_v >= 2:
    _ES = {}
else:
    _ES = {"early_stopping_rounds": 150}
lgb_cbs = [lgb_lib.early_stopping(150,verbose=False), lgb_lib.log_evaluation(-1)]

def train_group(name, Xg_tr, yg_tr, Xg_te, n_folds=None):
    """Train XGB+LGB+CAT on a group, return OOF and test predictions."""
    n = len(Xg_tr)
    if n_folds is None:
        n_folds = max(3, min(10, n // 15))

    p_xgb = dict(n_estimators=6000, learning_rate=0.01, max_depth=8,
        subsample=0.80, colsample_bytree=0.65, min_child_weight=1,
        reg_alpha=0.05, reg_lambda=0.5, random_state=42, n_jobs=-1, verbosity=0)
    p_lgb = dict(n_estimators=6000, learning_rate=0.01, num_leaves=256,
        subsample=0.80, colsample_bytree=0.65, reg_alpha=0.05, reg_lambda=0.5,
        min_child_samples=max(3, n//100), random_state=42, n_jobs=-1, verbose=-1)
    p_cat = dict(iterations=6000, learning_rate=0.01, depth=min(9, max(6, n//30)),
        l2_leaf_reg=1.0, subsample=0.80, early_stopping_rounds=150,
        random_seed=42, verbose=0)

    if _xgb_v >= 2: p_xgb['early_stopping_rounds'] = 150

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof = {k: np.zeros(n) for k in ["xgb","lgb","cat"]}
    pte = {k: np.zeros(len(Xg_te)) for k in ["xgb","lgb","cat"]}

    for tri, vai in kf.split(Xg_tr):
        Xf, Xv = Xg_tr[tri], Xg_tr[vai]
        yf, yv = yg_tr[tri], yg_tr[vai]

        m = XGBRegressor(**p_xgb)
        m.fit(Xf, yf, eval_set=[(Xv,yv)], verbose=False, **_ES)
        oof["xgb"][vai] = m.predict(Xv)
        pte["xgb"] += m.predict(Xg_te)/n_folds

        m = LGBMRegressor(**p_lgb)
        m.fit(Xf, yf, eval_set=[(Xv,yv)], callbacks=lgb_cbs)
        oof["lgb"][vai] = m.predict(Xv)
        pte["lgb"] += m.predict(Xg_te)/n_folds

        m = CatBoostRegressor(**p_cat)
        m.fit(Xf, yf, eval_set=(Xv,yv), verbose=False)
        oof["cat"][vai] = m.predict(Xv)
        pte["cat"] += m.predict(Xg_te)/n_folds

    # Ridge blend
    B_tr = np.column_stack([oof["xgb"], oof["lgb"], oof["cat"]])
    B_te = np.column_stack([pte["xgb"], pte["lgb"], pte["cat"]])
    ridge = Ridge(alpha=0.1); ridge.fit(B_tr, yg_tr)
    oof_blend = ridge.predict(B_tr)
    pte_blend = ridge.predict(B_te)
    r2_oof = r2_score(yg_tr, oof_blend)

    return oof_blend, pte_blend, r2_oof

# ── ALSO TRAIN GLOBAL MODEL (fallback) ──
print("\n" + "═"*60)
print("  PHASE 0: Global fallback model (all data)")
print("═"*60)
oof_global, pte_global, r2_global = train_group(
    "Global", X_tr_np, y_tr_np, X_te_np, n_folds=10)
print(f"  Global OOF R² = {r2_global:.4f}")

# ── TRAIN PER-GROUP SPECIALISTS ──
print("\n" + "═"*60)
print("  PHASE 1: Section-Type Specialist Models")
print("═"*60)

groups = sorted(df["SG"].unique())
final_oof = oof_global.copy()  # start with global as baseline
final_pte = pte_global.copy()

for g in groups:
    tr_mask = SG_tr == g
    te_mask = SG_te == g
    n_tr = tr_mask.sum(); n_te = te_mask.sum()
    print(f"\n  [{g}]  Train:{n_tr}  Test:{n_te}")

    if n_tr < 30:
        print(f"  → Too few, using global model")
        continue

    Xg_tr = X_tr_np[tr_mask]
    yg_tr = y_tr_np[tr_mask]
    Xg_te = X_te_np[te_mask] if n_te > 0 else np.zeros((1, X_tr_np.shape[1]))

    oof_g, pte_g, r2_g = train_group(g, Xg_tr, yg_tr, Xg_te)
    print(f"  OOF R² = {r2_g:.4f}")

    # Use specialist if better than global on this group
    global_r2_g = r2_score(yg_tr, oof_global[tr_mask])
    print(f"  Global R² on this group = {global_r2_g:.4f}")

    if r2_g > global_r2_g:
        # Blend: 70% specialist + 30% global
        final_oof[tr_mask] = 0.7 * oof_g + 0.3 * oof_global[tr_mask]
        if n_te > 0:
            final_pte[te_mask] = 0.7 * pte_g + 0.3 * pte_global[te_mask]
        print(f"  ✅ Specialist WINS → blended 70/30")
    else:
        print(f"  ❌ Global is better → keeping global")

# ── PHASE 2: Ridge re-blend specialist + global ──
print("\n" + "═"*60)
print("  PHASE 2: Final Ridge re-blend")
print("═"*60)
B_final_tr = np.column_stack([final_oof, oof_global])
B_final_te = np.column_stack([final_pte, pte_global])
ridge_final = Ridge(alpha=0.01)
ridge_final.fit(B_final_tr, y_tr_np)
y_pred_tr = ridge_final.predict(B_final_tr)
y_pred_te = ridge_final.predict(B_final_te)
print(f"  Final OOF R² = {r2_score(y_tr_np, y_pred_tr):.4f}")
print(f"  Ridge weights: {np.round(ridge_final.coef_,3)}")

# ── METRICS ──
def report(label, yt, yp, Py_v=None, fm_v=None):
    r2 = r2_score(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    mape = np.mean(np.abs((yt-yp)/(np.abs(yt)+1e-9)))*100
    ratio = yt/(yp+1e-9); mu=ratio.mean(); cov=ratio.std()/ratio.mean()
    print(f"\n{'═'*60}\n  {label}\n{'═'*60}")
    print(f"  R²(Pt/Py) = {r2:.6f}")
    print(f"  RMSE      = {rmse:.6f}")
    print(f"  MAE       = {mae:.6f}")
    print(f"  MAPE      = {mape:.3f}%")
    print(f"  Mean      = {mu:.4f}")
    print(f"  COV       = {cov:.4f}")
    if Py_v is not None:
        r2k = r2_score(yt*Py_v, yp*Py_v)
        print(f"  R²(Pt kN) = {r2k:.6f}")
        print(f"  RMSE(kN)  = {np.sqrt(mean_squared_error(yt*Py_v, yp*Py_v)):.1f}")
    print(f"  ── Benchmarks ──")
    print(f"  R²>0.980 {'✅' if r2>0.980 else '❌'} | R²>0.966 {'✅' if r2>0.966 else '❌'} | MAPE<5% {'✅' if mape<5 else '❌'} | COV<0.09 {'✅' if cov<0.09 else '⚠️ '}")
    if fm_v is not None:
        df_e=pd.DataFrame({"FM":fm_v,"err_%":np.abs((yt-yp)/yt)*100})
        print(f"\n  Error by FM:")
        print(df_e.groupby("FM")["err_%"].agg(["mean","median","max","count"]).round(2).to_string())
    return r2

report("TRAIN OOF", y_tr_np, y_pred_tr)
r2_te = report("TEST ◀ KEY", y_te.values, y_pred_te, Py_te, FM_te)

# Error by Section Group
print("\nError by Section Group:")
df_sg = pd.DataFrame({"SG":SG_te, "err_%":np.abs((y_te.values-y_pred_te)/y_te.values)*100})
print(df_sg.groupby("SG")["err_%"].agg(["mean","median","max","count"]).round(2).to_string())

# ── PLOTS ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
sg_colors = {"G1_O2C":"#01696f","G2_O2U":"#d4380d","G3_CUC":"#7a39bb",
             "G4_HC":"#0958d9","G5_C2C":"#389e0d","G6_Open":"#cf1322","G7_Other":"gray"}
c_arr = [sg_colors.get(s,"gray") for s in SG_te]
ax.scatter(y_te.values, y_pred_te, alpha=0.6, s=25, c=c_arr)
lim=[0,max(y_te.max(),y_pred_te.max())*1.06]
ax.plot(lim,lim,"k--",lw=1.5)
ax.fill_between(lim,[x*.95 for x in lim],[x*1.05 for x in lim],alpha=0.08,color="green",label="±5%")
ax.set_xlabel("Pt/Py Experimental"); ax.set_ylabel("Pt/Py Predicted")
ax.set_title(f"v9 Section-Type Specialists  R²={r2_te:.4f}")
# Legend for groups
for g,c in sg_colors.items():
    n = (np.array(SG_te)==g).sum()
    if n>0: ax.scatter([],[],c=c,s=25,label=f"{g} (n={n})")
ax.legend(fontsize=7,loc="lower right"); ax.grid(alpha=0.25)

ax = axes[1]
pct=(y_te.values-y_pred_te)/y_te.values*100
ax.scatter(y_pred_te, pct, alpha=0.6, s=25, c=c_arr)
ax.axhline(0,c="k"); ax.axhline(5,c="orange",ls="--"); ax.axhline(-5,c="orange",ls="--")
ax.set_xlabel("Predicted"); ax.set_ylabel("Residual(%)"); ax.set_title("Residuals by Group")
ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig("v9_scatter.png",bbox_inches="tight"); plt.show()

# SHAP
print("\nSHAP…")
xf=XGBRegressor(n_estimators=3000,learning_rate=0.01,max_depth=8,
    subsample=0.8,colsample_bytree=0.65,random_state=42,n_jobs=-1,verbosity=0)
xf.fit(X_tr,y_tr)
sv=shap.TreeExplainer(xf).shap_values(X_te)
plt.figure(figsize=(10,7))
shap.summary_plot(sv,X_te,feature_names=FEATURES,show=False)
plt.tight_layout(); plt.savefig("v9_shap.png",bbox_inches="tight"); plt.show()
imp=pd.DataFrame({"Feature":FEATURES,"SHAP":np.abs(sv).mean(0)}).sort_values("SHAP",ascending=False)
print(f"Top-5: {imp.head(5)['Feature'].tolist()}")

pd.DataFrame({"PtPy_actual":y_te.values,"PtPy_pred":y_pred_te,
    "SG":SG_te,"FM":FM_te,"Py":Py_te}).to_csv("v9_predictions.csv",index=False)

print(f"\n{'═'*60}")
print(f"  🎯 v9 Complete — Test R²(Pt/Py) = {r2_te:.6f}")
print(f"{'═'*60}")
