#!/usr/bin/env python3
"""
CFS v9-fix — Section-Type Specialist Models (PATCHED)
═══════════════════════════════════════════════════════
Fixes vs v9-original:
  1. Target encoding computed AFTER split (no leakage)
  2. early_stopping_rounds = 300 (was 150)
  3. Per-group Ridge blend (learned, not fixed 70/30)
  4. G7_Other split into G7a_Box + G7b_Rest

Strategy:
  Group 1: O-2C              ← largest single type
  Group 2: O-2U
  Group 3: C-U+C
  Group 4: HC-*
  Group 5: C-2C-* / C-2Σ
  Group 6: O-* (remaining open)
  Group 7a: Box / I sections
  Group 7b: Remainder

Each group: XGB + LGB + CatBoost → Ridge blend (learned)
Final:       per-group specialist Ridge-blended with global fallback
"""
import subprocess, sys
def pip(p): subprocess.run([sys.executable,"-m","pip","install","-q",p])
pip("catboost"); pip("lightgbm"); pip("shap")

import numpy as np, pandas as pd, matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["figure.dpi"] = 150
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap, lightgbm as lgb_lib
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import xgboost as _xgb
_xgb_v = int(_xgb.__version__.split(".")[0])
print("✅ Libraries imported")

# ═══════════════════════════════════════════════════════════
# 1. DATA
# ═══════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════
# 2. SECTION GROUPS  (FIX #4: G7 split into G7a + G7b)
# ═══════════════════════════════════════════════════════════
_BOX_KEYWORDS = ["box","Box","BOX","-I-","-I "," I-","built-up I","Built-up I"]
def assign_group(st):
    if st == "O-2C":          return "G1_O2C"
    if st == "O-2U":          return "G2_O2U"
    if st == "C-U+C":         return "G3_CUC"
    if st.startswith("HC"):   return "G4_HC"
    if (st.startswith("C-2C") or st.startswith("C-2Σ")
            or st.startswith("C-2\u03a3")): return "G5_C2C"
    if st.startswith("O-"):   return "G6_Open"
    if any(k in st for k in _BOX_KEYWORDS): return "G7a_Box"
    return "G7b_Rest"

df["SG"] = df["Section Types"].apply(assign_group)
print("\nSection Groups:")
print(df["SG"].value_counts().to_string())

# ═══════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING  (no target encoding yet — FIX #1)
# ═══════════════════════════════════════════════════════════
df["h_t"]       = df["h"]          / df["t"].replace(0, np.nan)
df["b_t"]       = df["b"]          / df["t"].replace(0, np.nan)
df["L_h"]       = df["L"]          / df["h"].replace(0, np.nan)
df["L_t"]       = df["L"]          / df["t"].replace(0, np.nan)
df["h_b"]       = df["h"]          / df["b"].replace(0, np.nan)
df["Pcrl_Py"]   = df["P(crl,crd)"] / df["Py"].replace(0, np.nan)
df["Pne_Py"]    = df["Pne"]        / df["Py"].replace(0, np.nan)
df["λc_sq"]     = df["λc"] ** 2
df["λled_sq"]   = df["λ(le-d)"] ** 2
df["λc_λled"]   = df["λc"]     * df["λ(le-d)"]
df["Pcrl_λled"] = df["Pcrl_Py"] * df["λ(le-d)"]
df["KL_λc"]     = df["KL_r"]   * df["λc"]
df["λc_Pne"]    = df["λc"]     * df["Pne_Py"]
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
df["A_t2"]      = df["A"] / (df["t"] ** 2).replace(0, np.nan)
df["Pcrl_Pne"]  = df["Pcrl_Py"] / df["Pne_Py"].replace(0, np.nan)
df["sqrt_Pcrl"] = np.sqrt(np.abs(df["Pcrl_Py"]))
df["sqrt_Pne"]  = np.sqrt(np.abs(df["Pne_Py"]))
df["DSM_l"]     = np.where(df["Pcrl_Py"] >= 0.776**2, 1.0,
                    (1 - 0.15 * df["Pcrl_Py"]**0.4) * df["Pcrl_Py"]**0.4)

le_fm = LabelEncoder(); df["FM_enc"] = le_fm.fit_transform(df["FM"])
le_bc = LabelEncoder(); df["BC_enc"] = le_bc.fit_transform(df["BC"].astype(str))
le_st = LabelEncoder(); df["ST_enc"] = le_st.fit_transform(df["Section Types"].astype(str))
le_sg = LabelEncoder(); df["SG_enc"] = le_sg.fit_transform(df["SG"])

for c in df.select_dtypes(include=np.number).columns:
    df[c].fillna(df[c].median(), inplace=True)

BASE_FEATURES = [
    "λc", "λ(le-d)", "KL_r", "h_t", "b_t", "L_h", "L_t", "h_b",
    "Pcrl_Py", "Pne_Py", "λc_sq", "λled_sq", "λc_λled", "Pcrl_λled",
    "KL_λc", "λc_Pne", "λled_Pcrl", "Pne_Pcrl", "λc3", "λled3",
    "inv_λc", "λc_inv_led", "Pcrl_sq", "Pne_sq", "h_t_b_t",
    "Fy_norm", "A_t2", "Pcrl_Pne", "sqrt_Pcrl", "sqrt_Pne", "DSM_l",
    "FM_enc", "BC_enc", "ST_enc", "SG_enc",
]
# Target encoding columns — added AFTER split
TE_FEATURES = ["section_te", "bc_te", "fm_te", "sg_te"]
FEATURES = BASE_FEATURES + TE_FEATURES
print(f"✅ Features: {len(FEATURES)} (incl. {len(TE_FEATURES)} TE columns)")

# ═══════════════════════════════════════════════════════════
# 4. SPLIT — stratified by SG
# ═══════════════════════════════════════════════════════════
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
tr_idx, te_idx = next(sss.split(df, df["SG"]))

df_tr_raw = df.iloc[tr_idx].copy()
df_te_raw = df.iloc[te_idx].copy()

# ── FIX #1: Target encoding on TRAIN only, then map to test ──
gm = df_tr_raw[TARGET].mean()
for col, grp in [("section_te","Section Types"),("bc_te","BC"),
                 ("fm_te","FM"),("sg_te","SG")]:
    tr_map = df_tr_raw.groupby(grp)[TARGET].mean()
    df_tr_raw[col] = df_tr_raw[grp].map(tr_map).fillna(gm)
    df_te_raw[col] = df_te_raw[grp].map(tr_map).fillna(gm)

X_tr = df_tr_raw[FEATURES].copy()
X_te = df_te_raw[FEATURES].copy()
y_tr = df_tr_raw[TARGET].copy()
y_te = df_te_raw[TARGET].copy()
Py_te  = df_te_raw["Py"].values
FM_te  = df_te_raw["FM"].values
SG_tr  = df_tr_raw["SG"].values
SG_te  = df_te_raw["SG"].values

y_tr_np = y_tr.values
X_tr_np = X_tr.values
X_te_np = X_te.values

print(f"Train:{len(X_tr)}  Test:{len(X_te)}")
print(f"Test SG distribution:\n{pd.Series(SG_te).value_counts().to_string()}")

# ═══════════════════════════════════════════════════════════
# 5. TRAIN FUNCTION  (FIX #2 + #3: ES=300, Ridge blend per group)
# ═══════════════════════════════════════════════════════════
ES = 300  # FIX #2: was 150

if _xgb_v >= 2:
    _xgb_es = {"early_stopping_rounds": ES}
    _xgb_fit_es = {}
else:
    _xgb_es = {}
    _xgb_fit_es = {"early_stopping_rounds": ES}

lgb_cbs = [lgb_lib.early_stopping(ES, verbose=False), lgb_lib.log_evaluation(-1)]

def train_group(name, Xg_tr, yg_tr, Xg_te, n_folds=None):
    n = len(Xg_tr)
    if n_folds is None:
        n_folds = max(3, min(10, n // 15))

    p_xgb = dict(n_estimators=8000, learning_rate=0.008, max_depth=8,
        subsample=0.80, colsample_bytree=0.65, min_child_weight=1,
        reg_alpha=0.05, reg_lambda=0.5,
        random_state=42, n_jobs=-1, verbosity=0, **_xgb_es)
    p_lgb = dict(n_estimators=8000, learning_rate=0.008, num_leaves=255,
        subsample=0.80, colsample_bytree=0.65, reg_alpha=0.05, reg_lambda=0.5,
        min_child_samples=max(3, n//100),
        random_state=42, n_jobs=-1, verbose=-1)
    p_cat = dict(iterations=8000, learning_rate=0.008,
        depth=min(9, max(6, n//30)), l2_leaf_reg=1.5, subsample=0.80,
        early_stopping_rounds=ES, random_seed=42, verbose=0)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof = {k: np.zeros(n) for k in ["xgb","lgb","cat"]}
    pte = {k: np.zeros(len(Xg_te)) for k in ["xgb","lgb","cat"]}

    for fold, (tri, vai) in enumerate(kf.split(Xg_tr), 1):
        Xf, Xv = Xg_tr[tri], Xg_tr[vai]
        yf, yv = yg_tr[tri], yg_tr[vai]

        m = XGBRegressor(**p_xgb)
        m.fit(Xf, yf, eval_set=[(Xv,yv)], verbose=False, **_xgb_fit_es)
        oof["xgb"][vai] = m.predict(Xv)
        pte["xgb"] += m.predict(Xg_te) / n_folds

        m = LGBMRegressor(**p_lgb)
        m.fit(Xf, yf, eval_set=[(Xv,yv)], callbacks=lgb_cbs)
        oof["lgb"][vai] = m.predict(Xv)
        pte["lgb"] += m.predict(Xg_te) / n_folds

        m = CatBoostRegressor(**p_cat)
        m.fit(Xf, yf, eval_set=(Xv,yv), verbose=False)
        oof["cat"][vai] = m.predict(Xv)
        pte["cat"] += m.predict(Xg_te) / n_folds

    # FIX #3: Ridge learns blend weights (not fixed 70/30)
    B_tr = np.column_stack([oof["xgb"], oof["lgb"], oof["cat"]])
    B_te = np.column_stack([pte["xgb"], pte["lgb"], pte["cat"]])
    ridge = Ridge(alpha=0.01)
    ridge.fit(B_tr, yg_tr)
    oof_blend = ridge.predict(B_tr)
    pte_blend = ridge.predict(B_te)
    r2_oof = r2_score(yg_tr, oof_blend)
    print(f"    Ridge weights [{name}]: {np.round(ridge.coef_, 3)}")

    return oof_blend, pte_blend, r2_oof

# ═══════════════════════════════════════════════════════════
# 6. PHASE 0: GLOBAL FALLBACK (10-fold)
# ═══════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  PHASE 0: Global fallback model")
print("═"*60)
oof_global, pte_global, r2_global = train_group(
    "Global", X_tr_np, y_tr_np, X_te_np, n_folds=10)
print(f"  Global OOF R²(Pt/Py) = {r2_global:.4f}")

# ═══════════════════════════════════════════════════════════
# 7. PHASE 1: SPECIALIST MODELS (per Section Group)
# ═══════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  PHASE 1: Specialist Models per Section Group")
print("═"*60)

groups = sorted(df["SG"].unique())
final_oof = oof_global.copy()
final_pte = pte_global.copy()

for g in groups:
    tr_mask = SG_tr == g
    te_mask = SG_te == g
    n_tr = tr_mask.sum()
    n_te = te_mask.sum()
    print(f"\n  [{g}]  Train:{n_tr}  Test:{n_te}")

    if n_tr < 25:
        print(f"  → Too few samples, keeping global")
        continue

    Xg_tr_g = X_tr_np[tr_mask]
    yg_tr_g  = y_tr_np[tr_mask]
    Xg_te_g  = X_te_np[te_mask] if n_te > 0 else np.zeros((1, X_tr_np.shape[1]))

    oof_g, pte_g, r2_g = train_group(g, Xg_tr_g, yg_tr_g, Xg_te_g)
    global_r2_g = r2_score(yg_tr_g, oof_global[tr_mask])
    print(f"  Specialist OOF R² = {r2_g:.4f}  |  Global R² = {global_r2_g:.4f}")

    if r2_g > global_r2_g - 0.005:   # allow up to 0.005 worse (bias-variance tradeoff)
        # FIX #3: Per-group Ridge blend of specialist + global
        B_blend_tr = np.column_stack([oof_g, oof_global[tr_mask]])
        B_blend_te = np.column_stack(
            [pte_g, pte_global[te_mask]] if n_te > 0
            else [pte_g, pte_global[te_mask]]
        )
        ridge_blend = Ridge(alpha=0.01)
        ridge_blend.fit(B_blend_tr, yg_tr_g)
        final_oof[tr_mask] = ridge_blend.predict(B_blend_tr)
        if n_te > 0:
            final_pte[te_mask] = ridge_blend.predict(B_blend_te)
        w = np.round(ridge_blend.coef_, 3)
        print(f"  ✅ Blended — Ridge learned weights: spec={w[0]:.3f} / global={w[1]:.3f}")
    else:
        print(f"  ❌ Global dominates → keeping global")

# ═══════════════════════════════════════════════════════════
# 8. PHASE 2: FINAL RIDGE RE-BLEND
# ═══════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  PHASE 2: Final Ridge re-blend (specialist_blend + global)")
print("═"*60)
B_final_tr = np.column_stack([final_oof, oof_global])
B_final_te = np.column_stack([final_pte, pte_global])
ridge_final = Ridge(alpha=0.01)
ridge_final.fit(B_final_tr, y_tr_np)
y_pred_tr = ridge_final.predict(B_final_tr)
y_pred_te = ridge_final.predict(B_final_te)
print(f"  OOF R²(Pt/Py) = {r2_score(y_tr_np, y_pred_tr):.4f}")
print(f"  Final Ridge weights: {np.round(ridge_final.coef_, 3)}")

# ═══════════════════════════════════════════════════════════
# 9. METRICS
# ═══════════════════════════════════════════════════════════
def report(label, yt, yp, Py_v=None, fm_v=None, sg_v=None):
    r2   = r2_score(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae  = mean_absolute_error(yt, yp)
    mape = np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-9))) * 100
    ratio = yt / (yp + 1e-9)
    mu = ratio.mean(); cov = ratio.std() / ratio.mean()
    print(f"\n{'═'*62}\n  {label}\n{'═'*62}")
    print(f"  R²(Pt/Py)  = {r2:.6f}")
    print(f"  RMSE       = {rmse:.6f}")
    print(f"  MAE        = {mae:.6f}")
    print(f"  MAPE       = {mape:.3f} %")
    print(f"  Mean(Pt/Pp)= {mu:.4f}")
    print(f"  COV        = {cov:.4f}")
    if Py_v is not None:
        r2k  = r2_score(yt*Py_v, yp*Py_v)
        rmsk = np.sqrt(mean_squared_error(yt*Py_v, yp*Py_v))
        print(f"  R²(Pt kN)  = {r2k:.6f}")
        print(f"  RMSE(kN)   = {rmsk:.2f}")
    print(f"  ── Benchmarks ──")
    for thr, lbl in [(0.985,"R²>0.985"),(0.980,"R²>0.980"),(0.966,"R²>0.966")]:
        print(f"  {lbl}: {'✅' if r2>=thr else '❌'}  ({r2:.4f})")
    print(f"  MAPE<5%: {'✅' if mape<5 else '❌'}  ({mape:.2f}%)")
    print(f"  COV<0.09: {'✅' if cov<0.09 else '⚠️ '}  ({cov:.4f})")
    print(f"  Mean 0.98-1.02: {'✅' if 0.98<=mu<=1.02 else '⚠️ '}  ({mu:.4f})")
    if fm_v is not None:
        df_e = pd.DataFrame({"FM": fm_v,
                             "err_%": np.abs((yt - yp) / (np.abs(yt)+1e-9)) * 100})
        print(f"\n  Error by FM:")
        print(df_e.groupby("FM")["err_%"]
              .agg(["mean","median","max","count"]).round(2).to_string())
    if sg_v is not None:
        df_s = pd.DataFrame({"SG": sg_v,
                             "err_%": np.abs((yt - yp) / (np.abs(yt)+1e-9)) * 100})
        print(f"\n  Error by Section Group:")
        print(df_s.groupby("SG")["err_%"]
              .agg(["mean","median","max","count"]).round(2).to_string())
    return r2

report("TRAIN OOF", y_tr_np, y_pred_tr)
r2_te = report("TEST ◀ KEY RESULT",
               y_te.values, y_pred_te,
               Py_v=Py_te, fm_v=FM_te, sg_v=SG_te)

# ═══════════════════════════════════════════════════════════
# 10. PLOTS
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sg_colors = {"G1_O2C":"#01696f", "G2_O2U":"#d4380d", "G3_CUC":"#7a39bb",
             "G4_HC":"#0958d9",  "G5_C2C":"#389e0d", "G6_Open":"#cf1322",
             "G7a_Box":"#fa8c16","G7b_Rest":"gray"}
c_arr = [sg_colors.get(s, "gray") for s in SG_te]

ax = axes[0]
ax.scatter(y_te.values, y_pred_te, alpha=0.6, s=25, c=c_arr)
lim = [0, max(float(y_te.max()), float(y_pred_te.max())) * 1.06]
ax.plot(lim, lim, "k--", lw=1.5)
ax.fill_between(lim, [x*.95 for x in lim], [x*1.05 for x in lim],
                alpha=0.07, color="green", label="±5%")
ax.fill_between(lim, [x*.98 for x in lim], [x*1.02 for x in lim],
                alpha=0.10, color="blue", label="±2%")
for g, c in sg_colors.items():
    cnt = (np.array(SG_te)==g).sum()
    if cnt > 0:
        ax.scatter([], [], c=c, s=25, label=f"{g} (n={cnt})")
ax.set_xlabel("Pt/Py  Experimental"); ax.set_ylabel("Pt/Py  Predicted")
ax.set_title(f"v9-fix  R²={r2_te:.4f}")
ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.25)

ax = axes[1]
pct = (y_te.values - y_pred_te) / y_te.values * 100
ax.scatter(y_pred_te, pct, alpha=0.6, s=25, c=c_arr)
for v, ls, lbl in [(0,"-",""),(5,"--","±5%"),(-5,"--",""),(2,":","±2%"),(-2,":","")]:
    ax.axhline(v, c="k" if v==0 else ("orange" if abs(v)==5 else "green"),
               ls=ls, lw=0.9, label=lbl if lbl else None)
ax.set_xlabel("Predicted"); ax.set_ylabel("Residual (%)")
ax.set_title("Residuals by Section Group"); ax.legend(fontsize=8); ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("v9_scatter.png", bbox_inches="tight")
plt.show()
print("✅  v9_scatter.png saved")

# ═══════════════════════════════════════════════════════════
# 11. SHAP
# ═══════════════════════════════════════════════════════════
print("\nSHAP analysis…")
xf = XGBRegressor(n_estimators=3000, learning_rate=0.01, max_depth=8,
                  subsample=0.8, colsample_bytree=0.65,
                  random_state=42, n_jobs=-1, verbosity=0)
xf.fit(X_tr, y_tr)
sv = shap.TreeExplainer(xf).shap_values(X_te)
plt.figure(figsize=(10, 7))
shap.summary_plot(sv, X_te, feature_names=FEATURES, show=False)
plt.tight_layout()
plt.savefig("v9_shap.png", bbox_inches="tight")
plt.show()
imp = (pd.DataFrame({"Feature": FEATURES, "SHAP": np.abs(sv).mean(0)})
       .sort_values("SHAP", ascending=False))
print(f"Top-10 SHAP features:\n{imp.head(10).to_string(index=False)}")

# ═══════════════════════════════════════════════════════════
# 12. SAVE
# ═══════════════════════════════════════════════════════════
pd.DataFrame({
    "PtPy_actual": y_te.values,
    "PtPy_pred":   y_pred_te,
    "SG": SG_te, "FM": FM_te,
    "Py_kN": Py_te,
    "Pt_actual_kN": y_te.values * Py_te,
    "Pt_pred_kN":   y_pred_te  * Py_te,
    "err_%": np.abs((y_te.values - y_pred_te) / y_te.values) * 100
}).to_csv("v9_predictions.csv", index=False)
print("✅  v9_predictions.csv saved")

print(f"\n{'═'*62}")
print(f"  🎯  v9-fix COMPLETE")
print(f"  Test R²(Pt/Py) = {r2_te:.6f}")
print(f"  Target: R² ≥ 0.980")
print(f"{'═'*62}")
