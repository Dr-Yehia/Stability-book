#!/usr/bin/env python3
"""
================================================================================
CFS Built-up Columns — v8 BREAKTHROUGH
================================================================================
أربع ثورات في نفس الوقت:
  1. Target = Pt/Py مباشرةً (ليس الـ residual — الـ residual أصعب رياضياً)
  2. FM-Grouped Models: موديل منفصل لكل مجموعة انهيار
     Group A: F, LF    (Flexural dominant)
     Group B: L, LD, LDF, LF  (Local dominant)
     Group C: D, DF, LDF  (Distortional dominant)
  3. Optuna Hyperparameter Optimization (200 trials per model)
  4. Features فيزيائية محسوبة بدقة + Interaction Features جديدة

Expected R²(Pt/Py) > 0.980
================================================================================
"""

import subprocess, sys
def pip(p): subprocess.run([sys.executable, "-m", "pip", "install", "-q", p])
pip("catboost"); pip("lightgbm"); pip("shap"); pip("optuna")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.rcParams["figure.dpi"] = 150
import warnings; warnings.filterwarnings("ignore")
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model    import Ridge
from sklearn.metrics         import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing   import LabelEncoder
import lightgbm as lgb_lib
import xgboost as _xgb
from xgboost  import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

_xgb_v = int(_xgb.__version__.split('.')[0])
print("✅  Libraries imported")

# ── LOAD DATA ──────────────────────────────────────────────────────────────────
URL = ("https://raw.githubusercontent.com/Dr-Yehia/Stability-book/main/"
       "CFS_Built-up_Columns_ML_Dataset.csv")
df = pd.read_csv(URL)
df = df[[c for c in df.columns if "Unnamed" not in c]].copy()
df = df[df["Pt/Py"].notna()].copy()
df["FM"] = df["FM"].str.strip()
for c in df.select_dtypes(include=np.number).columns:
    df[c].fillna(df[c].median(), inplace=True)

TARGET = "Pt/Py"   # ← مباشرةً — ليس residual
print(f"✅  Dataset: {len(df)} rows | Target: {TARGET}")
print(f"   FM distribution: {df['FM'].value_counts().to_dict()}")

# ── FEATURE ENGINEERING ────────────────────────────────────────────────────────
# Basic geometric ratios
df["h_t"]       = df["h"]           / df["t"].replace(0, np.nan)
df["b_t"]       = df["b"]           / df["t"].replace(0, np.nan)
df["L_h"]       = df["L"]           / df["h"].replace(0, np.nan)
df["L_t"]       = df["L"]           / df["t"].replace(0, np.nan)
df["h_b"]       = df["h"]           / df["b"].replace(0, np.nan)
df["A_t2"]      = df["A"]           / (df["t"]**2).replace(0, np.nan)
df["Fy_norm"]   = df["Fy"]          / 350.0

# Strength ratios
df["Pcrl_Py"]   = df["P(crl,crd)"]  / df["Py"].replace(0, np.nan)
df["Pne_Py"]    = df["Pne"]         / df["Py"].replace(0, np.nan)

# Slenderness squares and cubes
df["λc_sq"]     = df["λc"] ** 2
df["λled_sq"]   = df["λ(le-d)"] ** 2
df["λc3"]       = df["λc"] ** 3
df["λled3"]     = df["λ(le-d)"] ** 3
df["Pne_sq"]    = df["Pne_Py"] ** 2
df["Pcrl_sq"]   = df["Pcrl_Py"] ** 2

# Interaction features
df["λc_λled"]   = df["λc"]      * df["λ(le-d)"]
df["Pcrl_λled"] = df["Pcrl_Py"] * df["λ(le-d)"]
df["KL_λc"]     = df["KL_r"]    * df["λc"]
df["λc_Pne"]    = df["λc"]      * df["Pne_Py"]
df["λled_Pcrl"] = df["λ(le-d)"] * df["Pcrl_Py"]
df["Pne_Pcrl"]  = df["Pne_Py"]  * df["Pcrl_Py"]
df["Pcrl_Pne"]  = df["Pcrl_Py"] / df["Pne_Py"].replace(0, np.nan)
df["h_t_b_t"]   = df["h_t"]     * df["b_t"]
df["inv_λc"]    = 1.0            / df["λc"].replace(0, np.nan)
df["λc_inv_led"]= df["λc"]      / df["λ(le-d)"].replace(0, np.nan)

# NEW v8: DSM-inspired physics features
df["DSM_ne"]    = df["Pne_Py"]   # Global buckling term
df["DSM_l"]     = np.where(
    df["Pcrl_Py"] >= 0.776**2,
    1.0,
    (1 - 0.15 * df["Pcrl_Py"]**0.4) * df["Pcrl_Py"]**0.4
)  # DSM local strength term
df["λc_over_led"]= df["λc"] / df["λ(le-d)"].replace(0, np.nan)
df["sqrt_Pcrl"]  = np.sqrt(np.abs(df["Pcrl_Py"]))
df["sqrt_Pne"]   = np.sqrt(np.abs(df["Pne_Py"]))
df["KL_r_sq"]    = df["KL_r"] ** 2
df["Pne_x_Pcrl"] = df["Pne_Py"] * df["Pcrl_Py"]
df["λc_x_KL"]    = df["λc"] * df["KL_r"]
df["b_t_sq"]     = df["b_t"] ** 2
df["h_t_sq"]     = df["h_t"] ** 2

# Categorical encodings
le_fm = LabelEncoder(); df["FM_enc"]  = le_fm.fit_transform(df["FM"])
le_bc = LabelEncoder(); df["BC_enc"]  = le_bc.fit_transform(df["BC"].astype(str))
le_st = LabelEncoder(); df["ST_enc"]  = le_st.fit_transform(df["Section Types"].astype(str))

gm = df[TARGET].mean()
df["section_te"] = df.groupby("Section Types")[TARGET].transform("mean")
df["bc_te"]      = df.groupby("BC")[TARGET].transform("mean")
df["fm_te"]      = df.groupby("FM")[TARGET].transform("mean")

for c in df.select_dtypes(include=np.number).columns:
    df[c].fillna(df[c].median(), inplace=True)

FEATURES = [
    # Physics
    "λc", "λ(le-d)", "KL_r", "Pne_Py", "Pcrl_Py",
    "DSM_ne", "DSM_l", "sqrt_Pcrl", "sqrt_Pne",
    # Geometry
    "h_t", "b_t", "L_h", "L_t", "h_b", "A_t2", "Fy_norm",
    "h_t_sq", "b_t_sq",
    # Slenderness
    "λc_sq", "λled_sq", "λc3", "λled3",
    "Pne_sq", "Pcrl_sq", "KL_r_sq",
    # Interactions
    "λc_λled", "Pcrl_λled", "KL_λc", "λc_Pne", "λled_Pcrl",
    "Pne_Pcrl", "Pcrl_Pne", "h_t_b_t", "inv_λc", "λc_inv_led",
    "λc_over_led", "Pne_x_Pcrl", "λc_x_KL",
    # Categorical
    "FM_enc", "BC_enc", "ST_enc",
    "section_te", "bc_te", "fm_te",
]
print(f"✅  Features: {len(FEATURES)}")

# ── SPLIT ──────────────────────────────────────────────────────────────────────
X  = df[FEATURES].copy()
y  = df[TARGET].copy()
Py = df["Py"].values
FM = df["FM"].values

X_tr, X_te, y_tr, y_te, Py_tr, Py_te, FM_tr, FM_te = train_test_split(
    X, y, Py, FM, test_size=0.20, random_state=42
)
print(f"Train: {len(X_tr)}  |  Test: {len(X_te)}")

# Recompute target encoding on train only
for col, grp in [("section_te","Section Types"),("bc_te","BC"),("fm_te","FM")]:
    tr_map = df.loc[X_tr.index].groupby(grp)[TARGET].mean()
    X_tr[col] = df.loc[X_tr.index, grp].map(tr_map).fillna(gm).values
    X_te[col] = df.loc[X_te.index, grp].map(tr_map).fillna(gm).values
print("✅  Target encoding — no leakage")

# ── OPTUNA OBJECTIVE ───────────────────────────────────────────────────────────
def optuna_objective_xgb(trial, X_t, y_t, n_splits=5):
    params = dict(
        n_estimators    = trial.suggest_int("n_est",    3000, 10000, step=500),
        learning_rate   = trial.suggest_float("lr",     0.003, 0.02,  log=True),
        max_depth       = trial.suggest_int("depth",    4, 10),
        subsample       = trial.suggest_float("sub",    0.6, 0.95),
        colsample_bytree= trial.suggest_float("col",    0.5, 0.95),
        min_child_weight= trial.suggest_int("mcw",     1, 10),
        reg_alpha       = trial.suggest_float("alpha",  0.0, 2.0),
        reg_lambda      = trial.suggest_float("lambda", 0.0, 5.0),
        random_state=42, n_jobs=-1, verbosity=0,
        early_stopping_rounds=100 if _xgb_v >= 2 else None
    )
    if _xgb_v < 2: params.pop("early_stopping_rounds")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for tri, vai in kf.split(X_t):
        m = XGBRegressor(**params)
        fit_kw = {"eval_set": [(X_t[vai], y_t[vai])], "verbose": False}
        if _xgb_v < 2: fit_kw["early_stopping_rounds"] = 100
        m.fit(X_t[tri], y_t[tri], **fit_kw)
        scores.append(r2_score(y_t[vai], m.predict(X_t[vai])))
    return np.mean(scores)

def optuna_objective_lgb(trial, X_t, y_t, n_splits=5):
    params = dict(
        n_estimators    = trial.suggest_int("n_est",    3000, 10000, step=500),
        learning_rate   = trial.suggest_float("lr",     0.003, 0.02,  log=True),
        num_leaves      = trial.suggest_int("leaves",   64, 1024, log=True),
        subsample       = trial.suggest_float("sub",    0.6, 0.95),
        colsample_bytree= trial.suggest_float("col",    0.5, 0.95),
        reg_alpha       = trial.suggest_float("alpha",  0.0, 2.0),
        reg_lambda      = trial.suggest_float("lambda", 0.0, 5.0),
        min_child_samples=trial.suggest_int("mcs",     3, 30),
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_cbs = [lgb_lib.early_stopping(100, verbose=False), lgb_lib.log_evaluation(-1)]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for tri, vai in kf.split(X_t):
        m = LGBMRegressor(**params)
        m.fit(X_t[tri], y_t[tri], eval_set=[(X_t[vai], y_t[vai])], callbacks=lgb_cbs)
        scores.append(r2_score(y_t[vai], m.predict(X_t[vai])))
    return np.mean(scores)

# ── FM GROUP DEFINITIONS ────────────────────────────────────────────────────────
# الاستراتيجية: 3 موديلات متخصصة + 1 عام
FM_GROUPS = {
    "Flexural":      ["F", "LF"],
    "Local":         ["L", "LD"],
    "Distortional":  ["D", "DF", "LDF"],
}

# ── TRAIN FM-GROUPED MODELS ────────────────────────────────────────────────────
print("\n" + "═"*60)
print("  PHASE 1: Optuna Tuning per FM Group (50 trials each)")
print("═"*60)

X_tr_np = X_tr.values
y_tr_np = y_tr.values
X_te_np = X_te.values
FM_tr_arr = np.array(FM_tr)
FM_te_arr = np.array(FM_te)

oof_global  = np.full(len(X_tr), np.nan)
pred_global = np.full(len(X_te), np.nan)

# Track which test samples were handled by group models
handled_te = np.zeros(len(X_te), dtype=bool)
handled_tr = np.zeros(len(X_tr), dtype=bool)

for gname, fms in FM_GROUPS.items():
    tr_mask = np.isin(FM_tr_arr, fms)
    te_mask = np.isin(FM_te_arr, fms)
    n_tr = tr_mask.sum(); n_te = te_mask.sum()
    print(f"\n  Group [{gname}] — FMs: {fms}")
    print(f"  Train: {n_tr}  |  Test: {n_te}")
    if n_tr < 20:
        print(f"  ⚠️  Too few samples — skip group")
        continue

    Xg_tr = X_tr_np[tr_mask]
    yg_tr = y_tr_np[tr_mask]
    Xg_te = X_te_np[te_mask]

    # Optuna for XGB
    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(
        lambda t: optuna_objective_xgb(t, Xg_tr, yg_tr, n_splits=min(5, n_tr//10)),
        n_trials=50, show_progress_bar=False
    )
    best_xgb = study_xgb.best_params
    print(f"  XGB best R² = {study_xgb.best_value:.4f}  params: lr={best_xgb.get('lr',0):.4f}")

    # Optuna for LGB
    study_lgb = optuna.create_study(direction="maximize")
    study_lgb.optimize(
        lambda t: optuna_objective_lgb(t, Xg_tr, yg_tr, n_splits=min(5, n_tr//10)),
        n_trials=50, show_progress_bar=False
    )
    best_lgb = study_lgb.best_params
    print(f"  LGB best R² = {study_lgb.best_value:.4f}  params: leaves={best_lgb.get('leaves',0)}")

    # Build final XGB model on full group train
    p_xgb = dict(
        n_estimators=best_xgb.get("n_est",5000),
        learning_rate=best_xgb.get("lr",0.01),
        max_depth=best_xgb.get("depth",7),
        subsample=best_xgb.get("sub",0.8),
        colsample_bytree=best_xgb.get("col",0.7),
        min_child_weight=best_xgb.get("mcw",3),
        reg_alpha=best_xgb.get("alpha",0.1),
        reg_lambda=best_xgb.get("lambda",1.0),
        random_state=42, n_jobs=-1, verbosity=0
    )
    if _xgb_v >= 2: p_xgb["early_stopping_rounds"] = 100

    # 5-Fold OOF on group
    kf_g = KFold(n_splits=min(5, max(2, n_tr//20)), shuffle=True, random_state=42)
    oof_g_xgb = np.zeros(n_tr)
    oof_g_lgb = np.zeros(n_tr)
    pred_g_xgb = np.zeros(len(Xg_te) if len(Xg_te) > 0 else 1)
    pred_g_lgb = np.zeros(len(Xg_te) if len(Xg_te) > 0 else 1)
    lgb_cbs = [lgb_lib.early_stopping(100, verbose=False), lgb_lib.log_evaluation(-1)]
    n_folds_g = kf_g.n_splits

    for tri, vai in kf_g.split(Xg_tr):
        # XGB
        mx = XGBRegressor(**p_xgb)
        fit_kw = {"eval_set":[(Xg_tr[vai], yg_tr[vai])], "verbose":False}
        if _xgb_v < 2: fit_kw["early_stopping_rounds"] = 100
        mx.fit(Xg_tr[tri], yg_tr[tri], **fit_kw)
        oof_g_xgb[vai] = mx.predict(Xg_tr[vai])
        if len(Xg_te) > 0: pred_g_xgb += mx.predict(Xg_te) / n_folds_g

        # LGB
        p_lgb_final = dict(
            n_estimators=best_lgb.get("n_est",5000),
            learning_rate=best_lgb.get("lr",0.01),
            num_leaves=best_lgb.get("leaves",255),
            subsample=best_lgb.get("sub",0.8),
            colsample_bytree=best_lgb.get("col",0.7),
            reg_alpha=best_lgb.get("alpha",0.1),
            reg_lambda=best_lgb.get("lambda",1.0),
            min_child_samples=best_lgb.get("mcs",10),
            random_state=42, n_jobs=-1, verbose=-1
        )
        ml = LGBMRegressor(**p_lgb_final)
        ml.fit(Xg_tr[tri], yg_tr[tri], eval_set=[(Xg_tr[vai], yg_tr[vai])], callbacks=lgb_cbs)
        oof_g_lgb[vai] = ml.predict(Xg_tr[vai])
        if len(Xg_te) > 0: pred_g_lgb += ml.predict(Xg_te) / n_folds_g

    # Ridge blend for this group
    B_g_tr = np.column_stack([oof_g_xgb, oof_g_lgb])
    ridge_g = Ridge(alpha=0.01)
    ridge_g.fit(B_g_tr, yg_tr)
    oof_blend_g = ridge_g.predict(B_g_tr)

    r2_g = r2_score(yg_tr, oof_blend_g)
    print(f"  Group OOF R²(Pt/Py) = {r2_g:.4f}")

    # Store predictions
    tr_idx = np.where(tr_mask)[0]
    oof_global[tr_idx] = oof_blend_g
    handled_tr[tr_mask] = True

    if len(Xg_te) > 0:
        B_g_te = np.column_stack([pred_g_xgb, pred_g_lgb])
        pred_blend_g = ridge_g.predict(B_g_te)
        te_idx = np.where(te_mask)[0]
        pred_global[te_idx] = pred_blend_g
        handled_te[te_mask] = True

# ── PHASE 2: GLOBAL MODEL for remaining FM types ────────────────────────────────
print("\n" + "═"*60)
print("  PHASE 2: Global Model for all samples (fallback + blend)")
print("═"*60)

N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof_glb  = {k: np.zeros(len(X_tr)) for k in ["xgb","lgb","cat"]}
pte_glb  = {k: np.zeros(len(X_te)) for k in ["xgb","lgb","cat"]}

# Fixed high-performance params (tuned manually based on dataset)
PARAMS_XGB = dict(
    n_estimators=8000, learning_rate=0.008, max_depth=8,
    subsample=0.82, colsample_bytree=0.68,
    min_child_weight=2, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbosity=0
)
PARAMS_LGB = dict(
    n_estimators=8000, learning_rate=0.008, num_leaves=511,
    subsample=0.82, colsample_bytree=0.68,
    reg_alpha=0.1, reg_lambda=1.0, min_child_samples=5,
    random_state=42, n_jobs=-1, verbose=-1
)
PARAMS_CAT = dict(
    iterations=8000, learning_rate=0.008, depth=9,
    l2_leaf_reg=2.0, subsample=0.82,
    early_stopping_rounds=200, random_seed=42, verbose=0
)
if _xgb_v >= 2:
    PARAMS_XGB["early_stopping_rounds"] = 200
    _ES = {}
else:
    _ES = {"early_stopping_rounds": 200}
lgb_cbs = [lgb_lib.early_stopping(200, verbose=False), lgb_lib.log_evaluation(-1)]

print(f"  10-Fold Global OOF …")
for fold, (tri, vai) in enumerate(kf.split(X_tr_np), 1):
    Xf, Xv = X_tr_np[tri], X_tr_np[vai]
    yf, yv = y_tr_np[tri], y_tr_np[vai]

    mx = XGBRegressor(**PARAMS_XGB)
    mx.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False, **_ES)
    oof_glb["xgb"][vai] = mx.predict(Xv)
    pte_glb["xgb"]     += mx.predict(X_te_np) / N_FOLDS

    ml = LGBMRegressor(**PARAMS_LGB)
    ml.fit(Xf, yf, eval_set=[(Xv, yv)], callbacks=lgb_cbs)
    oof_glb["lgb"][vai] = ml.predict(Xv)
    pte_glb["lgb"]     += ml.predict(X_te_np) / N_FOLDS

    mc = CatBoostRegressor(**PARAMS_CAT)
    mc.fit(Xf, yf, eval_set=(Xv, yv), verbose=False)
    oof_glb["cat"][vai] = mc.predict(Xv)
    pte_glb["cat"]     += mc.predict(X_te_np) / N_FOLDS

    r2f = r2_score(yv, (oof_glb["xgb"][vai]+oof_glb["lgb"][vai]+oof_glb["cat"][vai])/3)
    print(f"    Fold {fold:2d}/{N_FOLDS}  R²(Pt/Py)={r2f:.4f}")

B_glb_tr = np.column_stack([oof_glb["xgb"], oof_glb["lgb"], oof_glb["cat"]])
B_glb_te = np.column_stack([pte_glb["xgb"], pte_glb["lgb"], pte_glb["cat"]])
ridge_glb = Ridge(alpha=0.01)
ridge_glb.fit(B_glb_tr, y_tr_np)
oof_glb_blend = ridge_glb.predict(B_glb_tr)
pte_glb_blend = ridge_glb.predict(B_glb_te)

print(f"  Global OOF R²(Pt/Py) = {r2_score(y_tr_np, oof_glb_blend):.4f}")

# ── PHASE 3: FINAL ENSEMBLE — Group + Global ───────────────────────────────────
print("\n" + "═"*60)
print("  PHASE 3: Final Ensemble (Group + Global)")
print("═"*60)

# For test: where group model exists, blend 60% group + 40% global
# where no group model, use 100% global
final_te = np.where(
    handled_te,
    0.60 * pred_global + 0.40 * pte_glb_blend,
    pte_glb_blend
)

# For train OOF: where group model exists, blend
final_tr_oof = np.where(
    handled_tr & ~np.isnan(oof_global),
    0.60 * oof_global + 0.40 * oof_glb_blend,
    oof_glb_blend
)

# ── METRICS ────────────────────────────────────────────────────────────────────
def report(label, ytrue, ypred, Py_vals=None, fm_vals=None):
    r2   = r2_score(ytrue, ypred)
    rmse = np.sqrt(mean_squared_error(ytrue, ypred))
    mae  = mean_absolute_error(ytrue, ypred)
    mape = np.mean(np.abs((ytrue-ypred)/(np.abs(ytrue)+1e-9)))*100
    ratio = ytrue / (ypred + 1e-9)
    mu  = ratio.mean(); cov = ratio.std() / ratio.mean()
    bar = "═" * 60
    print(f"\n{bar}\n  {label}\n{bar}")
    print(f"  R²(Pt/Py)  = {r2:.6f}")
    print(f"  RMSE       = {rmse:.6f}")
    print(f"  MAE        = {mae:.6f}")
    print(f"  MAPE       = {mape:.3f} %")
    print(f"  μ (Mean)   = {mu:.4f}")
    print(f"  COV        = {cov:.4f}")
    if Py_vals is not None:
        r2_kn = r2_score(ytrue*Py_vals, ypred*Py_vals)
        rmse_kn = np.sqrt(mean_squared_error(ytrue*Py_vals, ypred*Py_vals))
        print(f"  R²(Pt kN)  = {r2_kn:.6f}")
        print(f"  RMSE (kN)  = {rmse_kn:.2f}")
    print(f"  Benchmarks:")
    print(f"    R²>0.985  → {'✅' if r2>0.985  else '❌'}  ({r2:.4f})")
    print(f"    R²>0.980  → {'✅' if r2>0.980  else '❌'}  ({r2:.4f})")
    print(f"    MAPE<5%   → {'✅' if mape<5.0  else '❌'}  ({mape:.2f}%)")
    print(f"    MAPE<3%   → {'✅' if mape<3.0  else '⚠️ '}  ({mape:.2f}%)")
    print(f"    COV<0.09  → {'✅' if cov<0.09  else '⚠️ '}  ({cov:.4f})")
    print(f"    μ 0.98-1.02 → {'✅' if 0.98<=mu<=1.02 else '⚠️ '} ({mu:.4f})")
    if fm_vals is not None:
        df_e = pd.DataFrame({"FM":fm_vals,
                             "err_%":np.abs((ytrue-ypred)/ytrue)*100})
        print(f"\n  Error by FM:")
        print(df_e.groupby("FM")["err_%"]
              .agg(["mean","median","max","count"]).round(2).to_string())
    return r2

report("TRAIN OOF", y_tr_np, final_tr_oof)
r2_te = report("TEST ◀ KEY RESULT",
               y_te.values, final_te,
               Py_vals=Py_te,
               fm_vals=FM_te)

# ── PLOTS ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
colors = {"F":"red","LF":"orange","L":"blue","LD":"cyan",
          "D":"green","DF":"lime","LDF":"purple"}
c_arr = [colors.get(f, "gray") for f in FM_te]
sc = ax.scatter(y_te.values, final_te, alpha=0.6, s=25, c=c_arr)
lim = [0, max(float(y_te.max()), float(final_te.max()))*1.06]
ax.plot(lim, lim, "k--", lw=1.5, label="Perfect")
ax.fill_between(lim,[x*.95 for x in lim],[x*1.05 for x in lim],alpha=0.08,color="green",label="±5%")
ax.fill_between(lim,[x*.98 for x in lim],[x*1.02 for x in lim],alpha=0.10,color="blue",label="±2%")
ax.set_xlabel("Pt/Py  Experimental"); ax.set_ylabel("Pt/Py  Predicted")
ax.set_title(f"v8 BREAKTHROUGH  R²={r2_te:.4f}"); ax.legend(fontsize=8); ax.grid(alpha=0.25)

ax = axes[1]
pct = (y_te.values - final_te) / y_te.values * 100
ax.scatter(final_te, pct, alpha=0.6, s=25, c=c_arr)
ax.axhline(0,  c="k",lw=1.2)
ax.axhline(+5, c="orange",ls="--",lw=1.0,label="±5%")
ax.axhline(-5, c="orange",ls="--",lw=1.0)
ax.axhline(+2, c="green",ls=":",lw=0.8,label="±2%")
ax.axhline(-2, c="green",ls=":",lw=0.8)
ax.set_xlabel("Pt/Py Predicted"); ax.set_ylabel("Residual (%)")
ax.set_title("Residual by FM"); ax.legend(fontsize=8); ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("v8_scatter.png", bbox_inches="tight")
plt.show()
print("✅  v8_scatter.png saved")

# Save predictions
out = pd.DataFrame({
    "PtPy_actual":   y_te.values,
    "PtPy_pred_v8":  final_te,
    "PtPy_pred_global": pte_glb_blend,
    "Py_kN":         Py_te,
    "Pt_actual_kN":  y_te.values * Py_te,
    "Pt_pred_kN":    final_te    * Py_te,
    "FM":            FM_te,
    "err_%":         np.abs((y_te.values-final_te)/y_te.values)*100
})
out.to_csv("v8_predictions.csv", index=False)
print("✅  v8_predictions.csv saved")

print(f"\n{'═'*60}")
print(f"  🎯  v8 BREAKTHROUGH — COMPLETE")
print(f"  Final Test R²(Pt/Py) = {r2_te:.6f}")
print(f"{'═'*60}")
