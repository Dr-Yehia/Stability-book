#!/usr/bin/env python3
"""CFS v6 — Outlier removal + Huber loss + Direct Pt/Py prediction"""
import subprocess, sys
def pip(p): subprocess.run([sys.executable,"-m","pip","install","-q",p])
pip("catboost"); pip("lightgbm"); pip("shap")

import numpy as np, pandas as pd, matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["figure.dpi"]=150
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

# ── DATA ──
URL=("https://raw.githubusercontent.com/Dr-Yehia/Stability-book/main/"
     "CFS_Built-up_Columns_ML_Dataset.csv")
df=pd.read_csv(URL)
df=df[[c for c in df.columns if "Unnamed" not in c]].copy()
TARGET="Pt/Py"
df=df[df[TARGET].notna()].copy()
df["FM"]=df["FM"].str.strip()
for c in df.select_dtypes(include=np.number).columns:
    df[c].fillna(df[c].median(),inplace=True)

# ── REMOVE OUTLIERS ──
n_before=len(df)
df=df[(df[TARGET]>=0.08)&(df[TARGET]<=1.05)].copy()
print(f"✅ Outliers removed: {n_before}→{len(df)} ({n_before-len(df)} removed)")

# ── FEATURES ──
df["h_t"]=df["h"]/df["t"].replace(0,np.nan)
df["b_t"]=df["b"]/df["t"].replace(0,np.nan)
df["L_h"]=df["L"]/df["h"].replace(0,np.nan)
df["L_t"]=df["L"]/df["t"].replace(0,np.nan)
df["h_b"]=df["h"]/df["b"].replace(0,np.nan)
df["Pcrl_Py"]=df["P(crl,crd)"]/df["Py"].replace(0,np.nan)
df["Pne_Py"]=df["Pne"]/df["Py"].replace(0,np.nan)
df["λc_sq"]=df["λc"]**2
df["λled_sq"]=df["λ(le-d)"]**2
df["λc_λled"]=df["λc"]*df["λ(le-d)"]
df["Pcrl_λled"]=df["Pcrl_Py"]*df["λ(le-d)"]
df["KL_λc"]=df["KL_r"]*df["λc"]
df["λc_Pne"]=df["λc"]*df["Pne_Py"]
df["λled_Pcrl"]=df["λ(le-d)"]*df["Pcrl_Py"]
df["Pne_Pcrl"]=df["Pne_Py"]*df["Pcrl_Py"]
df["λc3"]=df["λc"]**3
df["λled3"]=df["λ(le-d)"]**3
df["inv_λc"]=1.0/df["λc"].replace(0,np.nan)
df["λc_inv_led"]=df["λc"]/df["λ(le-d)"].replace(0,np.nan)
df["log_Py"]=np.log1p(df["Py"])
df["log_Pne"]=np.log1p(df["Pne"])
df["log_Pcrl"]=np.log1p(df["P(crl,crd)"])
df["Pcrl_sq"]=df["Pcrl_Py"]**2
df["Pne_sq"]=df["Pne_Py"]**2
df["h_t_b_t"]=df["h_t"]*df["b_t"]
df["Fy_norm"]=df["Fy"]/350.0
df["A_t2"]=df["A"]/(df["t"]**2).replace(0,np.nan)

le_fm=LabelEncoder(); df["FM_enc"]=le_fm.fit_transform(df["FM"])
le_bc=LabelEncoder(); df["BC_enc"]=le_bc.fit_transform(df["BC"].astype(str))
le_st=LabelEncoder(); df["ST_enc"]=le_st.fit_transform(df["Section Types"].astype(str))

gm=df[TARGET].mean()
df["section_te"]=df.groupby("Section Types")[TARGET].transform("mean")
df["bc_te"]=df.groupby("BC")[TARGET].transform("mean")
df["fm_te"]=df.groupby("FM")[TARGET].transform("mean")

for c in df.select_dtypes(include=np.number).columns:
    df[c].fillna(df[c].median(),inplace=True)

FEATURES=[
    "λc","λ(le-d)","KL_r","h_t","b_t","L_h","L_t","h_b",
    "Pcrl_Py","Pne_Py","λc_sq","λled_sq","λc_λled","Pcrl_λled",
    "KL_λc","λc_Pne","λled_Pcrl","Pne_Pcrl","λc3","λled3",
    "inv_λc","λc_inv_led","log_Py","log_Pne","log_Pcrl",
    "Pcrl_sq","Pne_sq","h_t_b_t","Fy_norm","A_t2",
    "FM_enc","BC_enc","ST_enc","section_te","bc_te","fm_te",
]
print(f"✅ Features: {len(FEATURES)}")

# ── SPLIT ──
X=df[FEATURES].copy(); y=df[TARGET].copy()
Py_all=df["Py"].values; FM_all=df["FM"].values

X_tr,X_te,y_tr,y_te,Py_tr,Py_te,FM_tr,FM_te=train_test_split(
    X,y,Py_all,FM_all,test_size=0.20,random_state=42)
print(f"Train:{len(X_tr)} Test:{len(X_te)}")

for col,grp in [("section_te","Section Types"),("bc_te","BC"),("fm_te","FM")]:
    tr_map=df.loc[X_tr.index].groupby(grp)[TARGET].mean()
    X_tr[col]=df.loc[X_tr.index,grp].map(tr_map).fillna(gm).values
    X_te[col]=df.loc[X_te.index,grp].map(tr_map).fillna(gm).values

# ── HYPERPARAMS (Huber loss) ──
PARAMS_XGB=dict(
    n_estimators=8000,learning_rate=0.01,max_depth=9,
    subsample=0.80,colsample_bytree=0.65,
    min_child_weight=1,reg_alpha=0.05,reg_lambda=0.5,
    objective="reg:pseudohubererror",
    random_state=42,n_jobs=-1,verbosity=0)
PARAMS_LGB=dict(
    n_estimators=8000,learning_rate=0.01,num_leaves=512,
    max_depth=-1,subsample=0.80,colsample_bytree=0.65,
    reg_alpha=0.05,reg_lambda=0.5,min_child_samples=3,
    objective="huber",random_state=42,n_jobs=-1,verbose=-1)
PARAMS_CAT=dict(
    iterations=8000,learning_rate=0.01,depth=10,
    l2_leaf_reg=1.0,subsample=0.80,loss_function="Huber:delta=0.5",
    early_stopping_rounds=200,random_seed=42,verbose=0)

if _xgb_v>=2:
    PARAMS_XGB['early_stopping_rounds']=200; _ES={}
else:
    _ES={'early_stopping_rounds':200}
lgb_cbs=[lgb_lib.early_stopping(200,verbose=False),lgb_lib.log_evaluation(-1)]

# ── MULTI-SEED STACKING ──
N_FOLDS=10
SEEDS=[42,123,777]
oof_all=np.zeros(len(X_tr))
pte_all=np.zeros(len(X_te))
X_tr_np,y_tr_np,X_te_np=X_tr.values,y_tr.values,X_te.values

for seed_i,SEED in enumerate(SEEDS):
    fm_lab=LabelEncoder().fit_transform(df.loc[X_tr.index,"FM"].values)
    skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=SEED)
    oof={k:np.zeros(len(X_tr)) for k in ["xgb","lgb","cat"]}
    pte={k:np.zeros(len(X_te)) for k in ["xgb","lgb","cat"]}

    print(f"\n── Seed {SEED} ({seed_i+1}/{len(SEEDS)}) ──")
    for fold,(tri,vai) in enumerate(skf.split(X_tr_np,fm_lab),1):
        Xf,Xv=X_tr_np[tri],X_tr_np[vai]
        yf,yv=y_tr_np[tri],y_tr_np[vai]

        m=XGBRegressor(**PARAMS_XGB)
        m.fit(Xf,yf,eval_set=[(Xv,yv)],verbose=False,**_ES)
        oof["xgb"][vai]=m.predict(Xv); pte["xgb"]+=m.predict(X_te_np)/N_FOLDS

        m=LGBMRegressor(**PARAMS_LGB)
        m.fit(Xf,yf,eval_set=[(Xv,yv)],callbacks=lgb_cbs)
        oof["lgb"][vai]=m.predict(Xv); pte["lgb"]+=m.predict(X_te_np)/N_FOLDS

        m=CatBoostRegressor(**PARAMS_CAT)
        m.fit(Xf,yf,eval_set=(Xv,yv),verbose=False)
        oof["cat"][vai]=m.predict(Xv); pte["cat"]+=m.predict(X_te_np)/N_FOLDS

        avg=(oof["xgb"][vai]+oof["lgb"][vai]+oof["cat"][vai])/3
        print(f"  Fold {fold}/{N_FOLDS} R²={r2_score(yv,avg):.4f}")

    seed_oof=(oof["xgb"]+oof["lgb"]+oof["cat"])/3
    seed_pte=(pte["xgb"]+pte["lgb"]+pte["cat"])/3
    oof_all+=seed_oof/len(SEEDS)
    pte_all+=seed_pte/len(SEEDS)
    print(f"  Seed {SEED} OOF R²={r2_score(y_tr_np,seed_oof):.4f}")

# Clip predictions
oof_all=np.clip(oof_all,0.03,1.10)
pte_all=np.clip(pte_all,0.03,1.10)

# ── METRICS ──
def report(yt,yp,label,Py_v=None):
    r2=r2_score(yt,yp)
    rmse=np.sqrt(mean_squared_error(yt,yp))
    mae=mean_absolute_error(yt,yp)
    mape=np.mean(np.abs((yt-yp)/(np.abs(yt)+1e-9)))*100
    ratio=yt/(yp+1e-9); mu=ratio.mean(); cov=ratio.std()/ratio.mean()
    print(f"\n{'='*57}\n  {label}\n{'='*57}")
    print(f"  R²={r2:.6f}  RMSE={rmse:.6f}  MAE={mae:.6f}")
    print(f"  MAPE={mape:.3f}%  Mean={mu:.4f}  COV={cov:.4f}")
    print(f"  R²>0.981 {'✅' if r2>0.981 else '❌'}  R²>0.994 {'✅' if r2>0.994 else '⚠️ '}")
    print(f"  MAPE<5% {'✅' if mape<5 else '❌'}  MAPE<2.1% {'✅' if mape<2.1 else '⚠️ '}")
    print(f"  COV<0.09 {'✅' if cov<0.09 else '⚠️ '}  Mean∈[0.98,1.02] {'✅' if 0.98<=mu<=1.02 else '⚠️ '}")
    if Py_v is not None:
        rk=np.sqrt(mean_squared_error(yt*Py_v,yp*Py_v))
        print(f"  RMSE(kN)={rk:.1f}")
    return r2

report(y_tr_np,oof_all,"TRAIN (OOF)")
r2_te=report(y_te.values,pte_all,"TEST ◀ Key",Py_te)

# Error by FM
res=pd.DataFrame({"FM":FM_te,"actual":y_te.values,"pred":pte_all,
    "err_%":np.abs((y_te.values-pte_all)/y_te.values)*100})
print("\nError by FM:")
print(res.groupby("FM")["err_%"].agg(["mean","median","max","count"]).round(2).to_string())

# ── PLOTS ──
fig,axes=plt.subplots(1,2,figsize=(14,6))
ax=axes[0]
ax.scatter(y_te,pte_all,alpha=0.55,s=22,c="#01696f",label=f"R²={r2_te:.4f}")
lim=[0,max(y_te.max(),pte_all.max())*1.06]
ax.plot(lim,lim,"r--",lw=1.5); ax.set_xlabel("Experimental"); ax.set_ylabel("Predicted")
ax.set_title("v6.0 Huber+Outlier+MultiSeed"); ax.legend(); ax.grid(alpha=0.25)
ax=axes[1]
pct=(y_te.values-pte_all)/y_te.values*100
ax.scatter(pte_all,pct,alpha=0.55,s=22,c="#7a39bb")
ax.axhline(0,c="k"); ax.axhline(5,c="orange",ls="--"); ax.axhline(-5,c="orange",ls="--")
ax.set_xlabel("Predicted"); ax.set_ylabel("Residual(%)"); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig("v6_scatter.png",bbox_inches="tight"); plt.show()

# ── SHAP ──
print("\nSHAP…")
xf=XGBRegressor(n_estimators=3000,learning_rate=0.01,max_depth=9,
    subsample=0.8,colsample_bytree=0.65,random_state=42,n_jobs=-1,verbosity=0)
xf.fit(X_tr,y_tr)
sv=shap.TreeExplainer(xf).shap_values(X_te)
plt.figure(figsize=(10,7))
shap.summary_plot(sv,X_te,feature_names=FEATURES,show=False)
plt.tight_layout(); plt.savefig("v6_shap.png",bbox_inches="tight"); plt.show()
imp=pd.DataFrame({"Feature":FEATURES,"SHAP":np.abs(sv).mean(0)}).sort_values("SHAP",ascending=False)
print(f"Top-5: {imp.head(5)['Feature'].tolist()}")

pd.DataFrame({"actual":y_te.values,"pred":pte_all,"FM":FM_te}).to_csv("v6_predictions.csv",index=False)
print(f"\n{'='*57}\n  🎯 v6 Test R²(Pt/Py)={r2_te:.6f}\n{'='*57}")
