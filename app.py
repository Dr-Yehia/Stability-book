# app.py
# ============================================================
# CFS Built-up Column Strength Predictor
# V17 Streamlit Professional Interface
# Author: Dr-Yehia / Stability-book project
# Purpose:
#   - Load the V17 inference package from GitHub or uploaded ZIP.
#   - Run single-specimen prediction calculator.
#   - Run batch CSV prediction.
#   - Show official V16/V17 audit metrics.
#   - Show holdout, CV, Pareto, SHAP and group-error plots.
#   - Compare ML prediction with classical design-style curves.
# ============================================================

import os
import sys
import io
import json
import zipfile
import shutil
import tempfile
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================
# Page config
# ============================================================

st.set_page_config(
    page_title="CFS V17 Strength Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "CFS Built-up Column Strength Predictor — V17"
APP_SUBTITLE = (
    "No-fastener, no-direct-failure-mode, leakage-controlled ML framework "
    "for predicting normalized axial strength Pt/Py."
)

DEFAULT_GITHUB_ZIP_URL = (
    "https://github.com/Dr-Yehia/Stability-book/raw/"
    "claude/analyze-v16-results-AjbG2/results%20v17/cfs_v17_FINAL_PACKAGE.zip"
)

LOCAL_CACHE_DIR = Path(".cfs_v17_app_cache")
LOCAL_CACHE_DIR.mkdir(exist_ok=True)

PACKAGE_DIR = LOCAL_CACHE_DIR / "package"
ZIP_LOCAL_PATH = LOCAL_CACHE_DIR / "cfs_v17_FINAL_PACKAGE.zip"


# ============================================================
# Utilities
# ============================================================

def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def clean_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]].copy()


def find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for c in candidates:
        key = str(c).lower().strip()
        if key in lower_map:
            return lower_map[key]
    return None


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {}

    ratio = y_true / np.maximum(y_pred, 1e-12)

    return {
        "N": int(len(y_true)),
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-12))) * 100),
        "Mean_Test_over_Pred": float(np.mean(ratio)),
        "COV_Test_over_Pred": float(np.std(ratio) / np.maximum(np.mean(ratio), 1e-12)),
        "Unsafe_Overprediction_%": float(np.mean(y_pred > y_true) * 100),
        "Severe_Unsafe_5_%": float(np.mean(y_pred > 1.05 * y_true) * 100),
        "Severe_Unsafe_10_%": float(np.mean(y_pred > 1.10 * y_true) * 100),
    }


def metric_card(label, value, fmt="{:.4f}", help_text=None):
    if isinstance(value, (int, np.integer)):
        st.metric(label, f"{value:d}", help=help_text)
    elif isinstance(value, (float, np.floating)):
        st.metric(label, fmt.format(value), help=help_text)
    else:
        st.metric(label, str(value), help=help_text)


def download_file(url: str, dst: Path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as response:
        data = response.read()
    dst.write_bytes(data)
    return dst


def extract_zip(zip_path: Path, out_dir: Path):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    return out_dir


@st.cache_resource(show_spinner=False)
def load_package_from_path(zip_path_str: str):
    zip_path = Path(zip_path_str)
    extract_zip(zip_path, PACKAGE_DIR)

    sys.path.insert(0, str(PACKAGE_DIR))

    from cfs_v17_predict import CFSV17Predictor

    bundle_path = PACKAGE_DIR / "cfs_v17_inference_bundle.joblib"
    predictor = CFSV17Predictor.load(str(bundle_path))

    return predictor, PACKAGE_DIR


def load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default if default is not None else {}


def load_csv(path: Path, default=None):
    try:
        return pd.read_csv(path)
    except Exception:
        return default if default is not None else pd.DataFrame()


def read_package_files(pkg_dir: Path):
    files = {}

    files["summary"] = load_json(pkg_dir / "v16_final_summary.json", {})
    files["cv_summary"] = load_json(pkg_dir / "v16_repeated_cv_summary.json", {})
    files["feature_report"] = load_json(pkg_dir / "v16_feature_report.json", {})

    files["holdout"] = load_csv(pkg_dir / "v16_holdout_predictions.csv")
    files["cv_folds"] = load_csv(pkg_dir / "v16_repeated_cv_folds.csv")
    files["pareto"] = load_csv(pkg_dir / "v16_pareto_candidate_report.csv")
    files["err_group"] = load_csv(pkg_dir / "v16_error_by_design_group.csv")
    files["err_fm"] = load_csv(pkg_dir / "v16_error_by_true_FM_audit_only.csv")
    files["shap"] = load_csv(pkg_dir / "v16_shap_importance.csv")

    data_path = pkg_dir / "data" / "CFS_Built-up_Columns_ML_Dataset.csv"
    files["raw_data"] = clean_unnamed(load_csv(data_path))

    files["scatter_png"] = pkg_dir / "v16_scatter_holdout.png"
    files["residuals_png"] = pkg_dir / "v16_residuals_by_group.png"
    files["shap_png"] = pkg_dir / "v16_shap_summary.png"
    files["readme"] = (pkg_dir / "README_v17.md").read_text(encoding="utf-8", errors="ignore") if (pkg_dir / "README_v17.md").exists() else ""

    return files


# ============================================================
# Classical curve approximations
# ============================================================

def curve_aisc_ssrc(lambda_c):
    lc = np.maximum(np.asarray(lambda_c, dtype=float), 1e-12)
    return np.where(lc <= 1.5, 0.658 ** (lc ** 2), 0.877 / (lc ** 2))


def curve_ecp_lrfd_2012(lambda_c):
    lc = np.maximum(np.asarray(lambda_c, dtype=float), 1e-12)
    out = np.where(lc <= 1.10, 1.0 - 0.384 * (lc ** 2), 0.648 / (lc ** 2))
    return np.clip(out, 0.0, 1.5)


def curve_ec3(lambda_c, alpha=0.34, gamma=1.0):
    lc = np.maximum(np.asarray(lambda_c, dtype=float), 1e-12)
    phi = 0.5 * (1.0 + alpha * (lc - 0.2) + lc ** 2)
    rad = np.maximum(phi ** 2 - lc ** 2, 0.0)
    chi = 1.0 / np.maximum(phi + np.sqrt(rad), 1e-12)
    chi = np.minimum(chi, 1.0)
    return chi / gamma


def curve_euler_like(KLr):
    # Dimensionless visualization only.
    # Uses a normalized elastic curve shape. Not a substitute for code design.
    x = np.maximum(np.asarray(KLr, dtype=float), 1e-12)
    ref = 100.0
    out = (ref / x) ** 2
    return np.clip(out, 0.0, 1.5)


def dsm_global(lambda_c):
    return curve_aisc_ssrc(lambda_c)


def dsm_local(lambda_led, pne_py):
    lam = np.maximum(np.asarray(lambda_led, dtype=float), 1e-12)
    pne = np.asarray(pne_py, dtype=float)
    reduction = np.where(
        lam <= 0.776,
        1.0,
        (1.0 - 0.15 / (lam ** 0.8 + 1e-12)) / (lam ** 0.8 + 1e-12),
    )
    return np.clip(pne * reduction, 0.0, 1.5)


def compute_baseline_table(row):
    lambda_c = safe_float(row.get("lambda_c"))
    lambda_led = safe_float(row.get("lambda_led"))
    KLr = safe_float(row.get("KLr"))
    Py = safe_float(row.get("Py"))
    Pne = safe_float(row.get("Pne"))
    Pcrl = safe_float(row.get("Pcrl"))

    pne_py = Pne / Py if np.isfinite(Pne) and np.isfinite(Py) and abs(Py) > 1e-12 else np.nan

    methods = {
        "AISC / SSRC curve": curve_aisc_ssrc(lambda_c),
        "ECP LRFD 2012 curve": curve_ecp_lrfd_2012(lambda_c),
        "EC3 curve alpha=0.34": curve_ec3(lambda_c, alpha=0.34, gamma=1.0),
        "EC3 curve alpha=0.21": curve_ec3(lambda_c, alpha=0.21, gamma=1.0),
        "Euler-like reference": curve_euler_like(KLr),
        "DSM global baseline": dsm_global(lambda_c),
        "DSM local baseline": dsm_local(lambda_led, pne_py),
    }

    records = []
    for name, ratio in methods.items():
        ratio = float(np.asarray(ratio).reshape(-1)[0]) if np.isfinite(np.asarray(ratio)).all() else np.nan
        records.append({
            "Method": name,
            "Predicted Pt/Py": ratio,
            "Predicted Pt": ratio * Py if np.isfinite(Py) else np.nan,
        })
    return pd.DataFrame(records)


# ============================================================
# Plotting functions
# ============================================================

def plot_actual_vs_pred(df, actual_col, pred_col, title):
    temp = df[[actual_col, pred_col]].dropna().copy()
    y = temp[actual_col].astype(float)
    p = temp[pred_col].astype(float)
    m = compute_metrics(y, p)

    fig = px.scatter(
        temp,
        x=actual_col,
        y=pred_col,
        opacity=0.72,
        title=f"{title} | R²={m.get('R2', np.nan):.4f}, MAPE={m.get('MAPE_%', np.nan):.2f}%, N={m.get('N', 0)}",
        labels={actual_col: "Experimental Pt/Py", pred_col: "Predicted Pt/Py"},
    )

    mn = float(min(y.min(), p.min()))
    mx = float(max(y.max(), p.max()))
    fig.add_trace(go.Scatter(
        x=[mn, mx],
        y=[mn, mx],
        mode="lines",
        name="Ideal y=x",
        line=dict(color="black", dash="dash"),
    ))
    fig.update_layout(height=620)
    return fig


def plot_residuals(df, actual_col, pred_col):
    temp = df[[actual_col, pred_col]].dropna().copy()
    temp["Residual"] = temp[pred_col] - temp[actual_col]
    fig = px.histogram(
        temp,
        x="Residual",
        nbins=35,
        title="Residual Distribution: Predicted - Experimental",
    )
    fig.update_layout(height=480)
    return fig


def plot_abs_error(df):
    if "AbsErr_%" not in df.columns:
        return None
    fig = px.histogram(
        df,
        x="AbsErr_%",
        nbins=35,
        title="Absolute Percentage Error Distribution",
        labels={"AbsErr_%": "Absolute Error (%)"},
    )
    fig.update_layout(height=480)
    return fig


def plot_cv(cv):
    if cv.empty or "fold" not in cv.columns:
        return None, None

    fig_r2 = px.line(
        cv,
        x="fold",
        y="R2",
        markers=True,
        title=f"Repeated CV R² per Fold | mean={cv['R2'].mean():.4f}, std={cv['R2'].std():.4f}",
    )
    fig_r2.add_hline(y=cv["R2"].mean(), line_dash="dash")
    fig_r2.update_layout(height=450)

    fig_mape = px.line(
        cv,
        x="fold",
        y="MAPE_%",
        markers=True,
        title=f"Repeated CV MAPE per Fold | mean={cv['MAPE_%'].mean():.2f}%",
    )
    fig_mape.add_hline(y=cv["MAPE_%"].mean(), line_dash="dash")
    fig_mape.update_layout(height=450)

    return fig_r2, fig_mape


def plot_group_errors(df, group_col, title):
    if df.empty or group_col not in df.columns:
        return None

    if "AbsErr_%" in df.columns:
        g = df.groupby(group_col)["AbsErr_%"].agg(["mean", "median", "max", "count"]).reset_index()
    elif {"mean", "median", "max", "count"}.issubset(set(df.columns)):
        g = df.copy()
        if group_col not in g.columns:
            group_col = g.columns[0]
    else:
        return None

    g = g.sort_values("mean", ascending=False)

    fig = px.bar(
        g,
        x=group_col,
        y="mean",
        text="count",
        title=title,
        labels={"mean": "Mean Absolute Error (%)", "count": "N"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=520, xaxis_tickangle=-35)
    return fig


def plot_pareto(pareto):
    if pareto.empty:
        return None
    needed = {"Unsafe_Overprediction_%", "R2", "MAPE_%", "name"}
    if not needed.issubset(set(pareto.columns)):
        return None

    fig = px.scatter(
        pareto,
        x="Unsafe_Overprediction_%",
        y="R2",
        size=np.maximum(1, 12 - pareto["MAPE_%"]),
        color="MAPE_%",
        hover_name="name",
        title="Pareto Candidates: Accuracy vs Unsafe Overprediction",
        labels={
            "Unsafe_Overprediction_%": "Unsafe Overprediction (%)",
            "R2": "OOF R²",
            "MAPE_%": "MAPE (%)",
        },
    )
    fig.update_layout(height=600)
    return fig


def plot_shap(shap):
    if shap.empty:
        return None
    if "Feature" not in shap.columns:
        return None

    value_col = "MeanAbsSHAP" if "MeanAbsSHAP" in shap.columns else shap.columns[-1]
    top = shap.sort_values(value_col, ascending=False).head(25).copy()
    top = top.iloc[::-1]

    fig = px.bar(
        top,
        x=value_col,
        y="Feature",
        orientation="h",
        title="Top SHAP Feature Importance",
        labels={value_col: "Mean |SHAP|"},
    )
    fig.update_layout(height=700)
    return fig


# ============================================================
# Package loading section
# ============================================================

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

with st.sidebar:
    st.header("Package Source")

    source_mode = st.radio(
        "Load package from:",
        ["GitHub URL", "Upload ZIP manually"],
        index=0,
    )

    github_url = st.text_input(
        "GitHub package URL",
        value=DEFAULT_GITHUB_ZIP_URL,
        help="Direct GitHub raw ZIP URL for cfs_v17_FINAL_PACKAGE.zip.",
    )

    uploaded_zip = None
    if source_mode == "Upload ZIP manually":
        uploaded_zip = st.file_uploader(
            "Upload cfs_v17_FINAL_PACKAGE.zip",
            type=["zip"],
        )

    load_btn = st.button("Load / Reload Model Package", type="primary")

    st.divider()
    st.markdown("### Required raw inputs")
    st.markdown(
        """
        - SectionType / Section Types  
        - Sections  
        - BC  
        - L, t, h, b, A  
        - Fy, Py  
        - Pcrl / P(crl,crd)  
        - Pne  
        - KLr / KL_r  
        - lambda_c / λc  
        - lambda_led / λ(le-d)  
        """
    )
    st.warning("FM is not required for prediction. It is audit-only in the research pipeline.")
if load_btn or "package_loaded" not in st.session_state:
    try:
        repo_zip_candidates = [
            Path("results v17") / "cfs_v17_FINAL_PACKAGE.zip",
            Path("results%20v17") / "cfs_v17_FINAL_PACKAGE.zip",
            Path("cfs_v17_FINAL_PACKAGE.zip"),
        ]

        local_repo_zip = None
        for candidate in repo_zip_candidates:
            if candidate.exists():
                local_repo_zip = candidate
                break

        if source_mode == "Upload ZIP manually" and uploaded_zip is not None:
            ZIP_LOCAL_PATH.write_bytes(uploaded_zip.getvalue())
            package_source_msg = "uploaded ZIP"

        elif local_repo_zip is not None:
            shutil.copyfile(local_repo_zip, ZIP_LOCAL_PATH)
            package_source_msg = f"local repository ZIP: {local_repo_zip}"

        else:
            with st.spinner("Downloading V17 package from GitHub..."):
                download_file(github_url, ZIP_LOCAL_PATH)
            package_source_msg = "GitHub raw URL"

        with st.spinner(f"Extracting and loading model package from {package_source_msg}..."):
            predictor, pkg_dir = load_package_from_path(str(ZIP_LOCAL_PATH))
            files = read_package_files(pkg_dir)

        st.session_state["predictor"] = predictor
        st.session_state["pkg_dir"] = pkg_dir
        st.session_state["files"] = files
        st.session_state["package_loaded"] = True

        st.success(f"V17 package loaded successfully from {package_source_msg}.")

    except Exception as e:
        st.error("Could not load the package.")
        st.exception(e)
        st.stop()
predictor = st.session_state["predictor"]
pkg_dir = st.session_state["pkg_dir"]
files = st.session_state["files"]

summary = files["summary"]
cv_summary = files["cv_summary"]
holdout = files["holdout"]
cv_folds = files["cv_folds"]
pareto = files["pareto"]
err_group = files["err_group"]
err_fm = files["err_fm"]
shap = files["shap"]
raw_data = files["raw_data"]


# ============================================================
# Tabs
# ============================================================

tabs = st.tabs([
    "Executive Dashboard",
    "Single Specimen Calculator",
    "Batch Prediction",
    "Official Results & Graphs",
    "Full Dataset Explorer",
    "Classical Curves Comparison",
    "Model Documentation",
])


# ============================================================
# Executive Dashboard
# ============================================================

with tabs[0]:
    st.subheader("Executive Dashboard")

    holdout_metrics = summary.get("holdout_test_metrics", {})
    train_metrics = summary.get("holdout_train_metrics", {})
    v17_test_metrics = getattr(predictor, "bundle", {}).get("test_metrics", {}) if hasattr(predictor, "bundle") else {}

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Official Holdout R²", holdout_metrics.get("TEST_R2", np.nan))
    with c2:
        metric_card("Official Holdout MAPE", holdout_metrics.get("TEST_MAPE_%", np.nan), "{:.2f}%")
    with c3:
        metric_card("Repeated CV R² mean", cv_summary.get("R2_mean", np.nan))
    with c4:
        metric_card("Repeated CV MAPE mean", cv_summary.get("MAPE_%_mean", np.nan), "{:.2f}%")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        metric_card("Official Test COV", holdout_metrics.get("TEST_COV_Test_over_Pred", np.nan))
    with c6:
        metric_card("Official Unsafe", holdout_metrics.get("TEST_Unsafe_Overprediction_%", np.nan), "{:.2f}%")
    with c7:
        metric_card("Test N", holdout_metrics.get("TEST_N", 0))
    with c8:
        metric_card("Dataset rows in package", len(raw_data))

    st.markdown("### Scientific interpretation")

    st.info(
        """
        This app separates two models:
        
        1. **Official research pipeline**: reported by the V16 Pareto calibrated stack.
        2. **Production inference model**: loaded from `cfs_v17_inference_bundle.joblib` and used by this app.
        
        Use the official research metrics for publication tables. Use the V17 inference model for programmatic prediction.
        """
    )

    if not holdout.empty:
        fig = plot_actual_vs_pred(holdout, "PtPy_actual", "PtPy_pred", "Official Holdout: Experimental vs Predicted")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show official JSON summary"):
        st.json(summary)


# ============================================================
# Single Specimen Calculator
# ============================================================

with tabs[1]:
    st.subheader("Single Specimen Strength Calculator")

    st.markdown(
        """
        Enter raw specimen/design variables. The app will compute the predicted normalized strength **Pt/Py**
        and the predicted ultimate axial load **Pt**.
        """
    )

    # Build options from data
    section_type_col = find_col(raw_data, ["SectionType", "Section Types", "Section_Type"])
    sections_col = find_col(raw_data, ["Sections"])
    bc_col = find_col(raw_data, ["BC"])

    section_types = sorted(raw_data[section_type_col].dropna().astype(str).unique()) if section_type_col else ["O-2C", "O-2U", "C-U+C", "C-2C-1"]
    sections = sorted(raw_data[sections_col].dropna().astype(str).unique()) if sections_col else ["Open", "Closed"]
    bcs = sorted(raw_data[bc_col].dropna().astype(str).unique()) if bc_col else ["Pinned", "Fixed"]

    left, right = st.columns([1.1, 1.0])

    with left:
        st.markdown("#### Categorical inputs")
        SectionType = st.selectbox("Section Type", section_types, index=0)
        Sections = st.selectbox("Section Family", sections, index=0)
        BC = st.selectbox("Boundary Condition", bcs, index=0)

        st.markdown("#### Geometry")
        L = st.number_input("L", min_value=0.0, value=1000.0, step=10.0)
        t = st.number_input("t", min_value=0.0001, value=1.5, step=0.1)
        h = st.number_input("h", min_value=0.0, value=100.0, step=1.0)
        b = st.number_input("b", min_value=0.0, value=50.0, step=1.0)
        A = st.number_input("A", min_value=0.0, value=250.0, step=1.0)

    with right:
        st.markdown("#### Material and buckling inputs")
        Fy = st.number_input("Fy", min_value=0.0, value=350.0, step=1.0)
        Py = st.number_input("Py", min_value=0.0001, value=100.0, step=1.0)
        Pcrl = st.number_input("Pcrl / P(crl,crd)", min_value=0.0, value=80.0, step=1.0)
        Pne = st.number_input("Pne", min_value=0.0, value=70.0, step=1.0)
        KLr = st.number_input("KLr / KL_r", min_value=0.0, value=60.0, step=1.0)
        lambda_c = st.number_input("lambda_c / λc", min_value=0.0, value=0.9, step=0.01)
        lambda_led = st.number_input("lambda_led / λ(le-d)", min_value=0.0, value=0.8, step=0.01)

        st.markdown("#### Optional actual result")
        actual_ptpy = st.number_input("Experimental Pt/Py, if known", min_value=0.0, value=0.0, step=0.01)

    one = pd.DataFrame([{
        "SectionType": SectionType,
        "Section Types": SectionType,
        "Sections": Sections,
        "BC": BC,
        "L": L,
        "t": t,
        "h": h,
        "b": b,
        "A": A,
        "Fy": Fy,
        "Py": Py,
        "Pcrl": Pcrl,
        "P(crl,crd)": Pcrl,
        "Pne": Pne,
        "KLr": KLr,
        "KL_r": KLr,
        "lambda_c": lambda_c,
        "λc": lambda_c,
        "lambda_led": lambda_led,
        "λ(le-d)": lambda_led,
    }])

    if st.button("Predict strength", type="primary"):
        try:
            pred = predictor.predict(one)
            ptpy_pred = float(pred["PtPy_pred"].iloc[0])
            pt_pred = ptpy_pred * Py

            st.markdown("### Prediction result")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                metric_card("Predicted Pt/Py", ptpy_pred)
            with m2:
                metric_card("Predicted Pt", pt_pred, "{:.3f}")
            with m3:
                metric_card("Estimated MAPE reference", v17_test_metrics.get("V17_INFERENCE_TEST_MAPE_%", np.nan), "{:.2f}%")
            with m4:
                if actual_ptpy > 0:
                    err = abs((actual_ptpy - ptpy_pred) / actual_ptpy) * 100
                    metric_card("Absolute Error", err, "{:.2f}%")
                else:
                    st.metric("Absolute Error", "N/A")

            st.markdown("### Classical curve comparison")
            base_table = compute_baseline_table(one.iloc[0].to_dict())
            base_table = pd.concat([
                pd.DataFrame([{
                    "Method": "V17 ML inference model",
                    "Predicted Pt/Py": ptpy_pred,
                    "Predicted Pt": pt_pred,
                }]),
                base_table
            ], ignore_index=True)
            st.dataframe(base_table, use_container_width=True)

            fig = px.bar(
                base_table,
                x="Method",
                y="Predicted Pt/Py",
                title="Single Specimen: ML vs Classical Curves",
                text="Predicted Pt/Py",
            )
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig.update_layout(height=520, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Raw model output"):
                st.dataframe(pred, use_container_width=True)

        except Exception as e:
            st.error("Prediction failed. Check required inputs and package compatibility.")
            st.exception(e)


# ============================================================
# Batch Prediction
# ============================================================

with tabs[2]:
    st.subheader("Batch CSV Prediction")

    st.markdown(
        """
        Upload a CSV containing the required raw columns.  
        If the file includes an actual target column such as `Pt/Py`, the app will calculate evaluation metrics.
        """
    )

    sample_cols = [
        "SectionType", "Sections", "BC", "L", "t", "h", "b", "A", "Fy", "Py",
        "Pcrl", "Pne", "KLr", "lambda_c", "lambda_led"
    ]

    sample_df = pd.DataFrame([{
        "SectionType": "O-2C",
        "Sections": "Open",
        "BC": "Pinned",
        "L": 1000,
        "t": 1.5,
        "h": 100,
        "b": 50,
        "A": 250,
        "Fy": 350,
        "Py": 100,
        "Pcrl": 80,
        "Pne": 70,
        "KLr": 60,
        "lambda_c": 0.9,
        "lambda_led": 0.8,
    }])

    st.download_button(
        "Download input template CSV",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="cfs_v17_input_template.csv",
        mime="text/csv",
    )

    uploaded_csv = st.file_uploader("Upload input CSV", type=["csv"], key="batch_csv")

    if uploaded_csv is not None:
        batch = pd.read_csv(uploaded_csv)
        batch = clean_unnamed(batch)

        st.markdown("#### Uploaded data preview")
        st.dataframe(batch.head(20), use_container_width=True)

        if st.button("Run batch prediction", type="primary"):
            try:
                out = predictor.predict(batch)
                result = pd.concat([batch.reset_index(drop=True), out.reset_index(drop=True)], axis=1)

                if "Py" in result.columns and "PtPy_pred" in result.columns:
                    result["Pt_pred"] = result["PtPy_pred"] * pd.to_numeric(result["Py"], errors="coerce")

                target_col = find_col(result, ["Pt/Py", "PtPy_actual", "PtPy", "Ptest/Py"])
                if target_col is not None:
                    metrics = compute_metrics(result[target_col], result["PtPy_pred"])

                    st.markdown("### Batch Evaluation Metrics")
                    cols = st.columns(5)
                    keys = ["N", "R2", "RMSE", "MAPE_%", "Unsafe_Overprediction_%"]
                    for c, k in zip(cols, keys):
                        with c:
                            val = metrics.get(k, np.nan)
                            if k in ["MAPE_%", "Unsafe_Overprediction_%"]:
                                metric_card(k, val, "{:.2f}%")
                            elif k == "N":
                                metric_card(k, val)
                            else:
                                metric_card(k, val)

                    fig = plot_actual_vs_pred(result, target_col, "PtPy_pred", "Batch Prediction")
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Predictions")
                st.dataframe(result, use_container_width=True)

                st.download_button(
                    "Download predictions CSV",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="cfs_v17_batch_predictions.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error("Batch prediction failed.")
                st.exception(e)


# ============================================================
# Official Results & Graphs
# ============================================================

with tabs[3]:
    st.subheader("Official Results and Graphs")

    st.markdown("### Official holdout metrics")
    holdout_metrics = summary.get("holdout_test_metrics", {})
    cols = st.columns(4)
    for col, key in zip(cols, ["TEST_R2", "TEST_RMSE", "TEST_MAPE_%", "TEST_Unsafe_Overprediction_%"]):
        with col:
            fmt = "{:.2f}%" if key.endswith("%") or "Unsafe" in key else "{:.4f}"
            metric_card(key, holdout_metrics.get(key, np.nan), fmt)

    st.markdown("### Repeated CV summary")
    cv_cols = st.columns(4)
    for col, key in zip(cv_cols, ["R2_mean", "R2_std", "MAPE_%_mean", "Unsafe_Overprediction_%_mean"]):
        with col:
            fmt = "{:.2f}%" if "MAPE" in key or "Unsafe" in key else "{:.4f}"
            metric_card(key, cv_summary.get(key, np.nan), fmt)

    st.divider()

    fig = plot_actual_vs_pred(holdout, "PtPy_actual", "PtPy_pred", "Official Holdout")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_res = plot_residuals(holdout, "PtPy_actual", "PtPy_pred")
        st.plotly_chart(fig_res, use_container_width=True)
    with c2:
        fig_abs = plot_abs_error(holdout)
        if fig_abs:
            st.plotly_chart(fig_abs, use_container_width=True)

    fig_cv_r2, fig_cv_mape = plot_cv(cv_folds)
    if fig_cv_r2:
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(fig_cv_r2, use_container_width=True)
        with c4:
            st.plotly_chart(fig_cv_mape, use_container_width=True)

    st.markdown("### Error by design group")
    fig_g = plot_group_errors(holdout, "SG_design", "Holdout Mean Absolute Error by Design Group")
    if fig_g:
        st.plotly_chart(fig_g, use_container_width=True)

    st.markdown("### Error by failure mode")
    if "FM_true_not_directly_used" in holdout.columns:
        fig_fm = plot_group_errors(
            holdout,
            "FM_true_not_directly_used",
            "Holdout Mean Absolute Error by Failure Mode - Audit Only",
        )
        if fig_fm:
            st.plotly_chart(fig_fm, use_container_width=True)

    st.markdown("### Pareto candidate analysis")
    fig_p = plot_pareto(pareto)
    if fig_p:
        st.plotly_chart(fig_p, use_container_width=True)

    st.markdown("### SHAP importance")
    fig_s = plot_shap(shap)
    if fig_s:
        st.plotly_chart(fig_s, use_container_width=True)

    with st.expander("Original exported figures from the package"):
        img_cols = st.columns(3)
        image_paths = [files["scatter_png"], files["residuals_png"], files["shap_png"]]
        captions = ["Holdout scatter", "Residuals by group", "SHAP summary"]
        for col, p, cap in zip(img_cols, image_paths, captions):
            with col:
                if p.exists():
                    st.image(str(p), caption=cap, use_container_width=True)


# ============================================================
# Full Dataset Explorer
# ============================================================

with tabs[4]:
    st.subheader("Full Dataset Explorer")

    st.warning(
        """
        Full-dataset plots are useful for exploration, but not publication-grade validation because
        they include training specimens. Use holdout and repeated-CV results for official reporting.
        """
    )

    st.markdown(f"Dataset rows in package: **{len(raw_data)}**")
    st.dataframe(raw_data.head(30), use_container_width=True)

    if st.button("Run V17 inference on full dataset"):
        try:
            with st.spinner("Predicting full dataset..."):
                full_pred = predictor.predict(raw_data)
                full_result = pd.concat([raw_data.reset_index(drop=True), full_pred.reset_index(drop=True)], axis=1)

            target_col = find_col(full_result, ["Pt/Py", "PtPy_actual", "PtPy", "Ptest/Py"])
            if target_col is not None:
                metrics = compute_metrics(full_result[target_col], full_result["PtPy_pred"])
                st.markdown("### Full dataset inference metrics")
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    metric_card("N", metrics["N"])
                with c2:
                    metric_card("R²", metrics["R2"])
                with c3:
                    metric_card("MAPE", metrics["MAPE_%"], "{:.2f}%")
                with c4:
                    metric_card("COV", metrics["COV_Test_over_Pred"])
                with c5:
                    metric_card("Unsafe", metrics["Unsafe_Overprediction_%"], "{:.2f}%")

                fig = plot_actual_vs_pred(full_result, target_col, "PtPy_pred", "V17 Inference on Full Dataset")
                st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "Download full dataset predictions",
                data=full_result.to_csv(index=False).encode("utf-8"),
                file_name="v17_full_dataset_predictions.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error("Full dataset prediction failed.")
            st.exception(e)


# ============================================================
# Classical Curves Comparison
# ============================================================

with tabs[5]:
    st.subheader("Classical Curves Comparison")

    st.markdown(
        """
        This page compares the ML prediction with common column-style curves.
        These classical curves are implemented as research visualization baselines.
        For formal code design, verify all definitions, units, resistance factors, and code clauses.
        """
    )

    if not holdout.empty:
        comp = holdout.copy()

        # Need lambda values from raw dataset by original row index if available
        if "row_index_original" in comp.columns and not raw_data.empty:
            idx = comp["row_index_original"].astype(int).values
            raw_lookup = raw_data.reset_index(drop=True)
            valid = idx[(idx >= 0) & (idx < len(raw_lookup))]
            aux = raw_lookup.iloc[valid].copy()

            # Align safely
            comp_aux = raw_lookup.iloc[np.clip(idx, 0, len(raw_lookup) - 1)].reset_index(drop=True)

            lam_col = find_col(comp_aux, ["lambda_c", "λc"])
            led_col = find_col(comp_aux, ["lambda_led", "λ(le-d)"])
            klr_col = find_col(comp_aux, ["KLr", "KL_r", "KL/r"])
            pne_col = find_col(comp_aux, ["Pne"])
            py_col = find_col(comp_aux, ["Py"])

            if lam_col:
                lam = pd.to_numeric(comp_aux[lam_col], errors="coerce").values
                comp["AISC_SSRC"] = curve_aisc_ssrc(lam)
                comp["ECP_LRFD_2012"] = curve_ecp_lrfd_2012(lam)
                comp["EC3_alpha_034"] = curve_ec3(lam, alpha=0.34)
                comp["EC3_alpha_021"] = curve_ec3(lam, alpha=0.21)
                comp["DSM_global"] = dsm_global(lam)

            if led_col and pne_col and py_col:
                led = pd.to_numeric(comp_aux[led_col], errors="coerce").values
                pne = pd.to_numeric(comp_aux[pne_col], errors="coerce").values
                py = pd.to_numeric(comp_aux[py_col], errors="coerce").values
                comp["DSM_local_calc"] = dsm_local(led, pne / np.maximum(py, 1e-12))

            if klr_col:
                klr = pd.to_numeric(comp_aux[klr_col], errors="coerce").values
                comp["Euler_like"] = curve_euler_like(klr)

            methods = ["PtPy_pred", "AISC_SSRC", "ECP_LRFD_2012", "EC3_alpha_034", "EC3_alpha_021", "DSM_global", "DSM_local_calc", "Euler_like"]
            methods = [m for m in methods if m in comp.columns]

            records = []
            for m in methods:
                met = compute_metrics(comp["PtPy_actual"], comp[m])
                if met:
                    records.append({
                        "Method": "V17 Official Holdout Prediction" if m == "PtPy_pred" else m,
                        **met
                    })

            metrics_table = pd.DataFrame(records).sort_values("R2", ascending=False)
            st.dataframe(metrics_table, use_container_width=True)

            fig = px.bar(
                metrics_table,
                x="Method",
                y="R2",
                title="Holdout R²: ML vs Classical Curves",
                text="R2",
            )
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig.update_layout(height=520, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.bar(
                metrics_table,
                x="Method",
                y="MAPE_%",
                title="Holdout MAPE: ML vs Classical Curves",
                text="MAPE_%",
            )
            fig2.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            fig2.update_layout(height=520, xaxis_tickangle=-30)
            st.plotly_chart(fig2, use_container_width=True)

        else:
            st.info("Holdout row indexes or raw data are not available for classical curve comparison.")


# ============================================================
# Model Documentation
# ============================================================

with tabs[6]:
    st.subheader("Model Documentation")

    st.markdown(
        """
        ## What this app does
        
        This app loads the V17 production inference package and provides:
        
        - Single specimen strength prediction.
        - Batch CSV prediction.
        - Official holdout and repeated-CV review.
        - Error analysis by design group and failure mode.
        - SHAP feature-importance visualization.
        - Classical curve comparison.
        - Full-dataset exploratory prediction.
        
        ## Scientific status
        
        The official publication metrics should come from:
        
        - `v16_final_summary.json`
        - `v16_repeated_cv_summary.json`
        - `v16_repeated_cv_folds.csv`
        - `v16_holdout_predictions.csv`
        
        The Streamlit calculator uses:
        
        - `cfs_v17_inference_bundle.joblib`
        - `cfs_v17_predict.py`
        
        ## Important engineering note
        
        The V17 inference model is intended for prediction and software integration.
        The official scientific evidence remains the holdout and repeated-CV audit.
        For design-code adoption, a symbolic equation and safety calibration are still recommended.
        """
    )

    with st.expander("README from package"):
        st.markdown(files["readme"])

    with st.expander("Feature report JSON"):
        st.json(files["feature_report"])

    with st.expander("Package folder"):
        st.write(str(pkg_dir))
        st.write(sorted([p.name for p in pkg_dir.iterdir()]))
