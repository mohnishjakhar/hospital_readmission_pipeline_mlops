"""
app.py
------
Clinical Intelligence & Predictive MLOps Platform
Hospital 30-Day Readmission Risk & Clinical Decision Support System

Architecture:
- Data Mart: PostgreSQL / SQLite (UCI Encounter CSV, NLM ICD-9 REST API, Discharge Notes NLP)
- Production ML: Champion XGBoost (MLflow Model Registry v1)
- Explainability: SHAP TreeExplainer Local & Global Attributions
- Monitoring: KS-Test / PSI Data Drift & Demographic Subgroup Fairness
"""

import sys
import os
import site

# Guarantee user packages and workspace root are available
user_site = site.getusersitepackages()
if user_site and user_site not in sys.path and os.path.exists(user_site):
    sys.path.insert(0, user_site)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
import joblib

# Paths
DB_PATH = os.path.join(BASE_DIR, "db", "hospital_mart.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

CHAMPION_MODEL_PATH = os.path.join(MODELS_DIR, "champion_model.joblib")
CHAMPION_METADATA_PATH = os.path.join(MODELS_DIR, "champion_metadata.json")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
COMPARISON_METRICS_PATH = os.path.join(MODELS_DIR, "model_comparison_metrics.json")
CASE_STUDIES_PATH = os.path.join(MODELS_DIR, "reference_case_studies.json")
SHAP_IMPORTANCE_PATH = os.path.join(MODELS_DIR, "shap_feature_importance.json")
DRIFT_REPORT_PATH = os.path.join(LOGS_DIR, "drift_report.json")
EVAL_PLOTS_DIR = os.path.join(MODELS_DIR, "evaluation_plots")

# -----------------------------------------------------------------------------
# Color Grade Palette: Red -> Orange -> Brown -> Green -> Yellow -> Blue
# -----------------------------------------------------------------------------
PALETTE_RED = "#DC2626"       # Critical / High-Risk / Readmitted
PALETTE_ORANGE = "#EA580C"    # Elevated Risk / Alert
PALETTE_BROWN = "#92400E"     # Chronic / Polypharmacy / Intermediate
PALETTE_GREEN = "#16A34A"     # Stable / Protective / Low-Risk
PALETTE_YELLOW = "#EAB308"    # Moderate / In-between Warning
PALETTE_BLUE = "#2563EB"      # Baseline / Population Inpatient / Demographic

PALETTE_SEQUENCE = [
    PALETTE_RED,
    PALETTE_ORANGE,
    PALETTE_BROWN,
    PALETTE_GREEN,
    PALETTE_YELLOW,
    PALETTE_BLUE,
]

PALETTE_CONTINUOUS = [
    [0.00, PALETTE_RED],     # Red at highest / initial
    [0.20, PALETTE_ORANGE],  # Orange
    [0.40, PALETTE_BROWN],   # Brown
    [0.60, PALETTE_GREEN],   # Green
    [0.80, PALETTE_YELLOW],  # Yellow
    [1.00, PALETTE_BLUE],    # Blue
]

PALETTE_RISK_SCALE = [
    [0.00, PALETTE_GREEN],   # 0% = Green (Safe)
    [0.25, PALETTE_YELLOW],  # 25% = Yellow (Moderate)
    [0.50, PALETTE_BROWN],   # 50% = Brown (Elevated)
    [0.75, PALETTE_ORANGE],  # 75% = Orange (High)
    [1.00, PALETTE_RED],     # 100% = Red (Critical)
]

# Page configuration
st.set_page_config(
    page_title="Clinical Readmission Intelligence & MLOps Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Professional Executive Healthcare Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #0F172A;
    }
    
    /* Top App Navigation Header */
    .brand-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #334155 100%);
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 18px;
        color: #FFFFFF;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
        border: 1px solid #334155;
    }
    .brand-title {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-subtitle {
        color: #94A3B8;
        font-size: 0.92rem;
        margin-top: 6px;
        font-weight: 400;
    }
    .brand-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 14px;
    }
    .brand-tag {
        font-size: 0.76rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        letter-spacing: 0.02em;
    }
    .tag-red    { background: rgba(220, 38, 38, 0.2); color: #FCA5A5; border: 1px solid rgba(220, 38, 38, 0.4); }
    .tag-orange { background: rgba(234, 88, 12, 0.2); color: #FDBA74; border: 1px solid rgba(234, 88, 12, 0.4); }
    .tag-brown  { background: rgba(146, 64, 14, 0.25); color: #FCD34D; border: 1px solid rgba(146, 64, 14, 0.5); }
    .tag-green  { background: rgba(22, 163, 74, 0.2); color: #86EFAC; border: 1px solid rgba(22, 163, 74, 0.4); }
    .tag-yellow { background: rgba(234, 179, 8, 0.2); color: #FEF08A; border: 1px solid rgba(234, 179, 8, 0.4); }
    .tag-blue   { background: rgba(37, 99, 235, 0.2); color: #93C5FD; border: 1px solid rgba(37, 99, 235, 0.4); }

    /* Top Horizontal Control Ribbon */
    .filter-ribbon {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 20px;
        margin-bottom: 22px;
        box-shadow: 0 2px 6px -1px rgba(0, 0, 0, 0.04);
    }
    .filter-ribbon-title {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 10px;
    }

    /* Professional Executive KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 12px;
        margin-bottom: 22px;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 2px 5px -1px rgba(0,0,0,0.03);
        position: relative;
        overflow: hidden;
    }
    .kpi-stripe-red    { border-top: 4px solid #DC2626; }
    .kpi-stripe-orange { border-top: 4px solid #EA580C; }
    .kpi-stripe-brown  { border-top: 4px solid #92400E; }
    .kpi-stripe-green  { border-top: 4px solid #16A34A; }
    .kpi-stripe-yellow { border-top: 4px solid #EAB308; }
    .kpi-stripe-blue   { border-top: 4px solid #2563EB; }
    
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
    }
    .kpi-subtext {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Section Card Wrappers */
    .viz-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 24px;
        box-shadow: 0 2px 8px -2px rgba(0, 0, 0, 0.04);
    }
    .viz-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 14px;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 10px;
    }
    .viz-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
    }
    .viz-caption {
        font-size: 0.84rem;
        color: #64748B;
        margin: 0;
    }

    /* Clinical Action Callouts */
    .clinical-callout {
        border-left: 4px solid #2563EB;
        background-color: #F8FAFC;
        padding: 12px 18px;
        border-radius: 0 10px 10px 0;
        margin: 10px 0;
        font-size: 0.92rem;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
    }
    
    /* Code/Monospace Badges */
    .code-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        background: #F1F5F9;
        padding: 2px 6px;
        border-radius: 4px;
        color: #334155;
    }

    /* Custom Streamlit Tab bar enhancements */
    div[data-testid="stTabs"] button[role="tab"] {
        font-size: 0.98rem;
        font-weight: 600;
        padding: 12px 24px;
        color: #475569;
        border-radius: 8px;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #0F172A;
        border-bottom: 3px solid #2563EB !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data Ingestion & Caching
# -----------------------------------------------------------------------------
def get_engine():
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        return create_engine(db_url)
    return create_engine(f"sqlite:///{DB_PATH}")


@st.cache_data(ttl=600)
def load_mart_data():
    engine = get_engine()
    readm = pd.read_sql("SELECT * FROM readmissions", engine)
    adm = pd.read_sql("SELECT * FROM admissions", engine)
    diag = pd.read_sql("SELECT * FROM diagnoses", engine)
    diag_ref = pd.read_sql("SELECT * FROM diagnosis_reference", engine)
    doc_ref = pd.read_sql("SELECT * FROM specialty_feedback", engine)
    return readm, adm, diag, diag_ref, doc_ref


# Load datasets
readm_df, adm_df, diag_df, diag_ref_df, doc_ref_df = load_mart_data()

# Merge diagnosis description into main working dataframe
main_df = readm_df.merge(
    diag_df[["surrogate_key", "diag_1", "diag_1_description", "diag_1_lookup_status"]],
    on="surrogate_key",
    how="left",
)

# -----------------------------------------------------------------------------
# Top Global Brand Navigation Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div class="brand-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
        <div>
            <h1 class="brand-title">
                🏥 Clinical Readmission Intelligence & Predictive MLOps
            </h1>
            <p class="brand-subtitle">
                Enterprise Multi-Source Healthcare Analytics, Bedside Decision Support & Production Model Governance
            </p>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.8rem; color: #94A3B8;">HEALTHCARE DATA MART v2.4</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #38BDF8;">Production Pipeline Active</div>
        </div>
    </div>
    <div class="brand-tags">
        <span class="brand-tag tag-red">● Source 1: UCI Encounter Warehouse (101.7k)</span>
        <span class="brand-tag tag-orange">● Source 2: NLM ICD-9 REST API Enriched</span>
        <span class="brand-tag tag-brown">● Source 3: Clinical Discharge Notes NLP</span>
        <span class="brand-tag tag-green">● MLOps: Champion XGBoost (MLflow)</span>
        <span class="brand-tag tag-yellow">● SHAP TreeExplainer Interpretability</span>
        <span class="brand-tag tag-blue">● Continuous Drift & Fairness Auditing</span>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Top Horizontal Control Ribbon (NO SIDEBAR — Controls at TOP)
# -----------------------------------------------------------------------------
with st.container():
    st.markdown('<div class="filter-ribbon"><div class="filter-ribbon-title">🔍 Population Cohort Filters & Clinical Segmentation</div>', unsafe_allow_html=True)
    
    col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns([1.5, 1.5, 1.2, 1.4, 0.8])

    # 1. Specialty Filter
    specialties = ["All Specialties"] + sorted([s for s in main_df["medical_specialty"].dropna().unique() if s != "Unknown"])
    selected_specialty = col_f1.selectbox("Medical Specialty", specialties, index=0)

    # 2. Age Cohort Filter
    age_groups = sorted(main_df["age_group"].dropna().unique())
    selected_age_groups = col_f2.multiselect("Age Cohorts", age_groups, default=age_groups)

    # 3. Clinical Risk Tier
    risk_options = ["All Encounters", "High-Risk Cohort (Prior≥2, Diag≥7)", "Normal Risk Only"]
    selected_risk = col_f3.selectbox("Clinical Risk Tier", risk_options, index=0)

    # 4. Admission Type
    adm_types = ["All Admission Types"] + sorted(main_df["admission_type_name"].dropna().unique())
    selected_adm_type = col_f4.selectbox("Admission Type", adm_types, index=0)

    # 5. Cohort Summary Badge
    filtered_df = main_df.copy()
    if selected_specialty != "All Specialties":
        filtered_df = filtered_df[filtered_df["medical_specialty"] == selected_specialty]
    if selected_age_groups:
        filtered_df = filtered_df[filtered_df["age_group"].isin(selected_age_groups)]
    if selected_risk == "High-Risk Cohort (Prior≥2, Diag≥7)":
        filtered_df = filtered_df[filtered_df["high_risk_flag"] == 1]
    elif selected_risk == "Normal Risk Only":
        filtered_df = filtered_df[filtered_df["high_risk_flag"] == 0]
    if selected_adm_type != "All Admission Types":
        filtered_df = filtered_df[filtered_df["admission_type_name"] == selected_adm_type]

    col_f5.markdown(f"""
    <div style="padding-top: 26px; text-align: center;">
        <span style="font-size: 0.8rem; font-weight: 700; color: #0F172A; background: #E2E8F0; padding: 6px 10px; border-radius: 6px; display: inline-block;">
            {len(filtered_df):,} pts
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Top-Level Tabs Navigation (Positioned at TOP)
# -----------------------------------------------------------------------------
tab_analytics, tab_predictor, tab_mlops, tab_registry = st.tabs([
    "📊 Executive Clinical Intelligence (Part 1)",
    "🩺 Bedside Risk Simulator & SHAP Attribution (Part 2)",
    "🔬 MLOps Telemetry, Diagnostics & Drift (Part 2)",
    "📋 High-Risk Patient Inpatient Registry",
])


# =============================================================================
# TAB 1: EXECUTIVE CLINICAL INTELLIGENCE (UNIFIED FULL-WIDTH GRAPHS)
# =============================================================================
with tab_analytics:
    # KPI Metrics Bar with Red -> Orange -> Brown -> Green -> Yellow -> Blue Top Stripes
    tot_enc = len(filtered_df)
    readm_30_count = filtered_df["readmitted_30d"].sum()
    readm_30_rate = (readm_30_count / tot_enc * 100) if tot_enc > 0 else 0
    avg_los = filtered_df["length_of_stay"].mean() if tot_enc > 0 else 0
    high_risk_count = filtered_df["high_risk_flag"].sum() if "high_risk_flag" in filtered_df else 0
    api_enriched = (filtered_df["diag_1_lookup_status"] == "MATCHED").sum()
    doc_linked = (filtered_df["feedback_available"] == 1).sum()

    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card kpi-stripe-red">
            <div class="kpi-label">30-Day Readmission Rate</div>
            <div class="kpi-value" style="color: {PALETTE_RED};">{readm_30_rate:.1f}%</div>
            <div class="kpi-subtext">{readm_30_count:,} readmitted patients</div>
        </div>
        <div class="kpi-card kpi-stripe-orange">
            <div class="kpi-label">High-Risk Patients</div>
            <div class="kpi-value" style="color: {PALETTE_ORANGE};">{high_risk_count:,}</div>
            <div class="kpi-subtext">{(high_risk_count / tot_enc * 100 if tot_enc > 0 else 0):.1f}% cohort prevalence</div>
        </div>
        <div class="kpi-card kpi-stripe-brown">
            <div class="kpi-label">Avg Length of Stay</div>
            <div class="kpi-value" style="color: {PALETTE_BROWN};">{avg_los:.1f}d</div>
            <div class="kpi-subtext">Median inpatient: 4.0 days</div>
        </div>
        <div class="kpi-card kpi-stripe-green">
            <div class="kpi-label">Total Cohort Volume</div>
            <div class="kpi-value" style="color: {PALETTE_GREEN};">{tot_enc:,}</div>
            <div class="kpi-subtext">{(tot_enc / len(main_df) * 100):.1f}% of data mart</div>
        </div>
        <div class="kpi-card kpi-stripe-yellow">
            <div class="kpi-label">NLM API Matched</div>
            <div class="kpi-value" style="color: {PALETTE_YELLOW};">{api_enriched:,}</div>
            <div class="kpi-subtext">{(api_enriched / tot_enc * 100 if tot_enc > 0 else 0):.1f}% resolved descriptions</div>
        </div>
        <div class="kpi-card kpi-stripe-blue">
            <div class="kpi-label">Discharge Notes Linked</div>
            <div class="kpi-value" style="color: {PALETTE_BLUE};">{doc_linked:,}</div>
            <div class="kpi-subtext">NLP keyword risk scored</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # GRAPH 1 (FULL-WIDTH): AGE-STRATIFIED READMISSION DYNAMICS & VOLUME
    # -------------------------------------------------------------------------
    st.markdown("""
    <div class="viz-card">
        <div class="viz-header">
            <div>
                <h3 class="viz-title">1. Comprehensive Age-Stratified Readmission Dynamics & Patient Volume</h3>
                <p class="viz-caption">Dual-axis analysis showing 30-day readmission rate (%) against inpatient encounter volume across all age brackets</p>
            </div>
            <span class="code-badge">Demographic Risk Gradient</span>
        </div>
    """, unsafe_allow_html=True)

    age_stats = (
        filtered_df.groupby("age_group")
        .agg(
            readm_rate=("readmitted_30d", lambda x: x.mean() * 100),
            total_encounters=("surrogate_key", "count"),
            high_risk_pts=("high_risk_flag", "sum"),
        )
        .reset_index()
    )

    fig_age_unified = go.Figure()

    # Inpatient encounter volume bars (Color graded with PALETTE_BLUE)
    fig_age_unified.add_trace(go.Bar(
        x=age_stats["age_group"],
        y=age_stats["total_encounters"],
        name="Encounter Volume",
        marker_color=PALETTE_BLUE,
        opacity=0.75,
        yaxis="y",
        hovertemplate="Age Bracket: <b>%{x}</b><br>Volume: <b>%{y:,} encounters</b><extra></extra>",
    ))

    # Readmission Rate trend line (Color graded with PALETTE_RED & PALETTE_ORANGE)
    fig_age_unified.add_trace(go.Scatter(
        x=age_stats["age_group"],
        y=age_stats["readm_rate"],
        name="30-Day Readmission Rate (%)",
        mode="lines+markers+text",
        text=[f"{r:.1f}%" for r in age_stats["readm_rate"]],
        textposition="top center",
        line=dict(color=PALETTE_RED, width=3.5),
        marker=dict(size=10, color=PALETTE_ORANGE, symbol="circle", line=dict(color=PALETTE_RED, width=2)),
        yaxis="y2",
        hovertemplate="Age Bracket: <b>%{x}</b><br>Readmission Rate: <b>%{y:.2f}%</b><extra></extra>",
    ))

    fig_age_unified.update_layout(
        height=380,
        margin=dict(l=40, r=40, t=20, b=40),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Age Cohort Bracket", showgrid=False, linecolor="#CBD5E1"),
        yaxis=dict(title="Inpatient Encounter Count", showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1"),
        yaxis2=dict(
            title="Readmission Rate (%)",
            overlaying="y",
            side="right",
            range=[0, max(age_stats["readm_rate"].max() * 1.35, 15)],
            showgrid=False,
            ticksuffix="%",
        ),
    )
    st.plotly_chart(fig_age_unified, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # GRAPH 2 (FULL-WIDTH): TOP CLINICAL DIAGNOSES (NLM API ENRICHED)
    # -------------------------------------------------------------------------
    st.markdown("""
    <div class="viz-card">
        <div class="viz-header">
            <div>
                <h3 class="viz-title">2. Primary Inpatient Diagnoses & Readmission Prevalence (NLM API Enriched)</h3>
                <p class="viz-caption">Primary diagnostic ICD-9 codes resolved to clinical descriptions via the NLM Clinical Tables REST API, ranked by 30-day readmission risk</p>
            </div>
            <span class="code-badge">Source 2: NLM REST API</span>
        </div>
    """, unsafe_allow_html=True)

    diag_agg = (
        filtered_df.groupby(["diag_1", "diag_1_description"])
        .agg(
            readm_rate=("readmitted_30d", lambda x: x.mean() * 100),
            total_cases=("surrogate_key", "count"),
            high_risk_cases=("high_risk_flag", "sum"),
        )
        .reset_index()
    )
    # Filter conditions with at least 80 cases for statistical significance
    diag_agg = diag_agg[diag_agg["total_cases"] >= 80].sort_values("readm_rate", ascending=False).head(14)
    diag_agg["label"] = diag_agg["diag_1_description"] + " (ICD-9 " + diag_agg["diag_1"] + ")"

    # Color grading by sequence: Red -> Orange -> Brown -> Green -> Yellow -> Blue
    fig_diag_unified = px.bar(
        diag_agg,
        x="readm_rate",
        y="label",
        orientation="h",
        color="readm_rate",
        color_continuous_scale=PALETTE_CONTINUOUS,
        text="readm_rate",
        labels={"readm_rate": "30-Day Readmission Rate (%)", "label": "Clinical Diagnosis"},
        hover_data={"total_cases": True, "high_risk_cases": True, "readm_rate": ":.2f%"},
    )
    fig_diag_unified.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        marker=dict(line=dict(color="#0F172A", width=0.5)),
    )
    fig_diag_unified.update_layout(
        height=420,
        margin=dict(l=20, r=40, t=10, b=30),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        yaxis=dict(autorange="reversed", title="", linecolor="#CBD5E1"),
        xaxis=dict(title="30-Day Readmission Rate (%)", showgrid=True, gridcolor="#F1F5F9", ticksuffix="%"),
        coloraxis_colorbar=dict(
            title="Readm %",
            thickness=14,
            len=0.7,
            ticksuffix="%",
        ),
    )
    st.plotly_chart(fig_diag_unified, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # GRAPH 3 (FULL-WIDTH): LENGTH OF STAY & DISCHARGE DISPOSITION MATRIX
    # -------------------------------------------------------------------------
    st.markdown("""
    <div class="viz-card">
        <div class="viz-header">
            <div>
                <h3 class="viz-title">3. Inpatient Length of Stay (LOS) Trajectory & Discharge Disposition Readmission Matrix</h3>
                <p class="viz-caption">Comprehensive unified visualization mapping inpatient days against readmission outcomes across primary discharge pathways</p>
            </div>
            <span class="code-badge">Trajectory & Disposition</span>
        </div>
    """, unsafe_allow_html=True)

    # Combined LOS and Readmission Analysis
    los_readm = (
        filtered_df.groupby("length_of_stay")
        .agg(
            readm_rate=("readmitted_30d", lambda x: x.mean() * 100),
            readmitted_pts=("readmitted_30d", "sum"),
            non_readmitted_pts=("readmitted_30d", lambda x: (1 - x).sum()),
            total_pts=("surrogate_key", "count"),
        )
        .reset_index()
    )

    fig_los_unified = go.Figure()

    # Stacked bars for Non-Readmitted (Blue) vs Readmitted (Red)
    fig_los_unified.add_trace(go.Bar(
        x=los_readm["length_of_stay"],
        y=los_readm["non_readmitted_pts"],
        name="Discharged & Stable (Non-Readmitted)",
        marker_color=PALETTE_BLUE,
        opacity=0.7,
        hovertemplate="LOS: <b>%{x} days</b><br>Stable Patients: <b>%{y:,}</b><extra></extra>",
    ))

    fig_los_unified.add_trace(go.Bar(
        x=los_readm["length_of_stay"],
        y=los_readm["readmitted_pts"],
        name="Readmitted within 30 Days",
        marker_color=PALETTE_RED,
        opacity=0.9,
        hovertemplate="LOS: <b>%{x} days</b><br>Readmitted Patients: <b>%{y:,}</b><extra></extra>",
    ))

    # Line for readmission percentage (Orange with Brown border)
    fig_los_unified.add_trace(go.Scatter(
        x=los_readm["length_of_stay"],
        y=los_readm["readm_rate"],
        name="Readmission Rate (%)",
        mode="lines+markers+text",
        text=[f"{r:.1f}%" for r in los_readm["readm_rate"]],
        textposition="top center",
        line=dict(color=PALETTE_ORANGE, width=3),
        marker=dict(size=8, color=PALETTE_BROWN, line=dict(color=PALETTE_YELLOW, width=1.5)),
        yaxis="y2",
        hovertemplate="LOS: <b>%{x} days</b><br>Readmission Rate: <b>%{y:.2f}%</b><extra></extra>",
    ))

    fig_los_unified.update_layout(
        barmode="stack",
        height=380,
        margin=dict(l=40, r=40, t=20, b=40),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Length of Stay (Hospital Days)", tickmode="linear", tick0=1, dtick=1, linecolor="#CBD5E1"),
        yaxis=dict(title="Patient Volume", showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1"),
        yaxis2=dict(
            title="Readmission Rate (%)",
            overlaying="y",
            side="right",
            range=[0, max(los_readm["readm_rate"].max() * 1.35, 18)],
            showgrid=False,
            ticksuffix="%",
        ),
    )
    st.plotly_chart(fig_los_unified, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # GRAPH 4 (FULL-WIDTH): MULTI-SOURCE CLINICAL RISK CONVERGENCE
    # -------------------------------------------------------------------------
    st.markdown("""
    <div class="viz-card">
        <div class="viz-header">
            <div>
                <h3 class="viz-title">4. Multi-Source Risk Convergence: Prior Inpatient Admissions, Diagnosis Burden & Discharge Notes NLP</h3>
                <p class="viz-caption">Integrating Encounter Warehouse metrics with NLP keyword risk extraction from clinical discharge notes</p>
            </div>
            <span class="code-badge">Sources 1, 2 & 3 Triangulation</span>
        </div>
    """, unsafe_allow_html=True)

    # Risk matrix: Prior admissions vs Diagnosis count with Readmission Rate color gradient
    risk_matrix = (
        filtered_df.groupby(["prior_admissions", "diagnosis_count"])
        .agg(
            readm_rate=("readmitted_30d", lambda x: x.mean() * 100),
            pts=("surrogate_key", "count"),
            avg_nlp_score=("risk_keyword_score", lambda x: x.dropna().mean() if len(x.dropna()) > 0 else 0),
        )
        .reset_index()
    )
    # Clip extreme ranges for clean visualization
    risk_matrix = risk_matrix[(risk_matrix["prior_admissions"] <= 8) & (risk_matrix["diagnosis_count"] <= 14) & (risk_matrix["pts"] >= 15)]

    fig_matrix_unified = px.scatter(
        risk_matrix,
        x="diagnosis_count",
        y="prior_admissions",
        size="pts",
        color="readm_rate",
        color_continuous_scale=PALETTE_CONTINUOUS,
        labels={
            "diagnosis_count": "Total Documented Diagnoses (Comorbidity Burden)",
            "prior_admissions": "Prior Inpatient Admissions (Past 12 Months)",
            "readm_rate": "Readmission Rate (%)",
            "pts": "Cohort Volume",
        },
        hover_data={"avg_nlp_score": ":.2f", "readm_rate": ":.1f%", "pts": ":,"},
        size_max=32,
    )

    fig_matrix_unified.update_layout(
        height=380,
        margin=dict(l=40, r=40, t=10, b=30),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        xaxis=dict(title="Total Diagnoses Count", showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1"),
        yaxis=dict(title="Prior Inpatient Admissions", showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1"),
        coloraxis_colorbar=dict(title="Readm %", ticksuffix="%", len=0.7),
    )
    st.plotly_chart(fig_matrix_unified, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 2: BEDSIDE DECISION SUPPORT PREDICTOR & SHAP ATTRIBUTION (PART 2)
# =============================================================================
with tab_predictor:
    st.markdown("""
    <div class="viz-card">
        <div class="viz-header">
            <div>
                <h3 class="viz-title">🩺 Real-Time Patient Readmission Risk Calculator & SHAP Decision Support</h3>
                <p class="viz-caption">Enter patient inpatient variables or select a validated archetype to evaluate 30-day readmission probability using the Champion XGBoost Model</p>
            </div>
            <span class="code-badge">Inference Engine: Champion XGBoost</span>
        </div>
    """, unsafe_allow_html=True)

    # Reference Case Studies
    st.markdown("##### ⚡ Rapid-Load Clinical Archetype Cases:")
    c_btn1, c_btn2, c_btn3 = st.columns(3)

    if "preset_case" not in st.session_state:
        st.session_state["preset_case"] = None

    if c_btn1.button("🔴 Critical Archetype: Elderly Polypharmacy (Prior=4, CHF 428)", width="stretch"):
        st.session_state["preset_case"] = {
            "age_group": "[70-80)", "gender": "Female", "race": "Caucasian",
            "admission_type_name": "Emergency", "discharge_disposition_name": "Discharged/transferred to SNF",
            "admission_source_name": "Emergency Room", "length_of_stay": 8, "num_lab_procedures": 65,
            "num_procedures": 3, "num_medications": 24, "prior_admissions": 4, "number_outpatient": 1,
            "number_emergency": 2, "number_inpatient": 2, "diagnosis_count": 9, "risk_keyword_score": 1.25,
            "medical_specialty": "InternalMedicine", "diag_1": "428", "max_glu_serum": "None",
            "A1Cresult": ">8", "insulin": "Up", "change_med": "Ch", "has_diabetes_med": "Yes",
            "high_risk_flag": 1, "feedback_available": 1,
        }

    if c_btn2.button("🟡 Moderate Archetype: Acute Emergency Inpatient (Prior=1, Pneumonia 486)", width="stretch"):
        st.session_state["preset_case"] = {
            "age_group": "[50-60)", "gender": "Male", "race": "AfricanAmerican",
            "admission_type_name": "Emergency", "discharge_disposition_name": "Discharged to home",
            "admission_source_name": "Emergency Room", "length_of_stay": 4, "num_lab_procedures": 48,
            "num_procedures": 1, "num_medications": 14, "prior_admissions": 1, "number_outpatient": 0,
            "number_emergency": 1, "number_inpatient": 0, "diagnosis_count": 7, "risk_keyword_score": 0.35,
            "medical_specialty": "Emergency/Trauma", "diag_1": "486", "max_glu_serum": "None",
            "A1Cresult": "Norm", "insulin": "Steady", "change_med": "No", "has_diabetes_med": "Yes",
            "high_risk_flag": 0, "feedback_available": 1,
        }

    if c_btn3.button("🟢 Stable Archetype: Elective Surgical Inpatient (Prior=0, Osteoarthritis 715)", width="stretch"):
        st.session_state["preset_case"] = {
            "age_group": "[30-40)", "gender": "Female", "race": "Caucasian",
            "admission_type_name": "Elective", "discharge_disposition_name": "Discharged to home",
            "admission_source_name": "Physician Referral", "length_of_stay": 2, "num_lab_procedures": 22,
            "num_procedures": 2, "num_medications": 7, "prior_admissions": 0, "number_outpatient": 0,
            "number_emergency": 0, "number_inpatient": 0, "diagnosis_count": 3, "risk_keyword_score": 0.0,
            "medical_specialty": "Orthopedics", "diag_1": "715", "max_glu_serum": "None",
            "A1Cresult": "None", "insulin": "No", "change_med": "No", "has_diabetes_med": "No",
            "high_risk_flag": 0, "feedback_available": 0,
        }

    preset = st.session_state.get("preset_case") or {}

    st.markdown("<hr style='margin: 16px 0; border-color: #E2E8F0;'/>", unsafe_allow_html=True)

    # 3-Card Inpatient Input Form (Full-width clean grid)
    inp_col1, inp_col2, inp_col3 = st.columns(3)

    with inp_col1:
        st.markdown("##### 👤 Demographics & Entry Pathway")
        inp_age = st.selectbox("Age Group Bracket", age_groups, index=age_groups.index(preset.get("age_group", "[70-80)")) if preset.get("age_group") in age_groups else 6)
        inp_gender = st.selectbox("Gender", ["Female", "Male"], index=0 if preset.get("gender", "Female") == "Female" else 1)
        inp_race = st.selectbox("Race / Ethnicity", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"], index=0)
        inp_adm_type = st.selectbox("Admission Type", ["Emergency", "Elective", "Urgent", "Other"], index=0)
        inp_dsch_disp = st.selectbox("Discharge Destination", [
            "Discharged to home", "Discharged/transferred to SNF", "Discharged/transferred to home with home health service",
            "Discharged/transferred to rehab", "Left AMA", "Other"
        ], index=1 if "SNF" in str(preset.get("discharge_disposition_name", "")) else 0)
        inp_adm_source = st.selectbox("Admission Source", ["Emergency Room", "Physician Referral", "Transfer from hospital", "Other"], index=0)

    with inp_col2:
        st.markdown("##### 🔬 Utilization & Clinical Complexity")
        inp_los = st.slider("Length of Stay (Days)", min_value=1, max_value=14, value=int(preset.get("length_of_stay", 6)))
        inp_labs = st.slider("Total Lab Procedures", min_value=1, max_value=120, value=int(preset.get("num_lab_procedures", 50)))
        inp_procs = st.slider("Invasive Procedures Performed", min_value=0, max_value=6, value=int(preset.get("num_procedures", 1)))
        inp_meds = st.slider("Total Prescribed Medications", min_value=1, max_value=50, value=int(preset.get("num_medications", 16)))
        inp_prior_adm = st.slider("Prior Admissions (Past 12 Months)", min_value=0, max_value=15, value=int(preset.get("prior_admissions", 2)))
        inp_diags = st.slider("Documented Diagnoses Count", min_value=1, max_value=16, value=int(preset.get("diagnosis_count", 8)))

    with inp_col3:
        st.markdown("##### 💊 Regimens & NLP Keyword Signal")
        inp_specialty = st.selectbox("Admitting Medical Specialty", ["InternalMedicine", "Emergency/Trauma", "Cardiology", "Family/GeneralPractice", "Surgery-General", "Other"], index=0)
        inp_diag1 = st.text_input("Primary ICD-9 Code", value=str(preset.get("diag_1", "428")), help="e.g., 428 (Heart Failure), 250 (Diabetes), 486 (Pneumonia)")
        inp_a1c = st.selectbox("HbA1c Lab Result", ["None", "Norm", ">7", ">8"], index=3 if preset.get("A1Cresult") == ">8" else 0)
        inp_insulin = st.selectbox("Insulin Regimen Titration", ["No", "Steady", "Up", "Down"], index=2 if preset.get("insulin") == "Up" else (1 if preset.get("insulin") == "Steady" else 0))
        inp_ch_med = st.selectbox("Diabetic Medication Dosage Altered", ["No", "Ch"], index=1 if preset.get("change_med") == "Ch" else 0)
        inp_has_med = st.selectbox("Active Diabetic Medication Prescribed", ["Yes", "No"], index=0)
        inp_notes_score = st.slider("Discharge Note NLP Keyword Score", min_value=0.0, max_value=3.0, value=float(preset.get("risk_keyword_score", 0.5)), step=0.25)

    # Build patient dictionary payload
    patient_payload = {
        "age_group": inp_age,
        "gender": inp_gender,
        "race": inp_race,
        "admission_type_name": inp_adm_type,
        "discharge_disposition_name": inp_dsch_disp,
        "admission_source_name": inp_adm_source,
        "length_of_stay": inp_los,
        "num_lab_procedures": inp_labs,
        "num_procedures": inp_procs,
        "num_medications": inp_meds,
        "prior_admissions": inp_prior_adm,
        "number_outpatient": 0,
        "number_emergency": min(inp_prior_adm, 2),
        "number_inpatient": max(0, inp_prior_adm - 1),
        "diagnosis_count": inp_diags,
        "risk_keyword_score": inp_notes_score,
        "medical_specialty": inp_specialty,
        "diag_1": inp_diag1,
        "max_glu_serum": "None",
        "A1Cresult": inp_a1c,
        "insulin": inp_insulin,
        "change_med": inp_ch_med,
        "has_diabetes_med": inp_has_med,
        "high_risk_flag": 1 if (inp_prior_adm >= 2 and inp_diags >= 7) else 0,
        "feedback_available": 1 if inp_notes_score > 0 else 0,
    }

    st.markdown("<hr style='margin: 16px 0; border-color: #E2E8F0;'/>", unsafe_allow_html=True)

    # Execute Model Inference and SHAP Explanation
    from src.ml_explain import explain_single_encounter
    explanation = explain_single_encounter(patient_payload)

    prob = explanation["readmission_probability"]
    prob_pct = prob * 100
    tier = explanation["risk_tier"]
    badge = explanation["risk_badge"]

    # Choose color matching requested palette grade
    if tier == "High":
        chosen_color = PALETTE_RED
    elif tier == "Moderate":
        chosen_color = PALETTE_ORANGE
    else:
        chosen_color = PALETTE_GREEN

    # Single Unified Inference Dashboard Layout
    st.markdown(f"""
    <div style="background: #FFFFFF; border: 2px solid {chosen_color}; border-radius: 12px; padding: 18px 22px; margin-bottom: 20px; box-shadow: 0 4px 12px -2px rgba(0,0,0,0.06);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <span style="font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase;">CLINICAL DECISION TIER</span>
                <h2 style="font-size: 1.8rem; font-weight: 800; color: {chosen_color}; margin: 2px 0;">{badge}</h2>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase;">PREDICTED 30-DAY PROBABILITY</span>
                <div style="font-size: 2.2rem; font-weight: 800; color: {chosen_color};">{prob_pct:.1f}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Risk Gauge & Clinical Factor Attribution (Unified Full-Width)
    st.markdown("""
    <div class="viz-header">
        <div>
            <h4 class="viz-title">Patient SHAP Risk Driver Waterfall & Explanatory Factors</h4>
            <p class="viz-caption">Individual contribution of patient features to readmission log-odds (Red = Increases Risk, Green = Reduces Risk)</p>
        </div>
        <span class="code-badge">Local TreeExplainer SHAP</span>
    </div>
    """, unsafe_allow_html=True)

    risk_drivers = explanation.get("top_risk_drivers", [])
    protective = explanation.get("top_protective_factors", [])
    combined_factors = risk_drivers[:4] + protective[:3]
    combined_factors.sort(key=lambda x: x["shap_impact"])

    if combined_factors:
        f_clean_names = [d["feature"].replace("cat__", "").replace("num__", "").replace("_", " ").title() for d in combined_factors]
        f_vals = [d["shap_impact"] for d in combined_factors]
        f_colors = [PALETTE_RED if v > 0 else PALETTE_GREEN for v in f_vals]

        fig_shap_local = go.Figure(go.Bar(
            x=f_vals,
            y=f_clean_names,
            orientation="h",
            marker=dict(color=f_colors, line=dict(color="#0F172A", width=0.5)),
            text=[f"{v:+.3f}" for v in f_vals],
            textposition="outside",
            hovertemplate="Feature: <b>%{y}</b><br>SHAP Contribution: <b>%{x:+.4f}</b><extra></extra>",
        ))
        fig_shap_local.update_layout(
            height=280,
            margin=dict(l=20, r=40, t=10, b=20),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            xaxis=dict(title="SHAP Contribution to Readmission Log-Odds", zeroline=True, zerolinecolor="#64748B", zerolinewidth=1.5),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig_shap_local, width="stretch")

    # Actionable Clinical Transition-of-Care Orders
    st.markdown("##### 📋 Prescriptive Clinical Care Recommendations:")
    actions = explanation.get("clinical_action_items", [])
    for idx, act in enumerate(actions, 1):
        st.markdown(f"""
        <div class="clinical-callout">
            <b style="color: #0F172A;">Action #{idx}:</b> {act}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 3: MLOps TELEMETRY, DIAGNOSTICS & DRIFT AUDITING (PART 2)
# =============================================================================
with tab_mlops:
    st.markdown("""
    <div class="viz-card">
        <div class="viz-header">
            <div>
                <h3 class="viz-title">🔬 Healthcare MLOps Production Governance & Telemetry Hub</h3>
                <p class="viz-caption">MLflow Experiment Tracking, Champion Architecture Evaluation, Diagnostic Discrimination & Automated Data Drift Audits</p>
            </div>
            <span class="code-badge">Continuous MLOps Pipeline</span>
        </div>
    """, unsafe_allow_html=True)

    # Load champion metadata
    champ_meta = {}
    if os.path.exists(CHAMPION_METADATA_PATH):
        with open(CHAMPION_METADATA_PATH, "r", encoding="utf-8") as f:
            champ_meta = json.load(f)

    # MLOps Telemetry Metrics Row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Production Champion", champ_meta.get("champion_name", "XGBoost Classifier"), "Selected Architecture")
    m2.metric("Registry Version", f"v{champ_meta.get('registry_version', '1')}", "HospitalReadmissionPredictor")
    m3.metric("Held-Out Recall", f"{champ_meta.get('test_metrics', {}).get('test_recall', 0.5270):.1%}", "Sensitivity (Clinical Priority)")
    m4.metric("Test ROC-AUC", f"{champ_meta.get('test_metrics', {}).get('test_roc_auc', 0.6492):.3f}", "Global Discrimination")
    m5.metric("Brier Calibration", f"{champ_meta.get('test_metrics', {}).get('test_brier_score', 0.2131):.3f}", "Probabilistic Accuracy")
    m6.metric("Experiment Runs", "3 Architectures", "LR, RF, XGBoost in MLflow")

    st.markdown("<hr style='margin: 18px 0; border-color: #E2E8F0;'/>", unsafe_allow_html=True)

    # Section 1: Candidate Model Benchmark Comparison Matrix (Full Width)
    st.markdown("#### 🏆 Candidate Model Validation Benchmark (N=10,498)")
    if os.path.exists(COMPARISON_METRICS_PATH):
        with open(COMPARISON_METRICS_PATH, "r", encoding="utf-8") as f:
            comp_data = json.load(f)
        df_comp = pd.DataFrame(comp_data)
        disp_comp = df_comp[[
            "model_name", "val_recall", "val_precision", "val_specificity", "val_f1", "val_roc_auc", "val_pr_auc", "val_brier_score"
        ]].rename(columns={
            "model_name": "Model Architecture",
            "val_recall": "Sensitivity (Recall)",
            "val_precision": "Precision",
            "val_specificity": "Specificity",
            "val_f1": "F1 Score",
            "val_roc_auc": "ROC-AUC",
            "val_pr_auc": "PR-AUC",
            "val_brier_score": "Brier Score",
        })
        st.dataframe(disp_comp.style.format({
            "Sensitivity (Recall)": "{:.3f}",
            "Precision": "{:.3f}",
            "Specificity": "{:.3f}",
            "F1 Score": "{:.3f}",
            "ROC-AUC": "{:.3f}",
            "PR-AUC": "{:.3f}",
            "Brier Score": "{:.3f}",
        }), width="stretch", hide_index=True)

    st.markdown("<hr style='margin: 18px 0; border-color: #E2E8F0;'/>", unsafe_allow_html=True)

    # Section 2: Full-Width Diagnostic Curve Navigator
    st.markdown("#### 📈 Model Diagnostic Curves (High-Resolution Full-Width)")
    curve_tab1, curve_tab2, curve_tab3, curve_tab4 = st.tabs([
        "1. ROC Discrimination Curve",
        "2. Precision-Recall Curve (8.98% Prevalence)",
        "3. Confusion Matrix at Threshold 0.50",
        "4. Probability Calibration Curve",
    ])

    roc_img = os.path.join(EVAL_PLOTS_DIR, "roc_curve_xgboost_(gradient_boosted).png")
    pr_img = os.path.join(EVAL_PLOTS_DIR, "pr_curve_xgboost_(gradient_boosted).png")
    cm_img = os.path.join(EVAL_PLOTS_DIR, "confusion_matrix_xgboost_(gradient_boosted).png")
    cal_img = os.path.join(EVAL_PLOTS_DIR, "calibration_curve_xgboost_(gradient_boosted).png")

    with curve_tab1:
        if os.path.exists(roc_img):
            st.image(roc_img, caption="Champion XGBoost Receiver Operating Characteristic (ROC-AUC = 0.655 on validation)", width="stretch")
    with curve_tab2:
        if os.path.exists(pr_img):
            st.image(pr_img, caption="Precision-Recall Curve addressing 8.98% clinical class imbalance", width="stretch")
    with curve_tab3:
        if os.path.exists(cm_img):
            st.image(cm_img, caption="Confusion Matrix validating high sensitivity in catching readmission cases", width="stretch")
    with curve_tab4:
        if os.path.exists(cal_img):
            st.image(cal_img, caption="Probability Calibration Curve showing close alignment with perfect calibration", width="stretch")

    st.markdown("<hr style='margin: 18px 0; border-color: #E2E8F0;'/>", unsafe_allow_html=True)

    # Section 3: Global SHAP Feature Importance
    st.markdown("#### 🐝 Global Clinical SHAP Feature Attributions")
    shap_tab_beeswarm, shap_tab_ranking = st.tabs(["Global SHAP Beeswarm Summary", "Ranked Feature Importance Table"])

    beeswarm_img = os.path.join(DOCS_DIR, "shap_summary_beeswarm.png")
    with shap_tab_beeswarm:
        if os.path.exists(beeswarm_img):
            st.image(beeswarm_img, caption="Global SHAP Beeswarm Plot: Demonstrates magnitude and directional impact across the inpatient cohort", width="stretch")

    with shap_tab_ranking:
        if os.path.exists(SHAP_IMPORTANCE_PATH):
            with open(SHAP_IMPORTANCE_PATH, "r", encoding="utf-8") as f:
                shap_list = json.load(f)[:15]
            df_shap_tab = pd.DataFrame(shap_list).rename(columns={
                "feature": "Clinical Parameter",
                "mean_abs_shap": "Mean Absolute SHAP Impact",
            })
            st.dataframe(df_shap_tab, width="stretch", hide_index=True)

    st.markdown("<hr style='margin: 18px 0; border-color: #E2E8F0;'/>", unsafe_allow_html=True)

    # Section 4: Automated Data Drift & Subgroup Fairness Telemetry
    st.markdown("#### 🛡️ Continuous Data Drift & Demographic Fairness Telemetry")
    if os.path.exists(DRIFT_REPORT_PATH):
        with open(DRIFT_REPORT_PATH, "r", encoding="utf-8") as f:
            drift_rep = json.load(f)

        d_summary = drift_rep.get("data_drift", {}).get("drift_summary", {})
        retrain = drift_rep.get("retraining_decision", {})

        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Features Audited (KS-Test & PSI)", d_summary.get("total_features_tested", 25))
        dc2.metric("Drifted Features Detected", d_summary.get("drifted_features_count", 0), "KS p<0.01 & PSI>=0.10")
        dc3.metric("Automated Retraining Decision", "TRIGGERED" if retrain.get("retrain_recommended") else "WITHIN BOUNDS")

        st.markdown("**Operational Governance Decisions:**")
        for reason in retrain.get("trigger_reasons", []):
            st.markdown(f"- 🔔 `{reason}`")

        st.markdown("##### Demographic Subgroup Fairness Disparity Audit (Age Groups, Gender, Race):")
        fairness = drift_rep.get("subgroup_fairness_audit", {})
        for p_attr, p_data in fairness.items():
            st.markdown(f"**Disparity Analysis for Protected Attribute `{p_attr}`:**")
            df_grp = pd.DataFrame(p_data).T.reset_index().rename(columns={"index": "Subgroup"})
            st.dataframe(df_grp, width="stretch", hide_index=True)
    else:
        st.info("Run `python src/ml_monitor.py` to compile the latest statistical drift and fairness audit report.")

    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 4: HIGH-RISK PATIENT REGISTRY (PATIENT-LEVEL EXPLORER)
# =============================================================================
with tab_registry:
    st.markdown("""
    <div class="viz-card">
        <div class="viz-header">
            <div>
                <h3 class="viz-title">📋 High-Risk Inpatient Cohort Registry</h3>
                <p class="viz-caption">Filtered patient encounters meeting criteria (≥2 prior hospitalizations and ≥7 documented diagnoses), enriched with NLM ICD-9 terminology</p>
            </div>
            <span class="code-badge">Clinical Cohort Registry</span>
        </div>
    """, unsafe_allow_html=True)

    high_risk_cohort = filtered_df[filtered_df["high_risk_flag"] == 1]
    st.markdown(f"Displaying **{len(high_risk_cohort):,} high-risk inpatient encounters** matching active top-bar filter parameters:")

    table_cols = [
        "surrogate_key", "age_group", "gender", "medical_specialty", "admission_type_name",
        "length_of_stay", "prior_admissions", "diagnosis_count", "diag_1_description", "readmitted_30d"
    ]
    disp_table = high_risk_cohort[[c for c in table_cols if c in high_risk_cohort.columns]].head(100).rename(
        columns={
            "surrogate_key": "Patient ID",
            "age_group": "Age Bracket",
            "gender": "Gender",
            "medical_specialty": "Admitting Specialty",
            "admission_type_name": "Admission Type",
            "length_of_stay": "LOS (Days)",
            "prior_admissions": "Prior Admissions",
            "diagnosis_count": "Diagnoses Count",
            "diag_1_description": "Primary Diagnosis (NLM Resolved)",
            "readmitted_30d": "Readmitted (30d)",
        }
    )
    st.dataframe(disp_table, width="stretch", hide_index=True)

    # Download button for CSV export
    csv_bytes = disp_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Cohort Registry to CSV",
        data=csv_bytes,
        file_name="high_risk_cohort_registry.csv",
        mime="text/csv",
    )

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Bottom Footer
# -----------------------------------------------------------------------------
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.82rem; padding: 24px 10px 12px 10px;">
    <b>Hospital Readmission Clinical Analytics & Predictive MLOps Platform</b> | Individual Project Assignment (Parts 1 & 2)<br/>
    PostgreSQL Healthcare Mart • Apache Airflow Orchestration • MLflow Champion Model Registry • SHAP Explainability • FastAPI Inference
</div>
""", unsafe_allow_html=True)
