"""
app.py
------
Healthcare Analytical Dashboard & Clinical Decision Support System:
Hospital Readmission Analytics & Prediction (End-to-End MLOps Solution)
Course: Data Engineering & MLOps | Individual Project (Parts 1 & 2 Deliverables)

Features:
- Tab 1: Multi-Source Healthcare Data Mart & Population Analytics (Part 1 Foundation)
  * Integrates UCI Encounter CSV, NLM ICD-9 REST API, and Clinical Discharge Notes
  * 6 Core Clinical Analytics Views + Interactive Cohort Segmentation
- Tab 2: Clinical Decision Support & Readmission Risk Predictor (Part 2 MLOps)
  * Interactive Single Encounter Risk Calculator (Demographics, Labs, Meds, Regimens)
  * Pre-loaded canonical clinical case archetypes (High, Moderate, Low Risk)
  * Visual Risk Probability Gauge & Tier Badging (Low, Moderate, High)
  * Local SHAP Patient Risk Drivers & Protective Factors Attribution
  * Actionable Clinical Transition-of-Care Interventions
- Tab 3: MLOps Telemetry, Diagnostic Curves & Drift Auditing (Part 2 MLOps)
  * Model Comparison Matrix (Logistic Regression, Random Forest, Champion XGBoost)
  * Diagnostic ROC, PR, Confusion Matrix & Calibration Curves
  * Global SHAP Beeswarm & Feature Importance Rankings
  * Automated Data Drift (KS-Test/PSI) & Demographic Subgroup Fairness Telemetry
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
import joblib

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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

st.set_page_config(
    page_title="Hospital Readmission Analytics & Clinical AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern clinical analytics aesthetic
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border: 1px solid #DBEAFE;
        border-radius: 10px;
        padding: 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .badge-source {
        display: inline-block;
        font-size: 0.75rem;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-file { background-color: #E0E7FF; color: #3730A3; }
    .badge-api { background-color: #D1FAE5; color: #065F46; }
    .badge-doc { background-color: #FEF3C7; color: #92400E; }
    .badge-ml { background-color: #FCE7F3; color: #9D174D; }
    .clinical-callout {
        border-left: 4px solid #2563EB;
        background-color: #F0F7FF;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


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

# ---------------------------------------------------------------------
# Sidebar Navigation & Filters
# ---------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/hospital.png", width=64)
st.sidebar.title("Clinical Navigation")
st.sidebar.caption("Data Mart & MLOps Engine")

# Filter: Medical Specialty
specialties = ["All Specialties"] + sorted([s for s in main_df["medical_specialty"].dropna().unique() if s != "Unknown"])
selected_specialty = st.sidebar.selectbox("Medical Specialty", specialties)

# Filter: Age Group
age_groups = sorted(main_df["age_group"].dropna().unique())
selected_age_groups = st.sidebar.multiselect("Age Groups", age_groups, default=age_groups)

# Filter: Risk Level
risk_options = ["All Encounters", "High-Risk Cohort Only", "Normal Risk Only"]
selected_risk = st.sidebar.radio("Clinical Risk Tier", risk_options)

# Filter: Admission Type
adm_types = ["All Admission Types"] + sorted(main_df["admission_type_name"].dropna().unique())
selected_adm_type = st.sidebar.selectbox("Admission Type", adm_types)

# Apply filters
filtered_df = main_df.copy()

if selected_specialty != "All Specialties":
    filtered_df = filtered_df[filtered_df["medical_specialty"] == selected_specialty]

if selected_age_groups:
    filtered_df = filtered_df[filtered_df["age_group"].isin(selected_age_groups)]

if selected_risk == "High-Risk Cohort Only":
    filtered_df = filtered_df[filtered_df["high_risk_flag"] == 1]
elif selected_risk == "Normal Risk Only":
    filtered_df = filtered_df[filtered_df["high_risk_flag"] == 0]

if selected_adm_type != "All Admission Types":
    filtered_df = filtered_df[filtered_df["admission_type_name"] == selected_adm_type]

# ---------------------------------------------------------------------
# Header & Multi-Source Badges
# ---------------------------------------------------------------------
st.markdown('<div class="main-title">🏥 Hospital Readmission Clinical Intelligence & Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    '<span class="badge-source badge-file">Source 1: File (UCI Encounter CSV)</span>'
    '<span class="badge-source badge-api">Source 2: REST API (NLM ICD-9 Search)</span>'
    '<span class="badge-source badge-doc">Source 3: Documents (Discharge Notes)</span>'
    '<span class="badge-source badge-ml">MLOps: Champion XGBoost (MLflow Registered)</span>'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Top-Level Tabs Navigation
# ---------------------------------------------------------------------
tab_analytics, tab_predictor, tab_mlops = st.tabs([
    "🏥 Data Mart Analytics (Part 1)",
    "🔮 Clinical Decision Support Predictor (Part 2)",
    "📊 MLOps Telemetry & Drift Audit (Part 2)",
])


# =====================================================================
# TAB 1: DATA MART ANALYTICS (PART 1 DELIVERABLE)
# =====================================================================
with tab_analytics:
    # KPI Metric Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    tot_enc = len(filtered_df)
    readm_30_count = filtered_df["readmitted_30d"].sum()
    readm_30_rate = (readm_30_count / tot_enc * 100) if tot_enc > 0 else 0
    avg_los = filtered_df["length_of_stay"].mean() if tot_enc > 0 else 0
    api_enriched = (filtered_df["diag_1_lookup_status"] == "MATCHED").sum()
    doc_linked = (filtered_df["feedback_available"] == 1).sum()

    kpi1.metric("Total Inpatient Encounters", f"{tot_enc:,}", f"{(tot_enc/len(main_df)*100):.1f}% of Mart")
    kpi2.metric("30-Day Readmission Rate", f"{readm_30_rate:.1f}%", f"{readm_30_count:,} patients")
    kpi3.metric("Avg Length of Stay (LOS)", f"{avg_los:.1f} days", "Median: 4.0d")
    kpi4.metric("NLM API Enriched Encounters", f"{api_enriched:,}", f"{(api_enriched/tot_enc*100):.1f}% Matched" if tot_enc > 0 else "0%")
    kpi5.metric("Linked Clinical Notes", f"{doc_linked:,}", f"{(doc_linked/tot_enc*100):.1f}% Matched" if tot_enc > 0 else "0%")

    st.divider()

    # Row 1: Views 1 & 2
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("1. Readmission Rate by Age Group")
        st.caption("Demonstrating progressive age risk across inpatient cohorts")
        age_summary = (
            filtered_df.groupby("age_group")
            .agg(readm_rate=("readmitted_30d", lambda x: x.mean() * 100), count=("surrogate_key", "count"))
            .reset_index()
        )
        fig_age = px.bar(
            age_summary,
            x="age_group",
            y="readm_rate",
            labels={"age_group": "Age Bracket", "readm_rate": "30-Day Readmission Rate (%)"},
            color="readm_rate",
            color_continuous_scale="Blues",
            text="readm_rate",
        )
        fig_age.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_age.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_age, use_container_width=True)

    with c2:
        st.subheader("2. Top Diagnoses by Readmission Rate (NLM API Enriched)")
        st.caption("Clinical terminology resolved via NLM Clinical Tables REST API (Source 2)")
        diag_grouped = (
            filtered_df.groupby("diag_1_description")
            .agg(readm_rate=("readmitted_30d", lambda x: x.mean() * 100), encounter_count=("surrogate_key", "count"))
            .reset_index()
        )
        diag_filtered = diag_grouped[diag_grouped["encounter_count"] >= 150].sort_values("readm_rate", ascending=False).head(10)

        fig_diag = px.bar(
            diag_filtered,
            x="readm_rate",
            y="diag_1_description",
            orientation="h",
            labels={"readm_rate": "30-Day Readmission Rate (%)", "diag_1_description": "Primary Diagnosis"},
            color="readm_rate",
            color_continuous_scale="Reds",
            text="readm_rate",
        )
        fig_diag.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_diag.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig_diag, use_container_width=True)

    st.divider()

    # Row 2: Views 3 & 4
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("3. Length of Stay (LOS) Distribution")
        st.caption("Comparison: Readmitted (<30d) vs Non-Readmitted Patients")
        filtered_df["readmitted_label"] = filtered_df["readmitted_30d"].map({1: "Readmitted (<30d)", 0: "Not Readmitted"})
        fig_los = px.histogram(
            filtered_df,
            x="length_of_stay",
            color="readmitted_label",
            barmode="overlay",
            labels={"length_of_stay": "Days in Hospital", "readmitted_label": "Cohort"},
            color_discrete_map={"Readmitted (<30d)": "#EF4444", "Not Readmitted": "#3B82F6"},
            opacity=0.65,
        )
        fig_los.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_los, use_container_width=True)

    with c4:
        st.subheader("4. Admission & Discharge Clinical Patterns")
        st.caption("Distribution of readmissions across discharge destinations")
        dsch_grouped = (
            filtered_df.groupby("discharge_disposition_name")
            .agg(readm_rate=("readmitted_30d", lambda x: x.mean() * 100), total=("surrogate_key", "count"))
            .reset_index()
            .sort_values("total", ascending=False)
            .head(8)
        )
        fig_dsch = px.bar(
            dsch_grouped,
            x="discharge_disposition_name",
            y="readm_rate",
            labels={"discharge_disposition_name": "Discharge Destination", "readm_rate": "Readmission Rate (%)"},
            color="readm_rate",
            color_continuous_scale="Purples",
            text="readm_rate",
        )
        fig_dsch.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_dsch.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_dsch, use_container_width=True)

    st.divider()

    # Row 3: Views 5 & 6
    c5, c6 = st.columns([1.2, 1])

    with c5:
        st.subheader("5. High-Risk Cohort Intelligence Dashboard")
        st.caption("Criteria: Patients with ≥2 prior admissions and ≥7 documented diagnoses")
        high_risk_sub = filtered_df[filtered_df["high_risk_flag"] == 1]
        normal_sub = filtered_df[filtered_df["high_risk_flag"] == 0]

        hr_rate = high_risk_sub["readmitted_30d"].mean() * 100 if len(high_risk_sub) > 0 else 0
        norm_rate = normal_sub["readmitted_30d"].mean() * 100 if len(normal_sub) > 0 else 0
        risk_multiplier = (hr_rate / norm_rate) if norm_rate > 0 else 0

        st.markdown(
            f"**High-Risk Readmission Rate**: `{hr_rate:.1f}%` vs Normal Cohort: `{norm_rate:.1f}%` "
            f"(**{risk_multiplier:.2f}x Risk Multiplier**)"
        )
        table_cols = [
            "surrogate_key", "age_group", "medical_specialty", "admission_type_name",
            "length_of_stay", "prior_admissions", "diagnosis_count", "diag_1_description", "readmitted_30d"
        ]
        disp_df = high_risk_sub[[c for c in table_cols if c in high_risk_sub.columns]].head(25).rename(
            columns={
                "surrogate_key": "Patient Key",
                "age_group": "Age",
                "medical_specialty": "Specialty",
                "admission_type_name": "Admission",
                "length_of_stay": "LOS",
                "prior_admissions": "Prior Adm",
                "diagnosis_count": "Diag Count",
                "diag_1_description": "Primary Diagnosis",
                "readmitted_30d": "Readmitted (30d)",
            }
        )
        st.dataframe(disp_df, use_container_width=True, hide_index=True)

    with c6:
        st.subheader("6. Multi-Source Document Signal (Source 3)")
        st.caption("NLP keyword risk score from clinical discharge notes vs 30-day readmission")
        notes_df = filtered_df[filtered_df["feedback_available"] == 1]
        missing_notes = filtered_df[filtered_df["feedback_available"] == 0]

        sc1, sc2 = st.columns(2)
        sc1.metric("Linked Notes", f"{len(notes_df):,}", "Document Raw Matches")
        sc2.metric("Missing Notes", f"{len(missing_notes):,}", "Kept as NULL (not 0)")

        if len(notes_df) > 0:
            doc_risk_summary = (
                notes_df.groupby("risk_keyword_score")
                .agg(readm_rate=("readmitted_30d", lambda x: x.mean() * 100), count=("surrogate_key", "count"))
                .reset_index()
            )
            fig_doc = px.line(
                doc_risk_summary,
                x="risk_keyword_score",
                y="readm_rate",
                markers=True,
                labels={"risk_keyword_score": "Discharge Note Risk Keyword Count", "readm_rate": "30-Day Readmission Rate (%)"},
            )
            fig_doc.update_traces(line_color="#E11D48", line_width=3, marker_size=10)
            fig_doc.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_doc, use_container_width=True)


# =====================================================================
# TAB 2: CLINICAL DECISION SUPPORT PREDICTOR (PART 2 MLOps)
# =====================================================================
with tab_predictor:
    st.subheader("🔮 Clinical Decision Support System: Real-Time Patient Risk Inference")
    st.markdown(
        "Enter inpatient clinical parameters below to score 30-day readmission risk using the **Champion XGBoost Model** "
        "registered in MLflow. The system renders patient-level **SHAP factor attribution** and **prescriptive transition-of-care actions**."
    )

    # Pre-Loaded Case Studies Selector
    st.markdown("#### ⚡ Quick-Select Reference Clinical Case Studies:")
    cc1, cc2, cc3 = st.columns(3)

    if "preset_case" not in st.session_state:
        st.session_state["preset_case"] = None

    if cc1.button("🔴 Case 1: Elderly High-Risk Polypharmacy Inpatient", use_container_width=True):
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

    if cc2.button("🟡 Case 2: Moderate-Risk Emergency Inpatient", use_container_width=True):
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

    if cc3.button("🟢 Case 3: Low-Risk Elective Surgical Inpatient", use_container_width=True):
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

    st.markdown("---")
    f_col1, f_col2, f_col3 = st.columns(3)

    with f_col1:
        st.markdown("##### 👤 Demographics & Admissions")
        inp_age = st.selectbox("Age Group", age_groups, index=age_groups.index(preset.get("age_group", "[70-80)")) if preset.get("age_group") in age_groups else 6)
        inp_gender = st.selectbox("Gender", ["Female", "Male"], index=0 if preset.get("gender", "Female") == "Female" else 1)
        inp_race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"], index=0)
        inp_adm_type = st.selectbox("Admission Type", ["Emergency", "Elective", "Urgent", "Other"], index=0)
        inp_dsch_disp = st.selectbox("Discharge Disposition", [
            "Discharged to home", "Discharged/transferred to SNF", "Discharged/transferred to home with home health service",
            "Discharged/transferred to rehab", "Left AMA", "Other"
        ], index=1 if "SNF" in str(preset.get("discharge_disposition_name", "")) else 0)
        inp_adm_source = st.selectbox("Admission Source", ["Emergency Room", "Physician Referral", "Transfer from hospital", "Other"], index=0)

    with f_col2:
        st.markdown("##### 🔬 Stay & Clinical Utilization")
        inp_los = st.slider("Length of Stay (Days)", min_value=1, max_value=14, value=int(preset.get("length_of_stay", 6)))
        inp_labs = st.slider("Number of Lab Procedures", min_value=1, max_value=120, value=int(preset.get("num_lab_procedures", 50)))
        inp_procs = st.slider("Number of Procedures", min_value=0, max_value=6, value=int(preset.get("num_procedures", 1)))
        inp_meds = st.slider("Number of Medications", min_value=1, max_value=50, value=int(preset.get("num_medications", 16)))
        inp_prior_adm = st.slider("Prior Admissions (Past 12 Months)", min_value=0, max_value=15, value=int(preset.get("prior_admissions", 2)))
        inp_diags = st.slider("Total Diagnosis Count", min_value=1, max_value=16, value=int(preset.get("diagnosis_count", 8)))

    with f_col3:
        st.markdown("##### 💊 Medications, Regimens & Notes")
        inp_specialty = st.selectbox("Medical Specialty", ["InternalMedicine", "Emergency/Trauma", "Cardiology", "Family/GeneralPractice", "Surgery-General", "Other"], index=0)
        inp_diag1 = st.text_input("Primary ICD-9 Code", value=str(preset.get("diag_1", "428")), help="e.g. 428 (Congestive Heart Failure), 250 (Diabetes), 486 (Pneumonia)")
        inp_a1c = st.selectbox("HbA1c Test Result", ["None", "Norm", ">7", ">8"], index=3 if preset.get("A1Cresult") == ">8" else 0)
        inp_insulin = st.selectbox("Insulin Regimen", ["No", "Steady", "Up", "Down"], index=2 if preset.get("insulin") == "Up" else (1 if preset.get("insulin") == "Steady" else 0))
        inp_ch_med = st.selectbox("Change in Diabetic Medication", ["No", "Ch"], index=1 if preset.get("change_med") == "Ch" else 0)
        inp_has_med = st.selectbox("Has Diabetes Medication Prescribed", ["Yes", "No"], index=0)
        inp_notes_score = st.slider("Clinical Notes Keyword Risk Score", min_value=0.0, max_value=3.0, value=float(preset.get("risk_keyword_score", 0.5)), step=0.25)

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

    st.markdown("---")
    calc_col, res_col = st.columns([1, 2])

    with calc_col:
        st.markdown("#### Clinical Risk Assessment")
        from src.ml_explain import explain_single_encounter
        explanation = explain_single_encounter(patient_payload)
        prob = explanation["readmission_probability"]
        prob_pct = prob * 100
        tier = explanation["risk_tier"]
        badge = explanation["risk_badge"]
        color = explanation["tier_color"]

        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Predicted Readmission Risk<br><span style='font-size:0.9em;color:{color}'>{badge}</span>", 'font': {'size': 18}},
            number={'suffix': "%", 'font': {'size': 36, 'color': color}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 15], 'color': '#DCFCE7'},
                    {'range': [15, 35], 'color': '#FEF3C7'},
                    {'range': [35, 100], 'color': '#FEE2E2'},
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 35
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div style="background-color: {color}15; border: 1px solid {color}; border-radius: 8px; padding: 12px; text-align: center;">
            <b style="color: {color}; font-size: 1.1rem;">Clinical Recommendation Status</b><br/>
            Decision Threshold: <b>50.0%</b> | Alert Tier: <b>{tier}</b>
        </div>
        """, unsafe_allow_html=True)

    with res_col:
        st.markdown("#### Patient SHAP Factor Attribution & Care Interventions")
        t_drivers, t_interventions = st.tabs(["🔬 Factor Attribution (SHAP)", "📋 Actionable Care Plan"])

        with t_drivers:
            risk_drivers = explanation.get("top_risk_drivers", [])
            protective = explanation.get("top_protective_factors", [])

            # Plot waterfall horizontal bar
            attr_items = risk_drivers[:3] + protective[:2]
            attr_items.sort(key=lambda x: x["shap_impact"])
            if attr_items:
                f_names = [d["feature"].replace("cat__", "").replace("num__", "") for d in attr_items]
                f_vals = [d["shap_impact"] for d in attr_items]
                f_colors = ["#EF4444" if v > 0 else "#10B981" for v in f_vals]

                fig_attr = px.bar(
                    x=f_vals,
                    y=f_names,
                    orientation="h",
                    labels={"x": "SHAP Risk Contribution (Log-Odds)", "y": "Clinical Feature"},
                    color=f_vals,
                    color_continuous_scale=[[0, "#10B981"], [0.5, "#94A3B8"], [1, "#EF4444"]],
                )
                fig_attr.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
                st.plotly_chart(fig_attr, use_container_width=True)

            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.markdown("**🔴 Top Risk Drivers (Pushing Risk UP):**")
                for rd in risk_drivers[:4]:
                    clean_name = rd["feature"].replace("cat__", "").replace("num__", "").replace("_", " ")
                    st.markdown(f"- `{clean_name}`: **+{rd['shap_impact']:.3f}**")
            with d_col2:
                st.markdown("**🟢 Protective Factors (Pushing Risk DOWN):**")
                for pf in protective[:4]:
                    clean_name = pf["feature"].replace("cat__", "").replace("num__", "").replace("_", " ")
                    st.markdown(f"- `{clean_name}`: **{pf['shap_impact']:.3f}**")

        with t_interventions:
            st.markdown("**Prescriptive Transition-of-Care Interventions:**")
            actions = explanation.get("clinical_action_items", [])
            for idx, act in enumerate(actions, 1):
                st.markdown(f"""
                <div class="clinical-callout">
                    <b>Recommendation #{idx}:</b> {act}
                </div>
                """, unsafe_allow_html=True)


# =====================================================================
# TAB 3: MLOps TELEMETRY, DIAGNOSTICS & DRIFT AUDIT (PART 2)
# =====================================================================
with tab_mlops:
    st.subheader("📊 Healthcare MLOps Telemetry & Production Governance")
    st.markdown(
        "Real-time governance dashboard tracking **MLflow Experiment Tracking**, **Champion Model Diagnostics**, "
        "**Global SHAP Clinical Drivers**, and **Continuous Data Drift / Subgroup Fairness Auditing**."
    )

    # Load metadata
    champ_meta = {}
    if os.path.exists(CHAMPION_METADATA_PATH):
        with open(CHAMPION_METADATA_PATH, "r", encoding="utf-8") as f:
            champ_meta = json.load(f)

    # Champion telemetry cards
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Champion Architecture", champ_meta.get("champion_name", "XGBoost Classifier"), "MLflow Champion")
    mc2.metric("Registry Target", champ_meta.get("registry_name", "HospitalReadmissionPredictor"), f"v{champ_meta.get('registry_version', '1')}")
    mc3.metric("Held-Out Test Recall", f"{champ_meta.get('test_metrics', {}).get('test_recall', 0.5270):.1%}", "Sensitivity Priority")
    mc4.metric("Held-Out Test ROC-AUC", f"{champ_meta.get('test_metrics', {}).get('test_roc_auc', 0.6492):.3f}", "Global Discrimination")
    mc5.metric("Brier Calibration Score", f"{champ_meta.get('test_metrics', {}).get('test_brier_score', 0.2131):.3f}", "Probabilistic Accuracy")

    st.divider()

    m_tab1, m_tab2, m_tab3 = st.tabs([
        "🏆 Model Comparison & Curves",
        "🐝 Global Clinical SHAP Drivers",
        "🛡️ Continuous Drift & Subgroup Fairness",
    ])

    with m_tab1:
        st.markdown("#### Candidate Model Comparison on Validation Cohort (N=10,498)")
        if os.path.exists(COMPARISON_METRICS_PATH):
            with open(COMPARISON_METRICS_PATH, "r", encoding="utf-8") as f:
                comp_data = json.load(f)
            df_comp = pd.DataFrame(comp_data)
            disp_comp = df_comp[[
                "model_name", "val_recall", "val_precision", "val_specificity", "val_f1", "val_roc_auc", "val_pr_auc", "val_brier_score"
            ]].rename(columns={
                "model_name": "Model Architecture",
                "val_recall": "Recall (Sensitivity)",
                "val_precision": "Precision",
                "val_specificity": "Specificity",
                "val_f1": "F1 Score",
                "val_roc_auc": "ROC-AUC",
                "val_pr_auc": "PR-AUC",
                "val_brier_score": "Brier Score",
            })
            st.dataframe(disp_comp.style.format({
                "Recall (Sensitivity)": "{:.3f}",
                "Precision": "{:.3f}",
                "Specificity": "{:.3f}",
                "F1 Score": "{:.3f}",
                "ROC-AUC": "{:.3f}",
                "PR-AUC": "{:.3f}",
                "Brier Score": "{:.3f}",
            }), use_container_width=True, hide_index=True)

        st.markdown("#### Diagnostic Clinical Curves")
        diag_c1, diag_c2 = st.columns(2)
        roc_img = os.path.join(EVAL_PLOTS_DIR, "roc_curve_xgboost_(gradient_boosted).png")
        pr_img = os.path.join(EVAL_PLOTS_DIR, "pr_curve_xgboost_(gradient_boosted).png")
        cm_img = os.path.join(EVAL_PLOTS_DIR, "confusion_matrix_xgboost_(gradient_boosted).png")
        cal_img = os.path.join(EVAL_PLOTS_DIR, "calibration_curve_xgboost_(gradient_boosted).png")

        if os.path.exists(roc_img):
            diag_c1.image(roc_img, caption="Champion ROC Curve (Discrimination across thresholds)")
        if os.path.exists(pr_img):
            diag_c2.image(pr_img, caption="Precision-Recall Curve (Imbalanced 8.98% Prevalence)")
        if os.path.exists(cm_img):
            diag_c1.image(cm_img, caption="Confusion Matrix at Default Clinical Threshold 0.50")
        if os.path.exists(cal_img):
            diag_c2.image(cal_img, caption="Probability Calibration Curve (Brier Score = 0.212)")

    with m_tab2:
        st.markdown("#### Global Clinical Feature Attribution via SHAP")
        shap_c1, shap_c2 = st.columns(2)
        beeswarm_img = os.path.join(DOCS_DIR, "shap_summary_beeswarm.png")
        bar_img = os.path.join(DOCS_DIR, "shap_importance_bar.png")

        if os.path.exists(beeswarm_img):
            shap_c1.image(beeswarm_img, caption="SHAP Summary Beeswarm Plot: Feature Impact Directionality")
        if os.path.exists(bar_img):
            shap_c2.image(bar_img, caption="Mean Absolute SHAP Feature Importance Ranking")

        if os.path.exists(SHAP_IMPORTANCE_PATH):
            with open(SHAP_IMPORTANCE_PATH, "r", encoding="utf-8") as f:
                shap_list = json.load(f)[:10]
            st.markdown("##### Top 10 Clinical Risk Drivers Ranked by SHAP Magnitude:")
            st.dataframe(pd.DataFrame(shap_list), use_container_width=True, hide_index=True)

    with m_tab3:
        st.markdown("#### Continuous Data Drift & Subgroup Fairness Telemetry")
        if os.path.exists(DRIFT_REPORT_PATH):
            with open(DRIFT_REPORT_PATH, "r", encoding="utf-8") as f:
                drift_rep = json.load(f)

            d_summary = drift_rep.get("data_drift", {}).get("drift_summary", {})
            retrain = drift_rep.get("retraining_decision", {})

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Features Tested for Drift", d_summary.get("total_features_tested", 25))
            sc2.metric("Drifted Features Detected", d_summary.get("drifted_features_count", 0), "KS p<0.01 & PSI>=0.10")
            sc3.metric("Retraining Trigger Status", "Active Trigger" if retrain.get("retrain_recommended") else "Within Parameters")

            st.markdown("**Automated Retraining Operational Policy Status:**")
            for reason in retrain.get("trigger_reasons", []):
                st.markdown(f"- 🔔 `{reason}`")

            st.markdown("##### Demographic Subgroup Fairness Audit (Age Groups, Gender, Race)")
            fairness = drift_rep.get("subgroup_fairness_audit", {})
            for p_attr, p_data in fairness.items():
                st.markdown(f"**Disparity Audit for `{p_attr}`:**")
                df_grp = pd.DataFrame(p_data).T.reset_index().rename(columns={"index": "Subgroup"})
                st.dataframe(df_grp, use_container_width=True, hide_index=True)
        else:
            st.info("Run `python src/ml_monitor.py` to generate the latest drift and fairness audit report.")

st.divider()

# Footer with student project metadata
st.markdown("""
<div style="text-align: center; color: #9CA3AF; font-size: 0.85rem; padding: 10px;">
    Hospital Readmission Analytics Pipeline | Individual Project Assignment — Data Engineering & MLOps (Parts 1 & 2)<br/>
    PostgreSQL Mart • Airflow DAG • MLflow Experiment Tracking • SHAP Clinical Explainability • FastAPI Microservice
</div>
""", unsafe_allow_html=True)
