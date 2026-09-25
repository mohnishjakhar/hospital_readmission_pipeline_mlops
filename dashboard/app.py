"""
app.py
------
Healthcare Analytical Dashboard: Hospital Readmission Analytics & Prediction
Course: Data Engineering & MLOps | Individual Project (Part 1 Deliverable)

Features:
- Live analytical connection to PostgreSQL / SQLite healthcare data mart.
- Integrates 3 segregated sources:
  1. UCI Diabetes 130-US Hospitals (CSV encounter dataset)
  2. NLM Clinical Tables ICD-9-CM API (REST API diagnosis enrichment)
  3. Clinical Feedback & Discharge Documents (Unstructured text NLP scoring)
- 5 Core Required Clinical Views + Multi-Source Lineage & Document Signal View.
- Interactive multi-parameter filtering (Specialty, Age Group, Risk Level, Admission Type).
"""

import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "hospital_mart.db")

st.set_page_config(
    page_title="Hospital Readmission Analytics Mart",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern clinical analytics aesthetic
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border: 1px solid #DBEAFE;
        border-radius: 10px;
        padding: 16px;
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
</style>
""", unsafe_allow_html=True)


def get_engine():
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        return create_engine(db_url)
    return create_engine(f"sqlite:///{DB_PATH}")


@st.cache_data(ttl=600)
def load_data():
    engine = get_engine()
    readm = pd.read_sql("SELECT * FROM readmissions", engine)
    adm = pd.read_sql("SELECT * FROM admissions", engine)
    diag = pd.read_sql("SELECT * FROM diagnoses", engine)
    diag_ref = pd.read_sql("SELECT * FROM diagnosis_reference", engine)
    doc_ref = pd.read_sql("SELECT * FROM specialty_feedback", engine)
    return readm, adm, diag, diag_ref, doc_ref


# Load datasets
readm_df, adm_df, diag_df, diag_ref_df, doc_ref_df = load_data()

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
st.sidebar.title("Filter Mart Data")
st.sidebar.caption("Interactive Cohort Segmentation")

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
st.markdown('<div class="main-title">🏥 Hospital Readmission Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    '<span class="badge-source badge-file">Source 1: File (UCI Encounter CSV)</span>'
    '<span class="badge-source badge-api">Source 2: REST API (NLM ICD-9 Search)</span>'
    '<span class="badge-source badge-doc">Source 3: Documents (Clinical Discharge Notes)</span>'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# KPI Metric Cards
# ---------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

total_cohort = len(filtered_df)
readm_count = int(filtered_df["readmitted_30d"].sum()) if total_cohort > 0 else 0
readm_rate = (readm_count / total_cohort * 100) if total_cohort > 0 else 0.0
high_risk_count = int(filtered_df["high_risk_flag"].sum()) if total_cohort > 0 else 0
avg_los = filtered_df["length_of_stay"].mean() if total_cohort > 0 else 0.0
doc_coverage = (filtered_df["feedback_available"].sum() / total_cohort * 100) if total_cohort > 0 else 0.0

baseline_rate = (main_df["readmitted_30d"].mean() * 100)
rate_delta = readm_rate - baseline_rate

kpi1.metric("Filtered Encounters", f"{total_cohort:,}", delta=f"{total_cohort - len(main_df):,}" if total_cohort != len(main_df) else "Full Cohort")
kpi2.metric("30-Day Readmission Rate", f"{readm_rate:.1f}%", delta=f"{rate_delta:+.2f}% vs baseline", delta_color="inverse")
kpi3.metric("High-Risk Encounters", f"{high_risk_count:,}", f"{high_risk_count/total_cohort*100:.1f}% of cohort" if total_cohort > 0 else "0%")
kpi4.metric("Avg Length of Stay", f"{avg_los:.1f} Days", delta=f"{avg_los - main_df['length_of_stay'].mean():+.1f} days")
kpi5.metric("Clinical Notes Linked", f"{doc_coverage:.1f}%", "Source 3 Coverage")

st.divider()

# ---------------------------------------------------------------------
# Row 1: Views 1 & 2
# ---------------------------------------------------------------------
c1, c2 = st.columns([1, 1.2])

with c1:
    st.subheader("1. Readmission Rate by Age Group")
    st.caption("Percentage of patients readmitted within 30 days across age deciles")
    if total_cohort > 0:
        age_summary = (
            filtered_df.groupby("age_group")
            .agg(readm_rate=("readmitted_30d", lambda x: x.mean() * 100), total_patients=("surrogate_key", "count"))
            .reset_index()
            .sort_values("age_group")
        )
        fig_age = px.bar(
            age_summary,
            x="age_group",
            y="readm_rate",
            text=age_summary["readm_rate"].apply(lambda v: f"{v:.1f}%"),
            labels={"age_group": "Age Bracket", "readm_rate": "30-Day Readmission Rate (%)"},
            color="readm_rate",
            color_continuous_scale="Reds",
        )
        fig_age.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), coloraxis_showscale=False)
        fig_age.update_traces(textposition="outside")
        st.plotly_chart(fig_age, use_container_width=True)
    else:
        st.warning("No records match the current filter selection.")

with c2:
    st.subheader("2. Readmission by Diagnosis (Enriched via NLM API)")
    st.caption("Top primary diagnoses enriched with official NLM definitions (Source 2)")
    known_diag = filtered_df[filtered_df["diag_1_description"].notna() & (filtered_df["diag_1_description"] != "")]
    if len(known_diag) > 0:
        top_dx = known_diag["diag_1_description"].value_counts().head(8).index
        dx_summary = (
            known_diag[known_diag["diag_1_description"].isin(top_dx)]
            .groupby("diag_1_description")
            .agg(readm_rate=("readmitted_30d", lambda x: x.mean() * 100), case_count=("surrogate_key", "count"))
            .reset_index()
            .sort_values("readm_rate", ascending=True)
        )
        # Clean label length for display
        dx_summary["short_desc"] = dx_summary["diag_1_description"].apply(
            lambda x: x[:36] + "..." if len(x) > 36 else x
        )
        fig_dx = px.bar(
            dx_summary,
            x="readm_rate",
            y="short_desc",
            orientation="h",
            text=dx_summary["readm_rate"].apply(lambda v: f"{v:.1f}%"),
            labels={"short_desc": "Primary Diagnosis (NLM Description)", "readm_rate": "30-Day Readmission Rate (%)"},
            color="readm_rate",
            color_continuous_scale="Teal",
        )
        fig_dx.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), coloraxis_showscale=False)
        fig_dx.update_traces(textposition="outside")
        st.plotly_chart(fig_dx, use_container_width=True)
    else:
        st.info("No NLM diagnosis records found for the selected cohort.")

st.divider()

# ---------------------------------------------------------------------
# Row 2: Views 3 & 4
# ---------------------------------------------------------------------
c3, c4 = st.columns([1, 1.2])

with c3:
    st.subheader("3. Length-of-Stay Distribution")
    st.caption("Hospital stay duration (days) stratified by 30-day readmission status")
    if total_cohort > 0:
        los_summary = (
            filtered_df.groupby(["length_of_stay", "readmitted_30d"])
            .size()
            .reset_index(name="encounter_count")
        )
        los_summary["Readmitted"] = los_summary["readmitted_30d"].map({1: "Readmitted (<30d)", 0: "Not Readmitted"})
        fig_los = px.bar(
            los_summary,
            x="length_of_stay",
            y="encounter_count",
            color="Readmitted",
            barmode="group",
            labels={"length_of_stay": "Days in Hospital", "encounter_count": "Patient Encounters"},
            color_discrete_map={"Readmitted (<30d)": "#DC2626", "Not Readmitted": "#2563EB"},
        )
        fig_los.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_los, use_container_width=True)

with c4:
    st.subheader("4. Admission & Discharge Clinical Patterns")
    tab_adm, tab_dsch = st.tabs(["Admission Types", "Discharge Dispositions"])

    with tab_adm:
        st.caption("Encounter distribution across standardized admission categories")
        adm_summary = filtered_df["admission_type_name"].value_counts().reset_index()
        adm_summary.columns = ["Admission Type", "Encounters"]
        fig_adm = px.pie(
            adm_summary.head(6),
            names="Admission Type",
            values="Encounters",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig_adm.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_adm, use_container_width=True)

    with tab_dsch:
        st.caption("Post-discharge destinations and care transitions")
        dsch_summary = filtered_df["discharge_disposition_name"].value_counts().head(7).reset_index()
        dsch_summary.columns = ["Discharge Disposition", "Encounters"]
        fig_dsch = px.bar(
            dsch_summary,
            x="Encounters",
            y="Discharge Disposition",
            orientation="h",
            color="Encounters",
            color_continuous_scale="Blues",
        )
        fig_dsch.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_dsch, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------
# Row 3: Views 5 & 6
# ---------------------------------------------------------------------
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
    st.caption(f"Showing top 25 of {len(high_risk_sub):,} high-risk patient records.")

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
        st.caption("Notice the clear positive trend: encounters with higher discharge confusion/adherence risk keywords exhibit higher 30-day readmission rates.")
    else:
        st.info("No linked document records for the selected filters.")

st.divider()

# Footer with student project metadata
st.markdown("""
<div style="text-align: center; color: #9CA3AF; font-size: 0.85rem; padding: 10px;">
    Hospital Readmission Analytics Pipeline | Individual Project Assignment — Data Engineering & MLOps<br/>
    PostgreSQL Analytical Mart • Apache Airflow Orchestration • Multi-Source Segregated Ingestion
</div>
""", unsafe_allow_html=True)
