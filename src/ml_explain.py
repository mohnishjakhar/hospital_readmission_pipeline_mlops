"""
ml_explain.py
-------------
Phase 3: Clinical Explainability (XAI) with SHAP
for Hospital Readmission Prediction (Part 2: MLOps Extension).

Key Deliverables:
1. Global Explainability (Cohort Level):
   - TreeExplainer on Champion XGBoost model.
   - SHAP Summary Beeswarm Plot: Feature value impact directionality across cohort.
   - SHAP Global Feature Importance Bar Plot.
   - Extraction of Top 15 clinical drivers into 'models/shap_feature_importance.json'.
2. Local Explainability (Encounter Level):
   - 'explain_single_encounter()': Clinical waterfall plot and plain-English risk factors.
   - Distinguishes Risk Multipliers (positive SHAP pushing probability UP)
     from Protective Factors (negative SHAP pushing probability DOWN).
3. Reference Clinical Case Studies:
   - Pre-computes 3 canonical patient scenarios:
     * High-Risk Polypharmacy Inpatient
     * Moderate-Risk Emergency Inpatient
     * Low-Risk Elective Surgical Inpatient
"""

import os
import sys
import json
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ml_features import (
    transform_raw_records,
    load_saved_preprocessor,
    BASE_DIR,
    MODELS_DIR,
    DATA_STAGING_DIR,
    PREPROCESSOR_SAVE_PATH,
    FEATURE_NAMES_SAVE_PATH,
)
from src.ml_train import (
    CHAMPION_MODEL_PATH,
    CHAMPION_METADATA_PATH,
    EVALUATION_PLOTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ML_EXPLAIN] %(message)s")
logger = logging.getLogger(__name__)

DOCS_DIR = os.path.join(BASE_DIR, "docs")
SHAP_BEESWARM_PATH = os.path.join(EVALUATION_PLOTS_DIR, "shap_summary_beeswarm.png")
SHAP_BAR_PATH = os.path.join(EVALUATION_PLOTS_DIR, "shap_importance_bar.png")
SHAP_FEATURE_IMPORTANCE_JSON = os.path.join(MODELS_DIR, "shap_feature_importance.json")
CASE_STUDIES_JSON = os.path.join(MODELS_DIR, "reference_case_studies.json")


# ---------------------------------------------------------------------------
# SHAP & XGBoost 3.x Compatibility Patch
# ---------------------------------------------------------------------------
import shap.explainers._tree as _st

if hasattr(_st, "decode_ubjson_buffer"):
    _orig_decode = _st.decode_ubjson_buffer

    def _patched_decode(fd):
        res = _orig_decode(fd)
        try:
            param = res.get("learner", {}).get("learner_model_param", {})
            if "base_score" in param and isinstance(param["base_score"], str):
                param["base_score"] = param["base_score"].strip("[]")
        except Exception:
            pass
        return res

    _st.decode_ubjson_buffer = _patched_decode


# ---------------------------------------------------------------------------
# Explainer Loader & Cache
# ---------------------------------------------------------------------------
_cached_explainer = None
_cached_champion_model = None
_cached_feature_names = None


def get_champion_and_explainer() -> Tuple[Any, shap.TreeExplainer, List[str]]:
    """
    Loads and caches the champion model, TreeExplainer, and feature names.
    """
    global _cached_explainer, _cached_champion_model, _cached_feature_names

    if _cached_explainer is None:
        if not os.path.exists(CHAMPION_MODEL_PATH):
            raise FileNotFoundError(f"Champion model not found at: {CHAMPION_MODEL_PATH}. Run src/ml_train.py first.")
        
        logger.info("Loading Champion Model from: %s", CHAMPION_MODEL_PATH)
        _cached_champion_model = joblib.load(CHAMPION_MODEL_PATH)

        with open(FEATURE_NAMES_SAVE_PATH, "r", encoding="utf-8") as f:
            _cached_feature_names = json.load(f)

        logger.info("Initializing SHAP TreeExplainer for Champion Model...")
        _cached_explainer = shap.TreeExplainer(_cached_champion_model)

    return _cached_champion_model, _cached_explainer, _cached_feature_names


# ---------------------------------------------------------------------------
# Global Explainability Engine
# ---------------------------------------------------------------------------
def compute_global_shap_explanations(
    sample_size: int = 1500,
    save_docs_copy: bool = True,
) -> Dict[str, Any]:
    """
    Computes global SHAP values over a representative sample of the validation cohort,
    generates Beeswarm and Bar summary charts, and extracts top clinical risk drivers.
    """
    os.makedirs(EVALUATION_PLOTS_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    champion_model, explainer, feature_names = get_champion_and_explainer()

    # Load validation set
    val_raw_path = os.path.join(DATA_STAGING_DIR, "X_val_raw.parquet")
    preproc_path = PREPROCESSOR_SAVE_PATH

    if not os.path.exists(val_raw_path):
        raise FileNotFoundError(f"Validation parquet not found at: {val_raw_path}. Run src/ml_train.py first.")

    preprocessor = load_saved_preprocessor(preproc_path)
    df_val_raw = pd.read_parquet(val_raw_path)

    # Sample for fast, accurate SHAP computation
    if len(df_val_raw) > sample_size:
        df_sample = df_val_raw.sample(n=sample_size, random_state=42)
    else:
        df_sample = df_val_raw

    logger.info("Transforming %d validation encounters for SHAP explanation...", len(df_sample))
    X_sample_trans = preprocessor.transform(df_sample)

    logger.info("Computing TreeExplainer SHAP values (N=%d, features=%d)...",
                X_sample_trans.shape[0], X_sample_trans.shape[1])
    shap_values = explainer.shap_values(X_sample_trans)

    # Handle binary classifier output dimension if needed
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_matrix = shap_values[1]
    else:
        shap_matrix = shap_values

    # 1. Generate SHAP Beeswarm Plot
    logger.info("Generating SHAP Beeswarm Summary Plot...")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_matrix,
        X_sample_trans,
        feature_names=feature_names,
        max_display=15,
        plot_type="dot",
        show=False,
    )
    plt.title("Global Clinical Drivers of 30-Day Hospital Readmission (SHAP Beeswarm)", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(SHAP_BEESWARM_PATH, dpi=200, bbox_inches="tight")
    if save_docs_copy:
        plt.savefig(os.path.join(DOCS_DIR, "shap_summary_beeswarm.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 2. Generate SHAP Feature Importance Bar Plot
    logger.info("Generating SHAP Feature Importance Bar Plot...")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_matrix,
        X_sample_trans,
        feature_names=feature_names,
        max_display=15,
        plot_type="bar",
        show=False,
    )
    plt.title("Top 15 Predictors by Mean |SHAP| (Impact on Readmission Risk)", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(SHAP_BAR_PATH, dpi=200, bbox_inches="tight")
    if save_docs_copy:
        plt.savefig(os.path.join(DOCS_DIR, "shap_importance_bar.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 3. Calculate Mean Absolute SHAP Ranking JSON
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:20]

    ranking = []
    for rank, idx in enumerate(top_indices, 1):
        ranking.append({
            "rank": rank,
            "feature": feature_names[idx],
            "mean_abs_shap": float(round(mean_abs_shap[idx], 4)),
        })

    with open(SHAP_FEATURE_IMPORTANCE_JSON, "w", encoding="utf-8") as f:
        json.dump(ranking, f, indent=2)

    logger.info("Saved Top Clinical Risk Drivers JSON to: %s", SHAP_FEATURE_IMPORTANCE_JSON)
    return {
        "beeswarm_path": SHAP_BEESWARM_PATH,
        "bar_path": SHAP_BAR_PATH,
        "ranking": ranking,
    }


# ---------------------------------------------------------------------------
# Local (Patient-Level) Explainability Engine
# ---------------------------------------------------------------------------
def explain_single_encounter(
    patient_record: Dict[str, Any],
    top_n: int = 5,
    save_plot_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes real-time patient-level clinical explainability for a single encounter:
      1. Transforms the raw patient dictionary through preprocessor.
      2. Computes patient-specific SHAP values.
      3. Classifies factors into Risk Drivers (positive SHAP) vs Protective Factors (negative SHAP).
      4. Generates an interpretable patient waterfall plot.
      5. Returns risk tier, calibrated probability, and actionable clinical talking points.
    """
    champion_model, explainer, feature_names = get_champion_and_explainer()
    preprocessor = load_saved_preprocessor(PREPROCESSOR_SAVE_PATH)

    # Convert single patient dict to DataFrame
    df_single = pd.DataFrame([patient_record])
    X_single_trans = transform_raw_records(df_single, preprocessor=preprocessor)

    # Compute probability & risk tier
    prob = float(champion_model.predict_proba(X_single_trans)[0, 1])

    if prob >= 0.35:
        risk_tier = "High Risk"
        risk_badge = "🔴 High Risk (>35%)"
        tier_color = "#DC2626"
    elif prob >= 0.15:
        risk_tier = "Moderate Risk"
        risk_badge = "🟡 Moderate Risk (15-35%)"
        tier_color = "#F59E0B"
    else:
        risk_tier = "Low Risk"
        risk_badge = "🟢 Low Risk (<15%)"
        tier_color = "#10B981"

    # Compute SHAP explanation
    shap_vals = explainer(X_single_trans)
    base_value = float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0.0

    if len(shap_vals.values.shape) == 2:
        encounter_shap = shap_vals.values[0]
    else:
        encounter_shap = shap_vals.values[0, :, 1]

    # Partition into positive risk drivers and protective factors
    indexed_shap = [(feature_names[i], float(encounter_shap[i])) for i in range(len(feature_names))]
    
    # Sort positive (pushing risk UP)
    risk_drivers = sorted([item for item in indexed_shap if item[1] > 0.001], key=lambda x: x[1], reverse=True)[:top_n]
    # Sort negative (protective, pushing risk DOWN)
    protective_factors = sorted([item for item in indexed_shap if item[1] < -0.001], key=lambda x: x[1])[:top_n]

    # Clinical action recommendations based on top drivers
    action_items = []
    driver_features = [d[0].lower() for d in risk_drivers]
    
    if any("prior_admissions" in f or "inpatient" in f for f in driver_features):
        action_items.append("Schedule 7-day post-discharge nurse home visit (High prior utilization).")
    if any("discharge_disposition" in f and ("snf" in f or "rehab" in f or "home health" in f) for f in driver_features):
        action_items.append("Confirm skilled nursing facility handoff protocol and medication reconciliation.")
    if any("num_medications" in f or "insulin" in f or "change" in f for f in driver_features):
        action_items.append("Pharmacist medication consultation to prevent post-discharge adverse drug events.")
    if any("risk_keyword_score" in f or "notes" in f for f in driver_features):
        action_items.append("Social work assessment for non-compliance / post-discharge living support.")
    if not action_items:
        action_items.append("Standard primary care outpatient follow-up within 14-21 days.")

    # Generate patient waterfall plot
    if save_plot_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_plot_path)), exist_ok=True)
        plt.figure(figsize=(9, 5))
        
        # Select top drivers & protective for visualization
        display_items = risk_drivers[:3] + protective_factors[:2]
        display_items.sort(key=lambda x: x[1])
        
        feats = [d[0] for d in display_items]
        vals = [d[1] for d in display_items]
        colors = ["#EF4444" if v > 0 else "#10B981" for v in vals]
        
        bars = plt.barh(feats, vals, color=colors, height=0.55)
        plt.axvline(0, color="#6B7280", linestyle="--", alpha=0.7)
        plt.title(f"Patient Risk Factor Attribution (Risk: {prob:.1%} - {risk_tier})", fontsize=11, fontweight="bold")
        plt.xlabel("SHAP Contribution (Log-Odds Impact on Readmission)", fontsize=10)
        plt.tight_layout()
        plt.savefig(save_plot_path, dpi=200, bbox_inches="tight")
        plt.close()

    return {
        "readmission_probability": round(prob, 4),
        "readmission_percentage": f"{round(prob * 100, 1)}%",
        "risk_tier": risk_tier,
        "risk_badge": risk_badge,
        "tier_color": tier_color,
        "base_value": round(base_value, 4),
        "top_risk_drivers": [{"feature": f, "shap_impact": round(s, 4)} for f, s in risk_drivers],
        "top_protective_factors": [{"feature": f, "shap_impact": round(s, 4)} for f, s in protective_factors],
        "clinical_action_items": action_items,
        "waterfall_plot_path": save_plot_path,
    }


# ---------------------------------------------------------------------------
# Reference Clinical Case Studies Builder
# ---------------------------------------------------------------------------
def generate_reference_case_studies() -> List[Dict[str, Any]]:
    """
    Creates and explains 3 canonical patient archetypes for clinical testing:
      Case 1: Elderly High-Risk Inpatient (Polypharmacy, frequent admissions)
      Case 2: Moderate-Risk Emergency Inpatient (Circulatory diagnosis, insulin)
      Case 3: Low-Risk Elective Surgery Inpatient (Short stay, young, first visit)
    """
    cases = [
        {
            "case_id": "CASE-01",
            "title": "Elderly High-Risk Polypharmacy Inpatient",
            "description": "75-year-old female with diabetes & cardiovascular complications, 4 prior admissions, discharged to skilled nursing facility with 24 medications.",
            "record": {
                "age_group": "[70-80)",
                "gender": "Female",
                "race": "Caucasian",
                "admission_type_name": "Emergency",
                "discharge_disposition_name": "Discharged/transferred to SNF",
                "admission_source_name": "Emergency Room",
                "length_of_stay": 8,
                "num_lab_procedures": 65,
                "num_procedures": 3,
                "num_medications": 24,
                "prior_admissions": 4,
                "number_outpatient": 1,
                "number_emergency": 2,
                "number_inpatient": 2,
                "diagnosis_count": 9,
                "risk_keyword_score": 1.25,
                "medical_specialty": "InternalMedicine",
                "diag_1": "428",  # Congestive heart failure (Circulatory)
                "max_glu_serum": "None",
                "A1Cresult": ">8",
                "insulin": "Up",
                "change_med": "Ch",
                "has_diabetes_med": "Yes",
                "high_risk_flag": 1,
                "feedback_available": 1,
            },
        },
        {
            "case_id": "CASE-02",
            "title": "Moderate-Risk Inpatient with Glycemic Instability",
            "description": "55-year-old male admitted through emergency for respiratory distress, 1 prior visit, steady insulin regimen.",
            "record": {
                "age_group": "[50-60)",
                "gender": "Male",
                "race": "AfricanAmerican",
                "admission_type_name": "Emergency",
                "discharge_disposition_name": "Discharged to home",
                "admission_source_name": "Emergency Room",
                "length_of_stay": 4,
                "num_lab_procedures": 48,
                "num_procedures": 1,
                "num_medications": 14,
                "prior_admissions": 1,
                "number_outpatient": 0,
                "number_emergency": 1,
                "number_inpatient": 0,
                "diagnosis_count": 7,
                "risk_keyword_score": 0.35,
                "medical_specialty": "Emergency/Trauma",
                "diag_1": "486",  # Pneumonia (Respiratory)
                "max_glu_serum": "None",
                "A1Cresult": "Norm",
                "insulin": "Steady",
                "change_med": "No",
                "has_diabetes_med": "Yes",
                "high_risk_flag": 0,
                "feedback_available": 1,
            },
        },
        {
            "case_id": "CASE-03",
            "title": "Low-Risk Young Elective Inpatient",
            "description": "35-year-old female elective orthopedic admission, 2-day stay, no prior hospital admissions, discharged home.",
            "record": {
                "age_group": "[30-40)",
                "gender": "Female",
                "race": "Caucasian",
                "admission_type_name": "Elective",
                "discharge_disposition_name": "Discharged to home",
                "admission_source_name": "Physician Referral",
                "length_of_stay": 2,
                "num_lab_procedures": 22,
                "num_procedures": 2,
                "num_medications": 7,
                "prior_admissions": 0,
                "number_outpatient": 0,
                "number_emergency": 0,
                "number_inpatient": 0,
                "diagnosis_count": 3,
                "risk_keyword_score": 0.0,
                "medical_specialty": "Orthopedics",
                "diag_1": "715",  # Osteoarthrosis (Musculoskeletal)
                "max_glu_serum": "None",
                "A1Cresult": "None",
                "insulin": "No",
                "change_med": "No",
                "has_diabetes_med": "No",
                "high_risk_flag": 0,
                "feedback_available": 0,
            },
        },
    ]

    results = []
    case_plots_dir = os.path.join(EVALUATION_PLOTS_DIR, "case_studies")
    os.makedirs(case_plots_dir, exist_ok=True)

    for case in cases:
        plot_path = os.path.join(case_plots_dir, f"{case['case_id'].lower()}_waterfall.png")
        explanation = explain_single_encounter(case["record"], save_plot_path=plot_path)
        combined = {**case, "explanation": explanation}
        results.append(combined)

    with open(CASE_STUDIES_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved %d reference clinical case studies to: %s", len(results), CASE_STUDIES_JSON)
    return results


# ---------------------------------------------------------------------------
# CLI Orchestrator
# ---------------------------------------------------------------------------
def run_explainability_pipeline():
    logger.info("Starting Phase 3 Clinical Explainability (XAI) Pipeline...")
    global_results = compute_global_shap_explanations(sample_size=1500)
    case_results = generate_reference_case_studies()
    logger.info("Phase 3 Clinical Explainability completed successfully!")
    return {
        "global_results": global_results,
        "case_studies": case_results,
    }


if __name__ == "__main__":
    run_explainability_pipeline()
