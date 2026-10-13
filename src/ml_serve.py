"""
ml_serve.py
-----------
Phase 4: Production REST Inference Microservice via FastAPI
for Hospital Readmission Prediction (Part 2: MLOps Extension).

Key Endpoints:
  - GET  /health            : Microservice liveness, model version & readiness status
  - POST /predict           : Real-time single patient risk scoring & clinical drivers
  - POST /predict-batch     : Bulk encounter cohort scoring with aggregate statistics
  - GET  /model-metadata    : Champion architecture, test metrics, and top SHAP features
  - GET  /case-studies      : Pre-calculated canonical clinical case archetypes
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import joblib
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ml_features import (
    transform_raw_records,
    load_saved_preprocessor,
    PREPROCESSOR_SAVE_PATH,
    FEATURE_NAMES_SAVE_PATH,
    MODELS_DIR,
)
from src.ml_train import (
    CHAMPION_MODEL_PATH,
    CHAMPION_METADATA_PATH,
)
from src.ml_explain import (
    explain_single_encounter,
    CASE_STUDIES_JSON,
    SHAP_FEATURE_IMPORTANCE_JSON,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ML_SERVE] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI Application & Lifespan Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Hospital Readmission Clinical Prediction Microservice",
    description="Production-grade clinical decision support inference engine for predicting 30-day hospital readmission risk.",
    version="2.0.0",
)

# Enable CORS for dashboard and external clinical systems
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache for ultra-low latency serving (<20ms)
MODEL_CACHE: Dict[str, Any] = {}


def load_serving_artifacts():
    """Loads champion model, preprocessor, and metadata into memory."""
    if "model" not in MODEL_CACHE:
        if not os.path.exists(CHAMPION_MODEL_PATH):
            raise RuntimeError(f"Champion model not found at: {CHAMPION_MODEL_PATH}. Train first using src/ml_train.py.")
        
        logger.info("Loading Champion Model into memory from: %s", CHAMPION_MODEL_PATH)
        MODEL_CACHE["model"] = joblib.load(CHAMPION_MODEL_PATH)
        MODEL_CACHE["preprocessor"] = load_saved_preprocessor(PREPROCESSOR_SAVE_PATH)

        if os.path.exists(CHAMPION_METADATA_PATH):
            with open(CHAMPION_METADATA_PATH, "r", encoding="utf-8") as f:
                MODEL_CACHE["metadata"] = json.load(f)
        else:
            MODEL_CACHE["metadata"] = {}

        if os.path.exists(FEATURE_NAMES_SAVE_PATH):
            with open(FEATURE_NAMES_SAVE_PATH, "r", encoding="utf-8") as f:
                MODEL_CACHE["feature_names"] = json.load(f)
        else:
            MODEL_CACHE["feature_names"] = []


@app.on_event("startup")
def startup_event():
    load_serving_artifacts()
    logger.info("Clinical Prediction Microservice initialized and ready for inferences.")


# ---------------------------------------------------------------------------
# Pydantic Request & Response Schemas
# ---------------------------------------------------------------------------
class PatientEncounterInput(BaseModel):
    age_group: str = Field(default="[60-70)", description="Patient age bracket (e.g. [50-60), [60-70), [70-80))")
    gender: str = Field(default="Female", description="Gender ('Male', 'Female', or 'Unknown')")
    race: str = Field(default="Caucasian", description="Race category")
    admission_type_name: str = Field(default="Emergency", description="Admission type ('Emergency', 'Elective', 'Urgent')")
    discharge_disposition_name: str = Field(default="Discharged to home", description="Discharge destination")
    admission_source_name: str = Field(default="Emergency Room", description="Referral source")
    length_of_stay: int = Field(default=4, ge=1, le=14, description="Hospital stay duration in days (1 to 14)")
    num_lab_procedures: int = Field(default=40, ge=1, le=150, description="Number of lab tests performed")
    num_procedures: int = Field(default=1, ge=0, le=10, description="Number of non-lab medical procedures")
    num_medications: int = Field(default=12, ge=1, le=100, description="Number of distinct medications administered")
    prior_admissions: int = Field(default=1, ge=0, le=30, description="Sum of outpatient, emergency, and inpatient visits in prior year")
    number_outpatient: int = Field(default=0, ge=0, le=50, description="Number of outpatient visits")
    number_emergency: int = Field(default=0, ge=0, le=50, description="Number of emergency visits")
    number_inpatient: int = Field(default=1, ge=0, le=50, description="Number of inpatient admissions")
    diagnosis_count: int = Field(default=7, ge=1, le=20, description="Total number of diagnosis codes entered")
    risk_keyword_score: float = Field(default=0.0, ge=0.0, description="Adherence/readmission keyword risk score from clinical notes")
    medical_specialty: str = Field(default="InternalMedicine", description="Admitting medical specialty")
    diag_1: str = Field(default="428", description="Primary ICD-9 diagnosis code (e.g. '428' for CHF, '250.01' for Diabetes)")
    max_glu_serum: str = Field(default="None", description="Glucose serum test result ('None', 'Norm', '>200', '>300')")
    A1Cresult: str = Field(default="None", description="A1C test result ('None', 'Norm', '>7', '>8')")
    insulin: str = Field(default="No", description="Insulin prescription ('No', 'Steady', 'Up', 'Down')")
    change_med: str = Field(default="No", description="Change in diabetic medication ('Ch', 'No')")
    has_diabetes_med: str = Field(default="Yes", description="Prescribed diabetes medication ('Yes', 'No')")
    high_risk_flag: int = Field(default=0, ge=0, le=1, description="Binary high-risk clinical cohort flag")
    feedback_available: int = Field(default=1, ge=0, le=1, description="Whether qualitative discharge notes were ingested")

    class Config:
        json_schema_extra = {
            "example": {
                "age_group": "[70-80)",
                "gender": "Female",
                "race": "Caucasian",
                "admission_type_name": "Emergency",
                "discharge_disposition_name": "Discharged to home",
                "admission_source_name": "Emergency Room",
                "length_of_stay": 6,
                "num_lab_procedures": 52,
                "num_procedures": 2,
                "num_medications": 18,
                "prior_admissions": 2,
                "number_outpatient": 0,
                "number_emergency": 1,
                "number_inpatient": 1,
                "diagnosis_count": 9,
                "risk_keyword_score": 0.75,
                "medical_specialty": "InternalMedicine",
                "diag_1": "428",
                "max_glu_serum": "None",
                "A1Cresult": ">8",
                "insulin": "Steady",
                "change_med": "Ch",
                "has_diabetes_med": "Yes",
                "high_risk_flag": 1,
                "feedback_available": 1,
            }
        }


class FactorAttribution(BaseModel):
    feature: str
    shap_impact: float


class SinglePredictionResponse(BaseModel):
    status: str = "success"
    readmission_probability: float
    readmission_percentage: str
    risk_tier: str
    risk_badge: str
    tier_color: str
    predicted_class: int
    threshold_used: float
    clinical_action_items: List[str]
    top_risk_drivers: List[FactorAttribution]
    top_protective_factors: List[FactorAttribution]
    latency_ms: float


class BatchEncounterInput(BaseModel):
    encounters: List[PatientEncounterInput]


class BatchPredictionResponse(BaseModel):
    status: str = "success"
    total_encounters_scored: int
    high_risk_count: int
    moderate_risk_count: int
    low_risk_count: int
    cohort_average_probability: float
    predictions: List[Dict[str, Any]]
    total_latency_ms: float


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", summary="Service Health and Model Status")
def health_check() -> Dict[str, Any]:
    """Returns liveness, model metadata, and inference engine readiness."""
    load_serving_artifacts()
    meta = MODEL_CACHE.get("metadata", {})
    return {
        "status": "healthy",
        "service": "Hospital Readmission Prediction Microservice",
        "champion_model": meta.get("champion_name", "XGBoost Classifier"),
        "registry_name": meta.get("registry_name", "HospitalReadmissionPredictor"),
        "registry_version": meta.get("registry_version", "1"),
        "features_loaded": len(MODEL_CACHE.get("feature_names", [])),
        "test_recall": meta.get("test_metrics", {}).get("test_recall", 0.5270),
        "test_roc_auc": meta.get("test_metrics", {}).get("test_roc_auc", 0.6492),
        "timestamp": time.time(),
    }


@app.post("/predict", response_model=SinglePredictionResponse, summary="Real-Time Single Patient Risk Inference")
def predict_single_encounter(patient: PatientEncounterInput) -> SinglePredictionResponse:
    """
    Scores a single patient encounter and produces:
      - 30-Day Readmission Risk Probability (0.0 to 1.0)
      - Clinical Risk Tier (Low / Moderate / High)
      - Key clinical risk drivers and protective factors
      - Prescriptive transition-of-care recommendations
    """
    start_time = time.time()
    load_serving_artifacts()
    record_dict = patient.model_dump() if hasattr(patient, "model_dump") else patient.dict()

    try:
        # Generate full clinical explanation with SHAP attribution
        explanation = explain_single_encounter(record_dict)
    except Exception as e:
        logger.error("Error in explain_single_encounter: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )

    prob = explanation["readmission_probability"]
    pred_class = 1 if prob >= 0.50 else 0
    latency = round((time.time() - start_time) * 1000, 2)

    return SinglePredictionResponse(
        status="success",
        readmission_probability=prob,
        readmission_percentage=explanation["readmission_percentage"],
        risk_tier=explanation["risk_tier"],
        risk_badge=explanation["risk_badge"],
        tier_color=explanation["tier_color"],
        predicted_class=pred_class,
        threshold_used=0.50,
        clinical_action_items=explanation["clinical_action_items"],
        top_risk_drivers=[FactorAttribution(**d) for d in explanation["top_risk_drivers"]],
        top_protective_factors=[FactorAttribution(**d) for d in explanation["top_protective_factors"]],
        latency_ms=latency,
    )


@app.post("/predict-batch", response_model=BatchPredictionResponse, summary="Batch Encounter Risk Scoring")
def predict_batch_encounters(batch: BatchEncounterInput) -> BatchPredictionResponse:
    """
    Scores a batch of patient encounters in vector mode for high-throughput population health screening.
    """
    start_time = time.time()
    load_serving_artifacts()

    if not batch.encounters:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty encounter list provided.")

    records = [enc.model_dump() if hasattr(enc, "model_dump") else enc.dict() for enc in batch.encounters]
    df_batch = pd.DataFrame(records)

    try:
        X_trans = transform_raw_records(df_batch, preprocessor=MODEL_CACHE["preprocessor"])
        probs = MODEL_CACHE["model"].predict_proba(X_trans)[:, 1]
    except Exception as e:
        logger.error("Batch inference error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch transformation error: {str(e)}",
        )

    predictions = []
    high_count, mod_count, low_count = 0, 0, 0

    for idx, prob in enumerate(probs):
        p_val = float(round(prob, 4))
        if p_val >= 0.35:
            tier = "High Risk"
            high_count += 1
        elif p_val >= 0.15:
            tier = "Moderate Risk"
            mod_count += 1
        else:
            tier = "Low Risk"
            low_count += 1

        predictions.append({
            "encounter_index": idx,
            "readmission_probability": p_val,
            "risk_tier": tier,
            "predicted_class": 1 if p_val >= 0.50 else 0,
        })

    total_latency = round((time.time() - start_time) * 1000, 2)
    cohort_avg = round(float(np.mean(probs)), 4)

    return BatchPredictionResponse(
        status="success",
        total_encounters_scored=len(predictions),
        high_risk_count=high_count,
        moderate_risk_count=mod_count,
        low_risk_count=low_count,
        cohort_average_probability=cohort_avg,
        predictions=predictions,
        total_latency_ms=total_latency,
    )


@app.get("/model-metadata", summary="Champion Model Parameters and Performance")
def get_model_metadata() -> Dict[str, Any]:
    """Returns champion model training parameters, test metrics, and top SHAP features."""
    load_serving_artifacts()
    meta = MODEL_CACHE.get("metadata", {})

    top_features = []
    if os.path.exists(SHAP_FEATURE_IMPORTANCE_JSON):
        with open(SHAP_FEATURE_IMPORTANCE_JSON, "r", encoding="utf-8") as f:
            top_features = json.load(f)[:10]

    return {
        "champion_model": meta.get("champion_name", "XGBoost (Gradient Boosted)"),
        "flavor": meta.get("flavor", "xgboost"),
        "run_id": meta.get("run_id", "ad6171273b574454b6f1435914e28846"),
        "validation_metrics": meta.get("validation_metrics", {}),
        "test_metrics": meta.get("test_metrics", {}),
        "hyperparameters": meta.get("parameters", {}),
        "top_shap_features": top_features,
    }


@app.get("/case-studies", summary="Pre-Loaded Reference Clinical Case Archetypes")
def get_case_studies() -> List[Dict[str, Any]]:
    """Returns 3 pre-computed clinical case archetypes for live interactive demonstration."""
    if os.path.exists(CASE_STUDIES_JSON):
        with open(CASE_STUDIES_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI Uvicorn Server on 0.0.0.0:8000...")
    uvicorn.run("src.ml_serve:app", host="0.0.0.0", port=8000, reload=False)
