"""
ml_features.py
--------------
Phase 1: Feature Engineering, Stratified Splitting, and Leakage-Free Preprocessing
for Hospital Readmission Prediction (Part 2: MLOps Extension).

Key Deliverables:
1. Target Separation & Identifier Isolation:
   - Target: readmitted_30d (Binary: 1 for <30 days, 0 for >=30 days or NO).
   - Target Leakage Prevention: Drops original 'readmitted' string column ('NO', '>30', '<30').
   - Identifier Isolation: Separates surrogate_key, document_id, diag_1_lookup_status, feedback_text.
2. Deliberate Column Categorization:
   - Maps ICD-9 primary diagnosis codes (diag_1) into 9 clinical categories (Strack et al. 2014).
   - Consolidates medical specialties into top clinical specialties + 'Other'.
   - Categorizes numeric features (labs, procedures, medications, utilization, notes risk score).
   - Categorizes clinical categorical features (demographics, admission/discharge details, regimens).
3. Class Imbalance Accounting:
   - Calculates target distribution (~8.98% positive minority class).
   - Computes scale_pos_weight (neg / pos ≈ 10.14) and balanced class weights.
   - Stores imbalance metadata for Stage 2 & 4 recall-first optimization.
4. Stratified Train / Validation / Test Split:
   - 70% Train (48,993), 15% Validation (10,498), 15% Test (10,499).
   - Stratified strictly on readmitted_30d to guarantee identical ~8.98% minority representation.
5. Reusable Preprocessor Pipeline (Fit Once, Reuse Everywhere):
   - Sklearn ColumnTransformer + Pipeline:
     * Numeric: Median Imputer + StandardScaler.
     * Categorical: Constant Imputer ('Unknown') + OneHotEncoder(handle_unknown='ignore').
   - Fits EXCLUSIVELY on training data to prevent data leakage.
   - Serializes fitted preprocessor to 'models/preprocessor.joblib' and feature names to 'models/feature_names.json'.
   - Provides 'transform_raw_records()' for real-time FastAPI inference and Streamlit calculator.
6. Unified MLflow Tracking Configuration:
   - get_tracking_uri(): Points to 'sqlite:///mlflow.db' (or $MLFLOW_TRACKING_URI).
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# ---------------------------------------------------------------------------
# Project Directories & MLflow Tracking Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
DATA_STAGING_DIR = os.path.join(BASE_DIR, "data", "staging")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

DEFAULT_INTEGRATED_DATA_PATH = os.path.join(DATA_CLEANED_DIR, "integrated_encounters.csv")
PREPROCESSOR_SAVE_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
FEATURE_NAMES_SAVE_PATH = os.path.join(MODELS_DIR, "feature_names.json")
IMBALANCE_METADATA_PATH = os.path.join(MODELS_DIR, "target_imbalance_metadata.json")

# Standardized MLflow Tracking Configuration:
# Default to SQLite database backend so Model Registry works offline without a separate server process.
# Can be overridden via MLFLOW_TRACKING_URI environment variable (e.g. http://localhost:5000 in Docker).
DEFAULT_SQLITE_DB = os.path.join(BASE_DIR, "mlflow.db")
DEFAULT_MLFLOW_TRACKING_URI = f"sqlite:///{DEFAULT_SQLITE_DB.replace(os.sep, '/')}"
DEFAULT_EXPERIMENT_NAME = "Hospital_Readmission_Prediction"
DEFAULT_MODEL_REGISTRY_NAME = "HospitalReadmissionPredictor"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ML_FEATURES] %(message)s")
logger = logging.getLogger(__name__)


def get_tracking_uri() -> str:
    """
    Returns the standardized MLflow tracking URI.
    Checks MLFLOW_TRACKING_URI environment variable first, defaulting to local sqlite:///mlflow.db.
    This guarantees that Stage 2 (training), Stage 4 (registry & FastAPI), and Stage 6 (Streamlit)
    all communicate with the exact same tracking store and registry.
    """
    uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI)
    return uri


def get_experiment_name() -> str:
    return os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)


def get_model_registry_name() -> str:
    return os.getenv("MLFLOW_MODEL_NAME", DEFAULT_MODEL_REGISTRY_NAME)


# ---------------------------------------------------------------------------
# Deliberate Column Categorization & Domain Mappings
# ---------------------------------------------------------------------------
TARGET_COL = "readmitted_30d"

# Identifiers to separate and track (not fed into model)
IDENTIFIER_COLS = [
    "surrogate_key",
    "document_id",
    "diag_1_lookup_status",
]

# Target leakage columns to DROP entirely
LEAKAGE_COLS = [
    "readmitted",      # Original string: 'NO', '>30', '<30' -> direct label leak!
    "readmitted_30d",  # Separated as target y
]

# Raw text columns dropped from tabular model (signal already captured in numerical scores)
TEXT_COLS = [
    "feedback_text",
    "document_type",
]

# Redundant or zero-variance columns dropped to avoid collinearity and noise
REDUNDANT_COLS = [
    "payer_code",                  # 100% Unknown
    "admission_type_id",           # Captured by admission_type_name
    "discharge_disposition_id",    # Captured by discharge_disposition_name
    "admission_source_id",         # Captured by admission_source_name
    "time_in_hospital",            # Duplicate of length_of_stay
    "age",                         # Duplicate of age_group
    "number_diagnoses",            # Duplicate of diagnosis_count
    "diag_1",                      # Mapped into diag_1_category
    "diag_2",                      # High cardinality ICD-9 codes
    "diag_3",                      # High cardinality ICD-9 codes
    "diag_1_description",          # Mapped via diag_1_category
    "has_insulin",                 # Redundant with insulin categorical feature
]

# Numeric Features (10 features)
NUMERIC_FEATURES = [
    "length_of_stay",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "prior_admissions",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "diagnosis_count",
    "risk_keyword_score",
]

# Categorical Features (14 features)
CATEGORICAL_FEATURES = [
    "age_group",
    "gender",
    "race",
    "admission_type_name",
    "discharge_disposition_name",
    "admission_source_name",
    "medical_specialty",
    "diag_1_category",
    "max_glu_serum",
    "A1Cresult",
    "insulin",
    "change_med",
    "has_diabetes_med",
    "high_risk_flag",
    "feedback_available",
]

TOP_SPECIALTIES = [
    "InternalMedicine",
    "Family/GeneralPractice",
    "Emergency/Trauma",
    "Cardiology",
    "Surgery-General",
    "Orthopedics",
    "Orthopedics-Reconstructive",
    "Radiologist",
    "Nephrology",
    "Psychiatry",
]


def map_icd9_to_category(val: Any) -> str:
    """
    Standard clinical mapping of ICD-9 diagnosis codes into 9 major disease categories
    based on established diabetes readmission methodology (Strack et al., 2014):
      - Circulatory: 390-459, 785
      - Respiratory: 460-519, 786
      - Digestive: 520-579, 787
      - Diabetes: 250.xx
      - Injury: 800-999
      - Musculoskeletal: 710-739
      - Genitourinary: 580-629, 788
      - Neoplasms: 140-239
      - Other: all others, V-codes, E-codes, unmapped
    """
    if pd.isna(val) or str(val).strip() in ("", "?", "Unknown", "None", "nan"):
        return "Other"
    
    val_str = str(val).strip()
    if val_str.startswith("V") or val_str.startswith("E"):
        return "Other"
    
    try:
        code = float(val_str)
    except ValueError:
        return "Other"

    if (390 <= code <= 459) or code == 785:
        return "Circulatory"
    elif (460 <= code <= 519) or code == 786:
        return "Respiratory"
    elif (520 <= code <= 579) or code == 787:
        return "Digestive"
    elif int(code) == 250:
        return "Diabetes"
    elif (800 <= code <= 999):
        return "Injury"
    elif (710 <= code <= 739):
        return "Musculoskeletal"
    elif (580 <= code <= 629) or code == 788:
        return "Genitourinary"
    elif (140 <= code <= 239):
        return "Neoplasms"
    else:
        return "Other"


def clean_medical_specialty(val: Any) -> str:
    """
    Consolidates rare medical specialties into 'Other' to prevent high-cardinality sparse one-hot encoding.
    """
    if pd.isna(val) or str(val).strip() in ("", "?", "Unknown", "None", "nan"):
        return "Unknown"
    val_str = str(val).strip()
    if val_str in TOP_SPECIALTIES:
        return val_str
    return "Other"


# ---------------------------------------------------------------------------
# Data Loading & Feature Preparation
# ---------------------------------------------------------------------------
def load_raw_integrated_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Loads the curated analytical integrated encounters dataset from Part 1.
    """
    path = filepath or DEFAULT_INTEGRATED_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Curated dataset not found at: {path}")
    logger.info("Loading integrated encounters from: %s", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %d encounters with %d raw columns.", len(df), len(df.columns))
    return df


def prepare_feature_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Cleans raw integrated encounters into:
      1. X (Feature DataFrame containing NUMERIC_FEATURES + CATEGORICAL_FEATURES)
      2. y (Target Series: readmitted_30d)
      3. identifiers (DataFrame with surrogate_key and tracking metadata)
    """
    df = df.copy()

    # 1. Validate target presence
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in DataFrame.")

    # 2. Extract target y
    y = df[TARGET_COL].astype(int)

    # 3. Extract identifiers for auditability and join tracking
    id_cols = [c for c in IDENTIFIER_COLS if c in df.columns]
    identifiers = df[id_cols].copy()

    # 4. Feature Engineering: Map ICD-9 primary diagnosis code to 9 clinical categories
    if "diag_1" in df.columns:
        df["diag_1_category"] = df["diag_1"].apply(map_icd9_to_category)
    elif "diag_1_category" not in df.columns:
        df["diag_1_category"] = "Other"

    # 5. Clean Medical Specialty
    if "medical_specialty" in df.columns:
        df["medical_specialty"] = df["medical_specialty"].apply(clean_medical_specialty)
    else:
        df["medical_specialty"] = "Unknown"

    # 6. Clean and impute Lab categorical tests
    for lab_col in ["max_glu_serum", "A1Cresult"]:
        if lab_col in df.columns:
            df[lab_col] = df[lab_col].fillna("None").astype(str)
        else:
            df[lab_col] = "None"

    # 7. Clean Notes risk keyword score (Source 3)
    if "risk_keyword_score" in df.columns:
        df["risk_keyword_score"] = df["risk_keyword_score"].fillna(0.0).astype(float)
    else:
        df["risk_keyword_score"] = 0.0

    # 8. Clean Categorical columns to string
    for cat_col in CATEGORICAL_FEATURES:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].fillna("Unknown").astype(str)
        else:
            df[cat_col] = "Unknown"

    # 9. Clean Numeric columns to float
    for num_col in NUMERIC_FEATURES:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(0.0)
        else:
            df[num_col] = 0.0

    # 10. Assemble X strictly from designated feature lists
    selected_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[selected_features].copy()

    logger.info("Features assembled: %d numeric, %d categorical. Total X shape: %s",
                len(NUMERIC_FEATURES), len(CATEGORICAL_FEATURES), X.shape)
    
    return X, y, identifiers


# ---------------------------------------------------------------------------
# Preprocessor Pipeline (Fit Once, Reuse Everywhere)
# ---------------------------------------------------------------------------
def build_preprocessor_pipeline() -> ColumnTransformer:
    """
    Constructs an enterprise-grade scikit-learn ColumnTransformer:
      - Numeric: Median Imputation -> StandardScaler
      - Categorical: Constant Imputation ('Unknown') -> OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


def extract_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts human-readable feature names from the fitted ColumnTransformer.
    """
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception as e:
        logger.warning("Could not extract feature names via get_feature_names_out(): %s", e)
        feature_names = [f"f_{i}" for i in range(100)]
    return feature_names


# ---------------------------------------------------------------------------
# Stratified Splitting & Class Weight Calculation
# ---------------------------------------------------------------------------
def perform_stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    identifiers: Optional[pd.DataFrame] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Performs a 70% Train / 15% Validation / 15% Test split stratified strictly on target y.
    Preserves exact positive minority class distribution across all three splits.
    """
    total_holdout = test_size + val_size  # 0.30
    
    # Split Train (70%) vs Holdout (30%)
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y,
        test_size=total_holdout,
        stratify=y,
        random_state=random_state,
    )

    # Split Holdout into Validation (15%) and Test (15%)
    val_fraction = val_size / total_holdout  # 0.50 of holdout
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout,
        test_size=(1.0 - val_fraction),
        stratify=y_holdout,
        random_state=random_state,
    )

    id_train, id_val, id_test = None, None, None
    if identifiers is not None:
        id_train = identifiers.loc[X_train.index]
        id_val = identifiers.loc[X_val.index]
        id_test = identifiers.loc[X_test.index]

    split_dict = {
        "X_train": X_train,
        "y_train": y_train,
        "id_train": id_train,
        "X_val": X_val,
        "y_val": y_val,
        "id_val": id_val,
        "X_test": X_test,
        "y_test": y_test,
        "id_test": id_test,
    }

    # Verify class balance across splits
    logger.info("Stratified Split Completed:")
    for split_name, s_y in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
        pos_cnt = int(s_y.sum())
        neg_cnt = int((s_y == 0).sum())
        pos_rate = float(s_y.mean())
        logger.info("  %10s Split: %5d rows | Pos: %4d (%.2f%%) | Neg: %5d",
                    split_name, len(s_y), pos_cnt, pos_rate * 100, neg_cnt)

    return split_dict


def compute_target_imbalance_metadata(y_train: pd.Series) -> Dict[str, Any]:
    """
    Computes class weights and scale_pos_weight for imbalance-aware model training.
    Essential for recall-first optimization to prevent false-negative clinical harm.
    """
    pos_count = int(y_train.sum())
    neg_count = int((y_train == 0).sum())
    total_count = len(y_train)
    pos_rate = float(pos_count / total_count)
    
    # scale_pos_weight = negative_samples / positive_samples (used by XGBoost)
    scale_pos_weight = float(neg_count / max(1, pos_count))
    
    # Balanced class weights for Logistic Regression & Random Forest:
    # weight_i = total_samples / (n_classes * class_count_i)
    weight_0 = float(total_count / (2.0 * neg_count))
    weight_1 = float(total_count / (2.0 * pos_count))

    metadata = {
        "total_train_samples": total_count,
        "positive_readmissions": pos_count,
        "negative_readmissions": neg_count,
        "positive_rate_percent": round(pos_rate * 100, 2),
        "imbalance_ratio": f"{round(scale_pos_weight, 2)} : 1",
        "scale_pos_weight_xgboost": round(scale_pos_weight, 4),
        "class_weight_balanced": {
            "0": round(weight_0, 4),
            "1": round(weight_1, 4),
        },
        "target_col": TARGET_COL,
        "primary_metric_priority": "Recall (High Sensitivity for High-Risk Patients)",
    }
    return metadata


# ---------------------------------------------------------------------------
# Reusable Inference Transformation Function (for FastAPI & Streamlit)
# ---------------------------------------------------------------------------
def transform_raw_records(
    records: List[Dict[str, Any]],
    preprocessor: Optional[ColumnTransformer] = None,
) -> np.ndarray:
    """
    Takes a list of raw encounter dictionaries (as sent by FastAPI or Streamlit UI),
    applies the ICD-9 mapping, specialty capping, default value imputation,
    and executes preprocessor.transform() returning a dense matrix ready for model.predict().
    """
    if preprocessor is None:
        if not os.path.exists(PREPROCESSOR_SAVE_PATH):
            raise FileNotFoundError(f"Fitted preprocessor not found at {PREPROCESSOR_SAVE_PATH}. Run ml_features.py first.")
        preprocessor = joblib.load(PREPROCESSOR_SAVE_PATH)

    df_raw = pd.DataFrame(records)

    # 1. Feature Engineering: Map ICD-9 primary diagnosis code if provided
    if "diag_1" in df_raw.columns:
        df_raw["diag_1_category"] = df_raw["diag_1"].apply(map_icd9_to_category)
    elif "diag_1_category" not in df_raw.columns:
        df_raw["diag_1_category"] = "Other"

    # 2. Specialty capping
    if "medical_specialty" in df_raw.columns:
        df_raw["medical_specialty"] = df_raw["medical_specialty"].apply(clean_medical_specialty)
    else:
        df_raw["medical_specialty"] = "Unknown"

    # 3. Lab categoricals
    for lab_col in ["max_glu_serum", "A1Cresult"]:
        if lab_col in df_raw.columns:
            df_raw[lab_col] = df_raw[lab_col].fillna("None").astype(str)
        else:
            df_raw[lab_col] = "None"

    # 4. Fill defaults for missing categorical features
    for cat_col in CATEGORICAL_FEATURES:
        if cat_col not in df_raw.columns:
            df_raw[cat_col] = "Unknown"
        else:
            df_raw[cat_col] = df_raw[cat_col].fillna("Unknown").astype(str)

    # 5. Fill defaults for missing numeric features
    for num_col in NUMERIC_FEATURES:
        if num_col not in df_raw.columns:
            df_raw[num_col] = 0.0
        else:
            df_raw[num_col] = pd.to_numeric(df_raw[num_col], errors="coerce").fillna(0.0)

    # Select features in exact order
    selected_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_input = df_raw[selected_features].copy()

    # Transform through fitted preprocessor
    X_transformed = preprocessor.transform(X_input)
    return X_transformed


# ---------------------------------------------------------------------------
# Pipeline Execution & Serialization
# ---------------------------------------------------------------------------
def run_feature_pipeline(
    data_path: Optional[str] = None,
    save_splits: bool = True,
) -> Dict[str, Any]:
    """
    Orchestrates Phase 1 end-to-end:
      1. Loads integrated encounters.
      2. Separates target and identifiers, drops leaks.
      3. Performs 70/15/15 stratified train/val/test split.
      4. Fits ColumnTransformer ONLY on X_train.
      5. Transforms X_train, X_val, X_test.
      6. Serializes preprocessor, feature names, and imbalance metadata.
      7. Saves staged datasets for Stage 2 training.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_STAGING_DIR, exist_ok=True)
    os.makedirs(MLRUNS_DIR, exist_ok=True)

    # 1. Load data
    df_raw = load_raw_integrated_data(data_path)

    # 2. Prepare features, target, and identifiers
    X, y, identifiers = prepare_feature_dataframe(df_raw)

    # 3. Stratified Split
    splits = perform_stratified_split(X, y, identifiers=identifiers)

    # 4. Compute and save target imbalance metadata
    imbalance_meta = compute_target_imbalance_metadata(splits["y_train"])
    with open(IMBALANCE_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(imbalance_meta, f, indent=2)
    logger.info("Saved Target Imbalance Metadata to: %s", IMBALANCE_METADATA_PATH)
    logger.info("Target Imbalance: %s (Positive Rate: %.2f%%, XGBoost scale_pos_weight: %.2f)",
                imbalance_meta["imbalance_ratio"],
                imbalance_meta["positive_rate_percent"],
                imbalance_meta["scale_pos_weight_xgboost"])

    # 5. Fit Preprocessor ONLY on training set (Leakage-free)
    logger.info("Fitting Preprocessor Pipeline exclusively on X_train...")
    preprocessor = build_preprocessor_pipeline()
    preprocessor.fit(splits["X_train"])

    # 6. Save fitted preprocessor
    joblib.dump(preprocessor, PREPROCESSOR_SAVE_PATH)
    logger.info("Saved fitted preprocessor to: %s", PREPROCESSOR_SAVE_PATH)

    # 7. Extract and save transformed feature names
    feature_names = extract_feature_names(preprocessor)
    with open(FEATURE_NAMES_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    logger.info("Saved %d transformed feature names to: %s", len(feature_names), FEATURE_NAMES_SAVE_PATH)

    # 8. Transform splits
    X_train_trans = preprocessor.transform(splits["X_train"])
    X_val_trans = preprocessor.transform(splits["X_val"])
    X_test_trans = preprocessor.transform(splits["X_test"])

    logger.info("Transformed Matrix Shapes: Train=%s, Val=%s, Test=%s",
                X_train_trans.shape, X_val_trans.shape, X_test_trans.shape)

    # 9. Optionally persist staged splits for reproducible training and auditing
    if save_splits:
        splits["X_train"].to_parquet(os.path.join(DATA_STAGING_DIR, "X_train_raw.parquet"), index=False)
        splits["X_val"].to_parquet(os.path.join(DATA_STAGING_DIR, "X_val_raw.parquet"), index=False)
        splits["X_test"].to_parquet(os.path.join(DATA_STAGING_DIR, "X_test_raw.parquet"), index=False)
        
        pd.DataFrame(splits["y_train"]).to_parquet(os.path.join(DATA_STAGING_DIR, "y_train.parquet"), index=False)
        pd.DataFrame(splits["y_val"]).to_parquet(os.path.join(DATA_STAGING_DIR, "y_val.parquet"), index=False)
        pd.DataFrame(splits["y_test"]).to_parquet(os.path.join(DATA_STAGING_DIR, "y_test.parquet"), index=False)

        if splits["id_train"] is not None:
            splits["id_train"].to_parquet(os.path.join(DATA_STAGING_DIR, "id_train.parquet"), index=False)
            splits["id_val"].to_parquet(os.path.join(DATA_STAGING_DIR, "id_val.parquet"), index=False)
            splits["id_test"].to_parquet(os.path.join(DATA_STAGING_DIR, "id_test.parquet"), index=False)

        logger.info("Saved raw feature splits and targets to data/staging/*.parquet")

    return {
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "imbalance_metadata": imbalance_meta,
        "splits": splits,
        "X_train_transformed": X_train_trans,
        "X_val_transformed": X_val_trans,
        "X_test_transformed": X_test_trans,
    }


if __name__ == "__main__":
    logger.info("Executing Phase 1 Feature Pipeline...")
    results = run_feature_pipeline()
    logger.info("Phase 1 Feature Engineering successfully completed.")
