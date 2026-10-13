"""
test_ml_features.py
-------------------
Unit tests for Phase 1 Feature Engineering, Stratified Splitting,
and Preprocessing Pipeline (src/ml_features.py).
"""

import os
import json
import unittest
import numpy as np
import pandas as pd
import joblib

from src.ml_features import (
    load_raw_integrated_data,
    prepare_feature_dataframe,
    build_preprocessor_pipeline,
    perform_stratified_split,
    compute_target_imbalance_metadata,
    transform_raw_records,
    map_icd9_to_category,
    get_tracking_uri,
    TARGET_COL,
    LEAKAGE_COLS,
    IDENTIFIER_COLS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    PREPROCESSOR_SAVE_PATH,
    FEATURE_NAMES_SAVE_PATH,
    IMBALANCE_METADATA_PATH,
)


class TestMLFeaturesPhase1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_raw = load_raw_integrated_data()

    def test_target_separation_and_leak_prevention(self):
        """Verifies target is separated as y and no label leaks remain in feature matrix X."""
        X, y, identifiers = prepare_feature_dataframe(self.df_raw)
        
        # Verify target is binary 0 and 1
        self.assertEqual(set(y.unique()), {0, 1})
        self.assertEqual(len(y), len(self.df_raw))

        # Verify no leakage columns exist in feature frame X
        for leak in LEAKAGE_COLS:
            self.assertNotIn(leak, X.columns, f"Leakage column '{leak}' found in X!")

        # Verify no identifiers exist in X
        for ident in IDENTIFIER_COLS:
            self.assertNotIn(ident, X.columns, f"Identifier column '{ident}' found in X!")

        # Verify feature set composition
        self.assertEqual(len(X.columns), len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES))

    def test_icd9_category_mapping(self):
        """Validates ICD-9 mapping into 9 major clinical clusters."""
        self.assertEqual(map_icd9_to_category(428), "Circulatory")
        self.assertEqual(map_icd9_to_category("410.1"), "Circulatory")
        self.assertEqual(map_icd9_to_category(486), "Respiratory")
        self.assertEqual(map_icd9_to_category("250.7"), "Diabetes")
        self.assertEqual(map_icd9_to_category(530), "Digestive")
        self.assertEqual(map_icd9_to_category(820), "Injury")
        self.assertEqual(map_icd9_to_category(715), "Musculoskeletal")
        self.assertEqual(map_icd9_to_category(599), "Genitourinary")
        self.assertEqual(map_icd9_to_category(162), "Neoplasms")
        self.assertEqual(map_icd9_to_category("V45"), "Other")
        self.assertEqual(map_icd9_to_category("Unknown"), "Other")

    def test_stratified_split_preserves_imbalance(self):
        """Verifies 70/15/15 stratified split preserves ~8.98% minority positive class."""
        X, y, identifiers = prepare_feature_dataframe(self.df_raw)
        splits = perform_stratified_split(X, y, identifiers=identifiers)

        total_rows = len(self.df_raw)
        self.assertEqual(len(splits["X_train"]), 48993)
        self.assertEqual(len(splits["X_val"]), 10498)
        self.assertEqual(len(splits["X_test"]), 10499)
        self.assertEqual(len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"]), total_rows)

        # Verify stratification consistency
        train_pos_rate = splits["y_train"].mean()
        val_pos_rate = splits["y_val"].mean()
        test_pos_rate = splits["y_test"].mean()
        global_pos_rate = y.mean()

        self.assertAlmostEqual(train_pos_rate, global_pos_rate, delta=0.005)
        self.assertAlmostEqual(val_pos_rate, global_pos_rate, delta=0.005)
        self.assertAlmostEqual(test_pos_rate, global_pos_rate, delta=0.005)

    def test_imbalance_metadata_calculation(self):
        """Verifies class weight and scale_pos_weight calculations for recall optimization."""
        X, y, _ = prepare_feature_dataframe(self.df_raw)
        splits = perform_stratified_split(X, y)
        meta = compute_target_imbalance_metadata(splits["y_train"])

        self.assertIn("scale_pos_weight_xgboost", meta)
        self.assertIn("class_weight_balanced", meta)
        self.assertGreater(meta["scale_pos_weight_xgboost"], 9.0)
        self.assertLess(meta["scale_pos_weight_xgboost"], 11.5)
        self.assertAlmostEqual(meta["positive_rate_percent"], 8.98, delta=0.1)

    def test_preprocessor_serialization_and_inference(self):
        """Verifies fitted preprocessor exists and transforms single records without error."""
        self.assertTrue(os.path.exists(PREPROCESSOR_SAVE_PATH), "Preprocessor file missing!")
        self.assertTrue(os.path.exists(FEATURE_NAMES_SAVE_PATH), "Feature names file missing!")

        preprocessor = joblib.load(PREPROCESSOR_SAVE_PATH)
        with open(FEATURE_NAMES_SAVE_PATH, "r") as f:
            feature_names = json.load(f)

        self.assertEqual(len(feature_names), 115)

        # Test single patient inference dictionary
        sample_dict = [{
            "age_group": "[70-80)",
            "gender": "Female",
            "race": "Caucasian",
            "admission_type_name": "Emergency",
            "discharge_disposition_name": "Discharged to home",
            "admission_source_name": "Emergency Room",
            "medical_specialty": "InternalMedicine",
            "diag_1": "428",
            "length_of_stay": 4,
            "num_lab_procedures": 45,
            "num_procedures": 1,
            "num_medications": 14,
            "prior_admissions": 0,
            "number_outpatient": 0,
            "number_emergency": 0,
            "number_inpatient": 0,
            "diagnosis_count": 8,
            "risk_keyword_score": 0.0,
            "max_glu_serum": "None",
            "A1Cresult": "None",
            "insulin": "No",
            "change_med": 0,
            "has_diabetes_med": 1,
            "high_risk_flag": 0,
            "feedback_available": 0,
        }]

        X_transformed = transform_raw_records(sample_dict, preprocessor=preprocessor)
        self.assertEqual(X_transformed.shape, (1, 115))
        self.assertFalse(np.isnan(X_transformed).any(), "Transformed record contains NaN!")

    def test_tracking_uri_configuration(self):
        """Verifies MLflow tracking URI points to standardized SQLite backend or environment override."""
        uri = get_tracking_uri()
        self.assertTrue(uri.startswith("sqlite:///") or uri.startswith("http://"))


if __name__ == "__main__":
    unittest.main()
