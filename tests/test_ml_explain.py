"""
test_ml_explain.py
------------------
Unit tests for Phase 3 Clinical Explainability (XAI) with SHAP (src/ml_explain.py).
"""

import os
import unittest
import json

from src.ml_explain import (
    get_champion_and_explainer,
    explain_single_encounter,
    generate_reference_case_studies,
    CHAMPION_MODEL_PATH,
)


class TestMLExplainPhase3(unittest.TestCase):
    def setUp(self):
        self.sample_patient = {
            "age_group": "[70-80)",
            "gender": "Female",
            "race": "Caucasian",
            "admission_type_name": "Emergency",
            "discharge_disposition_name": "Discharged to home",
            "admission_source_name": "Emergency Room",
            "length_of_stay": 5,
            "num_lab_procedures": 45,
            "num_procedures": 1,
            "num_medications": 18,
            "prior_admissions": 2,
            "number_outpatient": 0,
            "number_emergency": 1,
            "number_inpatient": 1,
            "diagnosis_count": 8,
            "risk_keyword_score": 0.5,
            "medical_specialty": "InternalMedicine",
            "diag_1": "428",
            "max_glu_serum": "None",
            "A1Cresult": "None",
            "insulin": "Steady",
            "change_med": "No",
            "has_diabetes_med": "Yes",
            "high_risk_flag": 1,
            "feedback_available": 1,
        }

    def test_champion_explainer_initialization(self):
        """Verifies champion model and TreeExplainer can be initialized."""
        model, explainer, feature_names = get_champion_and_explainer()
        self.assertIsNotNone(model)
        self.assertIsNotNone(explainer)
        self.assertGreater(len(feature_names), 50)

    def test_explain_single_encounter(self):
        """Verifies local encounter explanation produces calibrated probability, risk tier, and clinical drivers."""
        res = explain_single_encounter(self.sample_patient)

        self.assertIn("readmission_probability", res)
        self.assertIn("risk_tier", res)
        self.assertIn("top_risk_drivers", res)
        self.assertIn("clinical_action_items", res)

        self.assertGreaterEqual(res["readmission_probability"], 0.0)
        self.assertLessEqual(res["readmission_probability"], 1.0)
        self.assertIn(res["risk_tier"], ["Low Risk", "Moderate Risk", "High Risk"])
        self.assertGreater(len(res["clinical_action_items"]), 0)

    def test_reference_case_studies_generation(self):
        """Verifies 3 reference clinical case studies are created."""
        cases = generate_reference_case_studies()
        self.assertEqual(len(cases), 3)
        case_ids = [c["case_id"] for c in cases]
        self.assertIn("CASE-01", case_ids)
        self.assertIn("CASE-02", case_ids)
        self.assertIn("CASE-03", case_ids)


if __name__ == "__main__":
    unittest.main()
