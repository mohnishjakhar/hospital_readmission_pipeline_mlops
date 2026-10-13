"""
test_api.py
-----------
Unit tests for Phase 4 FastAPI REST Inference Microservice (src/ml_serve.py).
"""

import unittest
from fastapi.testclient import TestClient

from src.ml_serve import app, startup_event


class TestFastAPIMicroservice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        startup_event()
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        """Verifies GET /health returns 200 OK with model version and healthy status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("champion_model", data)
        self.assertIn("registry_name", data)
        self.assertIn("test_recall", data)

    def test_predict_single_encounter(self):
        """Verifies POST /predict computes calibrated probability, risk tier, and action items."""
        payload = {
            "age_group": "[70-80)",
            "gender": "Female",
            "race": "Caucasian",
            "admission_type_name": "Emergency",
            "discharge_disposition_name": "Discharged to home",
            "admission_source_name": "Emergency Room",
            "length_of_stay": 5,
            "num_lab_procedures": 50,
            "num_procedures": 1,
            "num_medications": 16,
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
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["status"], "success")
        self.assertIn("readmission_probability", data)
        self.assertIn("risk_tier", data)
        self.assertIn("clinical_action_items", data)
        self.assertGreaterEqual(data["readmission_probability"], 0.0)
        self.assertLessEqual(data["readmission_probability"], 1.0)
        self.assertIn(data["risk_tier"], ["Low Risk", "Moderate Risk", "High Risk"])

    def test_predict_batch_encounters(self):
        """Verifies POST /predict-batch correctly scores multiple encounters."""
        encounters = [
            {
                "age_group": "[30-40)",
                "length_of_stay": 2,
                "num_medications": 6,
                "prior_admissions": 0,
            },
            {
                "age_group": "[70-80)",
                "length_of_stay": 9,
                "num_medications": 25,
                "prior_admissions": 5,
            },
        ]
        response = self.client.post("/predict-batch", json={"encounters": encounters})
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["status"], "success")
        self.assertEqual(data["total_encounters_scored"], 2)
        self.assertIn("cohort_average_probability", data)
        self.assertEqual(len(data["predictions"]), 2)

    def test_model_metadata_endpoint(self):
        """Verifies GET /model-metadata returns training metrics and top SHAP features."""
        response = self.client.get("/model-metadata")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("champion_model", data)
        self.assertIn("validation_metrics", data)
        self.assertIn("top_shap_features", data)

    def test_case_studies_endpoint(self):
        """Verifies GET /case-studies returns 3 clinical reference archetypes."""
        response = self.client.get("/case-studies")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 3)


if __name__ == "__main__":
    unittest.main()
