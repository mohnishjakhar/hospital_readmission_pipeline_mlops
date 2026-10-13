"""
test_ml_train.py
----------------
Unit tests for Phase 2 Model Training, Clinical Metrics Computation,
Diagnostic Curves Generation, and MLflow Tracking (src/ml_train.py).
"""

import os
import unittest
import numpy as np
import pandas as pd

from src.ml_train import (
    compute_clinical_metrics,
    plot_diagnostic_curves,
    build_candidate_models,
    plot_model_comparison_bar,
    CHAMPION_MODEL_PATH,
    CHAMPION_METADATA_PATH,
    EVALUATION_PLOTS_DIR,
)


class TestMLTrainPhase2(unittest.TestCase):
    def setUp(self):
        # Synthetic binary classification ground truth and predictions
        np.random.seed(42)
        self.y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # 20% positive rate
        self.y_prob = np.array([0.1, 0.2, 0.15, 0.3, 0.4, 0.6, 0.05, 0.25, 0.85, 0.9])
        self.y_pred = (self.y_prob >= 0.5).astype(int)

    def test_compute_clinical_metrics(self):
        """Verifies calculation of clinical recall, precision, specificity, AUC, and Brier score."""
        metrics = compute_clinical_metrics(self.y_true, self.y_pred, self.y_prob, prefix="val")

        self.assertIn("val_recall", metrics)
        self.assertIn("val_precision", metrics)
        self.assertIn("val_specificity", metrics)
        self.assertIn("val_f1", metrics)
        self.assertIn("val_roc_auc", metrics)
        self.assertIn("val_pr_auc", metrics)
        self.assertIn("val_brier_score", metrics)

        # TP=2, FN=0, Recall=1.0
        self.assertEqual(metrics["val_recall"], 1.0)
        # FP=1, TN=7, Specificity=7/8=0.875
        self.assertAlmostEqual(metrics["val_specificity"], 0.875)
        # Brier score should be low for good predictions
        self.assertLess(metrics["val_brier_score"], 0.20)

    def test_build_candidate_models(self):
        """Validates that candidate model dictionary contains Logistic Regression, Random Forest, and XGBoost."""
        models = build_candidate_models(scale_pos_weight=10.0)

        self.assertIn("Logistic_Regression", models)
        self.assertIn("Random_Forest", models)
        self.assertIn("XGBoost", models)

        # Verify scale_pos_weight is passed to XGBoost
        xgb_params = models["XGBoost"]["params"]
        self.assertEqual(xgb_params["scale_pos_weight"], 10.0)

    def test_diagnostic_curves_generation(self):
        """Verifies that diagnostic ROC, PR, Confusion Matrix, and Calibration plots are generated."""
        test_plot_dir = os.path.join(EVALUATION_PLOTS_DIR, "unit_test_plots")
        plots = plot_diagnostic_curves(
            self.y_true,
            self.y_prob,
            model_name="Unit_Test_Model",
            output_dir=test_plot_dir,
        )

        self.assertIn("roc_curve", plots)
        self.assertIn("pr_curve", plots)
        self.assertIn("confusion_matrix", plots)
        self.assertIn("calibration_curve", plots)

        for p_key, p_path in plots.items():
            self.assertTrue(os.path.exists(p_path), f"Plot file {p_path} was not created!")
            self.assertGreater(os.path.getsize(p_path), 1000)

    def test_plot_model_comparison_bar(self):
        """Verifies comparative bar plot generation."""
        comp_df = pd.DataFrame([
            {"model_name": "Logistic Regression", "val_recall": 0.65, "val_roc_auc": 0.64, "val_pr_auc": 0.16, "val_f1": 0.22},
            {"model_name": "Random Forest", "val_recall": 0.58, "val_roc_auc": 0.65, "val_pr_auc": 0.17, "val_f1": 0.23},
            {"model_name": "XGBoost", "val_recall": 0.62, "val_roc_auc": 0.67, "val_pr_auc": 0.19, "val_f1": 0.25},
        ])
        test_out = os.path.join(EVALUATION_PLOTS_DIR, "unit_test_comparison.png")
        res_path = plot_model_comparison_bar(comp_df, output_path=test_out)
        self.assertTrue(os.path.exists(res_path))
        self.assertGreater(os.path.getsize(res_path), 1000)


if __name__ == "__main__":
    unittest.main()
