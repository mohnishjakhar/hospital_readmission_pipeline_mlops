"""
test_ml_monitor.py
------------------
Unit tests for Phase 7 Continuous Monitoring, Drift Detection,
and Subgroup Fairness Auditing (src/ml_monitor.py).
"""

import unittest
import numpy as np
import pandas as pd

from src.ml_monitor import (
    calculate_psi,
    evaluate_feature_drift,
    audit_subgroup_fairness,
    evaluate_retraining_policy,
)


class TestMLMonitorPhase7(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Identical distribution -> PSI should be ~0.0
        self.base_num = pd.Series(np.random.normal(5, 1, 1000))
        self.curr_num_stable = pd.Series(np.random.normal(5, 1, 1000))
        # Shifted distribution -> PSI should be elevated
        self.curr_num_shifted = pd.Series(np.random.normal(8, 2, 1000))

    def test_psi_calculation(self):
        """Verifies PSI calculation returns near 0 for stable distribution and >0.25 for shifted distribution."""
        psi_stable = calculate_psi(self.base_num, self.curr_num_stable)
        psi_shifted = calculate_psi(self.base_num, self.curr_num_shifted)

        self.assertLess(psi_stable, 0.10)
        self.assertGreater(psi_shifted, 0.25)

    def test_evaluate_retraining_policy(self):
        """Verifies policy flags retrain when triggers fire and stays quiet when stable."""
        # Case A: No triggers
        res_stable = evaluate_retraining_policy(
            drift_summary={"drifted_features_count": 0},
            test_recall=0.525,
            baseline_recall=0.527,
            new_records_count=2000,
        )
        self.assertFalse(res_stable["retrain_recommended"])

        # Case B: Significant recall drop (>5%)
        res_perf_drop = evaluate_retraining_policy(
            drift_summary={"drifted_features_count": 0},
            test_recall=0.45,  # dropped from 0.527 -> >7% drop!
            baseline_recall=0.527,
            new_records_count=2000,
        )
        self.assertTrue(res_perf_drop["retrain_recommended"])
        self.assertTrue(res_perf_drop["triggers_fired"]["performance_trigger"])


if __name__ == "__main__":
    unittest.main()
