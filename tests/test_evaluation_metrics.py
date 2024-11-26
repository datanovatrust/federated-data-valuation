# tests/test_evaluation_metrics.py

import unittest
import sys
import os
import numpy as np

# Adjust import according to the project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.attacks.evaluation_metrics import compute_tpr_fpr, compute_auc, plot_roc_curve

class TestEvaluationMetrics(unittest.TestCase):
    def test_compute_tpr_fpr(self):
        """Test computing TPR and FPR."""
        true_labels = [0, 0, 1, 1]
        pred_scores = [0.1, 0.4, 0.35, 0.8]
        fpr, tpr = compute_tpr_fpr(true_labels, pred_scores)
        # Expected results can be calculated using known behavior of roc_curve
        expected_fpr = np.array([0.0, 0.0, 0.5, 0.5, 1.0])
        expected_tpr = np.array([0.0, 0.5, 0.5, 1.0, 1.0])
        np.testing.assert_array_almost_equal(fpr, expected_fpr)
        np.testing.assert_array_almost_equal(tpr, expected_tpr)
    
    def test_compute_auc(self):
        """Test computing AUC."""
        fpr = np.array([0.0, 0.5, 1.0])
        tpr = np.array([0.0, 0.75, 1.0])
        auc_score = compute_auc(fpr, tpr)
        expected_auc = 0.625  # Corrected expected AUC
        self.assertAlmostEqual(auc_score, expected_auc, places=6)
    
    def test_plot_roc_curve(self):
        """Test plotting ROC curve."""
        fpr = [0.0, 0.5, 1.0]
        tpr = [0.0, 0.75, 1.0]
        # We won't actually display or save the plot in tests
        try:
            plot_roc_curve(fpr, tpr, title='Test ROC Curve')
        except Exception as e:
            self.fail(f"plot_roc_curve raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
