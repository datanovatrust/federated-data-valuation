# tests/test_statistical_tests.py

import unittest
import torch
import sys
import os

# Adjust import according to the project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.attacks.statistical_tests import (
    compute_likelihood_ratio,
    compute_average_probability,
    compute_score_mia,
    hypothesis_test,
)

class MockModel:
    def __init__(self, probabilities):
        # probabilities: a tensor of shape [num_classes]
        self.probabilities = probabilities.unsqueeze(0)  # Shape [1, num_classes]

    def eval(self):
        pass  # Mock method for compatibility

    def predict_proba(self, x):
        # x: input tensor, shape [batch_size, ...]
        batch_size = x.shape[0]
        # Return the same probabilities for each sample
        return self.probabilities.repeat(batch_size, 1)

class TestStatisticalTests(unittest.TestCase):
    def setUp(self):
        # Create mock data and models for testing
        self.num_classes = 3
        self.x = torch.randn(1, 3, 32, 32)  # Sample input x
        self.z = torch.randn(1, 3, 32, 32)  # Sample input z
        self.y_x = 1  # True label for x
        self.y_z = 2  # True label for z

        # Create a theta model with predefined probabilities
        theta_probabilities = torch.tensor([0.1, 0.7, 0.2])  # Sum to 1
        self.theta = MockModel(theta_probabilities)

        # Create reference models with different probabilities
        ref_probabilities1 = torch.tensor([0.3, 0.4, 0.3])
        ref_probabilities2 = torch.tensor([0.25, 0.5, 0.25])
        self.reference_models = [
            MockModel(ref_probabilities1),
            MockModel(ref_probabilities2),
        ]

    def test_compute_average_probability(self):
        avg_prob = compute_average_probability(self.x, self.y_x, self.reference_models)
        expected_prob = (0.4 + 0.5) / 2  # For label y_x = 1
        self.assertAlmostEqual(avg_prob.item(), expected_prob, places=6)

    def test_compute_likelihood_ratio(self):
        lr = compute_likelihood_ratio(
            self.theta, self.x, self.y_x, self.z, self.y_z, self.reference_models
        )
        # Manually compute expected values
        pr_x_given_theta = self.theta.predict_proba(self.x)[0, self.y_x]
        pr_x = compute_average_probability(self.x, self.y_x, self.reference_models)
        pr_z_given_theta = self.theta.predict_proba(self.z)[0, self.y_z]
        pr_z = compute_average_probability(self.z, self.y_z, self.reference_models)

        expected_lr = (pr_x_given_theta / pr_x) / (pr_z_given_theta / pr_z)
        self.assertAlmostEqual(lr.item(), expected_lr.item(), places=6)

    def test_compute_score_mia(self):
        likelihood_ratios = [1.5, 0.8, 2.0, 1.2]
        gamma = 1.0
        score = compute_score_mia(likelihood_ratios, gamma)
        expected_score = 3 / 4  # Three out of four ratios are >= gamma
        self.assertAlmostEqual(score, expected_score, places=6)

    def test_hypothesis_test(self):
        score_mia = 0.75
        beta = 0.5
        is_member = hypothesis_test(score_mia, beta)
        self.assertTrue(is_member)

        score_mia = 0.3
        is_member = hypothesis_test(score_mia, beta)
        self.assertFalse(is_member)

    def test_compute_likelihood_ratio_with_batch_inputs(self):
        # Test with batch inputs
        x_batch = torch.randn(5, 3, 32, 32)
        z_batch = torch.randn(5, 3, 32, 32)
        y_x_batch = [0, 1, 2, 1, 0]
        y_z_batch = [2, 0, 1, 2, 1]

        # Adjust the MockModel to handle batch predictions
        # For simplicity, we'll assume predict_proba returns the same probabilities for all samples

        # Compute likelihood ratios for each sample in the batch
        likelihood_ratios = []
        for i in range(len(x_batch)):
            lr = compute_likelihood_ratio(
                self.theta,
                x_batch[i],
                y_x_batch[i],
                z_batch[i],
                y_z_batch[i],
                self.reference_models,
            )
            likelihood_ratios.append(lr.item())

        # Check that we have the correct number of likelihood ratios
        self.assertEqual(len(likelihood_ratios), len(x_batch))

    def test_compute_average_probability_with_empty_models(self):
        # Test compute_average_probability with empty reference models
        with self.assertRaises(ZeroDivisionError):
            compute_average_probability(self.x, self.y_x, [])

if __name__ == '__main__':
    unittest.main()
