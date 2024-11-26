# tests/test_rmia_attack.py

import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Adjust import according to the project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.attacks.rmia_attack import RMIAttack
from src.attacks.statistical_tests import (
    compute_likelihood_ratio,
    compute_score_mia,
    hypothesis_test
)
from src.attacks.config import RMIAConfig

# Mock classes and functions

class MockModel(torch.nn.Module):
    def __init__(self, probabilities):
        super(MockModel, self).__init__()
        self.probabilities = probabilities.unsqueeze(0)  # Shape [1, num_classes]

    def to(self, device):
        return self  # Mock the to() method

    def eval(self):
        pass

    def predict_proba(self, x):
        batch_size = x.shape[0]
        return self.probabilities.repeat(batch_size, 1)

class MockDataSampler:
    def __init__(self, population_data, population_labels):
        self.population_data = population_data
        self.population_labels = population_labels

    def sample_population_data(self, num_samples, exclude_indices=None):
        exclude_indices = exclude_indices or []
        # Check if all exclude_indices are valid
        for idx in exclude_indices:
            if idx < 0 or idx >= len(self.population_data):
                raise IndexError(f"Index {idx} is out of bounds for population data of size {len(self.population_data)}")

        available_indices = [i for i in range(len(self.population_data)) if i not in exclude_indices]
        sampled_indices = available_indices[:num_samples]
        samples = self.population_data[sampled_indices]
        labels = self.population_labels[sampled_indices]
        return samples, labels

class TestRMIAttack(unittest.TestCase):
    def setUp(self):
        # Create mock data
        self.num_classes = 3
        self.num_samples = 10
        self.feature_size = (3, 32, 32)

        data = torch.randn(self.num_samples, *self.feature_size)
        labels = torch.randint(0, self.num_classes, (self.num_samples,))

        # Split data into members and non-members
        self.member_data = data[:5]
        self.member_labels = labels[:5]
        self.non_member_data = data[5:]
        self.non_member_labels = labels[5:]

        # Create target model
        target_probabilities = torch.tensor([0.1, 0.7, 0.2])  # Sum to 1
        self.target_model = MockModel(target_probabilities)

        # Create reference models
        ref_probabilities1 = torch.tensor([0.3, 0.4, 0.3])
        ref_probabilities2 = torch.tensor([0.25, 0.5, 0.25])
        self.reference_models = [
            MockModel(ref_probabilities1),
            MockModel(ref_probabilities2),
        ]

        # Create DataSampler
        population_data = torch.randn(20, *self.feature_size)
        population_labels = torch.randint(0, self.num_classes, (20,))
        self.data_sampler = MockDataSampler(population_data, population_labels)

        # Create RMIAConfig
        self.config = RMIAConfig()
        self.config.NUM_Z_SAMPLES = 5  # For testing, use a small number

        # Create RMIAttack instance
        self.attack = RMIAttack(
            target_model=self.target_model,
            reference_models=self.reference_models,
            config=self.config,
            data_sampler=self.data_sampler
        )

    def test_initialization(self):
        """Test that RMIAttack initializes correctly."""
        self.assertIsNotNone(self.attack.target_model)
        self.assertEqual(len(self.attack.reference_models), 2)
        self.assertEqual(self.attack.config, self.config)
        self.assertEqual(self.attack.data_sampler, self.data_sampler)

    def test_compute_membership_score(self):
        """Test compute_membership_score method."""
        x = self.member_data[0]
        y = self.member_labels[0].item()
        index = 0
        score_mia = self.attack.compute_membership_score(x, y, index)
        self.assertIsInstance(score_mia, float)
        self.assertGreaterEqual(score_mia, 0.0)
        self.assertLessEqual(score_mia, 1.0)

    def test_perform_attack(self):
        """Test perform_attack method."""
        # Combine member and non-member data
        all_data = torch.cat([self.member_data, self.non_member_data], dim=0)
        all_labels = torch.cat([self.member_labels, self.non_member_labels], dim=0)
        true_membership_labels = [1]*len(self.member_data) + [0]*len(self.non_member_data)

        dataset = TensorDataset(all_data, all_labels)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        results = self.attack.perform_attack(data_loader, true_membership_labels)
        self.assertEqual(len(results), len(all_data))
        for true_label, score_mia in results:
            self.assertIn(true_label, [0, 1])
            self.assertIsInstance(score_mia, float)
            self.assertGreaterEqual(score_mia, 0.0)
            self.assertLessEqual(score_mia, 1.0)

    def test_evaluate(self):
        """Test evaluate method."""
        # Create mock results
        results = [(1, 0.8), (1, 0.6), (1, 0.9), (0, 0.2), (0, 0.3)]
        metrics = self.attack.evaluate(results)
        self.assertIn('fpr', metrics)
        self.assertIn('tpr', metrics)
        self.assertIn('auc', metrics)
        self.assertIsInstance(metrics['auc'], float)

    def test_full_attack_flow(self):
        """Test the full attack flow."""
        # Combine member and non-member data
        all_data = torch.cat([self.member_data, self.non_member_data], dim=0)
        all_labels = torch.cat([self.member_labels, self.non_member_labels], dim=0)
        true_membership_labels = [1]*len(self.member_data) + [0]*len(self.non_member_data)

        dataset = TensorDataset(all_data, all_labels)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        results = self.attack.perform_attack(data_loader, true_membership_labels)
        metrics = self.attack.evaluate(results)

        self.assertIn('auc', metrics)
        self.assertIsInstance(metrics['auc'], float)
        self.assertGreaterEqual(metrics['auc'], 0.0)
        self.assertLessEqual(metrics['auc'], 1.0)

    def test_compute_membership_score_with_invalid_index(self):
        """Test compute_membership_score with invalid index."""
        x = self.member_data[0]
        y = self.member_labels[0].item()
        index = 100  # Invalid index
        with self.assertRaises(IndexError):
            self.attack.compute_membership_score(x, y, index)

if __name__ == '__main__':
    unittest.main()
