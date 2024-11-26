# tests/test_data_sampler.py

import unittest
import sys
import os
import torch
import numpy as np

from torch.utils.data import TensorDataset

# Adjust import according to the project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.attacks.data_sampler import DataSampler

class TestDataSampler(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset
        self.num_samples = 100
        data = torch.randn(self.num_samples, 3, 32, 32)
        labels = torch.arange(self.num_samples)  # Labels are 0 to 99
        self.dataset = TensorDataset(data, labels)
        self.sampler = DataSampler(self.dataset)
    
    def test_sample_population_data(self):
        """Test sampling population data without exclusions."""
        num_samples = 10
        samples, labels = self.sampler.sample_population_data(num_samples)
        self.assertEqual(len(samples), num_samples)
        self.assertEqual(len(labels), num_samples)
        for sample in samples:
            self.assertIsInstance(sample, torch.Tensor)
        for label in labels:
            self.assertIsInstance(label, torch.Tensor)
    
    def test_sample_with_exclude_indices(self):
        """Test sampling with exclude indices."""
        num_samples = 10
        exclude_indices = [0, 1, 2, 3, 4]
        samples, labels = self.sampler.sample_population_data(num_samples, exclude_indices=exclude_indices)
        self.assertEqual(len(samples), num_samples)
        for label in labels:
            idx = label.item()  # Labels are indices
            self.assertNotIn(idx, exclude_indices)
    
    def test_set_dataset(self):
        """Test setting a new dataset."""
        new_data = torch.randn(50, 3, 32, 32)
        new_labels = torch.arange(50)
        new_dataset = TensorDataset(new_data, new_labels)
        self.sampler.set_dataset(new_dataset)
        self.assertEqual(len(self.sampler.dataset), 50)
    
    def test_sample_more_than_available(self):
        """Test sampling more samples than available."""
        num_samples = 150  # More than available
        with self.assertRaises(ValueError):
            self.sampler.sample_population_data(num_samples)
    
    def test_sample_with_all_indices_excluded(self):
        """Test sampling when all indices are excluded."""
        exclude_indices = list(range(self.num_samples))
        num_samples = 1
        with self.assertRaises(ValueError):
            self.sampler.sample_population_data(num_samples, exclude_indices=exclude_indices)
    
    def test_sample_with_no_dataset(self):
        """Test sampling when no dataset is set."""
        sampler = DataSampler()
        with self.assertRaises(TypeError):
            sampler.sample_population_data(10)
    
if __name__ == '__main__':
    unittest.main()
