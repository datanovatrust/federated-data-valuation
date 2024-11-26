# tests/test_partitioner.py

import unittest
import torch
from torch.utils.data import TensorDataset, Subset
import numpy as np
import sys
import os

# Adjust import according to the project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.partitioner import DataPartitioner

class TestDataPartitioner(unittest.TestCase):

    def setUp(self):
        # Create a simple dataset for testing
        self.data = torch.randn(1000, 10)
        self.labels = torch.randint(0, 10, (1000,))
        self.dataset = TensorDataset(self.data, self.labels)

    def test_partition_iid_normal(self):
        """Test IID partitioning with normal parameters."""
        partitioner = DataPartitioner(self.dataset, num_clients=10, partition_type='iid')
        client_datasets = partitioner.partition_data()
        self.assertEqual(len(client_datasets), 10)
        total_samples = sum(len(client_dataset) for client_dataset in client_datasets)
        self.assertEqual(total_samples, len(self.dataset))
        for client_dataset in client_datasets:
            self.assertGreater(len(client_dataset), 0)

    def test_partition_iid_zero_clients(self):
        """Test IID partitioning with zero clients."""
        with self.assertRaises(ValueError):
            partitioner = DataPartitioner(self.dataset, num_clients=0, partition_type='iid')
            partitioner.partition_data()

    def test_partition_iid_more_clients_than_samples(self):
        """Test IID partitioning with more clients than samples."""
        partitioner = DataPartitioner(self.dataset, num_clients=2000, partition_type='iid')
        client_datasets = partitioner.partition_data()
        self.assertEqual(len(client_datasets), 2000)
        total_samples = sum(len(client_dataset) for client_dataset in client_datasets)
        self.assertEqual(total_samples, len(self.dataset))

    def test_partition_non_iid_normal(self):
        """Test non-IID partitioning with normal parameters."""
        partitioner = DataPartitioner(self.dataset, num_clients=10, partition_type='non_iid', num_shards=20)
        client_datasets = partitioner.partition_data()
        self.assertEqual(len(client_datasets), 10)
        total_samples = sum(len(client_dataset) for client_dataset in client_datasets)
        self.assertEqual(total_samples, len(self.dataset))
        for client_dataset in client_datasets:
            self.assertGreater(len(client_dataset), 0)

    def test_partition_non_iid_zero_clients(self):
        """Test non-IID partitioning with zero clients."""
        with self.assertRaises(ValueError):
            partitioner = DataPartitioner(self.dataset, num_clients=0, partition_type='non_iid', num_shards=20)
            partitioner.partition_data()

    def test_partition_non_iid_zero_shards(self):
        """Test non-IID partitioning with zero shards."""
        with self.assertRaises(ValueError):
            partitioner = DataPartitioner(self.dataset, num_clients=10, partition_type='non_iid', num_shards=0)
            partitioner.partition_data()

    def test_partition_non_iid_shards_more_than_samples(self):
        """Test non-IID partitioning when number of shards exceeds number of samples."""
        with self.assertRaises(ValueError):
            partitioner = DataPartitioner(self.dataset, num_clients=10, partition_type='non_iid', num_shards=2000)
            partitioner.partition_data()

    def test_partition_non_iid_clients_more_than_shards(self):
        """Test non-IID partitioning when number of clients exceeds number of shards."""
        with self.assertRaises(ValueError):
            partitioner = DataPartitioner(self.dataset, num_clients=50, partition_type='non_iid', num_shards=20)
            partitioner.partition_data()

    def test_partition_non_iid_with_subset(self):
        """Test non-IID partitioning when dataset is a Subset."""
        indices = list(range(500))
        subset = Subset(self.dataset, indices)
        partitioner = DataPartitioner(subset, num_clients=5, partition_type='non_iid', num_shards=10)
        client_datasets = partitioner.partition_data()
        self.assertEqual(len(client_datasets), 5)
        total_samples = sum(len(client_dataset) for client_dataset in client_datasets)
        self.assertEqual(total_samples, len(subset))

    def test_partition_non_iid_empty_dataset(self):
        """Test non-IID partitioning with an empty dataset."""
        empty_dataset = TensorDataset(torch.tensor([]), torch.tensor([]))
        with self.assertRaises(ValueError):
            partitioner = DataPartitioner(empty_dataset, num_clients=2, partition_type='non_iid', num_shards=2)
            partitioner.partition_data()

    def test_partition_invalid_partition_type(self):
        """Test partitioning with an invalid partition type."""
        with self.assertRaises(ValueError):
            partitioner = DataPartitioner(self.dataset, num_clients=10, partition_type='random')
            partitioner.partition_data()

    def test_partition_iid_with_subset(self):
        """Test IID partitioning when dataset is a Subset."""
        indices = list(range(500))
        subset = Subset(self.dataset, indices)
        partitioner = DataPartitioner(subset, num_clients=5, partition_type='iid')
        client_datasets = partitioner.partition_data()
        self.assertEqual(len(client_datasets), 5)
        total_samples = sum(len(client_dataset) for client_dataset in client_datasets)
        self.assertEqual(total_samples, len(subset))

if __name__ == '__main__':
    unittest.main()
