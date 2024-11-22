import unittest
import torch
from src.utils.data_loader import get_mnist_datasets, partition_dataset_non_iid
from scripts.train_federated import compute_wasserstein_distances
import numpy as np

class TestFedBary(unittest.TestCase):
    def test_wasserstein_distances(self):
        train_dataset, test_dataset = get_mnist_datasets()
        client_datasets = partition_dataset_non_iid(train_dataset, num_clients=5)
        validation_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        validation_data, _ = next(iter(validation_loader))
        global_distribution = validation_data.view(len(test_dataset), -1)
        contributions = compute_wasserstein_distances(client_datasets, global_distribution)
        self.assertEqual(len(contributions), 5)
        for i, distance in contributions:
            self.assertIsInstance(distance, float)
            self.assertGreaterEqual(distance, 0)

if __name__ == '__main__':
    unittest.main()
