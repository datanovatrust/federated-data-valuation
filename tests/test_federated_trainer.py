# tests/test_federated_trainer.py

import unittest
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
import os
import tempfile
import shutil
import copy  # Add this import

# Adjust import according to the project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trainers.federated_trainer import FederatedTrainer, LocalClient
from src.models.image_classifier import ModelFactory

# Mock configurations
mock_config = {
    'federated_learning': {
        'num_clients': 2,
        'rounds': 1,
        'fraction_fit': 1.0,
        'num_shards': 2,
        'partition_type': 'iid',
    },
    'training': {
        'batch_size': 4,
        'epochs': 1,
        'learning_rate': 0.01,
    },
    'model': {
        'name': 'simple_cnn',
        'num_labels': 10,
    }
}

class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(28 * 28, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

class TestFederatedTrainer(unittest.TestCase):

    def setUp(self):
        # Create a simple dataset
        self.num_samples = 20
        self.num_features = 28 * 28
        self.num_classes = 10
        data = torch.randn(self.num_samples, 1, 28, 28)
        labels = torch.randint(0, self.num_classes, (self.num_samples,))
        self.train_dataset = TensorDataset(data, labels)
        self.test_dataset = TensorDataset(data, labels)
        self.wasserstein_train_dataset = TensorDataset(data, labels)
        self.wasserstein_test_dataset = TensorDataset(data, labels)

        # Mock model class
        self.model_class = SimpleCNN

    def test_initialization(self):
        """Test initialization of FederatedTrainer."""
        trainer = FederatedTrainer(
            config=mock_config,
            model_class=self.model_class,
            train_dataset=self.train_dataset,
            test_dataset=self.test_dataset,
            wasserstein_train_dataset=self.wasserstein_train_dataset,
            wasserstein_test_dataset=self.wasserstein_test_dataset,
            target_epsilon=None
        )
        self.assertEqual(trainer.num_clients, 2)
        self.assertEqual(trainer.rounds, 1)
        self.assertEqual(trainer.epochs, 1)
        self.assertEqual(trainer.learning_rate, 0.01)
        self.assertEqual(trainer.model_name, 'simple_cnn')
        self.assertFalse(trainer.use_dp)

    def test_partition_data(self):
        """Test data partitioning among clients."""
        trainer = FederatedTrainer(
            config=mock_config,
            model_class=self.model_class,
            train_dataset=self.train_dataset,
            test_dataset=self.test_dataset,
            wasserstein_train_dataset=self.wasserstein_train_dataset,
            wasserstein_test_dataset=self.wasserstein_test_dataset,
            target_epsilon=None
        )
        trainer.partition_data()
        self.assertEqual(len(trainer.clients), 2)
        total_samples = sum(len(client.dataset) for client in trainer.clients)
        self.assertEqual(total_samples, len(self.train_dataset))

    def test_initialize_global_model(self):
        """Test initialization of the global model."""
        trainer = FederatedTrainer(
            config=mock_config,
            model_class=self.model_class,
            train_dataset=self.train_dataset,
            test_dataset=self.test_dataset,
            wasserstein_train_dataset=self.wasserstein_train_dataset,
            wasserstein_test_dataset=self.wasserstein_test_dataset,
            target_epsilon=None
        )
        global_model = trainer.initialize_global_model()
        self.assertIsInstance(global_model, torch.nn.Module)

    def test_train_single_round(self):
        """Test training process for a single round."""
        trainer = FederatedTrainer(
            config=mock_config,
            model_class=self.model_class,
            train_dataset=self.train_dataset,
            test_dataset=self.test_dataset,
            wasserstein_train_dataset=self.wasserstein_train_dataset,
            wasserstein_test_dataset=self.wasserstein_test_dataset,
            target_epsilon=None
        )
        # We'll mock the methods that are not under test
        trainer.prepare_global_distribution = lambda: np.random.randn(1000, 784)
        trainer.compute_wasserstein_distances = lambda x: None
        trainer.select_clients = lambda: trainer.clients
        trainer.save_checkpoint = lambda model, round_num: None
        trainer.plot_confusion_matrix = lambda true_labels, pred_labels, round_num: None
        trainer.plot_results = lambda: None

        trainer.train()

        # Check that training accuracy was logged
        self.assertTrue(len(trainer.accuracy_list) > 0)

    def test_evaluate(self):
        """Test the evaluate function."""
        trainer = FederatedTrainer(
            config=mock_config,
            model_class=self.model_class,
            train_dataset=self.train_dataset,
            test_dataset=self.test_dataset,
            wasserstein_train_dataset=self.wasserstein_train_dataset,
            wasserstein_test_dataset=self.wasserstein_test_dataset,
            target_epsilon=None
        )
        model = self.model_class(num_classes=self.num_classes)
        test_loader = DataLoader(self.test_dataset, batch_size=4)
        trainer.evaluate(model, test_loader, round_num=1)
        self.assertTrue(len(trainer.accuracy_list) == 1)

    def test_aggregate_models(self):
        """Test model aggregation."""
        trainer = FederatedTrainer(
            config=mock_config,
            model_class=self.model_class,
            train_dataset=self.train_dataset,
            test_dataset=self.test_dataset,
            wasserstein_train_dataset=self.wasserstein_train_dataset,
            wasserstein_test_dataset=self.wasserstein_test_dataset,
            target_epsilon=None
        )
        global_model = self.model_class(num_classes=self.num_classes)
        local_models = [self.model_class(num_classes=self.num_classes).state_dict() for _ in range(2)]
        aggregated_model = trainer.aggregate_models(global_model, local_models)
        self.assertIsNotNone(aggregated_model)

    def test_training_with_dp(self):
        """Test training with differential privacy enabled."""
        dp_config = copy.deepcopy(mock_config)  # Use deepcopy
        trainer = FederatedTrainer(
            config=dp_config,
            model_class=self.model_class,
            train_dataset=self.train_dataset,
            test_dataset=self.test_dataset,
            wasserstein_train_dataset=self.wasserstein_train_dataset,
            wasserstein_test_dataset=self.wasserstein_test_dataset,
            target_epsilon=1.0
        )
        self.assertTrue(trainer.use_dp)

    def test_no_clients(self):
        """Test behavior when there are no clients."""
        zero_client_config = copy.deepcopy(mock_config)  # Use deepcopy
        zero_client_config['federated_learning']['num_clients'] = 0
        with self.assertRaises(ValueError):
            trainer = FederatedTrainer(
                config=zero_client_config,
                model_class=self.model_class,
                train_dataset=self.train_dataset,
                test_dataset=self.test_dataset,
                wasserstein_train_dataset=self.wasserstein_train_dataset,
                wasserstein_test_dataset=self.wasserstein_test_dataset,
                target_epsilon=None
            )
            trainer.partition_data()

    def test_clients_with_empty_datasets(self):
        """Test behavior when clients have empty datasets."""
        trainer = FederatedTrainer(
            config=mock_config,
            model_class=self.model_class,
            train_dataset=TensorDataset(torch.tensor([]), torch.tensor([])),
            test_dataset=self.test_dataset,
            wasserstein_train_dataset=TensorDataset(torch.tensor([]), torch.tensor([])),
            wasserstein_test_dataset=self.wasserstein_test_dataset,
            target_epsilon=None
        )
        trainer.partition_data()
        self.assertEqual(len(trainer.clients), 2)
        for client in trainer.clients:
            self.assertEqual(len(client.dataset), 0)
        trainer.prepare_global_distribution = lambda: np.random.randn(1000, 784)
        trainer.compute_wasserstein_distances = lambda x: None
        trainer.select_clients = lambda: trainer.clients
        trainer.save_checkpoint = lambda model, round_num: None
        trainer.plot_confusion_matrix = lambda true_labels, pred_labels, round_num: None
        trainer.plot_results = lambda: None
        trainer.train()
        # Should handle clients with empty datasets gracefully

if __name__ == '__main__':
    unittest.main()
