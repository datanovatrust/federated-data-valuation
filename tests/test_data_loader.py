# tests/test_data_loader.py

import unittest
import torch
from torch.utils.data import Subset, TensorDataset
import numpy as np
import os
import shutil
from torchvision import datasets, transforms
from unittest import mock
from PIL import Image
import tempfile
import pandas as pd

# Adjust import according to the project structure
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import data_loader

class TestDataLoader(unittest.TestCase):

    def test_get_mnist_datasets_default(self):
        """Test loading MNIST datasets with default transformation."""
        train_dataset, test_dataset = data_loader.get_mnist_datasets()
        self.assertGreater(len(train_dataset), 0)
        self.assertGreater(len(test_dataset), 0)
        sample, label = train_dataset[0]
        self.assertEqual(sample.shape, torch.Size([1, 28, 28]))
        self.assertIsInstance(label, int)

    def test_get_mnist_datasets_custom_transform(self):
        """Test loading MNIST datasets with a custom transformation."""
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        train_dataset, test_dataset = data_loader.get_mnist_datasets(transform=transform)
        sample, _ = train_dataset[0]
        self.assertIsInstance(sample, torch.Tensor)
        self.assertEqual(sample.shape, torch.Size([1, 28, 28]))

    def test_partition_dataset_non_iid_normal(self):
        """Test non-IID partitioning with normal parameters."""
        data = torch.randn(1000, 1, 28, 28)
        labels = torch.randint(0, 10, (1000,))
        dataset = TensorDataset(data, labels)
        num_clients = 10
        num_shards = 20

        client_datasets = data_loader.partition_dataset_non_iid(dataset, num_clients, num_shards)
        self.assertEqual(len(client_datasets), num_clients)
        total_samples = sum(len(client_dataset) for client_dataset in client_datasets)
        self.assertEqual(total_samples, len(dataset))
        for client_dataset in client_datasets:
            self.assertGreater(len(client_dataset), 0)

    def test_partition_dataset_non_iid_shards_more_than_samples(self):
        """Test partitioning when the number of shards exceeds the number of samples."""
        data = torch.randn(10, 1, 28, 28)
        labels = torch.randint(0, 10, (10,))
        dataset = TensorDataset(data, labels)
        num_clients = 2
        num_shards = 20

        with self.assertRaises(ValueError):
            data_loader.partition_dataset_non_iid(dataset, num_clients, num_shards)

    def test_partition_dataset_non_iid_clients_more_than_shards(self):
        """Test partitioning when the number of clients exceeds the number of shards."""
        data = torch.randn(100, 1, 28, 28)
        labels = torch.randint(0, 10, (100,))
        dataset = TensorDataset(data, labels)
        num_clients = 50
        num_shards = 10

        with self.assertRaises(ValueError):
            data_loader.partition_dataset_non_iid(dataset, num_clients, num_shards)

    def test_partition_dataset_non_iid_zero_clients(self):
        """Test partitioning with zero clients."""
        data = torch.randn(100, 1, 28, 28)
        labels = torch.randint(0, 10, (100,))
        dataset = TensorDataset(data, labels)
        num_clients = 0
        num_shards = 10

        with self.assertRaises(ValueError):
            data_loader.partition_dataset_non_iid(dataset, num_clients, num_shards)

    def test_partition_dataset_non_iid_zero_shards(self):
        """Test partitioning with zero shards."""
        data = torch.randn(100, 1, 28, 28)
        labels = torch.randint(0, 10, (100,))
        dataset = TensorDataset(data, labels)
        num_clients = 10
        num_shards = 0

        with self.assertRaises(ValueError):
            data_loader.partition_dataset_non_iid(dataset, num_clients, num_shards)

    def test_partition_dataset_non_iid_empty_dataset(self):
        """Test partitioning with an empty dataset."""
        data = torch.tensor([])
        labels = torch.tensor([])
        dataset = TensorDataset(data, labels)
        num_clients = 2
        num_shards = 2

        with self.assertRaises(ValueError):
            data_loader.partition_dataset_non_iid(dataset, num_clients, num_shards)

    def test_partition_dataset_non_iid_with_subset(self):
        """Test partitioning when the dataset is a Subset."""
        data = torch.randn(100, 1, 28, 28)
        labels = torch.randint(0, 10, (100,))
        full_dataset = TensorDataset(data, labels)
        indices = list(range(50))
        subset = Subset(full_dataset, indices)
        num_clients = 5
        num_shards = 10

        client_datasets = data_loader.partition_dataset_non_iid(subset, num_clients, num_shards)
        self.assertEqual(len(client_datasets), num_clients)
        total_samples = sum(len(client_dataset) for client_dataset in client_datasets)
        self.assertEqual(total_samples, len(subset))

    def test_load_custom_dataset_images_valid(self):
        """Test loading a valid custom image dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            img_size = (64, 64)
            num_images = 5
            for i in range(num_images):
                class_name = f'class{(i%3)+1}'
                img = Image.new('RGB', img_size, color=(i*10, i*10, i*10))
                img_file = os.path.join(temp_dir, f'{class_name}_image{i}.jpg')
                img.save(img_file)
            dataset = data_loader.load_custom_dataset(temp_dir, file_type='jpg')
            self.assertEqual(len(dataset), num_images)
            for _, label in dataset:
                self.assertIn(label.item(), [0, 1, 2])

    def test_load_custom_dataset_images_no_images(self):
        """Test loading from a directory with no images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                data_loader.load_custom_dataset(temp_dir, file_type='jpg')

    def test_load_custom_dataset_images_invalid_files(self):
        """Test loading when some image files are invalid."""
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_file = os.path.join(temp_dir, 'invalid_image.jpg')
            with open(invalid_file, 'w') as f:
                f.write('not an image')
            img = Image.new('RGB', (64, 64))
            img_file = os.path.join(temp_dir, 'class1_image0.jpg')
            img.save(img_file)
            dataset = data_loader.load_custom_dataset(temp_dir, file_type='jpg')
            self.assertEqual(len(dataset), 1)

    def test_load_custom_dataset_csv_valid(self):
        """Test loading a valid custom CSV dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = os.path.join(temp_dir, 'data.csv')
            df = pd.DataFrame({
                'feature1': np.random.randn(10),
                'feature2': np.random.randn(10),
                'label': np.random.randint(0, 2, size=10)
            })
            df.to_csv(csv_file, index=False)
            dataset = data_loader.load_custom_dataset(temp_dir, file_type='csv')
            self.assertEqual(len(dataset), 10)
            for data, label in dataset:
                self.assertEqual(len(data), 2)
                self.assertIn(label.item(), [0, 1])

    def test_load_custom_dataset_csv_missing_label(self):
        """Test loading a CSV dataset missing the 'label' column."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = os.path.join(temp_dir, 'data.csv')
            df = pd.DataFrame({
                'feature1': np.random.randn(10),
                'feature2': np.random.randn(10)
            })
            df.to_csv(csv_file, index=False)
            with self.assertRaises(ValueError):
                data_loader.load_custom_dataset(temp_dir, file_type='csv')

    def test_load_custom_dataset_csv_empty(self):
        """Test loading an empty CSV dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = os.path.join(temp_dir, 'data.csv')
            df = pd.DataFrame(columns=['feature1', 'feature2', 'label'])
            df.to_csv(csv_file, index=False)
            with self.assertRaises(ValueError):
                data_loader.load_custom_dataset(temp_dir, file_type='csv')

    def test_load_custom_dataset_unsupported_file_type(self):
        """Test loading with an unsupported file type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                data_loader.load_custom_dataset(temp_dir, file_type='txt')

    def test_extract_label_known_labels(self):
        """Test label extraction from filenames with known labels."""
        filename = 'class1_image1.jpg'
        label = data_loader.extract_label(filename)
        self.assertEqual(label, 0)
        filename = 'class2_image2.jpg'
        label = data_loader.extract_label(filename)
        self.assertEqual(label, 1)
        filename = 'class3_image3.jpg'
        label = data_loader.extract_label(filename)
        self.assertEqual(label, 2)

    def test_extract_label_unknown_label(self):
        """Test label extraction from filenames with unknown labels."""
        filename = 'unknown_image.jpg'
        label = data_loader.extract_label(filename)
        self.assertEqual(label, -1)

    def test_extract_label_invalid_format(self):
        """Test label extraction from filenames with an invalid format."""
        filename = 'image.jpg'
        label = data_loader.extract_label(filename)
        self.assertEqual(label, -1)

    def test_extract_label_none(self):
        """Test label extraction when the filename is None."""
        filename = None
        label = data_loader.extract_label(filename)
        self.assertEqual(label, -1)

if __name__ == '__main__':
    unittest.main()
