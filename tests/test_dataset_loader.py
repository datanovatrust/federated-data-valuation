# tests/test_dataset_loader.py

import unittest
import torch
from torch.utils.data import DataLoader
import os
import sys
import tempfile
import shutil
from PIL import Image
import pandas as pd
import numpy as np

# Adjust import according to the project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.dataset_loader import DatasetLoader, CustomImageDataset, CustomCSVDataset

from torchvision import transforms  # Added missing import

class TestDatasetLoader(unittest.TestCase):

    def test_load_mnist_default_transform(self):
        """Test loading MNIST dataset with default transformations."""
        loader = DatasetLoader('mnist')
        self.assertIn('train', loader.dataset)
        self.assertIn('test', loader.dataset)
        train_dataset = loader.dataset['train']
        test_dataset = loader.dataset['test']
        self.assertGreater(len(train_dataset), 0)
        self.assertGreater(len(test_dataset), 0)
        sample, label = train_dataset[0]
        self.assertEqual(sample.shape, torch.Size([1, 28, 28]))
        self.assertIsInstance(label, int)

    def test_load_mnist_custom_transform(self):
        """Test loading MNIST dataset with custom transformations."""
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        loader = DatasetLoader('mnist', transform=transform)
        train_dataset = loader.dataset['train']
        sample, _ = train_dataset[0]
        self.assertEqual(sample.shape, torch.Size([1, 28, 28]))

    def test_load_cifar10_default_transform(self):
        """Test loading CIFAR-10 dataset with default transformations."""
        loader = DatasetLoader('cifar10')
        self.assertIn('train', loader.dataset)
        self.assertIn('test', loader.dataset)
        train_dataset = loader.dataset['train']
        sample, label = train_dataset[0]
        self.assertEqual(sample.shape, torch.Size([3, 32, 32]))
        self.assertIsInstance(label, int)

    def test_load_cifar10_custom_transform(self):
        """Test loading CIFAR-10 dataset with custom transformations."""
        transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])
        loader = DatasetLoader('cifar10', transform=transform)
        train_dataset = loader.dataset['train']
        sample, _ = train_dataset[0]
        self.assertEqual(sample.shape, torch.Size([3, 32, 32]))

    def test_load_custom_image_dataset_valid(self):
        """Test loading a valid custom image dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            img_size = (64, 64)
            num_images = 5
            for i in range(num_images):
                class_name = f'class{(i % 3) + 1}'
                img = Image.new('RGB', img_size, color=(i * 10, i * 10, i * 10))
                img_file = os.path.join(temp_dir, f'{class_name}_image{i}.jpg')
                img.save(img_file)
            loader = DatasetLoader('custom', data_dir=temp_dir, file_type='jpg')
            dataset = loader.dataset
            self.assertEqual(len(dataset), num_images)
            for img, label in dataset:
                self.assertIsInstance(img, torch.Tensor)
                self.assertIn(label, [0, 1, 2])

    def test_load_custom_image_dataset_no_images(self):
        """Test loading a custom image dataset with no images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                DatasetLoader('custom', data_dir=temp_dir, file_type='jpg')

    def test_load_custom_image_dataset_invalid_file_type(self):
        """Test loading a custom image dataset with an unsupported file type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                DatasetLoader('custom', data_dir=temp_dir, file_type='txt')

    def test_load_custom_csv_dataset_valid(self):
        """Test loading a valid custom CSV dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = os.path.join(temp_dir, 'data.csv')
            df = pd.DataFrame({
                'feature1': np.random.randn(10),
                'feature2': np.random.randn(10),
                'label': np.random.randint(0, 2, size=10)
            })
            df.to_csv(csv_file, index=False)
            loader = DatasetLoader('custom', data_dir=temp_dir, file_type='csv')
            dataset = loader.dataset
            self.assertEqual(len(dataset), 10)
            for features, label in dataset:
                self.assertEqual(len(features), 2)
                self.assertIn(label.item(), [0, 1])

    def test_load_custom_csv_dataset_missing_label(self):
        """Test loading a custom CSV dataset missing the 'label' column."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = os.path.join(temp_dir, 'data.csv')
            df = pd.DataFrame({
                'feature1': np.random.randn(10),
                'feature2': np.random.randn(10)
            })
            df.to_csv(csv_file, index=False)
            with self.assertRaises(ValueError):
                DatasetLoader('custom', data_dir=temp_dir, file_type='csv')

    def test_load_custom_csv_dataset_empty(self):
        """Test loading an empty custom CSV dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = os.path.join(temp_dir, 'data.csv')
            df = pd.DataFrame(columns=['feature1', 'feature2', 'label'])
            df.to_csv(csv_file, index=False)
            with self.assertRaises(ValueError):
                DatasetLoader('custom', data_dir=temp_dir, file_type='csv')

    def test_load_custom_dataset_missing_data_dir(self):
        """Test loading a custom dataset without specifying data_dir."""
        with self.assertRaises(ValueError):
            DatasetLoader('custom', file_type='jpg')

    def test_load_unsupported_dataset(self):
        """Test loading an unsupported dataset."""
        with self.assertRaises(ValueError):
            DatasetLoader('unknown_dataset')

    def test_custom_image_dataset_invalid_images(self):
        """Test loading a custom image dataset with invalid images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_file = os.path.join(temp_dir, 'invalid_image.jpg')
            with open(invalid_file, 'w') as f:
                f.write('not an image')
            with self.assertRaises(ValueError):
                DatasetLoader('custom', data_dir=temp_dir, file_type='jpg')

    def test_custom_csv_dataset_nonexistent_file(self):
        """Test loading a custom CSV dataset when the file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                DatasetLoader('custom', data_dir=temp_dir, file_type='csv')

    def test_custom_image_dataset_with_transform(self):
        """Test loading a custom image dataset with a transformation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            img_size = (64, 64)
            num_images = 3
            for i in range(num_images):
                class_name = f'class{i+1}'
                img = Image.new('RGB', img_size, color=(i * 20, i * 20, i * 20))
                img_file = os.path.join(temp_dir, f'{class_name}_img{i}.jpg')
                img.save(img_file)
            transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])
            loader = DatasetLoader('custom', data_dir=temp_dir, file_type='jpg', transform=transform)
            dataset = loader.dataset
            for img, _ in dataset:
                self.assertEqual(img.shape[0], 1)  # Grayscale image

    def test_custom_csv_dataset_with_non_numeric_data(self):
        """Test loading a custom CSV dataset with non-numeric data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = os.path.join(temp_dir, 'data.csv')
            df = pd.DataFrame({
                'feature1': ['a', 'b', 'c'],
                'feature2': ['d', 'e', 'f'],
                'label': [0, 1, 0]
            })
            df.to_csv(csv_file, index=False)
            with self.assertRaises(ValueError):
                DatasetLoader('custom', data_dir=temp_dir, file_type='csv')

    def test_custom_image_dataset_unknown_labels(self):
        """Test loading a custom image dataset with unknown labels."""
        with tempfile.TemporaryDirectory() as temp_dir:
            img_size = (64, 64)
            img = Image.new('RGB', img_size)
            img_file = os.path.join(temp_dir, 'unknown_img.jpg')
            img.save(img_file)
            loader = DatasetLoader('custom', data_dir=temp_dir, file_type='jpg')
            dataset = loader.dataset
            self.assertEqual(len(dataset), 1)
            _, label = dataset[0]
            self.assertEqual(label, -1)

if __name__ == '__main__':
    unittest.main()
