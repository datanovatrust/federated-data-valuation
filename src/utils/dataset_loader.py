# src/utils/dataset_loader.py

import os
import logging
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Generalized dataset loader that supports multiple datasets.
    """

    def __init__(self, dataset_name, transform=None, **kwargs):
        self.dataset_name = dataset_name.lower()
        self.transform = transform
        self.kwargs = kwargs
        self.dataset = self.load_dataset()

    def load_dataset(self):
        if self.dataset_name == 'mnist':
            return self.load_mnist()
        elif self.dataset_name == 'cifar10':
            return self.load_cifar10()
        elif self.dataset_name == 'custom':
            return self.load_custom_dataset()
        else:
            logger.error(f"Dataset '{self.dataset_name}' is not supported.")
            raise ValueError("Unsupported dataset.")

    def load_mnist(self):
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        train_dataset = datasets.MNIST(
            root='data', train=True, download=True, transform=self.transform
        )
        test_dataset = datasets.MNIST(
            root='data', train=False, download=True, transform=self.transform
        )
        logger.info("MNIST dataset loaded successfully.")
        return {'train': train_dataset, 'test': test_dataset}

    def load_cifar10(self):
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                )
            ])
        train_dataset = datasets.CIFAR10(
            root='data', train=True, download=True, transform=self.transform
        )
        test_dataset = datasets.CIFAR10(
            root='data', train=False, download=True, transform=self.transform
        )
        logger.info("CIFAR-10 dataset loaded successfully.")
        return {'train': train_dataset, 'test': test_dataset}

    def load_custom_dataset(self):
        data_dir = self.kwargs.get('data_dir')
        file_type = self.kwargs.get('file_type', 'jpg')
        if data_dir is None:
            logger.error("Custom dataset requires 'data_dir' parameter.")
            raise ValueError("Missing 'data_dir' for custom dataset.")

        if file_type in ['jpg', 'png']:
            return CustomImageDataset(data_dir, file_type, transform=self.transform)
        elif file_type == 'csv':
            return CustomCSVDataset(data_dir)
        else:
            logger.error(f"Unsupported file type: {file_type}")
            raise ValueError("Unsupported file type.")


class CustomImageDataset(Dataset):
    """
    Custom dataset class for loading images from a directory.
    """

    def __init__(self, data_dir, file_type, transform=None):
        self.data_dir = data_dir
        self.file_type = file_type
        self.transform = transform
        self.images = []
        self.labels = []
        self.load_images()

    def load_images(self):
        for img_file in os.listdir(self.data_dir):
            if img_file.lower().endswith(self.file_type):
                try:
                    image = Image.open(
                        os.path.join(self.data_dir, img_file)
                    ).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    self.images.append(image)
                    label = self.extract_label(img_file)
                    self.labels.append(label)
                except Exception as e:
                    logger.warning(f"Failed to load image {img_file}: {e}")

        if not self.images:
            logger.error("No images found in the specified directory.")
            raise ValueError("Empty dataset.")

        logger.info("Custom image dataset loaded successfully.")

    def extract_label(self, img_file):
        label_str = img_file.split('_')[0].lower()
        label_mapping = {'class1': 0, 'class2': 1, 'class3': 2}
        return label_mapping.get(label_str, -1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class CustomCSVDataset(Dataset):
    """
    Custom dataset class for loading data from a CSV file.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = pd.read_csv(os.path.join(self.data_dir, 'data.csv'))
        if 'label' not in self.data.columns:
            logger.error("CSV file must contain a 'label' column.")
            raise ValueError("Missing 'label' column in CSV.")
        self.labels = self.data['label'].values
        self.features = self.data.drop('label', axis=1).values
        logger.info("Custom CSV dataset loaded successfully.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label
