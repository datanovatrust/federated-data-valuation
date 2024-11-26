# src/utils/data_loader.py

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np
from PIL import Image
import pandas as pd
import logging

# Configure logging for data_loader
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers if they don't exist
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def get_mnist_datasets(transform=None):
    """
    Load MNIST train and test datasets with optional transformations.
    
    Parameters:
    - transform: torchvision.transforms object to apply to the datasets.
    
    Returns:
    - train_dataset: Training dataset.
    - test_dataset: Testing dataset.
    """
    try:
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
            ])
        
        train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
        
        # Verify dataset integrity
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            logger.error("One of the MNIST datasets is empty.")
            raise ValueError("Empty dataset detected.")
        
        logger.info("MNIST datasets loaded successfully.")
        return train_dataset, test_dataset
    except Exception as e:
        logger.error(f"Failed to load MNIST datasets: {e}")
        raise

def partition_dataset_non_iid(train_dataset, num_clients, num_shards=200):
    """
    Partition the dataset into non-IID subsets for each client.

    Parameters:
    - train_dataset: PyTorch dataset object or Subset thereof.
    - num_clients: Number of clients.
    - num_shards: Total number of shards to divide the data into.

    Returns:
    - List of Subset objects, each corresponding to a client's data.
    """
    try:
        # Input validation
        if num_clients <= 0:
            logger.error("Number of clients must be a positive integer.")
            raise ValueError("Number of clients must be greater than zero.")
        if num_shards <= 0:
            logger.error("Number of shards must be a positive integer.")
            raise ValueError("Number of shards must be greater than zero.")

        num_samples = len(train_dataset)

        # Access labels appropriately based on the dataset type
        if isinstance(train_dataset, Subset):
            # If it's a Subset, access the underlying dataset and use the subset indices
            full_dataset = train_dataset.dataset
            indices = np.array(train_dataset.indices)
        else:
            full_dataset = train_dataset
            indices = np.arange(num_samples)

        # Extract labels from the dataset
        if hasattr(full_dataset, 'targets'):
            labels = np.array(full_dataset.targets)[indices]
        elif isinstance(full_dataset, TensorDataset):
            # Assuming labels are the second element in the dataset
            labels = full_dataset.tensors[1][indices].numpy()
        else:
            logger.error("The dataset does not have a targets attribute or is not a TensorDataset.")
            raise AttributeError("Dataset must have a 'targets' attribute or be a TensorDataset.")

        # Sort indices by label to create shards with similar labels
        indices_labels = np.vstack((indices, labels))
        sorted_indices = indices_labels[:, indices_labels[1, :].argsort()]
        sorted_indices = sorted_indices[0, :].astype(int)

        # Determine shard size
        shard_size = num_samples // num_shards
        if shard_size == 0:
            logger.error("Number of shards exceeds number of samples.")
            raise ValueError("Too many shards for the dataset size.")

        # Create shards
        shards = [sorted_indices[i * shard_size:(i + 1) * shard_size] for i in range(num_shards)]

        # Handle any remaining samples by adding them to the last shard
        remaining = num_samples % num_shards
        if remaining > 0:
            shards[-1] = np.concatenate((shards[-1], sorted_indices[-remaining:]))
            logger.warning(f"Added {remaining} remaining samples to the last shard.")

        # Shuffle shards to distribute them randomly
        np.random.shuffle(shards)

        # Assign shards to clients
        shards_per_client = num_shards // num_clients
        if shards_per_client == 0:
            logger.error("Number of clients exceeds number of shards.")
            raise ValueError("Too many clients for the number of shards.")

        client_indices = []
        for i in range(num_clients):
            assigned_shards = shards[i * shards_per_client:(i + 1) * shards_per_client]
            client_idx = np.concatenate(assigned_shards, axis=0)
            client_indices.append(client_idx)

        # Handle any remaining shards by distributing them one by one to the clients
        remaining_shards = shards[num_clients * shards_per_client:]
        for i, shard in enumerate(remaining_shards):
            client_indices[i % num_clients] = np.concatenate((client_indices[i % num_clients], shard), axis=0)
            logger.warning(f"Assigned remaining shard {i + 1} to client {i % num_clients}.")

        # Create Subset datasets for each client
        client_datasets = []
        for idx in client_indices:
            if len(idx) == 0:
                logger.warning("A client has been assigned an empty dataset.")
                client_subset = Subset(full_dataset, [])
            else:
                client_subset = Subset(full_dataset, idx)
            client_datasets.append(client_subset)

        logger.info("Data partitioned among clients successfully.")
        return client_datasets
    except Exception as e:
        logger.error(f"Failed to partition dataset: {e}")
        raise

def load_custom_dataset(data_dir, file_type='jpg', transform=None):
    """
    Load a custom dataset from the specified directory.
    
    Parameters:
    - data_dir: Directory containing the data files.
    - file_type: Type of files to load ('jpg', 'png', 'csv', etc.).
    - transform: Optional transformations to apply.
    
    Returns:
    - PyTorch Dataset object.
    """
    try:
        if file_type in ['jpg', 'png']:
            # Ensure a default transform is applied to convert images to tensors
            if transform is None:
                transform = transforms.Compose([
                    transforms.ToTensor()
                ])
            images = []
            labels = []
            for img_file in os.listdir(data_dir):
                if img_file.lower().endswith(file_type):
                    try:
                        image = Image.open(os.path.join(data_dir, img_file)).convert('RGB')
                        image = transform(image)
                        images.append(image)
                        # Extract label from filename or a separate label file
                        labels.append(extract_label(img_file))
                    except Exception as img_e:
                        logger.warning(f"Failed to load image {img_file}: {img_e}")
            if len(images) == 0:
                logger.error("No images loaded. Please check the data directory and file types.")
                raise ValueError("Empty image dataset.")
            dataset = torch.utils.data.TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long))
            logger.info("Custom image dataset loaded successfully.")
            return dataset
        elif file_type == 'csv':
            # Load tabular data
            data = pd.read_csv(os.path.join(data_dir, 'data.csv'))
            if 'label' not in data.columns:
                logger.error("CSV file must contain a 'label' column.")
                raise ValueError("Missing 'label' column in CSV.")
            labels = data['label'].values
            features = data.drop('label', axis=1).values
            # Handle empty dataset
            if features.size == 0 or labels.size == 0:
                logger.error("CSV file is empty.")
                raise ValueError("Empty CSV dataset.")
            dataset = torch.utils.data.TensorDataset(
                torch.tensor(features, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.long)
            )
            logger.info("Custom CSV dataset loaded successfully.")
            return dataset
        else:
            logger.error(f"Unsupported file type: {file_type}")
            raise ValueError("Unsupported file type.")
    except Exception as e:
        logger.error(f"Failed to load custom dataset: {e}")
        raise


def extract_label(img_file):
    """
    Extract label from the image filename.
    Modify this function based on your filename format.
    
    Example: 'class1_image1.jpg' -> label 0
    """
    try:
        label_str = img_file.split('_')[0].lower()
        label_mapping = {'class1': 0, 'class2': 1, 'class3': 2}  # Define your label mapping
        if label_str not in label_mapping:
            logger.warning(f"Unknown label '{label_str}' in file '{img_file}'. Assigning label -1.")
            return -1  # Unknown label
        return label_mapping[label_str]
    except Exception as e:
        logger.warning(f"Failed to extract label from filename '{img_file}': {e}")
        return -1  # Default to unknown label
