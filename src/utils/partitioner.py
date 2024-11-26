# src/utils/partitioner.py

import numpy as np
from torch.utils.data import Subset, TensorDataset
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers if they don't exist
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class DataPartitioner:
    """
    Class to encapsulate data partitioning logic.
    """

    def __init__(self, dataset, num_clients, partition_type='non_iid', **kwargs):
        self.dataset = dataset
        self.num_clients = num_clients
        self.partition_type = partition_type.lower()
        self.kwargs = kwargs

    def partition_data(self):
        if self.partition_type == 'non_iid':
            return self.partition_non_iid()
        elif self.partition_type == 'iid':
            return self.partition_iid()
        else:
            logger.error(f"Partition type '{self.partition_type}' is not supported.")
            raise ValueError("Unsupported partition type.")

    def partition_iid(self):
        if self.num_clients <= 0:
            logger.error("Number of clients must be a positive integer.")
            raise ValueError("Number of clients must be greater than zero.")
        num_items = len(self.dataset)
        all_indices = np.arange(num_items)
        np.random.shuffle(all_indices)
        client_indices = np.array_split(all_indices, self.num_clients)
        client_datasets = [Subset(self.dataset, indices) for indices in client_indices]
        logger.info("Data partitioned IID among clients.")
        return client_datasets

    def partition_non_iid(self):
        num_shards = self.kwargs.get('num_shards', 200)
        if self.num_clients <= 0:
            logger.error("Number of clients must be a positive integer.")
            raise ValueError("Number of clients must be greater than zero.")
        if num_shards <= 0:
            logger.error("Number of shards must be a positive integer.")
            raise ValueError("Number of shards must be greater than zero.")
        num_samples = len(self.dataset)

        # Access targets appropriately
        if isinstance(self.dataset, Subset):
            full_dataset = self.dataset.dataset
            indices = np.array(self.dataset.indices)
        else:
            full_dataset = self.dataset
            indices = np.arange(num_samples)

        if hasattr(full_dataset, 'targets'):
            targets = np.array(full_dataset.targets)[indices]
        elif isinstance(full_dataset, TensorDataset):
            # Assuming labels are the second element in the dataset
            targets = full_dataset.tensors[1][indices].numpy()
        else:
            logger.error("The dataset does not have a targets attribute or is not a TensorDataset.")
            raise AttributeError("Dataset must have a 'targets' attribute or be a TensorDataset.")

        num_classes = len(set(targets))

        # Sort indices by label
        indices_labels = np.vstack((indices, targets))
        sorted_indices = indices_labels[:, indices_labels[1, :].argsort()]
        sorted_indices = sorted_indices[0, :].astype(int)

        # Create shards
        shard_size = num_samples // num_shards
        if shard_size == 0:
            logger.error("Number of shards exceeds number of samples.")
            raise ValueError("Too many shards for the dataset size.")

        shards = [sorted_indices[i * shard_size:(i + 1) * shard_size] for i in range(num_shards)]

        # Handle remaining samples by adding them to the last shard
        remaining = num_samples % num_shards
        if remaining > 0:
            shards[-1] = np.concatenate((shards[-1], sorted_indices[-remaining:]))
            logger.warning(f"Added {remaining} remaining samples to the last shard.")

        # Assign shards to clients
        shards_per_client = num_shards // self.num_clients
        if shards_per_client == 0:
            logger.error("Number of clients exceeds number of shards.")
            raise ValueError("Too many clients for the number of shards.")

        client_indices = []
        for i in range(self.num_clients):
            assigned_shards = shards[i * shards_per_client:(i + 1) * shards_per_client]
            client_idx = np.concatenate(assigned_shards, axis=0)
            client_indices.append(client_idx)

        # Handle any remaining shards
        remaining_shards = shards[self.num_clients * shards_per_client:]
        for i, shard in enumerate(remaining_shards):
            client_indices[i % self.num_clients] = np.concatenate(
                (client_indices[i % self.num_clients], shard), axis=0
            )
            logger.warning(f"Assigned remaining shard {i + 1} to client {i % self.num_clients}.")

        client_datasets = [Subset(self.dataset, indices) for indices in client_indices]
        logger.info("Data partitioned non-IID among clients.")
        return client_datasets
