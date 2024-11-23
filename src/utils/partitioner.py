# src/utils/partitioner.py

import numpy as np
from torch.utils.data import Subset
import logging

# Configure logging
logger = logging.getLogger(__name__)


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
        num_items = len(self.dataset)
        all_indices = np.arange(num_items)
        np.random.shuffle(all_indices)
        client_indices = np.array_split(all_indices, self.num_clients)
        client_datasets = [Subset(self.dataset, indices) for indices in client_indices]
        logger.info("Data partitioned IID among clients.")
        return client_datasets

    def partition_non_iid(self):
        num_shards = self.kwargs.get('num_shards', 200)
        num_samples = len(self.dataset)

        # Access targets appropriately
        if isinstance(self.dataset, Subset):
            targets = np.array(self.dataset.dataset.targets)[self.dataset.indices]
            indices = np.array(self.dataset.indices)
        else:
            targets = np.array(self.dataset.targets)
            indices = np.arange(num_samples)

        num_classes = len(set(targets))

        # Sort indices by label
        indices_labels = np.vstack((indices, targets))
        sorted_indices = indices_labels[:, indices_labels[1, :].argsort()]
        sorted_indices = sorted_indices[0, :].astype(int)

        # Create shards
        shard_size = num_samples // num_shards
        shards = [sorted_indices[i * shard_size:(i + 1) * shard_size] for i in range(num_shards)]

        # Assign shards to clients
        shards_per_client = num_shards // self.num_clients
        client_indices = []
        for i in range(self.num_clients):
            assigned_shards = shards[i * shards_per_client:(i + 1) * shards_per_client]
            client_idx = np.concatenate(assigned_shards, axis=0)
            client_indices.append(client_idx)

        # Handle remaining shards
        remaining_shards = shards[self.num_clients * shards_per_client:]
        for i, shard in enumerate(remaining_shards):
            client_indices[i % self.num_clients] = np.concatenate(
                (client_indices[i % self.num_clients], shard), axis=0
            )
            logger.warning(f"Assigned remaining shard {i + 1} to client {i % self.num_clients}.")

        client_datasets = [Subset(self.dataset, indices) for indices in client_indices]
        logger.info("Data partitioned non-IID among clients.")
        return client_datasets
