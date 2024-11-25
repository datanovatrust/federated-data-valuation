# src/attacks/data_sampler.py

import numpy as np

class DataSampler:
    """
    Samples data points from the population distribution.
    """

    def __init__(self, dataset=None):
        """
        Initializes the DataSampler.

        Args:
            dataset: The dataset to sample from.
        """
        self.dataset = dataset

    def sample_population_data(self, num_samples, exclude_indices=None):
        """
        Samples data from the population dataset.

        Args:
            num_samples: Number of samples to draw.
            exclude_indices: List of indices to exclude from sampling.

        Returns:
            samples: List of data samples.
            labels: List of corresponding labels.
        """
        indices = list(range(len(self.dataset)))
        if exclude_indices is not None:
            indices = [i for i in indices if i not in exclude_indices]
        selected_indices = np.random.choice(indices, num_samples, replace=False)
        samples = []
        labels = []
        for idx in selected_indices:
            sample, label = self.dataset[idx]
            samples.append(sample)
            labels.append(label)
        return samples, labels

    def set_dataset(self, dataset):
        """
        Sets the dataset for sampling.

        Args:
            dataset: The dataset to sample from.
        """
        self.dataset = dataset
