# src/attacks/rmia_attack.py

import logging
import torch
from .statistical_tests import (
    compute_likelihood_ratio,
    compute_score_mia,
    hypothesis_test
)
from .reference_model_manager import ReferenceModelManager
from .data_sampler import DataSampler
from .evaluation_metrics import (
    compute_tpr_fpr,
    compute_auc
)
from .config import RMIAConfig

class RMIAttack:
    """
    Implements the Robust Membership Inference Attack (RMIA).
    """

    def __init__(self, target_model, reference_models, config=RMIAConfig(), data_sampler=None):
        """
        Initializes the RMIAttack instance.

        Args:
            target_model: The target model to attack.
            reference_models: A list of reference models.
            config: Configuration parameters.
            data_sampler: DataSampler instance to sample population data.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_model = target_model.to(self.device)
        self.reference_models = [model.to(self.device) for model in reference_models]
        self.config = config
        self.data_sampler = data_sampler  # Use the provided data_sampler
        self.reference_manager = ReferenceModelManager(reference_models)

        # Initialize logger for RMIAttack
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)  # Set the logger's level to INFO

    def compute_membership_score(self, x, y, index):
        """
        Computes the membership inference score for a data point x.

        Args:
            x: The data point to test.
            y: The true label of x.
            index: The index of x in the dataset.

        Returns:
            score_mia: The membership inference score.
        """
        likelihood_ratios = []
        z_samples, z_labels = self.data_sampler.sample_population_data(
            num_samples=self.config.NUM_Z_SAMPLES,
            exclude_indices=[index]
        )

        num_z_samples = len(z_samples)
        # Log the start of processing for this data point
        self.logger.debug(f"Computing membership score for sample {index + 1} with {num_z_samples} z samples.")

        for j, (z, y_z) in enumerate(zip(z_samples, z_labels)):
            z = z.to(self.device)
            lr = compute_likelihood_ratio(
                self.target_model,
                x,
                y,
                z,
                y_z,
                self.reference_models
            )
            likelihood_ratios.append(lr)

            # Optionally log progress every 500 z samples
            if (j + 1) % 500 == 0 or (j + 1) == num_z_samples:
                self.logger.debug(f"Processed {j + 1}/{num_z_samples} z samples for sample {index + 1}.")

        score_mia = compute_score_mia(likelihood_ratios, self.config.GAMMA)
        return score_mia

    def perform_attack(self, data_loader, true_membership_labels):
        """
        Performs the attack on a dataset.

        Args:
            data_loader: DataLoader providing the dataset to attack.
            true_membership_labels: List of true membership labels (1 for member, 0 for non-member).

        Returns:
            results: A list of tuples (true_label, predicted_score).
        """
        results = []
        total_samples = len(data_loader.dataset)
        self.logger.info(f"Total samples to attack: {total_samples}")

        for i, (data, label) in enumerate(data_loader):
            x = data.to(self.device)
            y = label.item()  # Assuming label is a scalar
            true_label = true_membership_labels[i]
            score_mia = self.compute_membership_score(x, y, index=i)
            results.append((true_label, score_mia))

            # Log progress every 100 samples
            if (i + 1) % 100 == 0 or (i + 1) == total_samples:
                self.logger.info(f"Processed {i + 1}/{total_samples} samples.")

        return results

    def evaluate(self, results):
        """
        Evaluates the attack's performance using evaluation metrics.

        Args:
            results: A list of tuples (true_label, predicted_score).

        Returns:
            metrics: A dictionary containing evaluation metrics.
        """
        true_labels, pred_scores = zip(*results)
        fpr, tpr = compute_tpr_fpr(true_labels, pred_scores)
        auc = compute_auc(fpr, tpr)
        metrics = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': auc
        }
        return metrics
