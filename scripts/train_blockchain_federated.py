# scripts/train_blockchain_federated.py

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import Subset
from src.utils.dataset_loader import DatasetLoader
import yaml
import logging
from torchvision import transforms
import argparse  # For command-line argument parsing
import sys

# ----------------- Logging Configuration -----------------

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG level to capture all logs

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler('logs/federated_training.log', mode='a')

# Set level for handlers
console_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)

# Add formatter to handlers
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger if none exist
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# Import the BlockchainFederatedTrainer for BCFL with DP support
from src.trainers.blockchain_federated_trainer import BlockchainFederatedTrainer

def main():
    logger.info("🚀 Starting the Blockchain-enabled Federated Training Script")

    try:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description='Blockchain Federated Learning Training Script')
        parser.add_argument('--epsilon', type=float, default=None,
                            help='Target epsilon for differential privacy (default: None)')
        args = parser.parse_args()

        # Determine the epsilon value to use
        if args.epsilon is not None:
            epsilon = args.epsilon
            logger.info(f"🔧 Epsilon value provided via command-line argument: epsilon={epsilon}")
        else:
            epsilon = None
            logger.info("🔧 No epsilon value provided; proceeding without Differential Privacy")

        # Load configuration
        logger.info("🔧 Loading configuration...")
        with open('src/config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully.")

        # Define transformations

        # Transform for Wasserstein computation (no resizing)
        wasserstein_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Transform for model training
        training_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load datasets using DatasetLoader
        dataset_name = config.get('dataset', {}).get('name', 'mnist')
        logger.info(f"📚 Loading {dataset_name.upper()} dataset...")

        # For Wasserstein computation
        wasserstein_dataset_loader = DatasetLoader(dataset_name=dataset_name, transform=wasserstein_transform)
        wasserstein_train_dataset = wasserstein_dataset_loader.dataset['train']
        wasserstein_test_dataset = wasserstein_dataset_loader.dataset['test']

        # For training
        training_dataset_loader = DatasetLoader(dataset_name=dataset_name, transform=training_transform)
        train_dataset = training_dataset_loader.dataset['train']
        test_dataset = training_dataset_loader.dataset['test']

        # Limit the training dataset to num_samples
        num_samples = int(config['federated_learning'].get('num_samples', len(train_dataset)))
        if num_samples < len(train_dataset):
            train_dataset = Subset(train_dataset, list(range(num_samples)))
            wasserstein_train_dataset = Subset(wasserstein_train_dataset, list(range(num_samples)))
            logger.info(f"✅ Limited training dataset to {num_samples} samples.")

        # Initialize model class
        logger.info("🧠 Initializing model class...")
        model_name = config['model']['name']

        # Choose model class based on model_name
        if model_name == 'vit':
            from src.models.vit_model import ViTModel
            model_class = ViTModel
        elif model_name == 'resnet':
            from src.models.resnet_model import ResNetModel
            model_class = ResNetModel
        else:
            logger.error(f"Model '{model_name}' not supported.")
            sys.exit(1)

        logger.info(f"Model class '{model_class.__name__}' initialized.")

        # Initialize BlockchainFederatedTrainer (BCFL with possible DP)
        trainer = BlockchainFederatedTrainer(
            config=config,
            model_class=model_class,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            wasserstein_train_dataset=wasserstein_train_dataset,
            wasserstein_test_dataset=wasserstein_test_dataset,
            target_epsilon=epsilon
        )

        # Log if DP is being used
        if trainer.target_epsilon is not None:
            logger.info(f"🏋️ Initializing BlockchainFederatedTrainer with DP (epsilon={trainer.target_epsilon})...")
        else:
            logger.info("🏋️ Initializing BlockchainFederatedTrainer without Differential Privacy...")

        # Start training
        logger.info("🎯 Starting Blockchain-enabled FL (and DP if enabled) training...")
        trainer.train()
        logger.info("Training completed successfully.")

    except Exception as e:
        logger.exception("An error occurred during training.")


if __name__ == "__main__":
    main()
