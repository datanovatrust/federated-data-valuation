# scripts/run_rmia_attack.py

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import Subset, DataLoader, ConcatDataset
from src.attacks.rmia_attack import RMIAttack
from src.attacks.config import RMIAConfig
from src.attacks.reference_model_manager import ReferenceModelManager
from src.utils.dataset_loader import DatasetLoader
import yaml
import logging
from torchvision import transforms
import argparse
import numpy as np
from tqdm import tqdm  # For progress bar

# ----------------- Logging Configuration -----------------

# Configure the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Or DEBUG, as needed

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('logs/rmia_attack.log')
file_handler.setLevel(logging.DEBUG)  # Log more details to the file

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)

# Add formatter to handlers
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the root logger
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# Now get the logger for this module
logger = logging.getLogger(__name__)

# ----------------- Main Execution Block -----------------


def main():
    logger.info("🚀 Starting the RMIA Attack Script")
    try:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description='RMIA Attack Script')
        parser.add_argument('--epsilon', type=float, default=None,
                            help='Target epsilon for differential privacy (default: None)')
        parser.add_argument('--gamma', type=float, default=None,
                            help='Threshold gamma for likelihood ratio test (default from config)')
        parser.add_argument('--beta', type=float, default=None,
                            help='Threshold beta for membership inference score (default from config)')
        parser.add_argument('--num_reference_models', type=int, default=None,
                            help='Number of reference models to use (default from config)')
        parser.add_argument('--num_z_samples', type=int, default=None,
                            help='Number of population samples z (default from config)')
        parser.add_argument('--model_checkpoint', type=str, default='checkpoints/global_model_round_5.pt',
                            help='Path to the trained model checkpoint')
        args = parser.parse_args()

        # Load configuration
        logger.info("🔧 Loading configuration...")
        with open('src/config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully.")

        # Ensure experiments directory exists
        experiments_dir = 'experiments'
        if not os.path.exists(experiments_dir):
            os.makedirs(experiments_dir)
            logger.info(f"Created experiments directory at '{experiments_dir}'.")

        # Override RMIAConfig parameters with command-line arguments if provided
        rmia_config = RMIAConfig()
        if args.gamma is not None:
            rmia_config.GAMMA = args.gamma
            logger.info(f"🔧 Overriding gamma value: GAMMA={rmia_config.GAMMA}")
        if args.beta is not None:
            rmia_config.BETA = args.beta
            logger.info(f"🔧 Overriding beta value: BETA={rmia_config.BETA}")
        if args.num_reference_models is not None:
            rmia_config.NUM_REFERENCE_MODELS = args.num_reference_models
            logger.info(f"🔧 Overriding number of reference models: NUM_REFERENCE_MODELS={rmia_config.NUM_REFERENCE_MODELS}")
        if args.num_z_samples is not None:
            rmia_config.NUM_Z_SAMPLES = args.num_z_samples
            logger.info(f"🔧 Overriding number of z samples: NUM_Z_SAMPLES={rmia_config.NUM_Z_SAMPLES}")

        # Define transformations
        attack_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load datasets using DatasetLoader
        dataset_name = config.get('dataset', {}).get('name', 'mnist')
        logger.info(f"📚 Loading {dataset_name.upper()} dataset for the attack...")

        dataset_loader = DatasetLoader(dataset_name=dataset_name, transform=attack_transform)
        full_dataset = dataset_loader.dataset['train']  # Assuming train dataset was used for training

        # Split the dataset into members and non-members
        num_samples = len(full_dataset)
        indices = list(range(num_samples))
        np.random.shuffle(indices)

        # Define the desired number of samples to attack
        subset_size = 2  # For example, attack 200 samples in total

        # Ensure subset_size is even
        if subset_size % 2 != 0:
            subset_size -= 1

        # Select subset indices
        member_indices = indices[:subset_size // 2]
        non_member_indices = indices[subset_size // 2: subset_size]

        member_dataset = Subset(full_dataset, member_indices)
        non_member_dataset = Subset(full_dataset, non_member_indices)

        # Combine datasets and create labels
        attack_dataset = ConcatDataset([member_dataset, non_member_dataset])
        membership_labels = [1] * len(member_dataset) + [0] * len(non_member_dataset)

        # Create DataLoader for the attack dataset
        attack_data_loader = DataLoader(attack_dataset, batch_size=1, shuffle=False)

        # Load the trained target model
        model_checkpoint = args.model_checkpoint
        logger.info(f"🧠 Loading the trained target model from {model_checkpoint}...")
        model_name = config['model']['name']
        if model_name == 'vit':
            from src.models.vit_model import ViTModel
            model_class = ViTModel
        elif model_name == 'resnet':
            from src.models.resnet_model import ResNetModel
            model_class = ResNetModel
        else:
            logger.error(f"Model '{model_name}' not supported.")
            sys.exit(1)

        target_model = model_class()
        target_model.load_state_dict(torch.load(model_checkpoint))
        target_model.eval()
        logger.info("Target model loaded successfully.")

        # Ensure target_model has predict_proba method
        if not hasattr(target_model, 'predict_proba'):
            logger.error("The target model does not have a 'predict_proba' method.")
            sys.exit(1)

        # Initialize ReferenceModelManager
        logger.info("🗄️ Initializing ReferenceModelManager...")
        reference_manager = ReferenceModelManager()
        reference_models = []

        # Load or train reference models
        # For simplicity, we'll assume that pre-trained reference models are available
        reference_model_paths = []  # Add paths to pre-trained reference models
        if reference_model_paths:
            logger.info("🔄 Loading pre-trained reference models...")
            reference_manager.load_pretrained_models(reference_model_paths)
            reference_models = reference_manager.get_reference_models()
            logger.info(f"{len(reference_models)} reference models loaded.")
        else:
            logger.info("⚠️ No pre-trained reference models provided.")
            # Optionally, train reference models here
            # For simplicity, let's create dummy reference models
            logger.info("🛠️ Training reference models...")
            for _ in range(rmia_config.NUM_REFERENCE_MODELS):
                ref_model = model_class()
                # Here you should train the ref_model on different data
                # For now, we'll load the same weights (this is not ideal)
                ref_model.load_state_dict(torch.load(model_checkpoint))
                ref_model.eval()
                reference_models.append(ref_model)
            logger.info(f"{len(reference_models)} reference models trained.")

        # Initialize DataSampler
        logger.info("📊 Initializing DataSampler...")
        from src.attacks.data_sampler import DataSampler
        data_sampler = DataSampler(dataset=full_dataset)

        # Initialize the RMIAttack
        logger.info("🛠️ Initializing RMIAttack...")
        attack = RMIAttack(
            target_model=target_model,
            reference_models=reference_models,
            config=rmia_config,
            data_sampler=data_sampler
        )
        logger.info("RMIAttack initialized successfully.")

        # Perform the attack
        logger.info("🎯 Starting the RMIA attack...")
        results = attack.perform_attack(attack_data_loader, membership_labels)
        logger.info("RMIA attack completed.")

        # Evaluate the attack
        logger.info("📈 Evaluating the attack results...")
        metrics = attack.evaluate(results)
        logger.info(f"AUC Score: {metrics['auc']:.4f}")

        # Plot ROC Curve
        logger.info("📊 Plotting ROC Curve...")
        from src.attacks.evaluation_metrics import plot_roc_curve

        # Define the save path for the ROC curve
        plot_path = os.path.join(experiments_dir, 'rmia_roc_curve.png')
        plot_roc_curve(metrics['fpr'], metrics['tpr'], title='RMIA Attack ROC Curve', save_path=plot_path)
        logger.info(f"ROC Curve saved to '{plot_path}'.")

    except Exception as e:
        logger.exception("An error occurred during the RMIA attack.")


if __name__ == "__main__":
    main()
