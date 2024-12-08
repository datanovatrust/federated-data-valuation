# src/trainers/blockchain_federated_trainer.py

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import wasserstein_distance
import logging
import os
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from sklearn.metrics import confusion_matrix
import seaborn as sns
import copy
import yaml

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Import the PrivacyEngine from FastDP
try:
    from src.utils.fastDP.privacy_engine import PrivacyEngine
except ImportError:
    logger.warning("PrivacyEngine not found. Differential Privacy will not be available.")
    PrivacyEngine = None

from src.utils.ipfs_utils import upload_to_ipfs
from src.utils.blockchain_utils import BlockchainClient

import torch.nn.functional as F

class LocalClient:
    """
    Represents a local client in federated learning.
    """
    def __init__(self, client_id, dataset, loader, model=None, optimizer=None, privacy_engine=None):
        self.client_id = client_id
        self.dataset = dataset
        self.loader = loader
        self.model = model
        self.optimizer = optimizer
        self.privacy_engine = privacy_engine

class BlockchainFederatedTrainer:
    """
    A trainer class to handle Blockchain-enabled Federated Learning with DP for privacy.
    Integrates differential privacy (if enabled) and records global model updates on-chain via IPFS.
    """

    def __init__(self, config, model_class, train_dataset, test_dataset,
                 wasserstein_train_dataset, wasserstein_test_dataset, target_epsilon=None):
        self.config = config
        self.model_class = model_class
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.wasserstein_train_dataset = wasserstein_train_dataset
        self.wasserstein_test_dataset = wasserstein_test_dataset

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training parameters
        self.num_clients = int(self.config['federated_learning']['num_clients'])
        self.batch_size = int(self.config['training']['batch_size'])
        self.rounds = int(self.config['federated_learning']['rounds'])
        self.epochs = int(self.config['training']['epochs'])
        self.learning_rate = float(self.config['training']['learning_rate'])
        self.model_name = str(self.config['model']['name'])
        self.fraction_fit = float(self.config['federated_learning']['fraction_fit'])
        self.num_shards = int(self.config['federated_learning'].get('num_shards', 50))
        self.num_classes = int(self.config['model']['num_labels'])

        # Validate num_clients
        if self.num_clients <= 0:
            logger.error("Number of clients must be greater than zero.")
            raise ValueError("Invalid number of clients.")

        # Clients and data loaders
        self.clients = []
        self.client_contributions = []
        self.accuracy_list = []

        # Initialize directories
        self.initialize_directories()

        # Initialize TensorBoard SummaryWriter
        self.writer = SummaryWriter(log_dir='runs')

        # Log training parameters as hyperparameters
        self.writer.add_hparams({
            'num_clients': self.num_clients,
            'batch_size': self.batch_size,
            'rounds': self.rounds,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'model_name': self.model_name,
            'fraction_fit': self.fraction_fit,
        }, {})

        # Differential Privacy parameters
        self.target_epsilon = target_epsilon
        self.use_dp = self.target_epsilon is not None and PrivacyEngine is not None

        # Logging training mode
        logger.info(f"📝 Training parameters: {self.num_clients} clients, {self.rounds} rounds, {self.epochs} epochs per round")
        logger.info(f"Batch size: {self.batch_size}, Learning rate: {self.learning_rate}")
        if self.use_dp:
            logger.info(f"🔒 Differential Privacy enabled with epsilon={self.target_epsilon}")
        else:
            logger.info("🚀 Training with standard Federated Learning")

        # Load Blockchain and IPFS configuration
        self._load_blockchain_config()

    def _load_blockchain_config(self):
        # Load from src/config/blockchain_config.yaml
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'blockchain_config.yaml')
        if not os.path.isfile(config_path):
            logger.error(f"Blockchain config file not found at {config_path}")
            raise FileNotFoundError("blockchain_config.yaml missing.")

        with open(config_path, 'r') as f:
            bc_config = yaml.safe_load(f)

        blockchain_conf = bc_config.get('blockchain', {})
        self.rpc_url = blockchain_conf.get('rpc_url')
        self.contract_address = blockchain_conf.get('contract_address')
        self.abi_file = blockchain_conf.get('abi_file')
        self.ipfs_gateway = blockchain_conf.get('ipfs_gateway')

        if not all([self.rpc_url, self.contract_address, self.abi_file, self.ipfs_gateway]):
            logger.error("Incomplete Blockchain configuration. Check blockchain_config.yaml.")
            raise ValueError("Incomplete Blockchain configuration.")

        # Initialize the Blockchain client without private key (view only) for now.
        # If you want to record updates on-chain, you need a private key for transactions.
        # For demonstration, we assume a PRIVATE_KEY env variable is set.
        private_key = os.getenv("PRIVATE_KEY", None)
        if not private_key:
            logger.warning("No PRIVATE_KEY provided. Blockchain client can only read data.")
        self.blockchain_client = BlockchainClient(self.rpc_url, self.contract_address, self.abi_file, private_key=private_key)

    def initialize_directories(self):
        os.makedirs('logs', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('experiments', exist_ok=True)

    def partition_data(self):
        from src.utils.partitioner import DataPartitioner

        logger.info("🔄 Partitioning data among clients...")

        partitioner = DataPartitioner(
            self.wasserstein_train_dataset,
            self.num_clients,
            partition_type=self.config['federated_learning'].get('partition_type', 'non_iid'),
            num_shards=self.num_shards
        )
        client_datasets = partitioner.partition_data()
        logger.info("✅ Data partitioned among clients successfully.")

        # Create corresponding datasets for training
        client_training_datasets = []
        for client_dataset in client_datasets:
            indices = client_dataset.indices if isinstance(client_dataset, Subset) else list(range(len(client_dataset)))
            client_training_dataset = Subset(self.train_dataset, indices)
            client_training_datasets.append(client_training_dataset)

        # Initialize local clients
        self.clients = []
        for i, dataset in enumerate(client_training_datasets):
            client_batch_size = min(self.batch_size, len(dataset)) if len(dataset) > 0 else self.batch_size
            shuffle = len(dataset) > 0
            loader = DataLoader(dataset, batch_size=client_batch_size, shuffle=shuffle)
            client = LocalClient(client_id=i, dataset=dataset, loader=loader)
            self.clients.append(client)
        logger.info("👥 Local clients set up successfully.")

    def prepare_global_distribution(self):
        num_test_samples = min(len(self.wasserstein_test_dataset), 1000)
        if num_test_samples == 0:
            logger.error("Wasserstein test dataset is empty.")
            raise ValueError("Empty Wasserstein test dataset.")
        validation_loader = DataLoader(
            self.wasserstein_test_dataset, batch_size=num_test_samples, shuffle=False
        )
        validation_data, _ = next(iter(validation_loader))
        global_distribution = validation_data.numpy().reshape(len(validation_data), -1)
        logger.info("🌐 Global distribution prepared successfully.")
        return global_distribution

    def compute_wasserstein_distances(self, global_distribution):
        logger.info("🔍 Computing Wasserstein distances between client datasets and global distribution...")
        client_datasets = [Subset(self.wasserstein_train_dataset, client.dataset.indices) for client in self.clients]

        self.client_contributions = []
        for i, client in enumerate(self.clients):
            client_id = client.client_id
            dataset = client_datasets[i]
            distance = self.compute_wasserstein_distance_client((client_id, dataset, global_distribution))
            self.client_contributions.append(distance)
            self.writer.add_scalar(f'Client_{client_id}/WassersteinDistance', distance[1], 0)

        logger.info("📈 Completed computation of Wasserstein distances")

    @staticmethod
    def compute_wasserstein_distance_client(args):
        client_id, dataset, global_distribution = args
        try:
            if len(dataset) == 0:
                logger.warning(f"Client {client_id} has no data. Assigning maximum distance.")
                return (client_id, float('inf'))

            data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
            data, _ = next(iter(data_loader))

            data = data.numpy().reshape(len(dataset), -1)
            global_data = global_distribution.reshape(len(global_distribution), -1)

            num_bins = 100
            num_features = data.shape[1]
            selected_features = np.random.choice(num_features, size=min(100, num_features), replace=False)

            distances = []
            for feature_idx in selected_features:
                hist_global, bin_edges = np.histogram(global_data[:, feature_idx], bins=num_bins, range=(-1, 1), density=True)
                hist_client, _ = np.histogram(data[:, feature_idx], bins=bin_edges, density=True)

                if hist_global.sum() == 0 or hist_client.sum() == 0:
                    distance = 0.0
                else:
                    distance = wasserstein_distance(hist_global, hist_client)
                distances.append(distance)

            avg_distance = np.mean(distances)
            logger.info(f"📊 Client {client_id} Wasserstein distance: {avg_distance:.4f}")
            return (client_id, avg_distance)
        except Exception as e:
            logger.error(f"Failed to compute Wasserstein distance for client {client_id}: {e}")
            return (client_id, float('inf'))

    def select_clients(self):
        self.client_contributions.sort(key=lambda x: x[1])
        num_selected_clients = max(1, int(self.num_clients * self.fraction_fit))
        selected_clients_indices = [i for i, _ in self.client_contributions[:num_selected_clients]]
        selected_clients = [self.clients[i] for i in selected_clients_indices]
        logger.info(f"🤝 Selected clients: {selected_clients_indices}")
        return selected_clients

    def initialize_global_model(self):
        global_model = self.model_class(num_classes=self.num_classes)
        global_model.to(self.device)
        logger.info(f"🧠 Global model '{self.model_name}' initialized on device: {self.device}")
        return global_model

    def setup_privacy_engine(self, client, optimizer, batch_size):
        if PrivacyEngine is None:
            logger.error("PrivacyEngine not available.")
            raise ImportError("PrivacyEngine not available.")
        privacy_engine = PrivacyEngine(
            module=client.model,
            batch_size=batch_size,
            sample_size=len(client.dataset),
            epochs=self.epochs * self.rounds,
            target_epsilon=self.target_epsilon,
            target_delta=1 / (2 * len(client.dataset)) if len(client.dataset) > 0 else 1e-5,
        )
        privacy_engine.attach(optimizer)
        return privacy_engine

    def train(self):
        logger.info("🚀 Starting Blockchain-enabled Federated Training with DP if enabled...")
        self.partition_data()
        global_distribution = self.prepare_global_distribution()
        self.compute_wasserstein_distances(global_distribution)
        selected_clients = self.select_clients()
        global_model = self.initialize_global_model()

        # Precompute test DataLoader
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=min(cpu_count(), 4),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        logger.info("✅ Test DataLoader prepared.")

        # Initialize client models and optimizers once
        for client in self.clients:
            client.model = self.model_class(num_classes=self.num_classes)
            client.model.to(self.device)

            # Initialize the optimizer
            client.optimizer = torch.optim.Adam(client.model.parameters(), lr=self.learning_rate)

            # Set up the PrivacyEngine if DP is enabled
            if self.use_dp:
                if len(client.dataset) == 0:
                    logger.warning(f"Client {client.client_id} has no data. Skipping privacy engine setup.")
                    client.privacy_engine = None
                else:
                    client_batch_size = min(self.batch_size, len(client.dataset))
                    client.privacy_engine = self.setup_privacy_engine(client, client.optimizer, client_batch_size)

        for round_num in range(1, self.rounds + 1):
            logger.info(f"\n🏁 Round {round_num}/{self.rounds} started")
            local_models = []

            for client in selected_clients:
                if len(client.dataset) == 0:
                    logger.warning(f"🔄 Client {client.client_id} has no data to train.")
                    continue

                client_batch_size = min(self.batch_size, len(client.dataset))
                # Update the client's DataLoader with the adjusted batch size
                client.loader = DataLoader(client.dataset, batch_size=client_batch_size, shuffle=True)

                logger.info(f"🔄 Client {client.client_id}: Training started")
                # Load the global model state into the client's model
                client.model.load_state_dict(global_model.state_dict())

                client.model.train()
                epoch_loss = 0.0
                loss_fn = torch.nn.CrossEntropyLoss()

                # Re-initialize optimizer after loading new model weights
                client.optimizer = torch.optim.Adam(client.model.parameters(), lr=self.learning_rate)

                # Re-attach privacy engine if needed
                if self.use_dp and client.privacy_engine is not None:
                    client.privacy_engine.attach(client.optimizer)

                for epoch in range(1, self.epochs + 1):
                    for data, target in client.loader:
                        data = data.to(self.device, non_blocking=True)
                        target = target.to(self.device, non_blocking=True)

                        client.optimizer.zero_grad()
                        output = client.model(data)
                        if hasattr(output, 'logits'):
                            output = output.logits
                        loss = loss_fn(output, target)
                        loss.backward()
                        client.optimizer.step()

                        epoch_loss += loss.item()

                    avg_loss = epoch_loss / len(client.loader)
                    logger.info(f"📚 Client {client.client_id}: Epoch {epoch}/{self.epochs} Loss: {avg_loss:.4f}")

                    # Log training loss for each client
                    self.writer.add_scalar(
                        f'Client_{client.client_id}/Train/Loss',
                        avg_loss,
                        (round_num - 1) * self.epochs + epoch
                    )

                local_models.append(client.model.state_dict())
                logger.info(f"✅ Client {client.client_id}: Training completed")

            if not local_models:
                logger.warning("No models were trained in this round. Skipping aggregation.")
                continue

            # Aggregate models
            logger.info("📥 Aggregating local models...")
            global_model = self.aggregate_models(global_model, local_models)

            # Save the global model checkpoint locally
            checkpoint_path = self.save_checkpoint(global_model, round_num)

            # Upload global model checkpoint to IPFS and record on the Blockchain
            ipfs_hash = upload_to_ipfs(checkpoint_path)
            if self.blockchain_client.account:
                # Record update on chain only if PRIVATE_KEY is provided
                logger.info("🔗 Recording global model update on Blockchain...")
                self.blockchain_client.record_update(round_num, ipfs_hash)

            # Log model weights histograms
            for name, param in global_model.named_parameters():
                self.writer.add_histogram(name, param, round_num)

            # Evaluate global model
            self.evaluate(global_model, test_loader, round_num)

        logger.info("\n🎉 Federated training with Blockchain and DP completed!")
        self.plot_results()
        # Close the SummaryWriter
        self.writer.close()

        # Report privacy spent for each selected client if DP was used
        if self.use_dp:
            for client in selected_clients:
                if client.privacy_engine is not None:
                    epsilon_spent = client.privacy_engine.get_privacy_spent()
                    logger.info(f"🔒 Client {client.client_id} privacy spent: {epsilon_spent}")

            # Detach privacy engine after training
            for client in self.clients:
                if client.privacy_engine is not None:
                    client.privacy_engine.detach()

    def aggregate_models(self, global_model, local_models):
        avg_state_dict = {}
        for key in global_model.state_dict().keys():
            stacked_params = torch.stack([model[key].float() for model in local_models], dim=0)
            avg_state_dict[key] = stacked_params.mean(dim=0)
        global_model.load_state_dict(avg_state_dict)
        logger.info("🔗 Model aggregation completed")
        return global_model

    def save_checkpoint(self, model, round_num):
        checkpoint_path = f'checkpoints/global_model_round_{round_num}.pt'
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"💾 Saved model checkpoint to {checkpoint_path}")
        return checkpoint_path

    def evaluate(self, model, test_loader, round_num):
        logger.info("🔎 Evaluating the global model...")
        model.eval()
        correct = 0
        total = 0
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                outputs = model(data)
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                all_targets.extend(target.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        accuracy = 100 * correct / total if total > 0 else 0
        self.accuracy_list.append(accuracy)
        logger.info(f"🌟 Accuracy after round {round_num}: {accuracy:.2f}%")

        # Log accuracy to TensorBoard
        self.writer.add_scalar('GlobalModel/Accuracy', accuracy, round_num)

        # Save confusion matrix
        self.plot_confusion_matrix(all_targets, all_predictions, round_num)

    def plot_confusion_matrix(self, true_labels, pred_labels, round_num):
        logger.info("🖼️ Generating confusion matrix...")
        try:
            cm = confusion_matrix(true_labels, pred_labels)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - Round {round_num}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'experiments/confusion_matrix_round_{round_num}.png')
            plt.close()
            logger.info(f"🖼️ Saved confusion matrix plot to 'experiments/confusion_matrix_round_{round_num}.png'")
        except Exception as e:
            logger.error(f"Failed to generate confusion matrix: {e}")

    def plot_results(self):
        logger.info("📊 Plotting client contributions...")
        try:
            client_ids = [i for i, _ in self.client_contributions]
            distances = [d for _, d in self.client_contributions]
            plt.figure(figsize=(10, 6))
            plt.bar(client_ids, distances, color='skyblue')
            plt.xlabel('Client ID')
            plt.ylabel('Wasserstein Distance')
            plt.title('Client Contributions')
            plt.xticks(client_ids)
            plt.savefig('experiments/client_contributions.png')
            plt.close()
            logger.info("🖼️ Saved client contributions plot to 'experiments/client_contributions.png'")
        except Exception as e:
            logger.error(f"Failed to plot client contributions: {e}")

        logger.info("📈 Plotting training accuracy over rounds...")
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.accuracy_list) + 1), self.accuracy_list, marker='o', linestyle='-', color='green')
            plt.xlabel('Round')
            plt.ylabel('Accuracy (%)')
            plt.title('Global Model Accuracy Over Rounds')
            plt.xticks(range(1, len(self.accuracy_list) + 1))
            plt.grid(True)
            plt.savefig('experiments/training_accuracy.png')
            plt.close()
            logger.info("🖼️ Saved training accuracy plot to 'experiments/training_accuracy.png'")
        except Exception as e:
            logger.error(f"Failed to plot training accuracy: {e}")
