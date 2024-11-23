# src/trainers/federated_trainer.py

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter
from scipy.stats import wasserstein_distance
import logging
import os
import matplotlib.pyplot as plt
from multiprocessing import cpu_count

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG level to capture all logs

# Create handlers if they don't exist
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class LocalClient:
    """
    Represents a local client in federated learning.
    """

    def __init__(self, client_id, dataset, loader):
        self.client_id = client_id
        self.dataset = dataset
        self.loader = loader
        self.model = None

class FederatedTrainer:
    """
    Trainer class to handle federated training.
    """

    def __init__(self, config, model_class, train_dataset, test_dataset,
                 wasserstein_train_dataset, wasserstein_test_dataset):
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

        # Log training parameters
        logger.info(f"📝 Training parameters: {self.num_clients} clients, {self.rounds} rounds, {self.epochs} epochs per round")
        logger.info(f"Batch size: {self.batch_size}, Learning rate: {self.learning_rate}")

    def initialize_directories(self):
        os.makedirs('logs', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('experiments', exist_ok=True)

    def partition_data(self):
        from src.utils.partitioner import DataPartitioner

        logger.info("🔄 Partitioning data among clients (non-IID)...")

        partitioner = DataPartitioner(
            self.wasserstein_train_dataset,  # Use the Wasserstein dataset for partitioning
            self.num_clients,
            partition_type=self.config['federated_learning'].get('partition_type', 'non_iid'),
            num_shards=self.num_shards
        )
        client_datasets = partitioner.partition_data()
        logger.info("✅ Data partitioned among clients successfully.")

        # Create corresponding datasets for training
        client_training_datasets = []
        for client_dataset in client_datasets:
            # Get indices from the Wasserstein dataset
            indices = client_dataset.indices if isinstance(client_dataset, Subset) else list(range(len(client_dataset)))
            # Create a Subset of the training dataset with these indices
            client_training_dataset = Subset(self.train_dataset, indices)
            client_training_datasets.append(client_training_dataset)
        client_loaders = [
            DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for dataset in client_training_datasets
        ]
        # Initialize local clients
        self.clients = [
            LocalClient(client_id=i, dataset=client_training_datasets[i], loader=client_loaders[i])
            for i in range(self.num_clients)
        ]
        logger.info("👥 Local clients set up successfully.")

    def prepare_global_distribution(self):
        num_test_samples = 1000  # Adjust as needed
        validation_loader = DataLoader(
            self.wasserstein_test_dataset, batch_size=num_test_samples, shuffle=False
        )
        validation_data, _ = next(iter(validation_loader))
        global_distribution = validation_data.numpy().reshape(len(validation_data), -1)
        logger.info("🌐 Global distribution prepared successfully.")
        return global_distribution

    def compute_wasserstein_distances(self, global_distribution):
        logger.info("🔍 Computing Wasserstein distances between client datasets and global distribution...")
        # Use the Wasserstein datasets for distance computation
        client_datasets = [Subset(self.wasserstein_train_dataset, client.dataset.indices) for client in self.clients]

        # Compute distances without multiprocessing
        self.client_contributions = []
        for i, client in enumerate(self.clients):
            client_id = client.client_id
            dataset = client_datasets[i]
            distance = self.compute_wasserstein_distance_client((client_id, dataset, global_distribution))
            self.client_contributions.append(distance)
            # Log Wasserstein distance for each client
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

            # To reduce computation, compute distances on a subset of features
            # For example, randomly select 100 features
            num_features = data.shape[1]
            selected_features = np.random.choice(num_features, size=min(100, num_features), replace=False)

            distances = []

            for feature_idx in selected_features:
                hist_global, bin_edges = np.histogram(
                    global_data[:, feature_idx], bins=num_bins, range=(-1, 1), density=True
                )
                hist_client, _ = np.histogram(
                    data[:, feature_idx], bins=bin_edges, density=True
                )

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
        from src.models.image_classifier import ModelFactory

        global_model = ModelFactory.create_model(
            model_name=self.model_name, num_classes=self.num_classes
        )
        global_model.to(self.device)
        logger.info(f"🧠 Global model '{self.model_name}' initialized on device: {self.device}")
        return global_model

    def train(self):
        logger.info("🚀 Starting federated training...")
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

        # Initialize client models once
        for client in self.clients:
            client.model = self.model_class(num_classes=self.num_classes)
            client.model.to(self.device)

        for round_num in range(1, self.rounds + 1):
            logger.info(f"\n🏁 Round {round_num}/{self.rounds} started")
            local_models = []

            for client in selected_clients:
                if len(client.dataset) == 0:
                    logger.warning(f"🔄 Client {client.client_id} has no data to train.")
                    continue

                logger.info(f"🔄 Client {client.client_id}: Training started")
                # Load the global model state into the client's model
                client.model.load_state_dict(global_model.state_dict())

                # Training code remains the same
                loss_fn = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(client.model.parameters(), lr=self.learning_rate)

                client.model.train()
                epoch_loss = 0.0

                for epoch in range(1, self.epochs + 1):
                    for data, target in client.loader:
                        data = data.to(self.device, non_blocking=True)
                        target = target.to(self.device, non_blocking=True)

                        optimizer.zero_grad()
                        output = client.model(data)
                        if hasattr(output, 'logits'):
                            output = output.logits
                        loss = loss_fn(output, target)
                        loss.backward()
                        optimizer.step()

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

            # Log model weights histograms
            for name, param in global_model.named_parameters():
                self.writer.add_histogram(name, param, round_num)

            self.save_checkpoint(global_model, round_num)
            self.evaluate(global_model, test_loader, round_num)

        logger.info("\n🎉 Federated training completed!")
        self.plot_results()
        # Close the SummaryWriter
        self.writer.close()

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

    def evaluate(self, model, test_loader, round_num):
        logger.info("🔎 Evaluating the global model...")
        model.eval()
        correct = 0
        total = 0
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

        accuracy = 100 * correct / total
        self.accuracy_list.append(accuracy)
        logger.info(f"🌟 Accuracy after round {round_num}: {accuracy:.2f}%")

        # Log accuracy to TensorBoard
        self.writer.add_scalar('GlobalModel/Accuracy', accuracy, round_num)

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
            plt.plot(range(1, self.rounds + 1), self.accuracy_list, marker='o', linestyle='-', color='green')
            plt.xlabel('Round')
            plt.ylabel('Accuracy (%)')
            plt.title('Global Model Accuracy Over Rounds')
            plt.xticks(range(1, self.rounds + 1))
            plt.grid(True)
            plt.savefig('experiments/training_accuracy.png')
            plt.close()
            logger.info("🖼️ Saved training accuracy plot to 'experiments/training_accuracy.png'")
        except Exception as e:
            logger.error(f"Failed to plot training accuracy: {e}")
