# src/trainers/peft_federated_trainer.py

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

# PEFT imports
from peft import LoraConfig, get_peft_model

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers if they don't exist
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

class FederatedTrainer:
    """
    Trainer class to handle federated training with PEFT and LoRA-A².
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

        # PEFT parameters
        self.lora_rank = int(self.config['peft']['lora_rank'])
        self.lora_alpha = int(self.config['peft']['lora_alpha'])
        self.lora_dropout = float(self.config['peft']['lora_dropout'])
        self.target_modules = self.config['peft']['target_modules']
        self.adaptive_rank = self.config['peft'].get('adaptive_rank', False)
        self.local_rank_budget = int(self.config['peft'].get('local_rank_budget', self.lora_rank))

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
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'adaptive_rank': self.adaptive_rank,
            'local_rank_budget': self.local_rank_budget,
        }, {})

        # Differential Privacy parameters
        self.target_epsilon = target_epsilon
        self.use_dp = self.target_epsilon is not None and PrivacyEngine is not None

        # Log training parameters
        logger.info(f"📝 Training parameters: {self.num_clients} clients, {self.rounds} rounds, {self.epochs} epochs per round")
        logger.info(f"Batch size: {self.batch_size}, Learning rate: {self.learning_rate}")
        logger.info(f"PEFT LoRA parameters: rank={self.lora_rank}, alpha={self.lora_alpha}, dropout={self.lora_dropout}")

        # Logging to indicate the training mode
        if self.use_dp:
            logger.info(f"🔒 Differential Privacy enabled with epsilon={self.target_epsilon}")
        else:
            logger.info("🚀 Training with standard Federated Learning")

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
            shuffle = len(dataset) > 0  # Only shuffle if dataset is not empty
            loader = DataLoader(dataset, batch_size=client_batch_size, shuffle=shuffle)
            client = LocalClient(client_id=i, dataset=dataset, loader=loader)
            self.clients.append(client)
        logger.info("👥 Local clients set up successfully.")

    def prepare_global_distribution(self):
        num_test_samples = min(len(self.wasserstein_test_dataset), 1000)  # Adjust as needed
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
        # Initialize the base model
        global_model = self.model_class(num_classes=self.num_classes)

        # Wrap the model with PEFT's LoRA
        peft_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CLASSIFICATION"
        )
        global_model.model = get_peft_model(global_model.model, peft_config)
        global_model.to(self.device)
        logger.info(f"🧠 Global model '{self.model_name}' with PEFT LoRA initialized on device: {self.device}")
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

        # Initialize client models and optimizers once
        for client in self.clients:
            client.model = copy.deepcopy(global_model)
            client.model.to(self.device)

            # Initialize the optimizer
            trainable_params = [p for p in client.model.parameters() if p.requires_grad]
            client.optimizer = torch.optim.Adam(trainable_params, lr=self.learning_rate)

            # Set up the PrivacyEngine if differential privacy is enabled
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

            # Determine whether to train B or A
            train_B = round_num % 2 == 1  # Train B in odd rounds
            train_A = not train_B

            for client in selected_clients:
                if len(client.dataset) == 0:
                    logger.warning(f"🔄 Client {client.client_id} has no data to train.")
                    continue

                client_batch_size = min(self.batch_size, len(client.dataset))

                # Update the client's DataLoader with the adjusted batch size
                client.loader = DataLoader(
                    client.dataset,
                    batch_size=client_batch_size,
                    shuffle=True
                )

                logger.info(f"🔄 Client {client.client_id}: Training started")
                # Load the global model state into the client's model
                client.model.load_state_dict(global_model.state_dict())

                # Freeze and unfreeze LoRA parameters accordingly
                for name, param in client.model.named_parameters():
                    if 'lora_B' in name:
                        param.requires_grad = train_B
                    elif 'lora_A' in name:
                        param.requires_grad = train_A
                    else:
                        param.requires_grad = False  # Freeze other parameters

                # Initialize optimizer with only trainable parameters
                trainable_params = [p for p in client.model.parameters() if p.requires_grad]
                client.optimizer = torch.optim.Adam(trainable_params, lr=self.learning_rate)

                # Re-attach privacy engine to optimizer
                if self.use_dp and client.privacy_engine is not None:
                    client.privacy_engine.attach(client.optimizer)

                client.model.train()

                # Perform a forward and backward pass on a batch to compute gradients
                client.optimizer.zero_grad()
                try:
                    data, target = next(iter(client.loader))
                except StopIteration:
                    logger.warning(f"Client {client.client_id} has no data in loader.")
                    continue  # Skip if no data

                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                output = client.model(data)
                if hasattr(output, 'logits'):
                    output = output.logits
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(output, target)
                loss.backward()

                # Now compute importance scores
                if self.adaptive_rank:
                    importance_scores = self.compute_importance_scores(client)
                    self.apply_rank_selection_and_masking(client, importance_scores)

                # Zero gradients again before training
                client.optimizer.zero_grad()

                epoch_loss = 0.0

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

                # Collect only the updated parameters for aggregation
                local_models.append(copy.deepcopy(client.model.state_dict()))
                logger.info(f"✅ Client {client.client_id}: Training completed")

            if not local_models:
                logger.warning("No models were trained in this round. Skipping aggregation.")
                continue

            # Aggregate models
            logger.info("📥 Aggregating local models...")
            global_model = self.aggregate_models(global_model, local_models, train_B=train_B, train_A=train_A)

            # Log model weights histograms
            for name, param in global_model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(name, param, round_num)

            self.save_checkpoint(global_model, round_num)
            self.evaluate(global_model, test_loader, round_num)

        logger.info("\n🎉 Federated training completed!")
        self.plot_results()
        # Close the SummaryWriter
        self.writer.close()

        # Report privacy spent for each client
        if self.use_dp:
            for client in selected_clients:
                if client.privacy_engine is not None:
                    epsilon_spent = client.privacy_engine.get_privacy_spent()
                    logger.info(f"🔒 Client {client.client_id} privacy spent: {epsilon_spent}")

            # Detach privacy engine after training
            for client in self.clients:
                if client.privacy_engine is not None:
                    client.privacy_engine.detach()

    def compute_importance_scores(self, client):
        importance_scores = {}
        for name, module in client.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Initialize scores for this module
                scores = None

                # Iterate over LoRA parameters in lora_A and lora_B
                for key in module.lora_A.keys():
                    lora_A_param = module.lora_A[key]
                    lora_B_param = module.lora_B[key]

                    # Ensure gradients are computed
                    if lora_A_param.weight.grad is None or lora_B_param.weight.grad is None:
                        continue  # Skip if gradients are None

                    delta_B = lora_B_param.weight.grad  # Gradient of B
                    A = lora_A_param.weight.data
                    delta_A = lora_A_param.weight.grad
                    B = lora_B_param.weight.data

                    # Compute importance per rank
                    r = delta_B.shape[1]  # Rank
                    importance_per_rank = []
                    for i in range(r):
                        delta_B_i = delta_B[:, i]  # Shape (out_features,)
                        A_i = A[i, :]              # Shape (in_features,)
                        delta_A_i = delta_A[i, :]  # Shape (in_features,)
                        B_i = B[:, i]              # Shape (out_features,)

                        # Compute importance for rank i
                        importance_B_i = torch.norm(delta_B_i) * torch.norm(A_i)
                        importance_A_i = torch.norm(B_i) * torch.norm(delta_A_i)

                        importance_i = (importance_B_i + importance_A_i) / 2
                        importance_per_rank.append(importance_i)

                    # Convert to tensor
                    module_scores = torch.tensor(importance_per_rank, device=self.device)
                    # Accumulate scores
                    if scores is None:
                        scores = module_scores
                    else:
                        scores += module_scores

                if scores is not None:
                    importance_scores[name] = scores
        return importance_scores

    def apply_rank_selection_and_masking(self, client, importance_scores):
        for name, module in client.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                scores = importance_scores.get(name)
                if scores is not None:
                    # Select top ranks based on local_rank_budget
                    top_ranks = torch.topk(scores, k=self.local_rank_budget)[1]

                    # Iterate over LoRA parameters
                    for key in module.lora_A.keys():
                        lora_A_param = module.lora_A[key]
                        lora_B_param = module.lora_B[key]

                        # Create masks
                        mask_B = torch.zeros_like(lora_B_param.weight)
                        mask_A = torch.zeros_like(lora_A_param.weight)
                        mask_B[:, top_ranks] = 1
                        mask_A[top_ranks, :] = 1

                        # Apply masks during training only if the parameter requires grad
                        if lora_B_param.weight.requires_grad:
                            lora_B_param.weight.register_hook(lambda grad: grad * mask_B)
                        if lora_A_param.weight.requires_grad:
                            lora_A_param.weight.register_hook(lambda grad: grad * mask_A)

    def aggregate_models(self, global_model, local_models, train_B=True, train_A=False):
        avg_state_dict = global_model.state_dict()
        for key in avg_state_dict.keys():
            if ('lora_B' in key and train_B) or ('lora_A' in key and train_A):
                stacked_params = torch.stack([model[key].float() for model in local_models], dim=0)
                avg_state_dict[key] = stacked_params.mean(dim=0)
            else:
                avg_state_dict[key] = avg_state_dict[key]
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
