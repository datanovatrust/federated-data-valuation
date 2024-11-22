# scripts/train_federated.py

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader, Subset
from src.models.model import ImageClassifier
from src.utils.data_loader import get_mnist_datasets, partition_dataset_non_iid
import yaml
import numpy as np
import logging
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from torchvision import transforms
from multiprocessing import Pool, cpu_count

# ----------------- Logging Configuration -----------------

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler('logs/federated_training.log')

# Set level for handlers
console_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.INFO)

# Create formatter with emojis
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Add formatter to handlers
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ----------------- Function Definitions -----------------

def compute_wasserstein_distance_client(args):
    """
    Compute Wasserstein distance for a single client.

    Parameters:
    - args: Tuple containing (client_id, client_dataset, global_distribution)

    Returns:
    - Tuple of (client_id, distance)
    """
    client_id, dataset, global_distribution = args
    try:
        if len(dataset) == 0:
            logger.warning(f"Client {client_id} has no data. Assigning maximum distance.")
            return (client_id, float('inf'))
        
        # Create DataLoader with batch_size equal to dataset size
        data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        data, _ = next(iter(data_loader))
        
        # Convert to numpy and reshape to (num_samples, features)
        data = data.numpy().reshape(len(dataset), -1)
        global_data = global_distribution.reshape(len(global_distribution), -1)
        
        # Compute Wasserstein distance between per-feature distributions
        # To reduce computation, compute distances on histograms
        # Define number of bins for histograms
        num_bins = 100
        
        # Initialize list to store per-feature distances
        distances = []
        
        for feature_idx in range(global_data.shape[1]):
            # Compute histograms for the feature with range (-1, 1)
            hist_global, bin_edges = np.histogram(global_data[:, feature_idx], bins=num_bins, range=(-1, 1), density=True)
            hist_client, _ = np.histogram(data[:, feature_idx], bins=bin_edges, density=True)
            
            # Handle cases where histograms sum to zero to prevent division by zero
            if hist_global.sum() == 0 or hist_client.sum() == 0:
                logger.warning(f"Client {client_id}, Feature {feature_idx}: Histogram sum is zero. Assigning distance 0.")
                distance = 0.0
            else:
                # Compute Wasserstein distance between histograms
                distance = wasserstein_distance(hist_global, hist_client)
            distances.append(distance)
        
        # Average the distances across all features
        avg_distance = np.mean(distances)
        
        logger.info(f"📊 Client {client_id} Wasserstein distance: {avg_distance:.4f}")
        return (client_id, avg_distance)
    except Exception as e:
        logger.error(f"Failed to compute Wasserstein distance for client {client_id}: {e}")
        return (client_id, float('inf'))

def compute_wasserstein_distances(client_datasets, global_distribution):
    logger.info("🔍 Computing Wasserstein distances between client datasets and global distribution...")
    # Prepare arguments for parallel computation
    args = [(i, client_datasets[i], global_distribution) for i in range(len(client_datasets))]
    
    # Utilize multiprocessing Pool for parallel computation
    try:
        with Pool(processes=cpu_count()) as pool:
            client_contributions = pool.map(compute_wasserstein_distance_client, args)
    except Exception as e:
        logger.error(f"Error during parallel Wasserstein distance computation: {e}")
        # Fallback to sequential computation
        client_contributions = []
        for arg in args:
            client_contributions.append(compute_wasserstein_distance_client(arg))
    
    logger.info("📈 Completed computation of Wasserstein distances")
    return client_contributions

# ----------------- Main Execution Block -----------------

def main():
    # 🚀 Starting the Federated Training Script
    logger.info("🚀 Starting the Federated Training Script")
    
    # Load configuration
    logger.info("🔧 Loading configuration...")
    try:
        with open('src/config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully.")
    except FileNotFoundError:
        logger.error("Configuration file 'config.yaml' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        sys.exit(1)
    
    # Define transformation to convert grayscale to RGB and resize
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ViT's expected input size
        transforms.Grayscale(num_output_channels=3),  # Convert 1-channel to 3-channel
        transforms.ToTensor(),  # Convert PIL Image to Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize all three channels
    ])
    
    # Load datasets with the new transform
    logger.info("📚 Loading datasets...")
    try:
        train_dataset, test_dataset = get_mnist_datasets(transform=transform)
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        sys.exit(1)
    
    # Limit the training dataset to num_samples
    try:
        num_samples = int(config['federated_learning'].get('num_samples', 100))  # Default to 100 if not set
        if num_samples < len(train_dataset):
            train_dataset = Subset(train_dataset, list(range(num_samples)))
            logger.info(f"✅ Limited training dataset to {num_samples} samples.")
        else:
            logger.info("✅ Using the full training dataset.")
    except ValueError as e:
        logger.error(f"Invalid 'num_samples' value in configuration: {e}")
        sys.exit(1)
    
    # Parameters
    try:
        num_clients = int(config['federated_learning']['num_clients'])
        batch_size = int(config['training']['batch_size'])
        rounds = int(config['federated_learning']['rounds'])
        epochs = int(config['training']['epochs'])
        learning_rate = float(config['training']['learning_rate'])  # Ensure it's a float
        model_name = str(config['model']['name'])
        fraction_fit = float(config['federated_learning']['fraction_fit'])
        num_shards = int(config['federated_learning'].get('num_shards', 50))  # Default to 50 if not set
        
        # Validate parameters
        if num_clients <= 0 or rounds <= 0 or epochs <= 0 or batch_size <= 0:
            logger.error("Numeric parameters must be positive integers.")
            sys.exit(1)
        if not (0.0 < fraction_fit <= 1.0):
            logger.error("fraction_fit must be between 0 (exclusive) and 1 (inclusive).")
            sys.exit(1)
    except KeyError as e:
        logger.error(f"Missing configuration parameter: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid configuration parameter type: {e}")
        sys.exit(1)
    
    logger.info(f"📝 Configuration: {num_clients} clients, {rounds} rounds, {epochs} epochs per round")
    
    # Partition data among clients (non-IID)
    logger.info("🔄 Partitioning data among clients (non-IID)...")
    try:
        client_datasets = partition_dataset_non_iid(train_dataset, num_clients, num_shards=num_shards)
        client_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in client_datasets]
    except Exception as e:
        logger.error(f"Failed to partition data among clients: {e}")
        sys.exit(1)
    
    # Simulate clients locally
    logger.info("👥 Setting up local clients...")
    class LocalClient:
        def __init__(self, client_id, dataset, loader):
            self.client_id = client_id
            self.dataset = dataset
            self.loader = loader
            self.model = None

    clients = []
    for i in range(num_clients):
        try:
            client = LocalClient(client_id=i, dataset=client_datasets[i], loader=client_loaders[i])
            if len(client.dataset) == 0:
                logger.warning(f"Client {i} has an empty dataset.")
            clients.append(client)
        except Exception as e:
            logger.error(f"Failed to initialize client {i}: {e}")
            sys.exit(1)
    
    logger.info("✅ Local clients set up successfully")
    
    # Implement FedBary method with optimizations
    # Prepare global distribution (e.g., validation set)
    logger.info("🌐 Preparing global distribution from validation set...")
    try:
        validation_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        validation_data, _ = next(iter(validation_loader))
        global_distribution = validation_data.numpy().reshape(len(test_dataset), -1)
        logger.info("Global distribution prepared successfully.")
    except Exception as e:
        logger.error(f"Failed to prepare global distribution: {e}")
        sys.exit(1)
    
    # Compute client contributions
    client_contributions = compute_wasserstein_distances(client_datasets, global_distribution)
    
    # Select clients based on contributions
    logger.info("🎯 Selecting clients based on Wasserstein distances...")
    try:
        client_contributions.sort(key=lambda x: x[1])  # Sort by distance (lower is better)
        num_selected_clients = max(1, int(num_clients * fraction_fit))
        selected_clients_indices = [i for i, _ in client_contributions[:num_selected_clients]]
        logger.info(f"🤝 Selected clients: {selected_clients_indices}")
    except Exception as e:
        logger.error(f"Failed to select clients based on contributions: {e}")
        sys.exit(1)
    
    # Initialize the global model
    logger.info(f"🧠 Initializing the global model '{model_name}'...")
    try:
        global_model = ImageClassifier(model_name=model_name, num_classes=10)
        # Move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        global_model.to(device)
        logger.info(f"Global model initialized on device: {device}")
    except Exception as e:
        logger.error(f"Failed to initialize the global model: {e}")
        sys.exit(1)
    
    accuracy_list = []
    
    # Create directories for logs, checkpoints, and experiments
    try:
        os.makedirs('logs', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('experiments', exist_ok=True)
        logger.info("Directories for logs, checkpoints, and experiments are ready.")
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        sys.exit(1)
    
    # 🔎 **Precompute the Test DataLoader Outside the Training Loop**
    try:
        # Use a larger batch size for faster evaluation
        eval_batch_size = 256  # Adjust as per your system's memory capacity
        # Set num_workers to the number of CPU cores for faster data loading
        eval_num_workers = cpu_count()
        # Enable pin_memory if using GPU to speed up data transfer
        eval_pin_memory = True if device.type == 'cuda' else False
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=eval_num_workers,
            pin_memory=eval_pin_memory
        )
        logger.info("✅ Test DataLoader optimized and created successfully.")
    except Exception as e:
        logger.error(f"Failed to create optimized Test DataLoader: {e}")
        sys.exit(1)
    
    # Training rounds
    for round in range(rounds):
        logger.info(f"\n🏁 Round {round+1}/{rounds} started")
        selected_clients = [clients[i] for i in selected_clients_indices]
        
        local_models = []
        for client in selected_clients:
            if len(client.dataset) == 0:
                logger.warning(f"🔄 Client {client.client_id} has no data to train.")
                continue  # Skip clients with no data
            
            logger.info(f"🔄 Client {client.client_id}: Training started")
            try:
                # Clone global model to client
                client.model = ImageClassifier(model_name=model_name, num_classes=10)
                client.model.load_state_dict(global_model.state_dict())
                client.model.to(device)
                
                # Define loss and optimizer
                loss_fn = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(client.model.parameters(), lr=learning_rate)
                
                # Training loop
                client.model.train()
                epoch_loss = 0.0
                for epoch in range(epochs):
                    for data, target in client.loader:
                        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                        
                        optimizer.zero_grad()
                        output = client.model(data)
                        if hasattr(output, 'logits'):
                            output = output.logits  # For models from transformers
                        loss = loss_fn(output, target)
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                    
                    avg_loss = epoch_loss / len(client.loader)
                    logger.info(f"📚 Client {client.client_id}: Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")
                
                # Append the updated model's state_dict
                local_models.append(client.model.state_dict())
                logger.info(f"✅ Client {client.client_id}: Training completed")
            except Exception as e:
                logger.error(f"Error during training on client {client.client_id}: {e}")
                continue  # Skip faulty client
        
        if len(local_models) == 0:
            logger.warning("No models were trained in this round. Skipping aggregation.")
            continue
        
        # Aggregate models (simple average)
        logger.info("📥 Aggregating local models...")
        try:
            avg_state_dict = {}
            for key in global_model.state_dict().keys():
                # Stack and compute mean across clients
                stacked_params = torch.stack([model[key].float() for model in local_models], dim=0)
                avg_state_dict[key] = stacked_params.mean(dim=0)
            global_model.load_state_dict(avg_state_dict)
            logger.info("🔗 Model aggregation completed")
        except Exception as e:
            logger.error(f"Failed to aggregate models: {e}")
            continue  # Proceed to next round
        
        # Save checkpoint
        try:
            checkpoint_path = f'checkpoints/global_model_round_{round+1}.pt'
            torch.save(global_model.state_dict(), checkpoint_path)
            logger.info(f"💾 Saved model checkpoint to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save model checkpoint: {e}")
        
        # 🔎 **Optimized Evaluation of the Global Model**
        logger.info("🔎 Evaluating the global model...")
        try:
            global_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                    outputs = global_model(data)
                    if hasattr(outputs, 'logits'):
                        outputs = outputs.logits  # For models from transformers
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            accuracy = 100 * correct / total
            accuracy_list.append(accuracy)
            logger.info(f"🌟 Accuracy after round {round+1}: {accuracy:.2f}%")
        except Exception as e:
            logger.error(f"Failed to evaluate the global model: {e}")
        
        # Optional: Dynamically select clients for next round based on current contributions
        # This can be implemented based on specific federated learning strategies
    
    logger.info("\n🎉 Training completed!")
    
    # At the end of training
    # Plot client contributions
    logger.info("📊 Plotting client contributions...")
    try:
        client_ids = [i for i, _ in client_contributions]
        distances = [d for _, d in client_contributions]
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
    
    # Plot training accuracy over rounds
    logger.info("📈 Plotting training accuracy over rounds...")
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, rounds + 1), accuracy_list, marker='o', linestyle='-', color='green')
        plt.xlabel('Round')
        plt.ylabel('Accuracy (%)')
        plt.title('Global Model Accuracy Over Rounds')
        plt.xticks(range(1, rounds + 1))
        plt.grid(True)
        plt.savefig('experiments/training_accuracy.png')
        plt.close()
        logger.info("🖼️ Saved training accuracy plot to 'experiments/training_accuracy.png'")
    except Exception as e:
        logger.error(f"Failed to plot training accuracy: {e}")
    
    logger.info("🚀 Federated Training Script completed successfully!")

if __name__ == "__main__":
    main()
