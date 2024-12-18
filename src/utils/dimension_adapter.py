import torch
import logging
import sys

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler with a higher debug level
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the logger if it doesn't already have handlers
if not logger.handlers:
    logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

class DimensionAdapter:
    """
    Adapts MNIST-like data (originally [B,1,28,28] or [1,28,28], etc.) 
    into a fixed-size input (5 dimensions) using adaptive average pooling.
    """

    def __init__(self, input_size=5, device='cpu'):
        self.input_size = input_size
        self.device = device
        self.adaptive_pool = torch.nn.AdaptiveAvgPool1d(input_size).to(self.device)

    def reduce_dimensions(self, x):
        """
        Reduces input dimensions to match the circuit requirement (5 features).
        
        Acceptable input shapes:
        - Single image: [C,H,W], typically [1,28,28] for MNIST, or already [784].
        - Batch of images: [B,C,H,W], typically [B,1,28,28] for MNIST, or already [B,784].
        
        Steps:
        1. If more than 2 dims, flatten to [B,784] or [784].
        2. If 1D -> [784], if 2D -> [B,784].
        3. Convert to [B,1,784] or [1,1,784] for adaptive pooling.
        4. Adaptive pooling reduces from 784 to 5: [B,1,5] or [1,1,5].
        5. Squeeze extra dims -> [B,5] or [5].

        Returns:
            torch.Tensor: [5] if single sample, [B,5] if batch.
        """
        original_shape = x.shape
        logger.debug(f"Original input shape: {original_shape}")

        # Move tensor to device
        x = x.to(self.device)

        # If input is more than 2D (e.g., [B,1,28,28] or [1,28,28]), flatten:
        if len(original_shape) == 4:
            # [B,C,H,W] -> [B, C*H*W], for MNIST C=1,H=28,W=28 => [B,784]
            B = original_shape[0]
            x = x.view(B, -1)  # Flatten all except batch
            logger.debug(f"Flattened 4D input to shape: {x.shape}")
        elif len(original_shape) == 3:
            # [C,H,W] -> [C*H*W], e.g. [1,28,28] => [784]
            x = x.view(-1)
            logger.debug(f"Flattened 3D input to shape: {x.shape}")

        # After possible flatten:
        new_shape = x.shape
        if len(new_shape) == 1:
            # Single sample: should be [784]
            if new_shape[0] != 784:
                raise ValueError(f"Expected single sample with 784 features, got {new_shape[0]}. Ensure input is MNIST-like.")
            # Prepare for pooling: [1,1,784]
            x = x.unsqueeze(0).unsqueeze(0)
            logger.debug("Prepared single sample for pooling")
        elif len(new_shape) == 2:
            # Batch: should be [B,784]
            if new_shape[1] != 784:
                raise ValueError(f"Expected batch of samples with 784 features each, got {new_shape[1]}.")
            # Prepare for pooling: [B,1,784]
            x = x.unsqueeze(1)
            logger.debug("Prepared batch for pooling")
        else:
            raise ValueError("After flattening, input must be 1D ([784]) or 2D ([B,784]).")

        # Apply adaptive pooling (from 784 -> 5)
        x = self.adaptive_pool(x)  # [B,1,5] or [1,1,5]
        logger.debug(f"Shape after pooling: {x.shape}")

        # Remove channel dimension: [B,5] or [1,5]
        x = x.squeeze(1)
        logger.debug(f"Shape after removing channel dimension: {x.shape}")

        # If originally single sample (detected if original was 3D or 1D),
        # we now have [1,5], squeeze to [5]
        was_single_sample = (len(original_shape) == 3) or (len(original_shape) == 1)
        if was_single_sample and x.dim() == 2 and x.size(0) == 1:
            x = x.squeeze(0)
            logger.debug("Squeezed single sample to final shape")

        logger.debug(f"Final output shape: {x.shape}")
        return x

class ZKPDimensionHandler:
    """
    Handles dimension adaptation and data preparation for ZKP proofs.
    This separates the ZKP-specific logic from the general dimension adaptation.
    """
    def __init__(self, input_size=5, output_size=3, device='cpu'):
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dimension_adapter = DimensionAdapter(input_size=input_size, device=device)

    def prepare_zkp_data(self, training_data, labels):
        """
        Prepares data specifically for ZKP proof generation.
        """
        logger.debug("=== ZKP Data Preparation Debug ===")
        logger.debug(f"Original training data shape: {training_data.shape}")
        logger.debug(f"Original labels shape: {labels.shape}")

        # Reduce dimensions for training data
        reduced_data = self.dimension_adapter.reduce_dimensions(training_data)
        
        # Handle batch dimension for training data
        if reduced_data.dim() == 2 and reduced_data.size(0) > 1:
            logger.debug("Selecting first sample from batch")
            reduced_data = reduced_data[0]

        # Handle labels
        if labels.dim() > 1 and labels.size(0) > 1:
            logger.debug("Selecting first label from batch")
            labels = labels[0]

        # Ensure labels have correct size
        if labels.numel() > self.output_size:
            logger.debug(f"Truncating labels to size {self.output_size}")
            labels = labels[:self.output_size]

        logger.debug(f"Final reduced data shape: {reduced_data.shape}")
        logger.debug(f"Final labels shape: {labels.shape}")

        return reduced_data, labels

def wrap_zkp_client(ZKPClientWrapper):
    """
    Creates a wrapped version of ZKPClientWrapper that handles dimension adaptation.
    This is now just a thin wrapper that uses ZKPDimensionHandler.
    """
    original_generate_training_proof = ZKPClientWrapper.generate_training_proof

    def new_generate_training_proof(self, global_model, local_model, training_data, 
                                labels, learning_rate, precision, global_hash, local_hash):
        """
        Modified training proof generation with proper dimension handling and signal preservation.
        """
        logger.critical("=== NEW GENERATE TRAINING PROOF STARTED ===")
        logger.critical(f"Input parameters:")
        logger.critical(f"Learning rate: {learning_rate}")
        logger.critical(f"Precision: {precision}")
        logger.critical(f"Global hash: {global_hash}")
        logger.critical(f"Local hash: {local_hash}")
        logger.critical(f"Training data shape: {training_data.shape}")
        logger.critical(f"Labels shape: {labels.shape}")

        # Ensure data is on correct device
        device = training_data.device
        adapter = DimensionAdapter(input_size=5, device=device)

        # Reduce dimensions for training data
        reduced_data = adapter.reduce_dimensions(training_data)
        logger.critical(f"Reduced data shape: {reduced_data.shape}")

        # Handle batch dimension
        if reduced_data.dim() == 2 and reduced_data.size(0) > 1:
            logger.critical("Selecting first sample from batch")
            reduced_data = reduced_data[0]

        # Process labels
        if labels.dim() > 1 and labels.size(0) > 1:
            logger.critical("Selecting first label from batch")
            labels = labels[0]

        # Ensure labels have correct size
        if labels.numel() > 3:
            logger.critical("Truncating labels to size 3")
            labels = labels[:3]

        logger.critical("=== Calling original training proof ===")
        logger.critical(f"Final reduced_data shape: {reduced_data.shape}")
        logger.critical(f"Final labels shape: {labels.shape}")
        logger.critical(f"Learning rate: {learning_rate}")
        logger.critical(f"Precision: {precision}")
        logger.critical(f"Global hash: {global_hash}")
        logger.critical(f"Local hash: {local_hash}")

        # Call original with processed data and named parameters
        result = original_generate_training_proof(
            self,
            global_model=global_model, 
            local_model=local_model,
            training_data=reduced_data,
            labels=labels,
            learning_rate=learning_rate,
            precision=precision,
            global_hash=global_hash,
            local_hash=local_hash
        )

        # Verify result contains required components
        logger.critical("=== Processing result ===")
        if result:
            logger.critical(f"Result keys: {result.keys() if result else 'None'}")
            if 'public' in result:
                logger.critical(f"Public signals in result: {result['public']}")
            if 'proof' in result:
                logger.critical("Proof present in result")
        else:
            logger.critical("No result received from original_generate_training_proof")

        return result

    ZKPClientWrapper.generate_training_proof = new_generate_training_proof
    return ZKPClientWrapper

# For backwards compatibility
modify_zkp_client_wrapper = wrap_zkp_client