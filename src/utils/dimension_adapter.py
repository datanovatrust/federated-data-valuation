# src/utils/dimension_adapter.py

import torch

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

        # Move tensor to device
        x = x.to(self.device)

        # If input is more than 2D (e.g., [B,1,28,28] or [1,28,28]), flatten:
        if len(original_shape) == 4:
            # [B,C,H,W] -> [B, C*H*W], for MNIST C=1,H=28,W=28 => [B,784]
            B = original_shape[0]
            x = x.view(B, -1)  # Flatten all except batch
        elif len(original_shape) == 3:
            # [C,H,W] -> [C*H*W], e.g. [1,28,28] => [784]
            x = x.view(-1)
        # If already 2D or 1D, do nothing here:
        # 1D should be [784], 2D should be [B,784]

        # After possible flatten:
        new_shape = x.shape
        if len(new_shape) == 1:
            # Single sample: should be [784]
            if new_shape[0] != 784:
                raise ValueError(f"Expected single sample with 784 features, got {new_shape[0]}. Ensure input is MNIST-like.")
            # Prepare for pooling: [1,1,784]
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(new_shape) == 2:
            # Batch: should be [B,784]
            if new_shape[1] != 784:
                raise ValueError(f"Expected batch of samples with 784 features each, got {new_shape[1]}.")
            # Prepare for pooling: [B,1,784]
            x = x.unsqueeze(1)
        else:
            raise ValueError("After flattening, input must be 1D ([784]) or 2D ([B,784]).")

        # Apply adaptive pooling (from 784 -> 5)
        x = self.adaptive_pool(x)  # [B,1,5] or [1,1,5]

        # Remove channel dimension: [B,5] or [1,5]
        x = x.squeeze(1)

        # If originally single sample (detected if original was 3D or 1D),
        # we now have [1,5], squeeze to [5]
        # Check conditions for single sample:
        was_single_sample = (len(original_shape) == 3) or (len(original_shape) == 1)
        if was_single_sample and x.dim() == 2 and x.size(0) == 1:
            x = x.squeeze(0)

        return x

def modify_zkp_client_wrapper(ZKPClientWrapper):
    """
    Modifies the ZKPClientWrapper class to handle dimension reduction.

    If multiple samples are provided in training_data, it will select the first sample.
    """
    original_generate_training_proof = ZKPClientWrapper.generate_training_proof

    def new_generate_training_proof(self, global_model, local_model, training_data, 
                                    labels, learning_rate, precision, global_hash, local_hash):
        # Ensure data is flattened and on correct device
        device = training_data.device
        adapter = DimensionAdapter(input_size=5, device=device)

        # Reduce dimensions for X
        reduced_data = adapter.reduce_dimensions(training_data)

        # If multiple samples are present (e.g., [B,5]), select the first sample
        if reduced_data.dim() == 2 and reduced_data.size(0) > 1:
            reduced_data = reduced_data[0]

        # If labels is a batch, select the first label as well (circuit expects single Y)
        if labels.dim() > 1 and labels.size(0) > 1:
            labels = labels[0]

        # If labels isn't exactly length 3, truncate to 3 (if larger)
        if labels.numel() > 3:
            labels = labels[:3]

        return original_generate_training_proof(
            self, global_model, local_model, reduced_data, 
            labels, learning_rate, precision, global_hash, local_hash
        )
    
    ZKPClientWrapper.generate_training_proof = new_generate_training_proof
    return ZKPClientWrapper
