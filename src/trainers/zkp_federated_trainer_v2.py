"""
src/trainers/zkp_federated_trainer_v2.py

A "from-scratch" Zero-Knowledge Proof-enabled Federated Trainer using our v2 circuits and
v2 ZKP utilities.

Key Features:
    - Demonstrates a simple Federated Learning process where multiple clients:
        (1) Download the current global model
        (2) Train locally (1 epoch or more)
        (3) Generate a ZKP proof that their new local model is computed correctly
        (4) Submit proof + local model hash to aggregator

    - Aggregator:
        (1) Collects verified local models (where proof checks out)
        (2) Averages them into a new global model
        (3) Generates a ZKP proof that aggregation is correct
        (4) Publishes the new global model + proof

    - This is a minimal example focusing on ZKP integration. 
      In practice, you'd have more sophisticated FL logic, data partitioning, 
      model definitions, etc.

Dependencies:
    - Torch for local training
    - zkp_utils_v2.py for proof generation & verification
    - aggregator_v2.circom and client_v2.circom circuits
    - Anvil or Ganache or Hardhat node for on-chain testing (optional)
"""

import os
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# You can adjust these imports depending on your project structure
from src.utils.zkp_utils_v2 import (
    ZKPClientWrapperV2,
    ZKPAggregatorWrapperV2,
    compute_model_hash,
    prepare_client_public_inputs,
    prepare_aggregator_public_inputs,
    float_to_fixed,
    normalize_to_field,
    FIELD_PRIME,
    tensor_to_field
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DummyNet(torch.nn.Module):
    """
    Simple one-layer feedforward net with no activation.
    Maintains floating point weights for gradients but handles integer conversion for field arithmetic.
    """
    def __init__(self, input_dim=5, output_dim=10):
        super(DummyNet, self).__init__()
        # Initialize as float32 for gradients
        self.weight = torch.nn.Parameter(torch.zeros((output_dim, input_dim)))
        self.bias = torch.nn.Parameter(torch.zeros(output_dim))
        self.true_output_dim = min(output_dim, 3)

    def get_quantized_params(self):
        """Get integer-quantized parameters for field arithmetic"""
        with torch.no_grad():
            w_int = torch.round(self.weight).to(torch.int64)
            b_int = torch.round(self.bias).to(torch.int64)
        return w_int, b_int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass handling both integer and float inputs.
        Returns floating point outputs for gradient computation.
        """
        # Convert input to float if needed
        x_f = x.float() if x.dtype == torch.int64 else x
        return x_f.matmul(self.weight.T) + self.bias


class LocalClientV2:
    """
    Simple container representing a single client in our FL system.
    """
    def __init__(self, client_id: int, data_x: torch.Tensor, data_y: torch.Tensor):
        self.client_id = client_id
        self.data_x = data_x
        self.data_y = data_y
        # We'll dynamically assign a local model
        self.model = None

    def __repr__(self):
        return f"LocalClientV2(id={self.client_id}, data_size={len(self.data_x)})"


class ZKPFederatedTrainerV2:
    """
    A simple example of how to orchestrate an FL process with ZKP verification.

    Steps:
     1) Initialize clients with local data.
     2) Create a global model (DummyNet).
     3) For each round:
        a) Each selected client loads global model, trains locally for 1 epoch (or more).
        b) Generate ZKP that this local update is correct (calls client circuit).
        c) Aggregator verifies the client proof. If valid, aggregator collects local model.
        d) Aggregator runs aggregator circuit to prove correct averaging -> new global model.
        e) All participants accept the new global model.
    """

    def __init__(
        self,
        num_clients: int = 4,
        input_dim: int = 5,
        hidden_dim: int = 10,
        output_dim: int = 3,
        precision: int = 1000,
        client_wasm_path: str = "./client_js/client.wasm",
        client_zkey_path: str = "./client_0000.zkey",
        client_vkey_path: str = "./build/circuits/client_vkey.json",
        aggregator_wasm_path: str = "./aggregator_js/aggregator.wasm",
        aggregator_zkey_path: str = "./aggregator_0000.zkey",
        aggregator_vkey_path: str = "./build/circuits/aggregator_vkey.json",
        device: str = None,
        debug_dir: str = None
    ):
        self.num_clients = num_clients
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.precision = precision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.debug_dir = debug_dir

        # ZKP wrappers
        self.client_wrapper = ZKPClientWrapperV2(
            circuit_wasm_path=client_wasm_path,
            proving_key_path=client_zkey_path,
            vkey_json_path=client_vkey_path,
            precision=self.precision
        )
        self.aggregator_wrapper = ZKPAggregatorWrapperV2(
            circuit_wasm_path=aggregator_wasm_path,
            proving_key_path=aggregator_zkey_path,
            vkey_json_path=aggregator_vkey_path
        )

        self.clients = []
        self.global_model = None

        logger.info(f"ZKPFederatedTrainerV2 initialized on device={self.device}.")
        if debug_dir:
            logger.info(f"Debug output will be saved to: {debug_dir}")

    def create_dummy_data(self):
        """
        Create a small synthetic dataset for demonstration.
        We'll distribute it among self.num_clients.
        Each client gets a small portion.
        """
        total_samples = 40  # total dataset size
        X_all = torch.randn(total_samples, self.input_dim)
        # For demonstration, let's do random integer labels for output_dim=3 classification
        # but we only have a hiddenDim=10 in the net. We'll keep it simple anyway.
        Y_all = torch.randint(0, 3, (total_samples,))

        # We'll just store X, Y as is. The circuit expects Y to have shape = output_size (like 3),
        # but for demonstration we do single-value classification. We'll keep it super minimal.
        # For a real match to the circuit, you'd want Y to be shape [3], or do a partial approach.

        # Partition among clients
        subset_size = total_samples // self.num_clients
        self.clients = []
        idx_start = 0
        for cid in range(self.num_clients):
            idx_end = idx_start + subset_size
            data_x = X_all[idx_start:idx_end]
            data_y = Y_all[idx_start:idx_end]
            idx_start = idx_end

            client = LocalClientV2(cid, data_x, data_y)
            self.clients.append(client)

        logger.info(f"Dummy data created with {total_samples} samples. {len(self.clients)} clients assigned.")

    def initialize_global_model(self):
        """
        Initialize model with safe field values.
        """
        net = DummyNet(input_dim=self.input_dim, output_dim=self.hidden_dim).to(self.device)
        
        # Use a smaller scale to avoid overflow
        init_scale = min(self.precision / 100.0, 1000.0)
        
        with torch.no_grad():
            # Initialize weights with smaller values
            torch.nn.init.uniform_(net.weight, -0.1, 0.1)
            net.weight.data *= init_scale
            net.weight.data = torch.round(net.weight.data)
            
            # Initialize biases
            torch.nn.init.uniform_(net.bias, -0.1, 0.1)
            net.bias.data *= init_scale
            net.bias.data = torch.round(net.bias.data)
            
            # Convert to field values
            net.weight.data = tensor_to_field(net.weight.data)
            net.bias.data = tensor_to_field(net.bias.data)
            
            # Convert large field values to smaller representations
            net.weight.data = torch.remainder(net.weight.data, 1e6).float()
            net.bias.data = torch.remainder(net.bias.data, 1e6).float()
        
        self.global_model = net
        
        logger.info(f"Global model initialized with field-normalized weights (precision={self.precision})")
        logger.debug(f"Weight range: [{net.weight.min().item():.2f}, {net.weight.max().item():.2f}]")
        logger.debug(f"Bias range: [{net.bias.min().item():.2f}, {net.bias.max().item():.2f}]")

    def local_training_step(self, client: LocalClientV2, epochs: int = 1, lr: float = 0.01):
        """
        Training step with safe field arithmetic.
        """
        if client.model is None:
            local_net = DummyNet(self.input_dim, self.hidden_dim).to(self.device)
            local_net.load_state_dict(self.global_model.state_dict())
            client.model = local_net

        local_net = client.model
        optimizer = torch.optim.SGD(local_net.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        # Use a smaller learning rate scaling to avoid overflow
        scaled_lr = min(lr * self.precision / 100.0, 1000.0)

        ds = TensorDataset(client.data_x, client.data_y)
        loader = DataLoader(ds, batch_size=4, shuffle=True)

        local_net.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                
                # Scale inputs safely
                xb_scaled = torch.round(torch.clamp(xb * self.precision / 100.0, -1e6, 1e6))
                
                # Forward pass
                outputs = local_net(xb_scaled)
                outputs = outputs[:, :self.output_dim]
                
                # Create safe targets
                y_onehot = torch.zeros(yb.size(0), self.output_dim, device=self.device)
                y_onehot.scatter_(1, yb.unsqueeze(1), self.precision / 100.0)
                
                # Compute loss
                loss = criterion(outputs, y_onehot)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Safe parameter updates
                with torch.no_grad():
                    for param in local_net.parameters():
                        param.grad *= scaled_lr
                        param.data -= param.grad
                        # Round and normalize to field
                        param.data = torch.round(param.data)
                        param.data = tensor_to_field(param.data)
                        # Keep values manageable
                        param.data = torch.remainder(param.data, 1e6).float()

            logger.debug(f"Client {client.client_id} epoch complete. Loss={loss.item():.4f}")

    def create_client_circuit_input(
        self,
        client: LocalClientV2,
        old_global_hash: int,
        new_local_hash: int,
        lr: float
    ) -> dict:
        """
        Build the dictionary for the client circuit input with proper field normalization.
        """
        # Helper function to normalize tensor values to field
        def normalize_tensor(tensor):
            return tensor.detach().cpu().numpy().astype(np.int64) % FIELD_PRIME

        # Get model states
        old_state = self.global_model.state_dict()
        new_state = client.model.state_dict()

        # Convert and normalize tensors
        GW = normalize_tensor(torch.round(old_state["weight"]))  # [hidden_dim, input_dim]
        GB = normalize_tensor(torch.round(old_state["bias"]))    # [hidden_dim]
        LWp = normalize_tensor(torch.round(new_state["weight"])) # [hidden_dim, input_dim]
        LBp = normalize_tensor(torch.round(new_state["bias"]))   # [hidden_dim]

        # Get a sample with proper scaling and normalization
        if len(client.data_x) > 0:
            x_sample = normalize_tensor(torch.round(client.data_x[0] * self.precision))
            y_val = client.data_y[0].item()
            # Create one-hot Y with proper scaling
            y_sample = np.zeros(self.output_dim)
            y_sample[y_val % self.output_dim] = self.precision
            y_sample = y_sample.astype(np.int64) % FIELD_PRIME
        else:
            x_sample = np.zeros(self.input_dim, dtype=np.int64)
            y_sample = np.zeros(self.output_dim, dtype=np.int64)

        # Convert normalized values to lists
        def to_int_list(arr):
            return arr.tolist()

        # Slice LWp to match output dimension
        lw_sliced = LWp[:, :self.output_dim]  # [hidden_dim, output_dim]
        lb_sliced = LBp[:self.output_dim]     # [output_dim]

        circuit_input = {
            "eta": str(float_to_fixed(lr, self.precision)),
            "pr": str(self.precision),
            "ldigest": str(normalize_to_field(new_local_hash)),
            "scgh": str(normalize_to_field(old_global_hash)),
            "GW": [to_int_list(row) for row in GW],  # [hidden_dim][input_dim]
            "GB": to_int_list(GB),                    # [hidden_dim]
            "LWp": [to_int_list(row) for row in lw_sliced], # [hidden_dim][output_dim]
            "LBp": to_int_list(lb_sliced),                  # [output_dim]
            "X": to_int_list(x_sample),                     # [input_dim]
            "Y": to_int_list(y_sample)                      # [output_dim]
        }

        # Verify all values are within field
        def verify_field_range(name, value):
            if isinstance(value, list):
                for i, subval in enumerate(value):
                    if isinstance(subval, list):
                        for j, val in enumerate(subval):
                            if val < 0 or val >= FIELD_PRIME:
                                logger.error(f"{name}[{i}][{j}] = {val} outside field range!")
                    else:
                        if subval < 0 or subval >= FIELD_PRIME:
                            logger.error(f"{name}[{i}] = {subval} outside field range!")
            else:
                if value < 0 or value >= FIELD_PRIME:
                    logger.error(f"{name} = {value} outside field range!")

        for key, value in circuit_input.items():
            if key not in ["eta", "pr", "ldigest", "scgh"]:  # Skip string values
                verify_field_range(key, value)

        # Log the shapes for verification
        logger.debug("Circuit input shapes:")
        logger.debug(f"GW: {len(circuit_input['GW'])}x{len(circuit_input['GW'][0])}")
        logger.debug(f"GB: {len(circuit_input['GB'])}")
        logger.debug(f"LWp: {len(circuit_input['LWp'])}x{len(circuit_input['LWp'][0])}")
        logger.debug(f"LBp: {len(circuit_input['LBp'])}")
        logger.debug(f"X: {len(circuit_input['X'])}")
        logger.debug(f"Y: {len(circuit_input['Y'])}")

        return circuit_input

    def train(
        self,
        fl_rounds: int = 3,
        client_epochs: int = 1,
        lr: float = 0.01,
        debug_dir: str = None
    ):
        """
        Main Federated Loop with debugging support
        """
        # Update debug directory if provided
        if debug_dir:
            self.debug_dir = debug_dir
            logger.info(f"Updated debug directory to: {debug_dir}")

        if not self.clients:
            self.create_dummy_data()
        if self.global_model is None:
            self.initialize_global_model()

        logger.info(f"==== Starting FL with {fl_rounds} rounds, {self.num_clients} clients each round ====")

        for round_i in range(fl_rounds):
            round_debug_dir = None
            if self.debug_dir:
                round_debug_dir = os.path.join(self.debug_dir, f"round_{round_i}")
                os.makedirs(round_debug_dir, exist_ok=True)
                logger.info(f"Created debug directory for round {round_i}: {round_debug_dir}")

            logger.info(f"\n--- Round {round_i+1}/{fl_rounds} ---")

            # 1) Compute old global hash
            old_global_hash = compute_model_hash(self.global_model.state_dict(), self.precision)

            # 2) Each client trains locally + produce ZKP
            verified_local_models = []
            verified_local_hashes = []
            for client_idx, client in enumerate(self.clients):
                self.local_training_step(client, epochs=client_epochs, lr=lr)

                # compute new local model hash
                new_local_hash = compute_model_hash(client.model.state_dict(), self.precision)

                # build circuit input
                client_circuit_input = self.create_client_circuit_input(
                    client, old_global_hash, new_local_hash, lr
                )

                # Set up client-specific debug directory
                client_debug_dir = None
                if round_debug_dir:
                    client_debug_dir = os.path.join(round_debug_dir, f"client_{client_idx}")
                    os.makedirs(client_debug_dir, exist_ok=True)

                # generate proof
                proof_dict = self.client_wrapper.generate_training_proof(
                    old_global_hash=old_global_hash,
                    new_local_hash=new_local_hash,
                    learning_rate=lr,
                    input_gw=client_circuit_input["GW"],
                    input_gb=client_circuit_input["GB"],
                    input_lwp=client_circuit_input["LWp"],
                    input_lbp=client_circuit_input["LBp"],
                    input_x=client_circuit_input["X"],
                    input_y=client_circuit_input["Y"],
                    debug_dir=client_debug_dir
                )

                # verify off-chain
                is_valid = self.client_wrapper.verify_client_proof(proof_dict)
                if is_valid:
                    logger.info(f"Client {client.client_id} proof verified. Accept local model.")
                    verified_local_models.append(client.model.state_dict())
                    verified_local_hashes.append(new_local_hash)
                else:
                    logger.warning(f"Client {client.client_id} proof invalid. Skipping update.")

            if not verified_local_models:
                logger.warning("No valid local models found this round; global model remains the same.")
                continue

            # 3) Aggregator proves correct averaging
            new_global_sd = self.average_models(verified_local_models)
            updated_global_hash = compute_model_hash(new_global_sd, self.precision)

            aggregator_input = self.create_aggregator_circuit_input(
                old_global_sd=self.global_model.state_dict(),
                verified_local_sds=verified_local_models,
                verified_local_hashes=verified_local_hashes,
                updated_global_sd=new_global_sd,
                updated_global_hash=updated_global_hash
            )

            # Set up aggregator debug directory
            agg_debug_dir = None
            if round_debug_dir:
                agg_debug_dir = os.path.join(round_debug_dir, "aggregator")
                os.makedirs(agg_debug_dir, exist_ok=True)

            # aggregator proof
            agg_proof_dict = self.aggregator_wrapper.generate_aggregation_proof(
                sc_lh=aggregator_input["ScLH"],
                g_digest=int(aggregator_input["gdigest"]),
                gw=aggregator_input["GW"],
                gb=aggregator_input["GB"],
                lwps=aggregator_input["LWp"],
                lbps=aggregator_input["LBp"],
                gwp=aggregator_input["GWp"],
                gbp=aggregator_input["GBp"],
                debug_dir=agg_debug_dir
            )

            # verify aggregator proof
            agg_valid = self.aggregator_wrapper.verify_aggregator_proof(agg_proof_dict)
            if agg_valid:
                logger.info("Aggregator proof verified. Updating global model.")
                self.global_model.load_state_dict(new_global_sd)
            else:
                logger.warning("Aggregator proof invalid. Discarding this global update.")

        logger.info("=== FL process completed. ===")

    def average_models(self, list_of_state_dicts):
        """
        Simple Python average of weight & bias parameters among a list of state_dicts.
        """
        if not list_of_state_dicts:
            raise ValueError("No state dicts provided to average.")

        # We'll assume they're all the same shape
        ref_sd = {k: v.clone() for k, v in list_of_state_dicts[0].items()}
        for k in ref_sd.keys():
            for i in range(1, len(list_of_state_dicts)):
                ref_sd[k] += list_of_state_dicts[i][k]
            ref_sd[k] /= len(list_of_state_dicts)
        return ref_sd

    def create_aggregator_circuit_input(
        self,
        old_global_sd: dict,
        verified_local_sds: list,
        verified_local_hashes: list,
        updated_global_sd: dict,
        updated_global_hash: int
    ) -> dict:
        """
        Build the aggregator circuit input matching aggregator_v2.circom:
        {
          "ScLH": [hash1, hash2, ...], // verified local models
          "gdigest": updated_global_hash,
          "GW": <old global weights>, "GB": <old global biases>,
          "LWp": <array of local weights>,
          "LBp": <array of local biases>,
          "GWp": <updated global weights>, "GBp": <updated global biases>
        }
        We'll do 2D arrays for old global, 3D arrays for local, etc.
        """
        # old global => shape [10,5], [10]
        oldW = old_global_sd["weight"].detach().cpu().numpy()
        oldB = old_global_sd["bias"].detach().cpu().numpy()

        # each local => shape [10,5], [10], but aggregator_v2 expects shape [numClients][10][5], [numClients][10]
        localW = []
        localB = []
        for sd in verified_local_sds:
            w_np = sd["weight"].detach().cpu().numpy()
            b_np = sd["bias"].detach().cpu().numpy()
            localW.append(w_np)
            localB.append(b_np)

        # updated global => shape [10,5], [10]
        newW = updated_global_sd["weight"].detach().cpu().numpy()
        newB = updated_global_sd["bias"].detach().cpu().numpy()

        # Convert them to Python int arrays
        def to2d(arr2d):
            return arr2d.astype(np.int64).tolist()

        def to1d(arr1d):
            return arr1d.astype(np.int64).tolist()

        # localW => shape [numClients][hiddenSize][inputSize]
        # localB => shape [numClients][hiddenSize]
        localW_3d = [to2d(w) for w in localW]
        localB_2d = [to1d(b) for b in localB]

        aggregator_input = {
            "ScLH": verified_local_hashes,
            "gdigest": str(updated_global_hash),
            "GW": to2d(oldW),
            "GB": to1d(oldB),
            "LWp": localW_3d,
            "LBp": localB_2d,
            "GWp": to2d(newW),
            "GBp": to1d(newB)
        }
        return aggregator_input
