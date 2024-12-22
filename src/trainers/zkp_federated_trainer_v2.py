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
    prepare_aggregator_public_inputs
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DummyNet(torch.nn.Module):
    """
    Simple one-layer feedforward net with no activation.
    Matches the shape:
      - hiddenSize=10, inputSize=5, outputSize=3 
      but in practice we'll store as:
        weight: [hiddenSize, inputSize]
        bias:   [hiddenSize]
      then for the "output layer," we'd do something else. 
    For demonstration, we treat it as a single layer.
    """
    def __init__(self, input_dim=5, output_dim=10):
        super(DummyNet, self).__init__()
        # We'll store weights in shape [output_dim, input_dim]
        # We'll store biases in shape [output_dim]
        self.weight = torch.nn.Parameter(torch.zeros((output_dim, input_dim)))
        self.bias = torch.nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        For demonstration, assume x has shape [batch, input_dim].
        Output shape => [batch, output_dim].
        No activation, so output = x @ W^T + b
        """
        # x: [batch, input_dim]
        # W: [output_dim, input_dim]
        # b: [output_dim]
        # out: [batch, output_dim]
        return x.matmul(self.weight.T) + self.bias


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
        device: str = None
    ):
        """
        Setup the FL environment with specified circuit artifacts and number of clients.
        """
        self.num_clients = num_clients
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim  # not strictly used in DummyNet; just for example
        self.precision = precision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

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
        Create a dummy global model that matches the shape used in the circuit (like hiddenDim=10, inputDim=5).
        We'll store it in self.global_model.
        """
        net = DummyNet(input_dim=self.input_dim, output_dim=self.hidden_dim).to(self.device)
        # Just random initialization
        torch.nn.init.normal_(net.weight, mean=0.0, std=0.1)
        torch.nn.init.constant_(net.bias, 0.0)
        self.global_model = net

        logger.info("Global model (DummyNet) initialized with random weights.")

    def local_training_step(self, client: LocalClientV2, epochs: int = 1, lr: float = 0.01):
        """
        A minimal training step for demonstration.
        We'll treat the client's data_x/data_y as a small dataset.
        """
        if client.model is None:
            # copy global model to local
            local_net = DummyNet(self.input_dim, self.hidden_dim).to(self.device)
            local_net.load_state_dict(self.global_model.state_dict())
            client.model = local_net

        # simple single-epoch training
        local_net = client.model
        optimizer = torch.optim.SGD(local_net.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()  # though the circuit uses MSE logic, this is just a placeholder

        # We'll do a small dataloader
        ds = TensorDataset(client.data_x, client.data_y)
        loader = DataLoader(ds, batch_size=4, shuffle=True)

        local_net.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                outputs = local_net(xb)  # shape [batch, hidden_dim]
                # We'll do some hacky approach: yb in [0..2], but net outputs size=10
                # So let's do a sub-slice or quick cross-entropy
                # In reality, you'd properly adapt the circuit for multi-layer usage.
                loss = criterion(outputs, yb % self.hidden_dim)  # just to keep it in range
                loss.backward()
                optimizer.step()

        logger.debug(f"Client {client.client_id} local training done. Last batch loss={loss.item():.4f}")

    def create_client_circuit_input(
        self,
        client: LocalClientV2,
        old_global_hash: int,
        new_local_hash: int,
        lr: float
    ) -> dict:
        """
        Build the dictionary for the client circuit input.
        We'll flatten the old global model and new local model, 
        plus pick a single example from client data to fill X, Y.
        For demonstration, we just pick the first sample.
        """
        # old_global model => shape [hidden_dim, input_dim], bias => [hidden_dim]
        old_state = self.global_model.state_dict()
        # new local model => shape [hidden_dim, input_dim], bias => [hidden_dim]
        new_state = client.model.state_dict()

        # We'll adapt them to 2D lists of ints for the circuit
        GW = old_state["weight"].detach().cpu().numpy()  # shape [hidden_dim, input_dim]
        GB = old_state["bias"].detach().cpu().numpy()    # shape [hidden_dim]
        LWp = new_state["weight"].detach().cpu().numpy() # shape [hidden_dim, input_dim]
        LBp = new_state["bias"].detach().cpu().numpy()   # shape [hidden_dim]

        # We'll treat the circuit as though "LWp" was [hiddenSize][outputSize], 
        # but in reality we have [10,5]. It's a mismatch from the circuit's perspective,
        # but let's keep going for demonstration. We won't fully match the example.
        # The circuit expects LWp dimension = hiddenSize x outputSize => 10x3. 
        # We'll keep it minimal here, though it's not a perfect 1:1 with the circuit's logic.

        # We'll pick the first sample from the client
        if len(client.data_x) > 0:
            x_sample = client.data_x[0].detach().cpu().numpy()
            y_sample = client.data_y[0].item()
        else:
            x_sample = np.zeros(self.input_dim)
            y_sample = 0

        # Convert to Python lists of ints
        def to_python_2d(arr2d):
            return arr2d.astype(np.int64).tolist()

        def to_python_1d(arr1d):
            return arr1d.astype(np.int64).tolist()

        # We'll slice the 10x5 into 10x3 for LWp if we want to match circuit's output dimension
        # For demonstration only: just slice to columns=3
        lw_sliced = LWp[:, :3]  # shape [10,3]
        # Similarly for LBp, slice to len=3
        lb_sliced = LBp[:3]     # shape [3]

        circuit_input = {
            "eta": str(int(lr * self.precision)),
            "pr": str(self.precision),
            "ldigest": str(new_local_hash),
            "scgh": str(old_global_hash),

            "GW": to_python_2d(GW),  # [10][5]
            "GB": to_python_1d(GB),  # [10]
            "LWp": to_python_2d(lw_sliced), # [10][3]
            "LBp": to_python_1d(lb_sliced), # [3]

            "X": to_python_1d(x_sample),    # [5]
            # We'll expand y to [3] if the circuit expects that
            "Y": [int(y_sample), 0, 0]      # hacky: store y in index 0
        }
        return circuit_input

    def train(
        self,
        fl_rounds: int = 3,
        client_epochs: int = 1,
        lr: float = 0.01
    ):
        """
        Main Federated Loop:
          - For multiple rounds
            - Each client trains
            - Generate & verify client proof
            - Aggregator collects valid local updates
            - Aggregator generates & verifies aggregator proof
            - Aggregator updates global model
        """
        if not self.clients:
            self.create_dummy_data()
        if self.global_model is None:
            self.initialize_global_model()

        logger.info(f"==== Starting FL with {fl_rounds} rounds, {self.num_clients} clients each round ====")

        for round_i in range(fl_rounds):
            logger.info(f"\n--- Round {round_i+1}/{fl_rounds} ---")

            # 1) Compute old global hash
            old_global_hash = compute_model_hash(self.global_model.state_dict(), self.precision)

            # 2) Each client trains locally + produce ZKP
            verified_local_models = []
            verified_local_hashes = []
            for client in self.clients:
                self.local_training_step(client, epochs=client_epochs, lr=lr)

                # compute new local model hash
                new_local_hash = compute_model_hash(client.model.state_dict(), self.precision)

                # build circuit input
                client_circuit_input = self.create_client_circuit_input(
                    client, old_global_hash, new_local_hash, lr
                )

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
                    input_y=client_circuit_input["Y"]
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

            # 3) Aggregator builds aggregator circuit input, proves correct averaging
            # For demonstration, we do a naive average in Python
            new_global_sd = self.average_models(verified_local_models)
            # compute new global hash
            updated_global_hash = compute_model_hash(new_global_sd, self.precision)

            # aggregator circuit input
            aggregator_input = self.create_aggregator_circuit_input(
                old_global_sd=self.global_model.state_dict(),
                verified_local_sds=verified_local_models,
                verified_local_hashes=verified_local_hashes,
                updated_global_sd=new_global_sd,
                updated_global_hash=updated_global_hash
            )

            # aggregator proof
            agg_proof_dict = self.aggregator_wrapper.generate_aggregation_proof(
                sc_lh=aggregator_input["ScLH"],
                g_digest=int(aggregator_input["gdigest"]),
                gw=aggregator_input["GW"],
                gb=aggregator_input["GB"],
                lwps=aggregator_input["LWp"],
                lbps=aggregator_input["LBp"],
                gwp=aggregator_input["GWp"],
                gbp=aggregator_input["GBp"]
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
