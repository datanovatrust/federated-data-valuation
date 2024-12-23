#!/usr/bin/env python3

"""
scripts/train_zkp_federated_v2.py

An example script that demonstrates how to run our Zero-Knowledge Proof-enabled
Federated Learning trainer (ZKPFederatedTrainerV2). This script:

1. Initializes the trainer with the appropriate circuit/witness/proving key paths.
2. Creates synthetic data and a dummy global model.
3. Runs a specified number of federated learning rounds.
4. Generates & verifies client proofs and aggregator proofs at each round.

Requires:
    - src/trainers/zkp_federated_trainer_v2.py
    - circuits/*_v2.circom compiled to WASM + zkey files
    - snarkjs on PATH
    - Node.js environment for generate_witness.js
"""

import os
import sys
import argparse
import logging

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Adjust if your project structure differs
from src.trainers.zkp_federated_trainer_v2 import ZKPFederatedTrainerV2

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def main():
    parser = argparse.ArgumentParser(description="ZKP Federated Learning (v2) Demo")
    parser.add_argument("--rounds", type=int, default=3, help="Number of FL rounds.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of local epochs per round.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for local updates.")
    parser.add_argument("--num_clients", type=int, default=4, help="Number of clients.")
    parser.add_argument("--precision", type=int, default=1000, help="Fixed-point precision for hashing.")

    # Updated defaults to point to the v2 artifacts (instead of the old v1 files).
    parser.add_argument(
        "--client_wasm",
        type=str,
        default="./client_v2_js/client_v2.wasm",
        help="Path to the Client v2 WASM file."
    )
    parser.add_argument(
        "--client_zkey",
        type=str,
        default="./client_v2_0000.zkey",
        help="Path to the Client v2 proving key (zkey)."
    )
    parser.add_argument(
        "--client_vkey",
        type=str,
        default="./build/circuits_v2/client_v2_vkey.json",
        help="Path to the Client v2 verification key (JSON)."
    )
    parser.add_argument(
        "--aggregator_wasm",
        type=str,
        default="./aggregator_v2_js/aggregator_v2.wasm",
        help="Path to the Aggregator v2 WASM file."
    )
    parser.add_argument(
        "--aggregator_zkey",
        type=str,
        default="./aggregator_v2_0000.zkey",
        help="Path to the Aggregator v2 proving key (zkey)."
    )
    parser.add_argument(
        "--aggregator_vkey",
        type=str,
        default="./build/circuits_v2/aggregator_v2_vkey.json",
        help="Path to the Aggregator v2 verification key (JSON)."
    )
    args = parser.parse_args()

    # Basic logging info
    logger.info("=== Starting ZKP Federated Learning (v2) ===")
    logger.info(f"Rounds: {args.rounds}, Epochs: {args.epochs}, LR: {args.lr}, Clients: {args.num_clients}")
    logger.info(f"Precision: {args.precision}")

    logger.info(f"Client WASM: {args.client_wasm}, ZKey: {args.client_zkey}, VKey: {args.client_vkey}")
    logger.info(f"Aggregator WASM: {args.aggregator_wasm}, ZKey: {args.aggregator_zkey}, VKey: {args.aggregator_vkey}")

    # Initialize trainer
    trainer = ZKPFederatedTrainerV2(
        num_clients=args.num_clients,
        precision=args.precision,
        client_wasm_path=args.client_wasm,
        client_zkey_path=args.client_zkey,
        client_vkey_path=args.client_vkey,
        aggregator_wasm_path=args.aggregator_wasm,
        aggregator_zkey_path=args.aggregator_zkey,
        aggregator_vkey_path=args.aggregator_vkey
    )

    # Create synthetic data
    trainer.create_dummy_data()

    # Initialize global model
    trainer.initialize_global_model()

    # Run FL
    trainer.train(
        fl_rounds=args.rounds,
        client_epochs=args.epochs,
        lr=args.lr
    )

    logger.info("=== ZKP Federated Learning (v2) finished ===")

if __name__ == "__main__":
    main()
