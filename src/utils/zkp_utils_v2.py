"""
src/utils/zkp_utils_v2.py

A from-scratch Zero-Knowledge Proof utilities module supporting:
    - MiMC-based hashing (must match our circom circuits)
    - Client proof generation (for verifying local training correctness)
    - Aggregator proof generation (for verifying global model aggregation)
    - Hash computations for PyTorch model parameters
    - Public input preparation
    - On-chain style verification stubs

Works in tandem with:
    circuits/client_v2.circom
    circuits/aggregator_v2.circom
    circuits/mimc_hash_v2.circom

Requires:
    - snarkjs (installed and on PATH)
    - Node.js environment for witness generation
    - A matching version of the MiMC functions/circuits in the .circom files

Disclaimer:
    - This is a minimal example. For production usage, handle errors, logs,
      security, ephemeral files, and environment checks more robustly.
"""

import os
import json
import logging
import subprocess
import tempfile
from typing import List, Dict, Any
import torch


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# =============================================================================
# Constants for the Ethereum (bn128) prime field used by Groth16 on many EVMs
# =============================================================================
FIELD_PRIME = 21888242871839275222246405745257275088548364400416034343698204186575808495617

# =============================================================================
# Utility Routines
# =============================================================================
def normalize_to_field(value: int) -> int:
    """
    Normalize an integer into the finite field [0..FIELD_PRIME-1],
    also handling negative values or large positives.
    """
    # This mod operation ensures the result is in [0, FIELD_PRIME-1].
    val_mod = value % FIELD_PRIME
    return val_mod

def float_to_fixed(val: float, precision: int) -> int:
    """
    Convert a Python float to an integer with scale factor = precision.
    Example: float_to_fixed(0.123, 1000) => 123
    Then store it mod FIELD_PRIME.
    """
    scaled = int(round(val * precision))
    return normalize_to_field(scaled)

def tensor_to_fixed(tensor: torch.Tensor, precision: int) -> List[int]:
    """
    Flatten a PyTorch tensor and convert each element to a field-scaled integer.
    """
    arr = tensor.detach().cpu().numpy().flatten()
    return [float_to_fixed(float(v), precision) for v in arr]


# =============================================================================
# MiMC Hash in Python (must match the circom version)
# =============================================================================
def mimc_hash_array(values: List[int], key: int = 0) -> int:
    """
    Minimal demonstration MiMC-based sponge hash for an array of inputs.
    - Must match our circuits/mimc_hash_v2.circom logic: 2-round, constants = [7919, 7927].
    - This is purely illustrative; real usage requires more rounds for security.
    """
    c = [7919, 7927]  # 2 round constants
    nRounds = 2

    # Start with "state = key"
    state = normalize_to_field(key)
    for v in values:
        # absorb the input
        after_add = normalize_to_field(state + v)
        # run 2 rounds
        t = after_add
        for r in range(nRounds):
            t = normalize_to_field(t + c[r])
            # t^3
            t = normalize_to_field(t * t * t)
        state = normalize_to_field(t)
    return state


# =============================================================================
# Public Inputs Preparation
# =============================================================================
def prepare_client_public_inputs(
    learning_rate: float,
    precision: int,
    local_model_hash: int,
    global_model_hash: int
) -> List[str]:
    """
    Creates the array of 4 public signals for the client circuit:
        [eta, pr, ldigest, scgh]
    Each must be a valid field element in string form.
    """
    # Scale learning_rate
    lr_scaled = float_to_fixed(learning_rate, precision)
    pr_scaled = normalize_to_field(precision)
    ld_hash = normalize_to_field(local_model_hash)
    scg_hash = normalize_to_field(global_model_hash)

    signals = [
        str(lr_scaled),
        str(pr_scaled),
        str(ld_hash),
        str(scg_hash)
    ]
    logger.debug(f"Prepared client public signals: {signals}")
    return signals

def prepare_aggregator_public_inputs(
    local_model_hashes: List[int],
    updated_global_hash: int
) -> List[str]:
    """
    Creates the array of public signals for the aggregator circuit:
        [ ...ScLH..., gdigest ]
    i.e., an arbitrary number of local model hashes plus a single global model hash.
    """
    signals = []
    for h in local_model_hashes:
        signals.append(str(normalize_to_field(h)))
    signals.append(str(normalize_to_field(updated_global_hash)))
    logger.debug(f"Prepared aggregator public signals: {signals}")
    return signals


# =============================================================================
# Model Hashing Utilities
# =============================================================================
def compute_model_hash(model_state: Dict[str, torch.Tensor], precision: int) -> int:
    """
    Flatten a PyTorch model (weights & biases), convert each param to field integers,
    then apply mimc_hash_array to produce a single integer in FIELD_PRIME.
    """
    all_fixed = []
    for param_name, param_tensor in model_state.items():
        param_fixed = tensor_to_fixed(param_tensor, precision)
        all_fixed.extend(param_fixed)

    # Finally, compute the MiMC-based sponge hash
    hval = mimc_hash_array(all_fixed, key=0)
    logger.debug(f"Computed model hash: {hval}")
    return hval


# =============================================================================
# File-based Groth16 Proof Generation
# =============================================================================
def generate_proof(
    input_json: Dict[str, Any],
    circuit_wasm_path: str,
    proving_key_path: str,
    output_proof_json: str
) -> Dict[str, Any]:
    """
    1. Writes the circuit input to a JSON file.
    2. Invokes Node.js to create a witness (witness.wtns).
    3. Runs snarkjs groth16 prove to create a proof.json + public.json
    4. Returns a dict with { 'proof': proof_data, 'public': [publicSignals...] }

    Expects:
      circuit_wasm_path: path/to/client.wasm or aggregator.wasm
      proving_key_path:  path/to/client_0000.zkey or aggregator_0000.zkey
      output_proof_json: name of the final proof file (e.g. "proof.json")

    Raises exceptions on failures. Returns the proof + public signals if successful.
    """
    logger.info("=== Starting proof generation ===")
    if not os.path.exists(circuit_wasm_path):
        raise FileNotFoundError(f"WASM file not found at: {circuit_wasm_path}")
    if not os.path.exists(proving_key_path):
        raise FileNotFoundError(f"Proving key not found at: {proving_key_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "input.json")
        witness_file = os.path.join(tmpdir, "witness.wtns")
        public_file = os.path.join(tmpdir, "public.json")
        proof_file = os.path.join(tmpdir, "proof.json")
        gen_witness_js = os.path.join(os.path.dirname(circuit_wasm_path), "generate_witness.js")

        # 1. Write input.json
        with open(input_file, "w") as f:
            json.dump(input_json, f, indent=2)

        # 2. Run node to produce witness
        if not os.path.exists(gen_witness_js):
            raise FileNotFoundError(f"generate_witness.js not found next to wasm: {gen_witness_js}")

        witness_cmd = [
            "node",
            gen_witness_js,
            circuit_wasm_path,
            input_file,
            witness_file
        ]
        logger.debug("Running witness generation command:")
        logger.debug(" ".join(witness_cmd))
        wproc = subprocess.run(witness_cmd, capture_output=True, text=True)
        if wproc.returncode != 0:
            logger.error(f"Failed to generate witness:\nStdout: {wproc.stdout}\nStderr: {wproc.stderr}")
            raise RuntimeError("Witness generation failed.")
        else:
            logger.debug("Witness generation successful.")

        # 3. Use snarkjs groth16 prove
        prove_cmd = [
            "snarkjs", "groth16", "prove",
            proving_key_path,
            witness_file,
            proof_file,
            public_file
        ]
        logger.debug("Running snarkjs prove command:")
        logger.debug(" ".join(prove_cmd))
        pproc = subprocess.run(prove_cmd, capture_output=True, text=True)
        if pproc.returncode != 0:
            logger.error(f"Proof generation failed:\nStdout: {pproc.stdout}\nStderr: {pproc.stderr}")
            raise RuntimeError("Proof generation via snarkjs failed.")
        else:
            logger.debug("Proof generation successful.")

        # 4. Read proof.json + public.json
        with open(proof_file, "r") as f:
            proof_data = json.load(f)
        with open(public_file, "r") as f:
            public_data = json.load(f)

        # Move proof.json out of tmp if desired
        final_proof = {}
        final_proof["proof"] = proof_data
        final_proof["public"] = public_data

        # Optionally write proof_file to the provided output path
        with open(output_proof_json, "w") as fout:
            json.dump(final_proof, fout, indent=2)

        return final_proof


# =============================================================================
# Verification (off-chain style via snarkjs)
# =============================================================================
def verify_proof(
    vkey_json_path: str,
    proof_object: Dict[str, Any]
) -> bool:
    """
    Calls `snarkjs groth16 verify vkey proof.public proof.proof` in a temporary workspace.
    We embed the proof + public signals into a 'verification_input.json' file
    and pass that + vkey to snarkjs for final check.
    Returns True if verification is successful, False otherwise.
    """
    if not os.path.exists(vkey_json_path):
        raise FileNotFoundError(f"Verification key not found: {vkey_json_path}")
    if "proof" not in proof_object or "public" not in proof_object:
        raise ValueError("Invalid proof object; must have 'proof' and 'public' keys.")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Save vkey.json into tmp
        vkey_tmp = os.path.join(tmpdir, "vkey.json")
        with open(vkey_json_path, "r") as src, open(vkey_tmp, "w") as dst:
            vkey_data = json.load(src)
            json.dump(vkey_data, dst, indent=2)

        # 2. Prepare 'verification_input.json'
        ver_input_path = os.path.join(tmpdir, "verification_input.json")
        verification_input = {
            "protocol": "groth16",
            "curve": "bn128",
            "pi_a": proof_object["proof"].get("pi_a", []),
            "pi_b": proof_object["proof"].get("pi_b", []),
            "pi_c": proof_object["proof"].get("pi_c", []),
            "public": proof_object["public"]
        }
        with open(ver_input_path, "w") as f:
            json.dump(verification_input, f, indent=2)

        # 3. Actually call "snarkjs groth16 verify"
        verify_cmd = [
            "snarkjs",
            "groth16",
            "verify",
            vkey_tmp,
            ver_input_path
        ]
        logger.debug(f"Verification command: {' '.join(verify_cmd)}")
        proc = subprocess.run(verify_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error(f"Verification returned error:\n{proc.stdout}\n{proc.stderr}")
            return False
        if "OK!" in proc.stdout:
            logger.info("Proof verification succeeded.")
            return True
        else:
            logger.error(f"Proof verification failed. Output:\n{proc.stdout}\n{proc.stderr}")
            return False


# =============================================================================
# High-Level Wrappers for Client & Aggregator
# =============================================================================
class ZKPClientWrapperV2:
    """
    Client wrapper that:
      - Constructs circuit input for the client circuit
      - Generates a proof
      - Returns { 'proof':..., 'public':[...] }
    """

    def __init__(
        self,
        circuit_wasm_path: str,
        proving_key_path: str,
        vkey_json_path: str,
        precision: int = 1000
    ):
        if not os.path.exists(circuit_wasm_path):
            raise FileNotFoundError(f"Client WASM not found: {circuit_wasm_path}")
        if not os.path.exists(proving_key_path):
            raise FileNotFoundError(f"Client proving key not found: {proving_key_path}")
        if not os.path.exists(vkey_json_path):
            raise FileNotFoundError(f"Client verification key not found: {vkey_json_path}")

        self.circuit_wasm = circuit_wasm_path
        self.pk_path = proving_key_path
        self.vkey_path = vkey_json_path
        self.precision = precision

        logger.info("ZKPClientWrapperV2 initialized.")

    def generate_training_proof(
        self,
        old_global_hash: int,
        new_local_hash: int,
        learning_rate: float,
        input_gw: List[List[int]],
        input_gb: List[int],
        input_lwp: List[List[int]],
        input_lbp: List[int],
        input_x: List[int],
        input_y: List[int]
    ) -> Dict[str, Any]:
        """
        Produces the input.json for the 'client_v2.circom' circuit and runs a proof.
        old_global_hash, new_local_hash: Already computed (e.g. from PyTorch model).
        learning_rate: float, must be scaled inside the circuit input or by the user.
        input_gw, input_gb: The old global model weights & biases
        input_lwp, input_lbp: The newly trained local model weights & biases
        input_x, input_y: A single training example (or partial batch) used for the demonstration
        """
        logger.debug("Preparing client circuit input for proof generation...")

        # Build the entire input structure for client circuit
        circuit_input = {
            # public signals
            "eta": str(float_to_fixed(learning_rate, self.precision)),
            "pr": str(self.precision),
            "ldigest": str(normalize_to_field(new_local_hash)),
            "scgh": str(normalize_to_field(old_global_hash)),

            # private inputs
            "GW": input_gw,       # shape [hiddenSize][inputSize]
            "GB": input_gb,       # shape [hiddenSize]
            "LWp": input_lwp,     # shape [hiddenSize][outputSize]
            "LBp": input_lbp,     # shape [outputSize]
            "X": input_x,         # shape [inputSize]
            "Y": input_y          # shape [outputSize]
        }

        # Save final proof in a local file "client_proof.json"
        proof_file = "client_proof.json"

        proof_dict = generate_proof(
            input_json=circuit_input,
            circuit_wasm_path=self.circuit_wasm,
            proving_key_path=self.pk_path,
            output_proof_json=proof_file
        )
        logger.debug(f"Client proof generated. Proof file at: {proof_file}")
        return proof_dict

    def verify_client_proof(self, proof_dict: Dict[str, Any]) -> bool:
        """
        Off-chain or local chain verification of the proof using the client vkey.
        """
        return verify_proof(self.vkey_path, proof_dict)


class ZKPAggregatorWrapperV2:
    """
    Aggregator wrapper that:
      - Constructs circuit input for the aggregator circuit
      - Generates a proof that the aggregator used verified local models 
        and computed the new global model properly
      - Verifies the proof with aggregator's vkey
    """

    def __init__(
        self,
        circuit_wasm_path: str,
        proving_key_path: str,
        vkey_json_path: str
    ):
        if not os.path.exists(circuit_wasm_path):
            raise FileNotFoundError(f"Aggregator WASM not found: {circuit_wasm_path}")
        if not os.path.exists(proving_key_path):
            raise FileNotFoundError(f"Aggregator proving key not found: {proving_key_path}")
        if not os.path.exists(vkey_json_path):
            raise FileNotFoundError(f"Aggregator verification key not found: {vkey_json_path}")

        self.circuit_wasm = circuit_wasm_path
        self.pk_path = proving_key_path
        self.vkey_path = vkey_json_path

        logger.info("ZKPAggregatorWrapperV2 initialized.")

    def generate_aggregation_proof(
        self,
        sc_lh: List[int],          # array of verified local model hashes
        g_digest: int,            # hash of updated global model
        gw: List[List[int]],       # old global weights
        gb: List[int],            # old global biases
        lwps: List[List[List[int]]], # local weights (numClients x hiddenSize x inputSize)
        lbps: List[List[int]],    # local biases (numClients x hiddenSize)
        gwp: List[List[int]],     # new global weights
        gbp: List[int]            # new global biases
    ) -> Dict[str, Any]:
        """
        Build aggregator circuit input and generate proof that:
         - aggregator used local models matching sc_lh
         - aggregator computed final global model => g_digest
         - aggregator performed the correct average from (gw, gb) + local models
        """
        aggregator_input = {
            "ScLH": [str(normalize_to_field(h)) for h in sc_lh],
            "gdigest": str(normalize_to_field(g_digest)),

            "GW": gw,   # hiddenSize x inputSize
            "GB": gb,   # hiddenSize

            # local model arrays
            "LWp": lwps, # shape [numClients][hiddenSize][inputSize]
            "LBp": lbps, # shape [numClients][hiddenSize]

            # updated global
            "GWp": gwp,  # hiddenSize x inputSize
            "GBp": gbp   # hiddenSize
        }

        proof_file = "aggregator_proof.json"
        proof_dict = generate_proof(
            input_json=aggregator_input,
            circuit_wasm_path=self.circuit_wasm,
            proving_key_path=self.pk_path,
            output_proof_json=proof_file
        )
        logger.debug(f"Aggregator proof generated. Proof file: {proof_file}")
        return proof_dict

    def verify_aggregator_proof(self, proof_dict: Dict[str, Any]) -> bool:
        """
        Locally verifies aggregator proof using aggregator's verification key.
        """
        return verify_proof(self.vkey_path, proof_dict)


# =============================================================================
# End of File
# =============================================================================
