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
    - A matching version of the MiMC functions/circom files

Disclaimer:
    - This is a minimal example. For production usage, handle errors, logs,
      security, ephemeral files, and environment checks more robustly.
"""

import os
import json
import logging
import subprocess
import tempfile
from typing import List, Dict, Any, Optional
import torch
import shutil
import numpy as np

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
    val_mod = value % FIELD_PRIME
    return val_mod

def float_to_fixed(val: float, precision: int) -> int:
    scaled = int(round(val * precision))
    return normalize_to_field(scaled)

def tensor_to_fixed(tensor: torch.Tensor, precision: int) -> List[int]:
    arr = tensor.detach().cpu().numpy().flatten()
    return [float_to_fixed(float(v), precision) for v in arr]

def safe_mod(x: np.ndarray, n: int) -> np.ndarray:
    """Safe modulo operation for large numbers that avoids object dtype"""
    # Convert to float64 first to handle larger numbers
    x_float = x.astype(np.float64)
    # Use fmod for floating point modulo
    result = np.fmod(x_float, float(n))
    # Handle negative values
    result = np.where(result < 0, result + n, result)
    # Convert back to a reasonable range for int64
    return (result % 1e9).astype(np.int64)

def tensor_to_field(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor values to field elements while keeping values manageable"""
    # Always return float tensor for gradient compatibility
    with torch.no_grad():
        # Convert to numpy array
        np_arr = tensor.detach().cpu().numpy().astype(np.int64)
        # Use safe modulo operation
        field_arr = safe_mod(np_arr, FIELD_PRIME)
        # Keep values in a reasonable range 
        field_arr = field_arr % int(1e9)
        # Back to tensor, ensuring float type
        return torch.from_numpy(field_arr).float().to(tensor.device)

def safe_float_to_fixed(val: float, precision: int) -> int:
    """Safely convert float to fixed point, handling large values."""
    try:
        if not np.isfinite(val):
            return 0
        # First truncate to avoid overflow
        truncated = float(np.clip(val, -1e9, 1e9))
        scaled = int(round(truncated * precision))
        return normalize_to_field(scaled)
    except (OverflowError, ValueError):
        logger.warning(f"Overflow in float_to_fixed, val={val}, precision={precision}")
        return 0

def safe_tensor_to_fixed(tensor: torch.Tensor, precision: int) -> List[int]:
    """Safely convert tensor to fixed point values."""
    arr = tensor.detach().cpu().numpy().flatten()
    return [safe_float_to_fixed(float(v), precision) for v in arr]


# =============================================================================
# MiMC Hash in Python (must match the circom version)
# =============================================================================
def mimc_hash_array(values: List[int], key: int = 0) -> int:
    """
    Minimal demonstration MiMC-based sponge hash for an array of inputs.
    Must match circuits/mimc_hash_v2.circom logic: 2-round, constants=[7919, 7927].
    """
    c = [7919, 7927]  # 2 round constants
    nRounds = 2

    state = normalize_to_field(key)
    for v in values:
        after_add = normalize_to_field(state + v)
        t = after_add
        for r in range(nRounds):
            t = normalize_to_field(t + c[r])
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
    all_fixed = []
    for param_name, param_tensor in model_state.items():
        param_fixed = safe_tensor_to_fixed(param_tensor, precision)
        all_fixed.extend(param_fixed)

    hval = mimc_hash_array(all_fixed, key=0)
    logger.debug(f"Computed model hash: {hval}")
    return hval


# =============================================================================
# File-based Groth16 Proof Generation (Updated with `debug_dir`)
# =============================================================================
def generate_proof(
    input_json: Dict[str, Any],
    circuit_wasm_path: str,
    proving_key_path: str,
    output_proof_json: str,
    debug_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    1. Writes the circuit input to a JSON file.
    2. Invokes Node.js to create a witness (witness.wtns).
    3. Runs snarkjs groth16 prove to create a proof.json + public.json
    4. Returns { 'proof': proof_data, 'public': [publicSignals...] }

    If debug_dir is provided, intermediate files (input.json, witness.wtns, etc.)
    are saved in that directory. Otherwise, we use a temporary directory
    that is cleaned up automatically.
    """
    logger.info("=== Starting proof generation ===")

    if not os.path.exists(circuit_wasm_path):
        raise FileNotFoundError(f"WASM file not found: {circuit_wasm_path}")
    if not os.path.exists(proving_key_path):
        raise FileNotFoundError(f"Proving key not found: {proving_key_path}")

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        input_file = os.path.join(debug_dir, "input.json")
        witness_file = os.path.join(debug_dir, "witness.wtns")
        public_file = os.path.join(debug_dir, "public.json")
        proof_file = os.path.join(debug_dir, "proof.json")
    else:
        tmpdir = tempfile.TemporaryDirectory()
        input_file = os.path.join(tmpdir.name, "input.json")
        witness_file = os.path.join(tmpdir.name, "witness.wtns")
        public_file = os.path.join(tmpdir.name, "public.json")
        proof_file = os.path.join(tmpdir.name, "proof.json")

    gen_witness_js = os.path.join(os.path.dirname(circuit_wasm_path), "generate_witness.js")

    # 1. Write input.json
    with open(input_file, "w") as f:
        json.dump(input_json, f, indent=2)

    logger.debug("Final JSON for witness generation:")
    logger.debug(json.dumps(input_json, indent=2))

    # 2. Generate witness with Node.js
    if not os.path.exists(gen_witness_js):
        raise FileNotFoundError(f"generate_witness.js not found next to wasm: {gen_witness_js}")

    witness_cmd = [
        "node",
        gen_witness_js,
        circuit_wasm_path,
        input_file,
        witness_file
    ]
    logger.debug(f"Running witness generation command: {' '.join(witness_cmd)}")
    wproc = subprocess.run(witness_cmd, capture_output=True, text=True)
    if wproc.returncode != 0:
        logger.error(f"Failed to generate witness:\nStdout: {wproc.stdout}\nStderr: {wproc.stderr}")
        # Optionally keep the input file for debugging if not using debug_dir
        raise RuntimeError("Witness generation failed.")
    else:
        logger.debug("Witness generation successful.")

    # 3. groth16 prove
    prove_cmd = [
        "snarkjs", "groth16", "prove",
        proving_key_path,
        witness_file,
        proof_file,
        public_file
    ]
    logger.debug(f"Running snarkjs prove command: {' '.join(prove_cmd)}")
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

    final_proof = {
        "proof": proof_data,
        "public": public_data
    }

    # Save proof_file as desired
    with open(output_proof_json, "w") as fout:
        json.dump(final_proof, fout, indent=2)

    # Clean up tempdir if no debug_dir was set
    if not debug_dir:
        tmpdir.cleanup()

    return final_proof


# =============================================================================
# Verification (off-chain style via snarkjs)
# =============================================================================
def verify_proof(
    vkey_json_path: str,
    proof_object: Dict[str, Any]
) -> bool:
    if not os.path.exists(vkey_json_path):
        raise FileNotFoundError(f"Verification key not found: {vkey_json_path}")
    if "proof" not in proof_object or "public" not in proof_object:
        raise ValueError("Invalid proof object; must have 'proof' and 'public' keys.")

    with tempfile.TemporaryDirectory() as tmpdir:
        vkey_tmp = os.path.join(tmpdir, "vkey.json")
        with open(vkey_json_path, "r") as src, open(vkey_tmp, "w") as dst:
            vkey_data = json.load(src)
            json.dump(vkey_data, dst, indent=2)

        # Prepare verification_input.json
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
# ZKPClientWrapperV2 & ZKPAggregatorWrapperV2
# =============================================================================
class ZKPClientWrapperV2:
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
        input_y: List[int],
        debug_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        if not isinstance(old_global_hash, int):
            logger.error(f"old_global_hash is not an int. Got {type(old_global_hash)}")
            raise ValueError("old_global_hash must be a single integer.")
        if not isinstance(new_local_hash, int):
            logger.error(f"new_local_hash is not an int. Got {type(new_local_hash)}")
            raise ValueError("new_local_hash must be a single integer.")

        logger.debug("Preparing client circuit input for proof generation...")
        circuit_input = {
            "eta": str(float_to_fixed(learning_rate, self.precision)),
            "pr": str(self.precision),
            "ldigest": str(normalize_to_field(new_local_hash)),
            "scgh": str(normalize_to_field(old_global_hash)),

            "GW": input_gw,
            "GB": input_gb,
            "LWp": input_lwp,
            "LBp": input_lbp,
            "X": input_x,
            "Y": input_y
        }

        # Add debug logging
        debug_circuit_inputs(circuit_input)

        logger.debug("Final client circuit input:")
        logger.debug(json.dumps(circuit_input, indent=2))

        proof_file = "client_proof.json"
        proof_dict = generate_proof(
            input_json=circuit_input,
            circuit_wasm_path=self.circuit_wasm,
            proving_key_path=self.pk_path,
            output_proof_json=proof_file,
            debug_dir=debug_dir
        )
        logger.debug("Client proof generated successfully.")
        return proof_dict

    def verify_client_proof(self, proof_dict: Dict[str, Any]) -> bool:
        return verify_proof(self.vkey_path, proof_dict)


class ZKPAggregatorWrapperV2:
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
        sc_lh: List[int],
        g_digest: int,
        gw: List[List[int]],
        gb: List[int],
        lwps: List[List[List[int]]],
        lbps: List[List[int]],
        gwp: List[List[int]],
        gbp: List[int],
        debug_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        aggregator_input = {
            "ScLH": [str(normalize_to_field(h)) for h in sc_lh],
            "gdigest": str(normalize_to_field(g_digest)),

            "GW": gw,
            "GB": gb,

            "LWp": lwps,
            "LBp": lbps,

            "GWp": gwp,
            "GBp": gbp
        }

        logger.debug("Final aggregator circuit input:")
        logger.debug(json.dumps(aggregator_input, indent=2))

        proof_file = "aggregator_proof.json"
        proof_dict = generate_proof(
            input_json=aggregator_input,
            circuit_wasm_path=self.circuit_wasm,
            proving_key_path=self.pk_path,
            output_proof_json=proof_file,
            debug_dir=debug_dir
        )
        logger.debug("Aggregator proof generated successfully.")
        return proof_dict

    def verify_aggregator_proof(self, proof_dict: Dict[str, Any]) -> bool:
        return verify_proof(self.vkey_path, proof_dict)

def debug_circuit_inputs(circuit_input: Dict[str, Any], field_prime: int = FIELD_PRIME) -> None:
    """Add detailed debugging for circuit inputs"""
    logger.debug("\n=== DETAILED CIRCUIT INPUT DEBUGGING ===")
    
    # Check scalar values
    for key in ['eta', 'pr', 'ldigest', 'scgh']:
        if key in circuit_input:
            val = int(circuit_input[key])
            logger.debug(f"{key}: {val}")
            if val >= field_prime:
                logger.error(f"WARNING: {key} exceeds field size!")

    # Check array dimensions
    if 'GW' in circuit_input:
        gw = circuit_input['GW']
        logger.debug(f"GW dimensions: {len(gw)}x{len(gw[0]) if gw else 0}")
        
    if 'GB' in circuit_input:
        gb = circuit_input['GB']
        logger.debug(f"GB dimensions: {len(gb)}")
        
    if 'LWp' in circuit_input:
        lwp = circuit_input['LWp']
        logger.debug(f"LWp dimensions: {len(lwp)}x{len(lwp[0]) if lwp else 0}")
        
    if 'LBp' in circuit_input:
        lbp = circuit_input['LBp']
        logger.debug(f"LBp dimensions: {len(lbp)}")
        
    if 'X' in circuit_input:
        x = circuit_input['X']
        logger.debug(f"X dimensions: {len(x)}")
        
    if 'Y' in circuit_input:
        y = circuit_input['Y']
        logger.debug(f"Y dimensions: {len(y)}")

    # Check for any values exceeding field size
    def check_array_values(arr, name):
        if isinstance(arr, list):
            for i, val in enumerate(arr):
                if isinstance(val, list):
                    check_array_values(val, f"{name}[{i}]")
                else:
                    try:
                        val_int = int(val)
                        if val_int >= field_prime:
                            logger.error(f"WARNING: {name}[{i}] = {val_int} exceeds field size!")
                    except ValueError:
                        logger.error(f"WARNING: Could not convert {name}[{i}] = {val} to integer!")

    for key, value in circuit_input.items():
        if isinstance(value, list):
            check_array_values(value, key)

    logger.debug("=== END DETAILED CIRCUIT INPUT DEBUGGING ===\n")