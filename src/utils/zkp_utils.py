# src/utils/zkp_utils.py

import json
import os
import logging
from typing import Dict, List
import torch
import time
import subprocess
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

FIELD_PRIME = 21888242871839275222246405745257275088548364400416034343698204186575808495617

def mimc_hash(values: List[int], key=0):
    if not values:
        return 0
    inputs = [int(v) % FIELD_PRIME for v in values]
    nInputs = len(inputs)
    nRounds = 110
    constants = [i+1 for i in range(nRounds)]

    currentStateInputs = [0]*(nInputs+1)
    currentStateInputs[0] = key % FIELD_PRIME

    for i in range(nInputs):
        afterAdd = (currentStateInputs[i] + inputs[i]) % FIELD_PRIME
        roundStates = [0]*(nRounds+1)
        roundStates[0] = afterAdd

        for j in range(nRounds):
            t = (roundStates[j] + constants[j]) % FIELD_PRIME
            t_cubed = (t * t * t) % FIELD_PRIME
            roundStates[j+1] = t_cubed

        currentStateInputs[i+1] = roundStates[nRounds]

    return currentStateInputs[nInputs]

########################################################
# Normal input generation functions for real trainer usage
########################################################

def generate_client_input(gw: List[int], gb: List[int],
                          x: List[int], y: List[int],
                          lwp: List[int], lbp: List[int],
                          eta: int, pr: int, scgh: int, ldigest: int,
                          delta2_input: List[int], dW_input: List[int], dB_input: List[int]):
    # This function expects fully sized arrays according to ClientCircuit(5,10,3).
    # For reference:
    # GW: hiddenSize(10) x inputSize(5) = 50 elements
    # GB: hiddenSize(10) = 10 elements
    # X: inputSize(5)
    # Y: outputSize(3)
    # LWp: hiddenSize(10) x inputSize(5) = 50 elements
    # LBp: hiddenSize(10) = 10 elements
    # delta2_input: outputSize(3)
    # dW_input: 10x5 = 50 elements
    # dB_input: 10 elements

    hiddenSize = 10
    inputSize = 5
    outputSize = 3

    def to_2d(arr, rows, cols):
        return [arr[i*cols:(i+1)*cols] for i in range(rows)]

    GW_2d = to_2d(gw, hiddenSize, inputSize)
    GB_1d = gb
    X_1d = x
    Y_1d = y
    LWp_2d = to_2d(lwp, hiddenSize, inputSize)
    LBp_1d = lbp
    dW_2d = to_2d(dW_input, hiddenSize, inputSize)
    dB_1d = dB_input

    client_input = {
        "GW": GW_2d,
        "GB": GB_1d,
        "X": X_1d,
        "Y": Y_1d,
        "LWp": LWp_2d,
        "LBp": LBp_1d,
        "eta": eta,
        "pr": pr,
        "ScGH": scgh,
        "ldigest": ldigest,
        "delta2_input": delta2_input,
        "dW_input": dW_2d,
        "dB_input": dB_1d
    }
    return client_input

def generate_aggregator_input(gw: List[int], gb: List[int],
                              lwps: List[int], lbps: List[int],
                              gwp: List[int], gbp: List[int],
                              sclh: List[int], gdigest: int):
    # AggregatorCircuit(4,5,10,3)
    # Need to reshape similarly:
    numClients=4
    hiddenSize=10
    inputSize=5

    def to_2d(arr, rows, cols):
        return [arr[i*cols:(i+1)*cols] for i in range(rows)]

    def to_3d(arr, dim1, dim2, dim3):
        # arr length must be dim1*dim2*dim3
        out = []
        idx = 0
        for i in range(dim1):
            slice_2d = []
            for j in range(dim2):
                row = arr[idx:idx+dim3]
                idx+=dim3
                slice_2d.append(row)
            out.append(slice_2d)
        return out

    GW_2d = to_2d(gw, hiddenSize, inputSize)
    GB_1d = gb
    LWp_3d = to_3d(lwps, numClients, hiddenSize, inputSize)
    LBp_2d = to_2d(lbps, numClients, hiddenSize)
    GWp_2d = to_2d(gwp, hiddenSize, inputSize)
    GBp_1d = gbp

    aggregator_input = {
        "ScLH": sclh,
        "gdigest": gdigest,
        "GW": GW_2d,
        "GB": GB_1d,
        "LWp": LWp_3d,
        "LBp": LBp_2d,
        "GWp": GWp_2d,
        "GBp": GBp_1d
    }
    return aggregator_input

########################################################
# Test versions of input generation (hardcoded zeros/dummy)
########################################################

def generate_client_input_for_test(gw, gb, x, y, lwp, lbp, eta, pr, scgh, ldigest):
    # For testing: zero out delta2_input, dW_input, dB_input
    hiddenSize=10
    inputSize=5
    outputSize=3

    delta2_input = [0]*outputSize
    dW_input = [0]*(hiddenSize*inputSize)
    dB_input = [0]*hiddenSize

    return generate_client_input(
        gw, gb, x, y, lwp, lbp, eta, pr, scgh, ldigest,
        delta2_input, dW_input, dB_input
    )

def generate_aggregator_input_for_test(gw, gb, lwps, lbps, gwp, gbp, sclh, gdigest):
    # For test, just pass as is. If needed, zero out something.
    return generate_aggregator_input(gw, gb, lwps, lbps, gwp, gbp, sclh, gdigest)

########################################################
def run_command(cmd):
    logger.debug(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {' '.join(cmd)}")
        logger.error(f"stdout: {result.stdout}")
        logger.error(f"stderr: {result.stderr}")
    return result

def generate_client_proof(client_inputs: dict, circuit_path: str, proving_key_path: str, js_dir: str):
    if not os.path.exists(circuit_path):
        raise FileNotFoundError(f"Circuit file not found at {circuit_path}")
    if not os.path.exists(proving_key_path):
        raise FileNotFoundError(f"Proving key file not found at {proving_key_path}")

    input_file = "client_input.json"
    witness_file = "witness.wtns"
    proof_file = "proof.json"
    public_file = "public.json"

    for f in [input_file, witness_file, proof_file, public_file]:
        if os.path.exists(f):
            os.remove(f)

    with open(input_file, "w") as f:
        json.dump(client_inputs, f)

    logger.info(f"🔑 Generating client proof using circuit {circuit_path} and pk {proving_key_path}...")
    start_time = time.time()

    wasm_path = os.path.join(js_dir, "client.wasm")
    gen_witness_js = os.path.join(js_dir, "generate_witness.js")

    if not os.path.exists(gen_witness_js) or not os.path.exists(wasm_path):
        raise FileNotFoundError("generate_witness.js or client.wasm not found in js_dir")

    cmd = ["node", gen_witness_js, wasm_path, input_file, witness_file]
    result = run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError("Client proof generation failed during witness generation")

    cmd = ["snarkjs", "groth16", "prove", proving_key_path, witness_file, proof_file, public_file]
    result = run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError("Client proof generation failed during proof generation")

    if not (os.path.exists(proof_file) and os.path.exists(public_file)):
        raise RuntimeError("Proof generation failed: proof.json or public.json not found.")

    gen_time = time.time() - start_time
    logger.info(f"✅ Client proof generated successfully in {gen_time:.2f}s")

    with open(proof_file) as pf:
        proof_data = json.load(pf)
    with open(public_file) as pubf:
        public_data = json.load(pubf)

    return proof_data, public_data

def generate_aggregator_proof(agg_inputs: dict, circuit_path: str, proving_key_path: str, js_dir: str):
    if not os.path.exists(circuit_path):
        raise FileNotFoundError(f"Circuit file not found at {circuit_path}")
    if not os.path.exists(proving_key_path):
        raise FileNotFoundError(f"Proving key file not found at {proving_key_path}")

    input_file = "aggregator_input.json"
    witness_file = "witness.wtns"
    proof_file = "proof.json"
    public_file = "public.json"

    for f in [input_file, witness_file, proof_file, public_file]:
        if os.path.exists(f):
            os.remove(f)

    logger.info(f"🔑 Generating aggregator proof using circuit {circuit_path} and pk {proving_key_path}...")
    start_time = time.time()

    wasm_path = os.path.join(js_dir, "aggregator.wasm")
    gen_witness_js = os.path.join(js_dir, "generate_witness.js")

    if not os.path.exists(gen_witness_js) or not os.path.exists(wasm_path):
        raise FileNotFoundError("generate_witness.js or aggregator.wasm not found in js_dir")

    with open(input_file, "w") as f:
        json.dump(agg_inputs, f)

    cmd = ["node", gen_witness_js, wasm_path, input_file, witness_file]
    result = run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError("Aggregator proof generation failed during witness generation")

    cmd = ["snarkjs", "groth16", "prove", proving_key_path, witness_file, proof_file, public_file]
    result = run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError("Aggregator proof generation failed during proof generation")

    if not (os.path.exists(proof_file) and os.path.exists(public_file)):
        raise RuntimeError("Proof generation failed: proof.json or public.json not found.")

    gen_time = time.time() - start_time
    logger.info(f"✅ Aggregator proof generated successfully in {gen_time:.2f}s")

    with open(proof_file) as pf:
        proof_data = json.load(pf)
    with open(public_file) as pubf:
        public_data = json.load(pubf)

    return proof_data, public_data

class ZKPVerifier:
    def __init__(self, client_vkey_path: str, aggregator_vkey_path: str):
        if not os.path.exists(client_vkey_path):
            raise FileNotFoundError(f"Client verification key not found: {client_vkey_path}")
        if not os.path.exists(aggregator_vkey_path):
            raise FileNotFoundError(f"Aggregator verification key not found: {aggregator_vkey_path}")

        with open(client_vkey_path, 'r') as f:
            self.client_vkey = json.load(f)
        with open(aggregator_vkey_path, 'r') as f:
            self.aggregator_vkey = json.load(f)

    def prepare_client_public_inputs(self, learning_rate: float, precision: int,
                                     local_model_hash: str, global_model_hash: str) -> List[str]:
        return [
            str(int(round(learning_rate))),
            str(int(round(precision))),
            str(int(local_model_hash)),
            str(int(global_model_hash))
        ]

    def prepare_aggregator_public_inputs(self, local_model_hashes: List[str],
                                         global_model_hash: str) -> List[str]:
        inputs = [str(int(h)) for h in local_model_hashes] + [str(int(global_model_hash))]
        return inputs

    def verify_client_proof(self, proof: Dict, public_signals: List[str]) -> bool:
        logger.warning("Client proof verification not implemented in this debug version.")
        return True

    def verify_aggregator_proof(self, proof: Dict, public_signals: List[str]) -> bool:
        logger.warning("Aggregator proof verification not implemented in this debug version.")
        return True

    @staticmethod
    def compute_model_hash(model_state: Dict[str, torch.Tensor]) -> str:
        params = []
        for param in model_state.values():
            arr = param.detach().cpu().numpy().flatten()
            ints = [int(round(v)) for v in arr]
            params.extend(ints)
        h = mimc_hash(params, key=0)
        return str(h)

class ZKPClientWrapper:
    def __init__(self, client_circuit_path: str, client_pk_path: str, client_wasm_path: str):
        if not os.path.exists(client_circuit_path):
            raise FileNotFoundError(f"Client circuit file not found: {client_circuit_path}")
        if not os.path.exists(client_pk_path):
            raise FileNotFoundError(f"Client proving key file not found: {client_pk_path}")
        if not os.path.exists(client_wasm_path):
            raise FileNotFoundError(f"Client WASM file not found: {client_wasm_path}")

        self.client_circuit_path = client_circuit_path
        self.client_pk_path = client_pk_path
        self.client_wasm_path = client_wasm_path
        self.client_js_dir = os.path.dirname(client_wasm_path)

    def generate_training_proof(self,
                              global_model: Dict[str, torch.Tensor],
                              local_model: Dict[str, torch.Tensor],
                              training_data: torch.Tensor,
                              labels: torch.Tensor,
                              learning_rate: float,
                              precision: int,
                              global_hash: str,
                              local_hash: str) -> Dict:
        logger.info("🔑 Preparing inputs for client training proof...")
        hiddenSize=10
        inputSize=5
        outputSize=3

        gw_t = global_model.get('weight', torch.zeros((hiddenSize,inputSize))).detach().cpu().numpy().flatten()
        gb_t = global_model.get('bias', torch.zeros(hiddenSize)).detach().cpu().numpy().flatten()

        lw_t = local_model.get('weight', torch.zeros((hiddenSize,inputSize))).detach().cpu().numpy().flatten()
        lb_t = local_model.get('bias', torch.zeros(hiddenSize)).detach().cpu().numpy().flatten()

        x = training_data.cpu().numpy().flatten()
        y = labels.cpu().numpy().flatten()

        # For real scenario, trainer should provide delta2, dW, dB inputs
        # Here let's assume trainer will handle real computations.
        # For now, pass zeros to not break the circuit:
        delta2_input = [0]*outputSize
        dW_input = [0]*(hiddenSize*inputSize)
        dB_input = [0]*hiddenSize

        client_inputs = generate_client_input(
            gw_t.tolist(), gb_t.tolist(),
            x.tolist(), y.tolist(),
            lw_t.tolist(), lb_t.tolist(),
            int(learning_rate), int(precision),
            int(global_hash), int(local_hash),
            delta2_input, dW_input, dB_input
        )

        logger.info(f"🔧 Using WASM file at: {self.client_wasm_path}, js_dir: {self.client_js_dir}")
        proof_data, public_data = generate_client_proof(
            client_inputs, 
            self.client_circuit_path, 
            self.client_pk_path,
            js_dir=self.client_js_dir
        )
        logger.info("✅ Client training proof generated and returned.")
        
        return {
            "proof": proof_data,
            "public": public_data
        }

class ZKPAggregatorWrapper:
    def __init__(self, aggregator_circuit_path: str, aggregator_pk_path: str, aggregator_wasm_path: str):
        if not os.path.exists(aggregator_circuit_path):
            raise FileNotFoundError(f"Aggregator circuit file not found: {aggregator_circuit_path}")
        if not os.path.exists(aggregator_pk_path):
            raise FileNotFoundError(f"Aggregator proving key file not found: {aggregator_pk_path}")
        if not os.path.exists(aggregator_wasm_path):
            raise FileNotFoundError(f"Aggregator WASM file not found: {aggregator_wasm_path}")

        self.aggregator_circuit_path = aggregator_circuit_path
        self.aggregator_pk_path = aggregator_pk_path
        self.aggregator_wasm_path = aggregator_wasm_path
        self.aggregator_js_dir = os.path.dirname(aggregator_wasm_path)

    def generate_aggregation_proof_with_hashes(self,
                                             global_model: Dict[str, torch.Tensor],
                                             local_models: List[Dict[str, torch.Tensor]],
                                             sclh: List[str],
                                             updated_global_hash: str) -> Dict:
        logger.info("🔑 Preparing inputs for aggregator proof...")
        # Normally you would pass actual aggregator inputs from the trainer.
        # For demonstration, let's assume trainer calls generate_aggregator_input with real params.
        # Here we just show structure; no zeroing needed.

        # Example:
        # Extract data from global_model and local_models, then call generate_aggregator_input.
        hiddenSize=10
        inputSize=5

        gw_t = global_model.get('weight', torch.zeros((hiddenSize,inputSize))).detach().cpu().numpy().flatten().tolist()
        gb_t = global_model.get('bias', torch.zeros(hiddenSize)).detach().cpu().numpy().flatten().tolist()

        # Flatten local models:
        lwps=[]
        lbps=[]
        for lm in local_models:
            lw_arr = lm.get('weight', torch.zeros((hiddenSize,inputSize))).detach().cpu().numpy().flatten().tolist()
            lb_arr = lm.get('bias', torch.zeros(hiddenSize)).detach().cpu().numpy().flatten().tolist()
            lwps.extend(lw_arr)
            lbps.extend(lb_arr)

        gwp = gw_t
        gbp = gb_t
        sclh_int = [int(h) for h in sclh]
        gdigest = int(updated_global_hash)

        agg_inputs = generate_aggregator_input(gw_t, gb_t, lwps, lbps, gwp, gbp, sclh_int, gdigest)
        
        logger.info(f"🔧 Using WASM file at: {self.aggregator_wasm_path}, js_dir: {self.aggregator_js_dir}")
        proof_data, public_data = generate_aggregator_proof(
            agg_inputs, 
            self.aggregator_circuit_path, 
            self.aggregator_pk_path,
            js_dir=self.aggregator_js_dir
        )
        logger.info("✅ Aggregator proof generated and returned.")
        
        return {
            "proof": proof_data,
            "public": public_data
        }

def test_proof_generation():
    """A simple test function to run locally for quick debugging."""
    # Use the test functions with controlled zero/dummy inputs
    gw = list(range(50))  # 10*5
    gb = list(range(10))
    x = list(range(5))
    y = list(range(3))
    lwp = list(range(50))  # 10*5
    lbp = list(range(10))
    eta = 1
    pr = 1000
    scgh = 123456
    ldigest = 654321

    # Use test function
    client_inputs = generate_client_input_for_test(gw, gb, x, y, lwp, lbp, eta, pr, scgh, ldigest)

    circuit_path = "client.r1cs"
    proving_key_path = "client_0000.zkey"
    js_dir = "client_js"

    if not (os.path.exists(circuit_path) and os.path.exists(proving_key_path) and os.path.isdir(js_dir)):
        logger.error("Test environment not set up. Please ensure circuit, pk, and js_dir exist.")
        return

    logger.info("Running test_proof_generation...")
    try:
        proof_data, public_data = generate_client_proof(client_inputs, circuit_path, proving_key_path, js_dir=js_dir)
        logger.info("Test proof generation succeeded.")
        logger.info(f"Proof Data: {proof_data}")
        logger.info(f"Public Data: {public_data}")
        logger.info("End of test_proof_generation reached with no exceptions.")
    except Exception as e:
        logger.error(f"Test proof generation failed: {e}")
    finally:
        logger.info("Done running zkp_utils.py script.")

if __name__ == "__main__":
    test_proof_generation()
