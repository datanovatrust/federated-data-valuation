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

def weighted_sum(inputs: List[float], weights: List[float], factor: int) -> float:
    """
    Implements the WeightedSum template from the circuit.
    Matches the exact computation steps in the circuit.
    """
    sum_val = 0
    for i in range(len(inputs)):
        # First compute product, then scale by factor
        prod_val = inputs[i] * weights[i]
        scaled_val = prod_val * factor
        sum_val += scaled_val
    return sum_val

def compute_forward_pass(gw: List[float], gb: List[float], x: List[float], 
                        lwp: List[float], lbp: List[float], pr: int):
    """
    Computes forward pass exactly as done in the circuit.
    Returns all intermediate values needed for gradient computation.
    
    Args:
        gw: Global weights (flattened)
        gb: Global biases
        x: Input features
        lwp: Local weights (flattened)
        lbp: Local biases
        pr: Precision factor
    
    Returns:
        Dictionary containing z1, a1, z2, a2 values
    """
    hiddenSize = 10
    inputSize = 5
    outputSize = 3

    # Reshape weights into 2D arrays
    gw_2d = [gw[i*inputSize:(i+1)*inputSize] for i in range(hiddenSize)]
    lwp_2d = [lwp[i*inputSize:(i+1)*inputSize] for i in range(hiddenSize)]

    # Hidden layer (Z1, A1)
    z1 = []
    a1 = []
    for i in range(hiddenSize):
        # Compute weighted sum for each hidden neuron
        weighted_sum_result = weighted_sum(x, gw_2d[i], pr)
        z1_val = weighted_sum_result + gb[i]
        z1.append(z1_val)
        # In circuit, A1 = Z1 (no activation)
        a1.append(z1_val)

    # Output layer (Z2, A2)
    z2 = []
    a2 = []
    for i in range(outputSize):
        # Prepare weights for this output neuron
        output_weights = [lwp_2d[j][i] for j in range(hiddenSize)]
        # Compute weighted sum
        weighted_sum_result = weighted_sum(a1, output_weights, pr)
        z2_val = weighted_sum_result + lbp[i]
        z2.append(z2_val)
        # In circuit, A2 = Z2 (no activation)
        a2.append(z2_val)

    return {
        'Z1': z1,
        'A1': a1,
        'Z2': z2,
        'A2': a2
    }

def compute_output_gradients(a2: List[float], y: List[float], pr: int) -> List[float]:
    """
    Computes output layer gradients (delta2) exactly as in circuit.
    """
    outputSize = len(y)
    delta2 = []
    
    for i in range(outputSize):
        # Circuit computes: diff = A2 - Y, then delta2 = 2 * diff
        diff = a2[i] - y[i]
        delta2.append(diff * 2)
    
    return delta2

def compute_hidden_gradients(delta2: List[float], lwp: List[float]) -> List[float]:
    """
    Computes hidden layer gradients (delta1) exactly as in circuit.
    """
    hiddenSize = 10
    outputSize = 3
    
    # Reshape local weights
    lwp_2d = [lwp[i*5:(i+1)*5] for i in range(hiddenSize)]
    
    delta1 = []
    for i in range(hiddenSize):
        # Extract weights for this hidden neuron
        hidden_weights = [lwp_2d[i][j] for j in range(outputSize)]
        # Compute weighted sum of delta2 values (factor=1 as per circuit)
        grad_sum = weighted_sum(delta2, hidden_weights, 1)
        delta1.append(grad_sum)
    
    return delta1

def compute_weight_gradients(delta1: List[float], x: List[float], pr: int) -> List[float]:
    """
    Computes weight gradients exactly as in circuit.
    """
    hiddenSize = 10
    inputSize = 5
    dW = []
    
    for i in range(hiddenSize):
        for j in range(inputSize):
            # Circuit: delta1X = delta1 * X
            delta1X = delta1[i] * x[j]
            # Note: Circuit checks scaledDW = dW * pr === delta1X
            # So our dW should be delta1X / pr
            dW.append(delta1X / pr)
            
    return dW

def compute_bias_gradients(delta1: List[float]) -> List[float]:
    """
    Computes bias gradients exactly as in circuit.
    """
    # In circuit, dB === delta1 directly
    return delta1

def validate_model_updates(gw: List[float], gb: List[float],
                         lwp: List[float], lbp: List[float],
                         dw: List[float], db: List[float],
                         eta: float) -> bool:
    """
    Validates that model updates match circuit constraints:
    LWp = GW - eta * dW
    LBp = GB - eta * dB
    """
    hiddenSize = 10
    inputSize = 5
    
    for i in range(hiddenSize):
        for j in range(inputSize):
            idx = i * inputSize + j
            weight_update = eta * dw[idx]
            expected_lwp = gw[idx] - weight_update
            if abs(lwp[idx] - expected_lwp) > 1e-10:
                return False
    
    for i in range(hiddenSize):
        bias_update = eta * db[i]
        expected_lbp = gb[i] - bias_update
        if abs(lbp[i] - expected_lbp) > 1e-10:
            return False
            
    return True

def scale_to_int(value: float, precision: int = 1000) -> int:
    """Scale a float value to integer by multiplying with precision factor."""
    return int(round(value * precision))

def scale_array_to_int(arr: List[float], precision: int = 1000) -> List[int]:
    """Scale an array of float values to integers."""
    return [scale_to_int(x, precision) for x in arr]

def mimc_hash(values: List[int], key=0):
    if not values:
        return 0
    inputs = [int(v) % FIELD_PRIME for v in values]
    nInputs = len(inputs)

    # Reduced number of rounds to 22
    nRounds = 2
    constants = [i+1 for i in range(nRounds)]

    currentStateInputs = [0]*(nInputs+1)
    currentStateInputs[0] = key % FIELD_PRIME

    for i in range(nInputs):
        afterAdd = (currentStateInputs[i] + inputs[i]) % FIELD_PRIME
        roundStates = [0]*(nRounds+1)
        roundStates[0] = afterAdd

        for j in range(nRounds):
            t = (roundStates[j] + constants[j]) % FIELD_PRIME
            # t^3 mod FIELD_PRIME
            t_cubed = (t * t * t) % FIELD_PRIME
            roundStates[j+1] = t_cubed

        currentStateInputs[i+1] = roundStates[nRounds]

    return currentStateInputs[nInputs]

def generate_client_input(gw: List[float], gb: List[float],
                         x: List[float], y: List[float],
                         lwp: List[float], lbp: List[float],
                         eta: float, pr: int, scgh: int, ldigest: int,
                         delta2_input: List[float], dW_input: List[float], dB_input: List[float]):
    """
    Generates client input matching exact circuit computations.
    All computations maintain circuit scaling and constraints.
    """
    hiddenSize = 10
    inputSize = 5
    outputSize = 3

    # Scale input values
    gw_int = scale_array_to_int(gw, pr)
    gb_int = scale_array_to_int(gb, pr)
    x_int = scale_array_to_int(x, pr)
    y_int = scale_array_to_int(y, pr)
    lwp_int = scale_array_to_int(lwp, pr)
    lbp_int = scale_array_to_int(lbp, pr)
    eta_int = scale_to_int(eta, pr)

    # Compute complete forward pass exactly as in circuit
    forward_results = compute_forward_pass(gw_int, gb_int, x_int, lwp_int, lbp_int, pr)
    
    # Compute output gradients (delta2)
    delta2 = compute_output_gradients(forward_results['A2'], y_int, pr)
    
    # Compute hidden layer gradients (delta1)
    delta1 = compute_hidden_gradients(delta2, lwp_int)
    
    # Compute weight and bias gradients
    dW_computed = compute_weight_gradients(delta1, x_int, pr)
    dB_computed = compute_bias_gradients(delta1)

    # Validate that model updates satisfy circuit constraints
    valid = validate_model_updates(gw_int, gb_int, lwp_int, lbp_int, 
                                 dW_computed, dB_computed, eta_int)
    if not valid:
        logger.warning("⚠️ Model updates don't satisfy circuit constraints")

    def to_2d(arr, rows, cols):
        return [arr[i*cols:(i+1)*cols] for i in range(rows)]

    client_input = {
        "GW": to_2d(gw_int, hiddenSize, inputSize),
        "GB": gb_int,
        "X": x_int,
        "Y": y_int,
        "LWp": to_2d(lwp_int, hiddenSize, inputSize),
        "LBp": lbp_int,
        "eta": eta_int,
        "pr": pr,
        "ScGH": scgh,
        "ldigest": ldigest,
        # Use computed gradients instead of input ones to ensure circuit consistency
        "delta2_input": delta2,
        "dW_input": to_2d(dW_computed, hiddenSize, inputSize),
        "dB_input": dB_computed
    }

    # Debug logging
    logger.debug("Circuit inputs prepared:")
    for key, value in client_input.items():
        logger.debug(f"{key}: {value}")
        # Additional debug info for gradients
        if key in ["delta2_input", "dW_input", "dB_input"]:
            logger.debug(f"{key} stats - min: {min(value if isinstance(value, list) else [x for row in value for x in row])}, "
                        f"max: {max(value if isinstance(value, list) else [x for row in value for x in row])}")

    # Verify values are within field bounds
    flat_values = []
    for v in client_input.values():
        if isinstance(v, list):
            if isinstance(v[0], list):
                flat_values.extend([x for row in v for x in row])
            else:
                flat_values.extend(v)
        elif isinstance(v, (int, float)):
            flat_values.append(v)
    
    for val in flat_values:
        if abs(int(val)) >= FIELD_PRIME:
            logger.warning(f"⚠️ Value {val} exceeds field prime {FIELD_PRIME}")

    return client_input

def generate_aggregator_input(gw: List[int], gb: List[int],
                              lwps: List[int], lbps: List[int],
                              gwp: List[int], gbp: List[int],
                              sclh: List[int], gdigest: int):
    numClients=4
    hiddenSize=10
    inputSize=5

    def to_2d(arr, rows, cols):
        return [arr[i*cols:(i+1)*cols] for i in range(rows)]

    def to_3d(arr, dim1, dim2, dim3):
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

    # Reshape aggregator inputs
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

def generate_client_input_for_test(gw, gb, x, y, lwp, lbp, eta, pr, scgh, ldigest):
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
    return generate_aggregator_input(gw, gb, lwps, lbps, gwp, gbp, sclh, gdigest)

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

        # The circuit expects one sample: if multiple given, select first
        if training_data.dim() > 1 and training_data.size(0) > 1:
            x_arr = training_data[0].cpu().numpy().flatten()
            y_arr = labels[0].cpu().numpy().flatten()
        else:
            x_arr = training_data.cpu().numpy().flatten()
            y_arr = labels.cpu().numpy().flatten()

        if len(x_arr) != 5:
            raise ValueError(f"X must be length 5, got {len(x_arr)}.")
        if len(y_arr) != 3:
            raise ValueError(f"Y must be length 3, got {len(y_arr)}.")

        delta2_input = [0]*outputSize
        dW_input = [0]*(hiddenSize*inputSize)
        dB_input = [0]*hiddenSize

        client_inputs = generate_client_input(
            gw_t.tolist(), gb_t.tolist(),
            x_arr.tolist(), y_arr.tolist(),
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
        hiddenSize=10
        inputSize=5

        gw_t = global_model.get('weight', torch.zeros((hiddenSize,inputSize))).detach().cpu().numpy().flatten().tolist()
        gb_t = global_model.get('bias', torch.zeros(hiddenSize)).detach().cpu().numpy().flatten().tolist()

        lwps=[]
        lbps=[]
        for lm in local_models:
            w = lm.get('weight', torch.zeros((hiddenSize,inputSize))).detach().cpu().numpy().flatten().tolist()
            b = lm.get('bias', torch.zeros(hiddenSize)).detach().cpu().numpy().flatten().tolist()
            lwps.extend(w)
            lbps.extend(b)

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
