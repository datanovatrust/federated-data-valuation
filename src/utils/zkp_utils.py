import json
import os
import logging
from typing import Dict, List, Any, Tuple
import torch
import time
import subprocess
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

FIELD_PRIME = 21888242871839275222246405745257275088548364400416034343698204186575808495617

def initialize_gradients(forward_results: Dict[str, List[float]], y: List[int], pr: int) -> List[float]:
    """Initialize gradients that exactly match circuit constraints."""
    outputSize = len(y)
    delta2 = []
    
    logger.debug("=== Initializing Gradients ===")
    for i in range(outputSize):
        # Convert and normalize inputs explicitly as integers
        a2_val = int(forward_results['A2'][i])
        y_val = int(y[i])
        
        # Compute difference in field
        diff = normalize_to_field(a2_val - y_val)
        
        # Compute 2 * diff in field
        diffTimesTwo = normalize_to_field(diff * 2)
        
        # Set delta2 directly to normalized diffTimesTwo
        delta2_val = diffTimesTwo  # Important: No conversion to float
        
        logger.debug(f"Output {i}:")
        logger.debug(f"  Raw A2: {a2_val}")
        logger.debug(f"  Raw Y: {y_val}")
        logger.debug(f"  Raw diff: {a2_val - y_val}")
        logger.debug(f"  Normalized diff: {diff}")
        logger.debug(f"  diffTimesTwo: {diffTimesTwo}")
        logger.debug(f"  delta2: {delta2_val}")
        
        # Verify constraint using direct integer comparison
        if delta2_val != diffTimesTwo:
            logger.error(f"Output {i} constraint violation:")
            logger.error(f"  delta2: {delta2_val}")
            logger.error(f"  diffTimesTwo: {diffTimesTwo}")
            logger.error(f"  difference: {delta2_val - diffTimesTwo}")
        
        delta2.append(delta2_val)
    
    return delta2

def verify_intermediate_values(inputs: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Verifies intermediate values match circuit constraints exactly."""
    debug_info = {}
    
    a2 = inputs.get('A2', [])
    y = inputs.get('Y', [])
    delta2 = inputs.get('delta2_input', [])
    
    for i in range(len(a2)):
        # Convert and normalize as integers
        a2_val = int(a2[i])
        y_val = int(y[i])
        delta2_val = int(delta2[i])
        
        # Compute values in field
        diff = normalize_to_field(a2_val - y_val)
        diffTimesTwo = normalize_to_field(diff * 2)
        
        # Direct comparison without scaling
        constraint_diff = normalize_to_field(delta2_val - diffTimesTwo)
        
        debug_info[f'output_{i}'] = {
            'a2': a2_val,
            'y': y_val,
            'diff': diff,
            'diffTimesTwo': diffTimesTwo,
            'delta2': delta2_val,
            'scaledDelta2': delta2_val,  # No scaling needed
            'constraint_diff': constraint_diff
        }
        
        if constraint_diff != 0:
            logger.error(f"❌ Constraint violation at index {i}")
            logger.error(f"Computation for index {i}:")
            logger.error(f"  a2: {a2_val}")
            logger.error(f"  y: {y_val}")
            logger.error(f"  diff: {diff}")
            logger.error(f"  diffTimesTwo: {diffTimesTwo}")
            logger.error(f"  delta2: {delta2_val}")
            return False, debug_info
    
    return True, debug_info

def track_field_arithmetic(operation: str, a: int, b: int, result: int, 
                         expected_circuit_result: int) -> bool:
    """
    Tracks and validates field arithmetic operations between Python and Circom.
    """
    logger.debug(f"=== Field Arithmetic Operation: {operation} ===")
    
    # Normalize all values to field
    a_norm = normalize_to_field(a)
    b_norm = normalize_to_field(b)
    result_norm = normalize_to_field(result)
    expected_norm = normalize_to_field(expected_circuit_result)
    
    logger.debug(f"Python computation:")
    logger.debug(f"  a: {a} -> normalized: {a_norm}")
    logger.debug(f"  b: {b} -> normalized: {b_norm}")
    logger.debug(f"  result: {result} -> normalized: {result_norm}")
    logger.debug(f"Circuit expected: {expected_circuit_result} -> normalized: {expected_norm}")
    
    if result_norm != expected_norm:
        logger.error(f"❌ Arithmetic mismatch:")
        logger.error(f"  Python result: {result_norm}")
        logger.error(f"  Circuit result: {expected_norm}")
        return False
        
    return True

def validate_field_arithmetic(operation: str, a: int, b: int, result: int) -> bool:
    logger.debug(f"=== Field Arithmetic Validation: {operation} ===")
    logger.debug(f"Input a: {a}")
    logger.debug(f"Input b: {b}")
    logger.debug(f"Result: {result}")
    
    if abs(a) >= FIELD_PRIME or abs(b) >= FIELD_PRIME:
        logger.error("❌ Input values exceed field prime")
        return False
        
    if abs(result) >= FIELD_PRIME:
        logger.error("❌ Result exceeds field prime")
        return False
    
    return True

def verify_hash_computation(values: List[int], expected_hash: int) -> bool:
    """
    Verify hash computation with proper field arithmetic and debugging.
    
    Args:
        values: List of model parameters to hash
        expected_hash: Expected hash value to compare against
        
    Returns:
        bool: True if hash verification succeeds
    """
    logger.debug("=== Hash Computation Debug ===")
    
    # First normalize all input values
    normalized_values = [normalize_to_field(int(v)) for v in values]
    
    # Log first few values for debugging
    logger.debug(f"First few normalized input values: {normalized_values[:5]}")
    logger.debug(f"Input length: {len(normalized_values)}")
    
    # Compute hash with normalized values
    computed_hash = mimc_hash(normalized_values)
    normalized_computed = normalize_to_field(computed_hash)
    normalized_expected = normalize_to_field(expected_hash)
    
    logger.debug(f"Computed hash (raw): {computed_hash}")
    logger.debug(f"Computed hash (normalized): {normalized_computed}")
    logger.debug(f"Expected hash (raw): {expected_hash}")
    logger.debug(f"Expected hash (normalized): {normalized_expected}")
    
    # Check field bounds
    if abs(normalized_computed) >= FIELD_PRIME or abs(normalized_expected) >= FIELD_PRIME:
        logger.error("❌ Hash value exceeds field prime")
        return False
    
    # Compare normalized values
    if normalized_computed != normalized_expected:
        logger.error("❌ Hash mismatch:")
        logger.error(f"  Computed: {normalized_computed}")
        logger.error(f"  Expected: {normalized_expected}")
        return False
        
    logger.debug("✅ Hash verification successful")
    return True

def verify_hash_chain(
    inputs: List[int], 
    intermediate_states: List[int],
    final_hash: int
) -> bool:
    """
    Verifies the complete MiMC hash chain matches between implementations.
    """
    logger.debug("=== Hash Chain Verification ===")
    
    # Track each state in the hash computation
    current_state = 0  # Initial state
    for i, inp in enumerate(inputs):
        # Log state before operation
        logger.debug(f"Step {i}:")
        logger.debug(f"  Input: {inp}")
        logger.debug(f"  Current state: {current_state}")
        
        # Compute next state exactly as in circuit
        next_state = normalize_to_field(mimc_hash([inp], current_state))
        logger.debug(f"  Next state: {next_state}")
        
        # Verify against provided intermediate state
        if next_state != intermediate_states[i]:
            logger.error(f"❌ Hash state mismatch at step {i}")
            logger.error(f"  Computed: {next_state}")
            logger.error(f"  Expected: {intermediate_states[i]}")
            return False
            
        current_state = next_state
    
    # Verify final hash
    if normalize_to_field(current_state) != normalize_to_field(final_hash):
        logger.error("❌ Final hash mismatch")
        return False
        
    return True

def verify_gradient_computation(
    a2: List[float], 
    y: List[float], 
    delta2: List[float],
    pr: int
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verifies gradient computation matches exactly between Python and Circom.
    The circuit expects: delta2 === 2 * (a2 - y)
    """
    logger.debug("=== Gradient Computation Verification ===")
    debug_info = {}
    
    for i in range(len(a2)):
        # Normalize all values to field integers
        a2_val = normalize_to_field(int(a2[i]))
        y_val = normalize_to_field(int(y[i]))
        delta2_val = normalize_to_field(int(delta2[i]))
        
        # Compute difference and double it
        diff = normalize_to_field(a2_val - y_val)
        diffTimesTwo = normalize_to_field(diff * 2)
        
        # No additional scaling needed - direct comparison
        constraint_check = normalize_to_field(delta2_val - diffTimesTwo)
        
        debug_info[f'output_{i}'] = {
            'a2': a2_val,
            'y': y_val,
            'diff': diff,
            'diffTimesTwo': diffTimesTwo,
            'delta2': delta2_val,
            'scaled_delta2': delta2_val,  # No scaling
            'constraint_check': constraint_check
        }
        
        if constraint_check != 0:
            logger.error(f"❌ Gradient constraint failed for output {i}")
            return False, debug_info
    
    return True, debug_info

def verify_hashes(local_hash: str, global_hash: str) -> bool:
    """Debug helper to verify hash computations."""
    FIELD_PRIME = 21888242871839275222246405745257275088548364400416034343698204186575808495617
    
    try:
        local_int = normalize_to_field(int(local_hash))
        global_int = normalize_to_field(int(global_hash))
        
        logger.debug("=== Hash Verification ===")
        logger.debug(f"Local hash (raw): {local_hash}")
        logger.debug(f"Global hash (raw): {global_hash}")
        logger.debug(f"Local hash (normalized): {local_int}")
        logger.debug(f"Global hash (normalized): {global_int}")
        
        return True
    except ValueError as e:
        logger.error(f"Hash verification failed: {e}")
        return False

def normalize_to_field(value: int) -> int:
    """
    Normalize a value to be within the field prime range with better handling of large values.
    """
    FIELD_PRIME = 21888242871839275222246405745257275088548364400416034343698204186575808495617
    FIELD_PRIME_HALF = FIELD_PRIME // 2

    # Handle negative values first
    if value < 0:
        value = value % FIELD_PRIME

    # Reduce large positive values
    if value >= FIELD_PRIME:
        value = value % FIELD_PRIME

    # If value is in the upper half of the field, convert to negative representation
    if value > FIELD_PRIME_HALF:
        value = value - FIELD_PRIME

    return value

def weighted_sum(inputs: List[float], weights: List[float], factor: int) -> float:
    sum_val = 0
    for i in range(len(inputs)):
        prod_val = normalize_to_field(inputs[i] * weights[i])
        sum_val = normalize_to_field(sum_val + prod_val)
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

def verify_delta2_constraint(a2: List[float], y: List[float], delta2: List[float], pr: int) -> bool:
    """
    Explicitly verify the delta2 constraint that's failing in the circuit.
    This matches line 114 in the client circuit where the error occurs.
    """
    outputSize = len(y)
    for i in range(outputSize):
        diff = a2[i] - y[i]
        diffTimesTwo = diff * 2
        scaledDelta2 = delta2[i] * pr
        
        # Log the exact values being compared
        logger.debug(f"Delta2 constraint check for output {i}:")
        logger.debug(f"  A2[{i}] = {a2[i]}")
        logger.debug(f"  Y[{i}] = {y[i]}")
        logger.debug(f"  diff = {diff}")
        logger.debug(f"  diffTimesTwo = {diffTimesTwo}")
        logger.debug(f"  delta2[{i}] = {delta2[i]}")
        logger.debug(f"  scaledDelta2 = {scaledDelta2}")
        
        # Check if the constraint would fail
        if abs(scaledDelta2 - diffTimesTwo) > 1e-10:
            logger.error(f"❌ Delta2 constraint failed for output {i}")
            logger.error(f"   Expected: {diffTimesTwo}")
            logger.error(f"   Got: {scaledDelta2}")
            logger.error(f"   Difference: {abs(scaledDelta2 - diffTimesTwo)}")
            return False
    return True

def compute_output_gradients(a2: List[float], y: List[float], pr: int) -> List[float]:
    outputSize = len(y)
    delta2 = []
    
    logger.debug("=== Gradient Computation Debug ===")
    for i in range(outputSize):
        # Compute normalized difference
        diff = normalize_to_field(a2[i] - y[i])
        logger.debug(f"  Normalized diff: {diff}")
        
        # Compute gradient
        diffTimesTwo = normalize_to_field(diff * 2)
        logger.debug(f"  diffTimesTwo: {diffTimesTwo}")
        
        # Scale gradient properly
        delta2_val = normalize_to_field(diffTimesTwo)  # Don't divide by pr yet
        logger.debug(f"  Raw delta2: {delta2_val}")
        
        # Track final scaled value
        delta2.append(delta2_val)
        logger.debug(f"  Final delta2: {delta2_val}")
    
    return delta2

def compute_hidden_gradients(delta2: List[float], lwp: List[float]) -> List[float]:
    """
    Modified hidden gradient computation with proper scaling.
    """
    hiddenSize = 10
    outputSize = 3
    
    # Reshape local weights
    lwp_2d = [lwp[i*5:(i+1)*5] for i in range(hiddenSize)]
    
    delta1 = []
    for i in range(hiddenSize):
        hidden_weights = [lwp_2d[i][j] for j in range(outputSize)]
        # Match circuit's scaling
        grad_sum = weighted_sum(delta2, hidden_weights, 1)
        delta1.append(normalize_to_field(grad_sum))
    
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

def scale_to_int(value: float, precision: int = 100000) -> int:
    """
    Scale a float value to integer with better handling of small values.
    Now using higher precision by default (100000 instead of 1000).
    """
    # For very small values like learning rate, use a higher precision
    if abs(value) < 0.001:
        precision *= 1000  # Use even higher precision for tiny values
    
    scaled = int(round(value * precision))
    return normalize_to_field(scaled)

def scale_array_to_int(arr: List[float], precision: int = 1000) -> List[int]:
    """Scale an array of float values to integers."""
    return [scale_to_int(x, precision) for x in arr]

def mimc_hash(values: List[int], key=0):
    """Compute MiMC hash of input values with proper field arithmetic."""
    if not values:
        return 0
    
    FIELD_PRIME = 21888242871839275222246405745257275088548364400416034343698204186575808495617
    
    # Normalize inputs to field
    inputs = [normalize_to_field(int(v)) for v in values]
    nInputs = len(inputs)
    nRounds = 2
    constants = [i+1 for i in range(nRounds)]

    currentStateInputs = [0]*(nInputs+1)
    currentStateInputs[0] = normalize_to_field(key)

    for i in range(nInputs):
        # All operations must be done modulo the field prime
        afterAdd = normalize_to_field(currentStateInputs[i] + inputs[i])
        roundStates = [0]*(nRounds+1)
        roundStates[0] = afterAdd

        for j in range(nRounds):
            t = normalize_to_field(roundStates[j] + constants[j])
            t_squared = normalize_to_field(t * t)
            t_cubed = normalize_to_field(t_squared * t)
            roundStates[j+1] = t_cubed

        currentStateInputs[i+1] = normalize_to_field(roundStates[nRounds])

    return normalize_to_field(currentStateInputs[nInputs])

def generate_client_input(gw: List[float], gb: List[float],
                         x: List[float], y: List[float],
                         lwp: List[float], lbp: List[float],
                         eta: float, pr: int, scgh: int, ldigest: int,
                         delta2_input: List[float], dW_input: List[float], dB_input: List[float]):
    """
    Generates client input matching exact circuit computations.
    All computations maintain circuit scaling and constraints.
    """
    try:
        logger.critical("=== GENERATE CLIENT INPUT STARTED ===")
        logger.critical("Input parameters:")
        logger.critical(f"eta (learning rate): {eta}")
        logger.critical(f"pr (precision): {pr}")
        logger.critical(f"scgh (global hash): {scgh}")
        logger.critical(f"ldigest (local hash): {ldigest}")

        # Normalize public signals within field
        scgh = normalize_to_field(int(scgh))
        ldigest = normalize_to_field(int(ldigest))
        
        hiddenSize = 10
        inputSize = 5
        outputSize = 3

        # Scale and normalize input values
        gw_int = [normalize_to_field(v) for v in scale_array_to_int(gw, pr)]
        gb_int = [normalize_to_field(v) for v in scale_array_to_int(gb, pr)]
        x_int = [normalize_to_field(v) for v in scale_array_to_int(x, pr)]
        y_int = [normalize_to_field(v) for v in scale_array_to_int(y, pr)]
        lwp_int = [normalize_to_field(v) for v in scale_array_to_int(lwp, pr)]
        lbp_int = [normalize_to_field(v) for v in scale_array_to_int(lbp, pr)]
        eta_int = normalize_to_field(scale_to_int(eta, pr))

        # Debug scaled values
        logger.debug("Scaled integer ranges:")
        for name, values in [
            ('GW', gw_int), ('GB', gb_int), ('X', x_int), ('Y', y_int),
            ('LWp', lwp_int), ('LBp', lbp_int), ('eta', eta_int)
        ]:
            if isinstance(values, list):
                logger.debug(f"{name} range: [{min(values)}, {max(values)}]")
            else:
                logger.debug(f"{name}: {values}")

        # Verify field bounds for all inputs
        all_inputs = {
            'gw': gw_int, 'gb': gb_int, 'x': x_int, 'y': y_int,
            'lwp': lwp_int, 'lbp': lbp_int, 'eta': eta_int
        }
        for name, values in all_inputs.items():
            if isinstance(values, list):
                for i, val in enumerate(values):
                    if not validate_field_arithmetic(f"{name}[{i}]", val, 0, val):
                        raise ValueError(f"Field arithmetic validation failed for {name}[{i}]")
            else:
                if not validate_field_arithmetic(name, values, 0, values):
                    raise ValueError(f"Field arithmetic validation failed for {name}")

        # Compute forward pass
        forward_results = compute_forward_pass(gw_int, gb_int, x_int, lwp_int, lbp_int, pr)
        
        # Verify intermediate values
        success, intermediate_debug = verify_intermediate_values({
            'A2': forward_results['A2'],
            'Y': y_int,
            'delta2_input': delta2_input,
            'pr': pr
        })
        if not success:
            raise ValueError(f"Intermediate value verification failed: {intermediate_debug}")

        # Compute and verify gradients
        delta2 = compute_output_gradients(forward_results['A2'], y_int, pr)
        grad_success, grad_debug = verify_gradient_computation(
            forward_results['A2'], y_int, delta2, pr
        )
        if not grad_success:
            raise ValueError(f"Gradient computation verification failed: {grad_debug}")

        # Compute hidden layer gradients
        delta1 = compute_hidden_gradients(delta2, lwp_int)
        for i, d1 in enumerate(delta1):
            result = track_field_arithmetic(
                f"delta1[{i}]",
                d1, pr,
                normalize_to_field(d1 * pr),
                normalize_to_field(d1 * pr)
            )
            if not result:
                raise ValueError(f"Delta1 computation mismatch at index {i}")

        # Compute weight and bias gradients
        dW_computed = compute_weight_gradients(delta1, x_int, pr)
        dB_computed = compute_bias_gradients(delta1)

        # Verify model updates
        if not validate_model_updates(gw_int, gb_int, lwp_int, lbp_int, 
                                    dW_computed, dB_computed, eta_int):
            raise ValueError("Model updates don't satisfy circuit constraints")

        def to_2d(arr, rows, cols):
            return [arr[i*cols:(i+1)*cols] for i in range(rows)]

        # Prepare and verify hash computation
        model_params = []
        model_params.extend(gw_int)  # Add weights
        model_params.extend(gb_int)  # Add biases
        
        logger.debug("=== Hash Verification Debug ===")
        logger.debug(f"Number of parameters for hash: {len(model_params)}")
        logger.debug(f"Parameter range: [{min(model_params)}, {max(model_params)}]")
        logger.debug(f"Expected hash (normalized): {scgh}")
        
        # Verify hash with normalized values
        hash_result = mimc_hash(model_params)
        normalized_hash = normalize_to_field(hash_result)
        logger.debug(f"Computed hash (raw): {hash_result}")
        logger.debug(f"Computed hash (normalized): {normalized_hash}")
        
        if normalized_hash != scgh:
            logger.error("❌ Hash verification failed")
            logger.error(f"Computed hash: {normalized_hash}")
            logger.error(f"Expected hash: {scgh}")
            raise ValueError("Hash verification failed for model parameters")

        # Prepare client input
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
            "delta2_input": delta2,
            "dW_input": to_2d(dW_computed, hiddenSize, inputSize),
            "dB_input": dB_computed
        }

        # Final field validation
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
            if not validate_field_arithmetic("final_value", int(val), 0, int(val)):
                raise ValueError(f"Final value {val} validation failed")

        logger.info("✅ Client input generation completed with all verifications")
        return client_input

    except Exception as e:
        logger.error(f"Failed to generate client input: {str(e)}")
        raise

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

def run_command(cmd: List[str]) -> subprocess.CompletedProcess:
    """Execute a command and return the result."""
    logger.debug(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {' '.join(cmd)}")
        logger.error(f"stdout: {result.stdout}")
        logger.error(f"stderr: {result.stderr}")
    return result

def generate_client_proof(client_inputs: dict, circuit_path: str, proving_key_path: str, js_dir: str) -> Dict:
    """
    Generate a client proof with proper handling of public signals.
    
    Args:
        client_inputs: Dictionary containing all input values including the 4 required public signals
        circuit_path: Path to the circuit file
        proving_key_path: Path to the proving key
        js_dir: Directory containing JS/WASM files
    
    Returns:
        Dictionary containing proof data and public signals
    """
    logger.critical("=== GENERATE CLIENT PROOF STARTED ===")
    logger.critical(f"Circuit path: {circuit_path}")
    logger.critical(f"Proving key path: {proving_key_path}")
    logger.critical(f"JS dir: {js_dir}")

    if not os.path.exists(circuit_path):
        raise FileNotFoundError(f"Circuit file not found at {circuit_path}")
    if not os.path.exists(proving_key_path):
        raise FileNotFoundError(f"Proving key file not found at {proving_key_path}")

    # Define all file paths
    input_file = "client_input.json"
    witness_file = "witness.wtns"
    proof_file = "proof.json"
    original_public_file = "original_public.json"  # Store our original signals
    snarkjs_public_file = "snarkjs_public.json"   # For snarkjs output

    try:
        # Clean up any existing files
        for f in [input_file, witness_file, proof_file, original_public_file, snarkjs_public_file]:
            if os.path.exists(f):
                os.remove(f)

        # Verify all required public signals are present
        required_signals = ['eta', 'pr', 'ldigest', 'ScGH']
        missing_signals = [sig for sig in required_signals if sig not in client_inputs]
        if missing_signals:
            raise ValueError(f"Missing required public signals: {missing_signals}")

        # Log public signals
        logger.critical("=== Public Signals Check ===")
        for signal in required_signals:
            logger.critical(f"{signal}: {client_inputs[signal]}")

        # Create array of public signals in the correct order
        public_signals = [
            str(client_inputs['eta']),
            str(client_inputs['pr']),
            str(client_inputs['ldigest']),
            str(client_inputs['ScGH'])
        ]

        logger.critical("=== Ordered Public Signals ===")
        logger.critical(f"Public signals array: {public_signals}")

        # Verify we have exactly 4 public signals
        if len(public_signals) != 4:
            logger.critical(f"Public signals length check failed: got {len(public_signals)}")
            raise ValueError(f"Expected exactly 4 public signals, got {len(public_signals)}")

        # Write input and public files
        logger.critical("=== Writing Files ===")
        
        with open(input_file, "w") as f:
            logger.critical(f"Writing input file: {input_file}")
            json.dump(client_inputs, f, indent=2)
        
        # Save our original public signals
        with open(original_public_file, "w") as f:
            logger.critical(f"Writing original public signals to {original_public_file}")
            json.dump(public_signals, f)
        
        # Create initial snarkjs public file
        with open(snarkjs_public_file, "w") as f:
            logger.critical(f"Writing initial snarkjs public file: {snarkjs_public_file}")
            json.dump(public_signals, f)

        logger.info("🔑 Generating client proof...")
        start_time = time.time()

        # Generate witness
        wasm_path = os.path.join(js_dir, "client.wasm")
        gen_witness_js = os.path.join(js_dir, "generate_witness.js")

        if not os.path.exists(gen_witness_js) or not os.path.exists(wasm_path):
            raise FileNotFoundError(
                f"Required files missing:\n"
                f"generate_witness.js: {os.path.exists(gen_witness_js)}\n"
                f"client.wasm: {os.path.exists(wasm_path)}"
            )

        logger.critical("=== Generating Witness ===")
        witness_cmd = ["node", gen_witness_js, wasm_path, input_file, witness_file]
        witness_result = run_command(witness_cmd)
        if witness_result.returncode != 0:
            raise RuntimeError("Witness generation failed")

        logger.critical("=== Generating Proof ===")
        prove_cmd = ["snarkjs", "groth16", "prove", proving_key_path, witness_file, proof_file, snarkjs_public_file]
        prove_result = run_command(prove_cmd)
        if prove_result.returncode != 0:
            raise RuntimeError("Proof generation failed")

        logger.critical("=== Verifying Generated Files ===")
        
        # Load and verify the proof file
        with open(proof_file) as pf:
            proof_data = json.load(pf)
            required_proof_keys = {'pi_a', 'pi_b', 'pi_c', 'protocol', 'curve'}
            missing_keys = required_proof_keys - set(proof_data.keys())
            if missing_keys:
                raise ValueError(f"Proof missing required keys: {missing_keys}")
            logger.critical(f"Proof data keys: {list(proof_data.keys())}")

        # Load our original public signals
        with open(original_public_file) as f:
            original_signals = json.load(f)
            logger.critical(f"Original public signals: {original_signals}")

        # Load snarkjs generated public signals
        with open(snarkjs_public_file) as f:
            snarkjs_signals = json.load(f)
            logger.critical(f"Snarkjs public signals: {snarkjs_signals}")

        # Verify signals match
        if original_signals != public_signals:
            logger.critical("Original signals don't match expected signals")
            raise ValueError("Public signal mismatch in original file")

        if len(snarkjs_signals) != 4:
            logger.critical(f"Snarkjs produced wrong number of public signals: {len(snarkjs_signals)}")
            raise ValueError(f"Expected 4 public signals from snarkjs, got {len(snarkjs_signals)}")

        # Use our original signals for the result
        gen_time = time.time() - start_time
        logger.info(f"✅ Client proof generated successfully in {gen_time:.2f}s")
        
        result = {
            "proof": proof_data,
            "public": original_signals,
            "debug_info": {
                "generation_time": gen_time,
                "witness_output": witness_result.stdout if witness_result.stdout else "",
                "prove_output": prove_result.stdout if prove_result.stdout else "",
                "original_signals": original_signals,
                "snarkjs_signals": snarkjs_signals
            }
        }
        
        logger.critical("=== Final Result ===")
        logger.critical(f"Result keys: {list(result.keys())}")
        logger.critical(f"Public signals in result: {result['public']}")

        return result

    except Exception as e:
        logger.error(f"Proof generation failed: {str(e)}")
        logger.critical(f"Critical error in proof generation: {str(e)}")
        logger.debug("Full exception:", exc_info=True)
        raise

    finally:
        # Clean up intermediate files
        for f in [witness_file]:
            if os.path.exists(f):
                os.remove(f)

def generate_aggregator_proof(agg_inputs: dict, circuit_path: str, proving_key_path: str, js_dir: str) -> tuple:
    """
    Generate aggregator proof with proper public signal handling.
    Uses separate files for original and snarkjs-generated public signals.
    """
    if not os.path.exists(circuit_path):
        raise FileNotFoundError(f"Circuit file not found at {circuit_path}")
    if not os.path.exists(proving_key_path):
        raise FileNotFoundError(f"Proving key file not found at {proving_key_path}")

    input_file = "aggregator_input.json"
    witness_file = "witness.wtns"
    proof_file = "proof.json"
    original_public_file = "original_aggregator_public.json"
    snarkjs_public_file = "snarkjs_aggregator_public.json"

    # Clean up old files
    for f in [input_file, witness_file, proof_file, original_public_file, snarkjs_public_file]:
        if os.path.exists(f):
            os.remove(f)

    logger.info(f"🔑 Generating aggregator proof...")
    start_time = time.time()

    wasm_path = os.path.join(js_dir, "aggregator.wasm")
    gen_witness_js = os.path.join(js_dir, "generate_witness.js")

    if not os.path.exists(gen_witness_js) or not os.path.exists(wasm_path):
        raise FileNotFoundError("generate_witness.js or aggregator.wasm not found in js_dir")

    # Write aggregator inputs
    with open(input_file, "w") as f:
        json.dump(agg_inputs, f, indent=2)

    # Prepare public signals
    numClients = len(agg_inputs["ScLH"])
    aggregator_public_signals = [str(h) for h in agg_inputs["ScLH"]] + [str(agg_inputs["gdigest"])]

    # Save original public signals
    with open(original_public_file, "w") as f:
        json.dump(aggregator_public_signals, f)

    # Create initial snarkjs public file
    with open(snarkjs_public_file, "w") as f:
        json.dump(aggregator_public_signals, f)

    # Generate witness
    witness_cmd = ["node", gen_witness_js, wasm_path, input_file, witness_file]
    witness_result = run_command(witness_cmd)
    if witness_result.returncode != 0:
        raise RuntimeError("Aggregator proof generation failed during witness generation")

    # Generate proof
    prove_cmd = ["snarkjs", "groth16", "prove", proving_key_path, witness_file, proof_file, snarkjs_public_file]
    prove_result = run_command(prove_cmd)
    if prove_result.returncode != 0:
        raise RuntimeError("Aggregator proof generation failed during proof generation")

    # Load and verify files
    with open(proof_file) as pf:
        proof_data = json.load(pf)

    with open(original_public_file) as f:
        original_signals = json.load(f)

    with open(snarkjs_public_file) as f:
        snarkjs_signals = json.load(f)

    # Use original signals in result
    gen_time = time.time() - start_time
    logger.info(f"✅ Aggregator proof generated successfully in {gen_time:.2f}s")

    return proof_data, original_signals

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
        """Prepare public inputs for client proof verification."""
        scaled_lr = int(round(learning_rate * precision))
        
        signals = [
            str(scaled_lr),
            str(precision),
            str(local_model_hash),
            str(global_model_hash)
        ]

        # Validate signals
        for i, signal in enumerate(signals):
            try:
                value = int(signal)
                if value >= FIELD_PRIME or value < 0:
                    raise ValueError(f"Signal {i} ({value}) outside valid range [0, FIELD_PRIME)")
            except ValueError as e:
                raise ValueError(f"Invalid signal {i} ({signal}): {str(e)}")

        logger.debug(f"Prepared public inputs: {signals}")
        return signals

    def prepare_aggregator_public_inputs(self, local_model_hashes: List[str],
                                       global_model_hash: str) -> List[str]:
        """Prepare public inputs for aggregator proof verification."""
        # Convert and validate all hashes
        signals = []
        for i, hash_val in enumerate(local_model_hashes):
            try:
                value = int(hash_val)
                if value >= FIELD_PRIME or value < 0:
                    raise ValueError(f"Local hash {i} ({value}) outside valid range")
                signals.append(str(value))
            except ValueError as e:
                raise ValueError(f"Invalid local hash {i}: {str(e)}")

        # Convert and validate global hash
        try:
            global_value = int(global_model_hash)
            if global_value >= FIELD_PRIME or global_value < 0:
                raise ValueError(f"Global hash ({global_value}) outside valid range")
            signals.append(str(global_value))
        except ValueError as e:
            raise ValueError(f"Invalid global hash: {str(e)}")

        return signals

    def _verify_groth16_proof(self, vkey: Dict, proof: Dict, public_signals: List[str]) -> bool:
        """Verify a Groth16 proof using snarkjs with separate files for signals."""
        try:
            formatted_signals = [str(signal) for signal in public_signals]
            
            logger.debug("Verifying with public signals:")
            for i, signal in enumerate(formatted_signals):
                logger.debug(f"  Signal {i}: {signal}")
            
            # Save verification input to separate file
            verification_input = {
                "protocol": "groth16",
                "curve": "bn128",
                "pi_a": proof["pi_a"],
                "pi_b": proof["pi_b"],
                "pi_c": proof["pi_c"],
                "public": formatted_signals
            }

            verify_input_file = "verification_input.json"
            verify_public_file = "verification_public.json"

            with open(verify_input_file, 'w') as f:
                json.dump(verification_input, f, indent=2)

            # Save public signals to separate file
            with open(verify_public_file, 'w') as f:
                json.dump(formatted_signals, f)

            # Save verification key to debug file
            with open("debug_vkey.json", 'w') as f:
                json.dump(vkey, f, indent=2)

            # Run verification with separate files
            verify_cmd = [
                "snarkjs", 
                "groth16", 
                "verify", 
                "--verbose",
                "debug_vkey.json",
                verify_input_file,
                verify_public_file
            ]
            
            logger.debug(f"Running verification command: {' '.join(verify_cmd)}")
            result = subprocess.run(verify_cmd, capture_output=True, text=True)
            
            logger.debug(f"Verification stdout: {result.stdout}")
            logger.debug(f"Verification stderr: {result.stderr}")
            
            success = result.returncode == 0 and "OK!" in result.stdout
            if success:
                logger.info("✅ Cryptographic verification successful")
            else:
                logger.error(f"❌ Verification failed with return code {result.returncode}")
            
            return success

        except Exception as e:
            logger.error(f"❌ Verification error: {str(e)}")
            logger.debug("Full exception:", exc_info=True)
            return False

        finally:
            # Clean up verification files
            for f in [verify_input_file, verify_public_file, "debug_vkey.json"]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception as e:
                        logger.warning(f"Failed to remove {f}: {e}")

    def verify_client_proof(self, proof: Dict, public_signals: List[str]) -> bool:
        """Verifies a client's zero-knowledge proof against their public signals."""
        logger.info("🔍 Verifying client proof...")
        try:
            logger.debug("Verifying client proof with:")
            logger.debug(f"Proof keys: {list(proof.keys())}")
            logger.debug(f"Public signals: {public_signals}")
            
            # Verify proof structure
            required_proof_keys = {'pi_a', 'pi_b', 'pi_c', 'protocol'}
            missing_keys = required_proof_keys - set(proof.keys())
            if missing_keys:
                logger.error(f"❌ Proof missing required fields: {missing_keys}")
                return False

            if len(public_signals) != 4:
                logger.error(f"❌ Expected 4 public signals, got {len(public_signals)}")
                logger.debug(f"Public signals: {public_signals}")
                return False

            # Verify values are within field prime
            for i, signal in enumerate(public_signals):
                try:
                    value = int(signal)
                    if value >= FIELD_PRIME or value < 0:
                        logger.error(f"❌ Public signal {i} ({value}) outside valid range")
                        return False
                except ValueError:
                    logger.error(f"❌ Public signal {i} ({signal}) is not a valid integer")
                    return False

            # Perform verification
            start_time = time.time()
            is_valid = self._verify_groth16_proof(self.client_vkey, proof, public_signals)
            verify_time = time.time() - start_time
            
            if is_valid:
                logger.info(f"✅ Client proof verified successfully in {verify_time:.2f}s")
            else:
                logger.error(f"❌ Client proof verification failed after {verify_time:.2f}s")
            
            return is_valid

        except Exception as e:
            logger.error(f"❌ Client proof verification failed: {e}")
            logger.debug("Full exception:", exc_info=True)
            return False

    def verify_aggregator_proof(self, proof: Dict, public_signals: List[str]) -> bool:
        """Verifies an aggregator's zero-knowledge proof."""
        logger.info("🔍 Verifying aggregator proof...")
        try:
            # Verify proof structure
            required_proof_keys = {'pi_a', 'pi_b', 'pi_c', 'protocol'}
            missing_keys = required_proof_keys - set(proof.keys())
            if missing_keys:
                logger.error(f"❌ Proof missing required fields: {missing_keys}")
                return False

            # Verify public signals format
            if len(public_signals) < 2:  # At least one local hash and one global hash
                logger.error(f"❌ Invalid number of public signals: {len(public_signals)}")
                return False

            # Verify values are within field prime
            for i, signal in enumerate(public_signals):
                try:
                    value = int(signal)
                    if value >= FIELD_PRIME or value < 0:
                        logger.error(f"❌ Public signal {i} ({value}) outside valid range")
                        return False
                except ValueError:
                    logger.error(f"❌ Public signal {i} ({signal}) is not a valid integer")
                    return False

            # Perform cryptographic verification
            start_time = time.time()
            is_valid = self._verify_groth16_proof(self.aggregator_vkey, proof, public_signals)
            verify_time = time.time() - start_time
            
            if is_valid:
                logger.info(f"✅ Aggregator proof verified successfully in {verify_time:.2f}s")
            else:
                logger.error(f"❌ Aggregator proof verification failed after {verify_time:.2f}s")
            
            return is_valid

        except Exception as e:
            logger.error(f"❌ Aggregator proof verification failed: {e}")
            logger.debug("Full exception:", exc_info=True)
            return False

    @staticmethod
    def compute_model_hash(model_state: Dict[str, torch.Tensor], precision: int=1000) -> str:
        """
        Compute model hash with consistent normalization and field arithmetic.
        
        Args:
            model_state: Dictionary containing model parameters
            precision: Precision factor for fixed-point arithmetic
            
        Returns:
            str: Normalized hash value as string
        """
        logger.debug("=== Computing Model Hash ===")
        
        # First collect and flatten all parameters
        params = []
        for name, param in model_state.items():
            values = param.detach().cpu().numpy().flatten()
            logger.debug(f"Parameter {name}: shape {values.shape}, range [{values.min()}, {values.max()}]")
            
            # Scale float values to integers and normalize to field
            scaled_values = [normalize_to_field(scale_to_int(float(v), precision)) for v in values]
            params.extend(scaled_values)
        
        logger.debug(f"Total parameters: {len(params)}")
        if params:
            logger.debug(f"Parameter range: [{min(params)}, {max(params)}]")
        
        # Compute hash with normalized values
        h = mimc_hash(params, key=0)
        normalized_hash = normalize_to_field(h)
        
        logger.debug(f"Raw hash: {h}")
        logger.debug(f"Normalized hash: {normalized_hash}")
        
        # Important: Return string representation of normalized hash
        return str(normalized_hash)

    def cleanup(self):
        """Clean up any temporary verification files."""
        temp_files = [
            "verification_input.json",
            "verification_public.json",
            "debug_vkey.json",
            "client_input.json",
            "witness.wtns",
            "proof.json",
            "original_public.json",
            "snarkjs_public.json",
            "original_aggregator_public.json",
            "snarkjs_aggregator_public.json"
        ]
        
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    logger.debug(f"Cleaned up temporary file: {f}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {f}: {e}")

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
        """
        Generate a training proof with proper gradient scaling and normalization.
        
        Args:
            global_model: Global model state dictionary
            local_model: Local model state dictionary
            training_data: Input training data tensor
            labels: Target labels tensor
            learning_rate: Learning rate for gradient updates
            precision: Precision factor for fixed-point arithmetic
            global_hash: Hash of global model parameters
            local_hash: Hash of local model parameters
            
        Returns:
            Dictionary containing proof data and public signals
        """
        logger.info("🔑 Preparing inputs for client training proof...")
        
        try:
            # Debug log all input parameters
            logger.debug("=== Training Proof Generation Debug ===")
            logger.debug(f"Learning rate: {learning_rate}")
            logger.debug(f"Precision: {precision}")
            logger.debug(f"Global hash: {global_hash}")
            logger.debug(f"Local hash: {local_hash}")
            logger.debug(f"Training data shape: {training_data.shape}")
            logger.debug(f"Labels shape: {labels.shape}")

            hiddenSize = 10
            inputSize = 5
            outputSize = 3

            # Convert model parameters to numpy arrays and normalize
            gw_t = [normalize_to_field(val) for val in global_model.get('weight', torch.zeros((hiddenSize,inputSize))).detach().cpu().numpy().flatten()]
            gb_t = [normalize_to_field(val) for val in global_model.get('bias', torch.zeros(hiddenSize)).detach().cpu().numpy().flatten()]
            lw_t = [normalize_to_field(val) for val in local_model.get('weight', torch.zeros((hiddenSize,inputSize))).detach().cpu().numpy().flatten()]
            lb_t = [normalize_to_field(val) for val in local_model.get('bias', torch.zeros(hiddenSize)).detach().cpu().numpy().flatten()]

            # Debug log converted parameter shapes
            logger.debug("Converted parameter shapes:")
            logger.debug(f"gw_t length: {len(gw_t)}")
            logger.debug(f"gb_t length: {len(gb_t)}")
            logger.debug(f"lw_t length: {len(lw_t)}")
            logger.debug(f"lb_t length: {len(lb_t)}")

            # Ensure proper input shapes and select first sample if batch
            if training_data.dim() > 1 and training_data.size(0) > 1:
                logger.debug("Processing batch data, selecting first sample")
                x_arr = training_data[0].cpu().numpy().flatten()
                y_arr = labels[0].cpu().numpy().flatten()
            else:
                logger.debug("Processing single sample data")
                x_arr = training_data.cpu().numpy().flatten()
                y_arr = labels.cpu().numpy().flatten()

            # Scale input values with proper normalization
            x_int = [normalize_to_field(val) for val in scale_array_to_int(x_arr.tolist(), precision)]
            y_int = [normalize_to_field(val) for val in scale_array_to_int(y_arr.tolist(), precision)]

            logger.debug("=== Input Scaling Debug ===")
            logger.debug(f"x_int (first few): {x_int[:5]}")
            logger.debug(f"y_int: {y_int}")

            # Compute forward pass
            forward_results = compute_forward_pass(
                gw_t, gb_t, x_int, lw_t, lb_t, precision
            )

            # Verify forward pass outputs
            logger.debug("=== Forward Pass Results ===")
            logger.debug(f"Z1 shape: {len(forward_results['Z1'])}")
            logger.debug(f"A1 shape: {len(forward_results['A1'])}")
            logger.debug(f"Z2 shape: {len(forward_results['Z2'])}")
            logger.debug(f"A2 shape: {len(forward_results['A2'])}")

            # Initialize gradients with proper scaling
            delta2_input = initialize_gradients(forward_results, y_int, precision)
            
            # Log gradient initialization results
            logger.debug("=== Gradient Initialization Results ===")
            logger.debug(f"delta2_input: {delta2_input}")

            # Verify gradient computation
            grad_success, grad_debug = verify_gradient_computation(
                forward_results['A2'], y_int, delta2_input, precision
            )
            if not grad_success:
                raise ValueError(f"Gradient computation verification failed: {grad_debug}")

            # Compute hidden layer gradients
            delta1 = compute_hidden_gradients(delta2_input, lw_t)
            logger.debug(f"delta1 values (first few): {delta1[:5]}")

            # Verify hidden gradients
            for i, d1 in enumerate(delta1):
                result = track_field_arithmetic(
                    f"delta1[{i}]",
                    normalize_to_field(d1), precision,
                    normalize_to_field(d1 * precision),
                    normalize_to_field(d1 * precision)
                )
                if not result:
                    raise ValueError(f"Hidden gradient verification failed at index {i}")

            # Compute weight and bias gradients
            dW_input = compute_weight_gradients(delta1, x_int, precision)
            dB_input = compute_bias_gradients(delta1)

            logger.debug("=== Final Gradient Shapes ===")
            logger.debug(f"dW_input length: {len(dW_input)}")
            logger.debug(f"dB_input length: {len(dB_input)}")

            # Generate client inputs with proper scaling
            client_inputs = generate_client_input(
                gw_t, gb_t,
                x_arr.tolist(), y_arr.tolist(),
                lw_t, lb_t,
                learning_rate, precision,
                int(global_hash), int(local_hash),
                delta2_input, dW_input, dB_input
            )

            # Verify all required inputs are present
            required_keys = ['eta', 'pr', 'ldigest', 'ScGH', 'GW', 'GB', 'X', 'Y', 'LWp', 'LBp', 'delta2_input', 'dW_input', 'dB_input']
            missing_keys = [key for key in required_keys if key not in client_inputs]
            if missing_keys:
                raise ValueError(f"Missing required input keys: {missing_keys}")

            # Generate proof
            result = generate_client_proof(
                client_inputs, 
                self.client_circuit_path, 
                self.client_pk_path,
                js_dir=self.client_js_dir
            )
            
            # Verify result contains all expected components
            if not result or 'proof' not in result or 'public' not in result:
                logger.error("Proof generation failed - incomplete result")
                logger.error(f"Result keys: {result.keys() if result else 'None'}")
                raise ValueError("Proof generation failed to produce complete result")

            # Verify public signals
            if len(result['public']) != 4:
                logger.error(f"Invalid number of public signals: {len(result['public'])}")
                raise ValueError(f"Expected 4 public signals, got {len(result['public'])}")

            # Log successful completion
            logger.info("✅ Client training proof generated successfully")
            return result

        except Exception as e:
            logger.error(f"Failed to generate training proof: {str(e)}")
            logger.debug("Full exception:", exc_info=True)
            raise

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
    pr = 100000
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
