# src/utils/zkp_utils.py

import json
import subprocess
import os
from zkpy.circuit import Circuit, GROTH

# If needed, implement MiMC hash in Python (matching circuit).
# Below is a placeholder. You must ensure this matches your circom code.
def mimc_hash(values, key=0, rounds=91):
    # This is a placeholder function. You must implement the exact MiMC 
    # permutation used in your circuit. Replace this stub with a correct MiMC.
    # For now, just sum values as a placeholder (not secure).
    # TODO: Implement actual MiMC as in your circom code.
    return sum(values) % (2**254)  # Mock

def generate_client_input(gw, gb, x, y, lwp, lbp, eta, pr, scgh, ldigest):
    # Prepare the input for the client circuit.
    # These must match exactly the input signals expected by client.circom
    # Arrays must be flattened if needed.
    client_input = {
        "GW": gw,       # e.g. list of integers
        "GB": gb,
        "X": x,
        "Y": y,
        "LWp": lwp,
        "LBp": lbp,
        "eta": eta,
        "pr": pr,
        "ScGH": scgh,
        "ldigest": ldigest
    }
    return client_input

def generate_aggregator_input(gw, gb, lwps, lbps, gwp, gbp, sclh, gdigest):
    aggregator_input = {
        "GW": gw,
        "GB": gb,
        "LWp": lwps,  # array of arrays if needed
        "LBp": lbps,
        "GWp": gwp,
        "GBp": gbp,
        "ScLH": sclh,
        "gdigest": gdigest
    }
    return aggregator_input

def generate_client_proof(client_inputs: dict, circuit_path: str, proving_key_path: str):
    # Write client_inputs to a JSON file
    with open("client_input.json", "w") as f:
        json.dump(client_inputs, f)

    client_circuit = Circuit(circuit_path)
    # client_circuit was compiled and set up during generate_zkeys
    # Here we just run witness & prove steps
    client_circuit.gen_witness("client_input.json")  # creates witness.wtns
    client_circuit.prove(GROTH)  # creates proof.json, public.json

    with open("proof.json") as pf:
        proof_data = json.load(pf)
    with open("public.json") as pubf:
        public_data = json.load(pubf)

    return proof_data, public_data

def generate_aggregator_proof(agg_inputs: dict, circuit_path: str, proving_key_path: str):
    with open("aggregator_input.json", "w") as f:
        json.dump(agg_inputs, f)

    aggregator_circuit = Circuit(circuit_path)
    aggregator_circuit.gen_witness("aggregator_input.json")
    aggregator_circuit.prove(GROTH)

    with open("proof.json") as pf:
        proof_data = json.load(pf)
    with open("public.json") as pubf:
        public_data = json.load(pubf)

    return proof_data, public_data
