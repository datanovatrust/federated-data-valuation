# scripts/generate_zkeys.py

import os
from zkpy.ptau import PTau
from zkpy.circuit import Circuit, GROTH

# Ensure the circuits directory and circom files exist.
# Ensure that circom is in PATH or set CIRCOM environment variable.

def main():
    # Prepare powers of tau
    ptau = PTau()
    print("Starting powers of tau...")
    ptau.start()        # creates something like pot.ptau
    print("Contributing to powers of tau...")
    ptau.contribute()
    ptau.beacon()
    ptau.prep_phase2()  # now we have a ptau file ready for groth16 setup

    # Set up client circuit
    print("Setting up client circuit...")
    client_circuit = Circuit("./circuits/client.circom")
    client_circuit.compile()
    client_circuit.setup(GROTH, ptau)
    client_circuit.export_vkey("client_vkey.json")
    # This might produce a client.zkey file internally.

    # Set up aggregator circuit
    print("Setting up aggregator circuit...")
    aggregator_circuit = Circuit("./circuits/aggregator.circom")
    aggregator_circuit.compile()
    aggregator_circuit.setup(GROTH, ptau)
    aggregator_circuit.export_vkey("aggregator_vkey.json")

    print("ZKey generation completed.")

if __name__ == "__main__":
    main()
