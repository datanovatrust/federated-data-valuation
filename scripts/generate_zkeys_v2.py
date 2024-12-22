#!/usr/bin/env python3

"""
scripts/generate_zkeys_v2.py

Script to:
1. Compile the v2 circuits (client_v2.circom, aggregator_v2.circom)
   into R1CS, WASM, and SYM files.
2. Run Groth16 setup using final.ptau to generate .zkey files.
3. Export verification keys to JSON for on-chain or local verification usage.

Usage:
    python scripts/generate_zkeys_v2.py

Requirements:
    - final.ptau must exist (created by powers_of_tau_ceremony_v2.py)
    - circom, snarkjs must be installed and in PATH
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(cmd, desc=""):
    logger.info(f"=== {desc} ===")
    logger.info(f"Command: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        logger.error(f"Error during {desc}")
        logger.error(f"stdout: {result.stdout}")
        logger.error(f"stderr: {result.stderr}")
        sys.exit(1)

    logger.info(f"Completed in {elapsed:.2f} seconds")
    if result.stdout:
        logger.info(f"STDOUT: {result.stdout.strip()}")
    if result.stderr:
        logger.info(f"STDERR: {result.stderr.strip()}")

def compile_circuit(circuit_path: Path):
    if not circuit_path.exists():
        logger.error(f"Circuit file not found: {circuit_path}")
        sys.exit(1)

    cmd = [
        "circom",
        str(circuit_path),
        "--r1cs",
        "--wasm",
        "--sym",
        "--O2"
    ]
    desc = f"Compile {circuit_path.name}"
    run_command(cmd, desc)

def main():
    build_dir = Path("./build/circuits_v2")
    build_dir.mkdir(parents=True, exist_ok=True)

    final_ptau = "final.ptau"
    if not Path(final_ptau).exists():
        logger.error(f"{final_ptau} not found. Please run powers_of_tau_ceremony_v2.py first.")
        sys.exit(1)

    # Circuits
    circuits = [
        "client_v2.circom",
        "aggregator_v2.circom"
    ]

    # 1) Compile each circuit
    for c in circuits:
        cpath = Path("./circuits") / c
        compile_circuit(cpath)

    # 2) Generate zkeys for each circuit
    for c in circuits:
        circuit_name = c.replace(".circom", "")
        r1cs_file = f"{circuit_name}.r1cs"
        zkey_file = f"{circuit_name}_0000.zkey"
        vkey_file = build_dir / f"{circuit_name}_vkey.json"

        # groth16 setup
        desc_setup = f"Groth16 Setup for {circuit_name}"
        cmd_setup = [
            "snarkjs", "groth16", "setup",
            r1cs_file,
            final_ptau,
            zkey_file
        ]
        run_command(cmd_setup, desc_setup)

        # export vkey
        desc_vkey = f"Export Verification Key for {circuit_name}"
        cmd_vkey = [
            "snarkjs", "zkey", "export", "verificationkey",
            zkey_file,
            str(vkey_file)
        ]
        run_command(cmd_vkey, desc_vkey)

        logger.info(f"Verification key saved to {vkey_file}")

    logger.info("=== All v2 circuits processed. ZKeys and verification keys generated. ===")

if __name__ == "__main__":
    main()
