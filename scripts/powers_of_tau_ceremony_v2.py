#!/usr/bin/env python3

"""
scripts/powers_of_tau_ceremony_v2.py

A minimal script to run the Groth16 Powers of Tau setup
(Phase 1 and Phase 2) for the v2 circuits. It:

1. Checks if final.ptau already exists.
   - If yes, it skips the ceremony, assuming it's already generated.
2. Otherwise, runs:
   - snarkjs powersoftau new bn128 <ptau_exponent> pot_0000.ptau
   - snarkjs powersoftau contribute pot_0000.ptau pot_0001.ptau ...
   - snarkjs powersoftau beacon ...
   - snarkjs powersoftau prepare phase2 pot_beacon.ptau final.ptau
3. Outputs final.ptau in the working directory.

Typical usage:
    python scripts/powers_of_tau_ceremony_v2.py
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(cmd, desc=""):
    logger.info(f"=== {desc} ===")
    logger.info(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Error during {desc}")
        logger.error(f"stdout: {result.stdout}")
        logger.error(f"stderr: {result.stderr}")
        sys.exit(1)
    logger.info(result.stdout)

def main():
    final_ptau = "final.ptau"
    if os.path.exists(final_ptau):
        logger.info(f"{final_ptau} already exists. Skipping ceremony.")
        return

    # Phase 1
    logger.info("=== Starting Powers of Tau Ceremony for v2 circuits ===")
    ptau_0 = "pot_0000.ptau"
    cmd_new = ["snarkjs", "powersoftau", "new", "bn128", "18", ptau_0]
    run_command(cmd_new, desc="Powers of Tau new")

    # Phase 1 contribution
    ptau_1 = "pot_0001.ptau"
    cmd_contrib = [
        "snarkjs", "powersoftau", "contribute", ptau_0, ptau_1,
        "--name=First_Contribution_v2", "--entropy=random_v2"
    ]
    run_command(cmd_contrib, desc="Contribute to Powers of Tau")

    # Add beacon
    ptau_beacon = "pot_beacon.ptau"
    cmd_beacon = [
        "snarkjs", "powersoftau", "beacon",
        ptau_1, ptau_beacon,
        "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f", "10", "--name=BeaconStep"
    ]
    run_command(cmd_beacon, desc="Beacon phase")

    # Prepare phase 2
    cmd_phase2 = ["snarkjs", "powersoftau", "prepare", "phase2", ptau_beacon, final_ptau]
    run_command(cmd_phase2, desc="Prepare Phase2")

    logger.info("=== Powers of Tau v2 ceremony completed successfully ===")

if __name__ == "__main__":
    main()
