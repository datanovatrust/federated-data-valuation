# scripts/generate_zkeys.py

import os
import sys
import subprocess
import time
from pathlib import Path

def log_step(emoji, message):
    """Print a step with emoji."""
    print(f"\n{emoji} {message}")

def log_progress(emoji, message):
    """Print a progress message with emoji."""
    print(f"  {emoji} {message}")

def run_command(cmd, description):
    """Run a command and capture output."""
    print(f"=== {description} ===")
    print("Command:", ' '.join(cmd))
    start_time = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True)

    elapsed = time.time() - start_time

    if result.stdout:
        print("STDOUT:", result.stdout.strip())
    if result.stderr:
        print("STDERR:", result.stderr.strip())
    if result.returncode != 0:
        print(f"❌ Error during {description}")
        sys.exit(1)
    else:
        print(f"✅ {description} completed in {elapsed:.2f}s")

def compile_circuit(circuit_path):
    """Compile a circuit to r1cs, wasm, and sym."""
    log_step("🔨", f"Compiling circuit {circuit_path}")
    cmd = [
        "circom", 
        "--r1cs", "--wasm", "--sym", 
        "--O2",
        circuit_path
    ]
    run_command(cmd, f"compile {circuit_path}")

def generate_zkey(circuit_name, ptau_file, build_dir):
    """Generate zkey and verification key for a given circuit."""
    r1cs_file = f"{circuit_name}.r1cs"
    zkey_file = f"{circuit_name}_0000.zkey"
    vkey_path = build_dir / f"{circuit_name}_vkey.json"

    # groth16 setup
    log_step("🔑", f"Setting up zkey for {circuit_name}")
    cmd = ["snarkjs", "groth16", "setup", r1cs_file, ptau_file, zkey_file]
    run_command(cmd, f"{circuit_name} zkey setup")

    # export verification key
    log_step("📤", f"Exporting verification key for {circuit_name}")
    cmd = ["snarkjs", "zkey", "export", "verificationkey", zkey_file, str(vkey_path)]
    run_command(cmd, f"{circuit_name} verification key export")

    return vkey_path

def main():
    # Paths and assumptions
    build_dir = Path("./build/circuits")
    build_dir.mkdir(parents=True, exist_ok=True)
    ptau_file = "final.ptau"

    if not os.path.exists(ptau_file):
        print("❌ final.ptau file not found. Please run the powers_of_tau_ceremony.py first.")
        sys.exit(1)

    # Circuits to process
    circuits = ["client", "aggregator"]
    
    # Recompile circuits
    for c in circuits:
        circuit_path = f"./circuits/{c}.circom"
        if not os.path.exists(circuit_path):
            print(f"❌ Circuit file {circuit_path} not found.")
            sys.exit(1)
        compile_circuit(circuit_path)

    # Generate zkeys and vkeys
    for c in circuits:
        vkey = generate_zkey(c, ptau_file, build_dir)
        log_progress("🔑", f"Verification key for {c} stored at {vkey}")

    print("\n✅ All circuits processed. ZKeys and verification keys generated successfully.")

if __name__ == "__main__":
    main()
