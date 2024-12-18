#!/usr/bin/env python3

import os
import sys
import subprocess
import time
from pathlib import Path
from zkpy.ptau import PTau
from zkpy.circuit import Circuit, GROTH

def print_banner(text):
    """Print a stylized banner."""
    width = len(text) + 4
    print("╔" + "═" * width + "╗")
    print(f"║ {text} ║")
    print("╚" + "═" * width + "╝")

def log_step(emoji, message):
    """Print a step with emoji."""
    print(f"\n{emoji} {message}")

def log_progress(emoji, message):
    """Print a progress message with emoji."""
    print(f"  {emoji} {message}")

def run_command(cmd, description, show_progress=False):
    """Run a command and capture detailed output with optional progress monitoring."""
    print(f"DEBUG: Running command for {description}: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        start_time = time.time()
        
        # Real-time output processing
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output and show_progress:
                log_progress("📋", output.strip())

        returncode = process.wait()
        
        # Capture any remaining output
        stdout, stderr = process.communicate()
        
        if stdout:
            print(f"DEBUG: STDOUT for {description}: {stdout}")
        if stderr:
            print(f"DEBUG: STDERR for {description}: {stderr}")

        if returncode != 0:
            print(f"Error during {description}:")
            print(f"Command: {' '.join(cmd)}")
            print(f"Return code: {returncode}")
            return False

        elapsed = time.time() - start_time
        if elapsed > 5:  # Show timing for longer operations
            log_progress("⏱️", f"Completed in {elapsed:.1f} seconds")
        return True
    except Exception as e:
        print(f"Exception during {description}: {e}")
        return False

def verify_snarkjs():
    """Detailed verification of snarkjs installation."""
    log_step("🔍", "Checking snarkjs installation")
    try:
        # Check if snarkjs exists in PATH
        result = subprocess.run(['which', 'snarkjs'], 
                                capture_output=True, 
                                text=True)
        if result.returncode != 0:
            log_progress("❌", "snarkjs not found in PATH")
            print(f"Current PATH: {os.environ.get('PATH', 'PATH not set')}")
            return False
        
        snarkjs_path = result.stdout.strip()
        log_progress("📍", f"Found snarkjs at: {snarkjs_path}")
        
        # Test snarkjs functionality by checking version string
        version_result = subprocess.run(['snarkjs', '--version'], 
                                        capture_output=True, 
                                        text=True)
        
        if version_result.stdout.strip():
            print(f"DEBUG: snarkjs version output: {version_result.stdout}")
        else:
            print("DEBUG: snarkjs produced no version output")

        # Check if output contains version information
        if 'snarkjs@' in version_result.stdout:
            version = version_result.stdout.split('\n')[0].strip()
            log_progress("✨", f"Detected {version}")
            return True
            
        log_progress("❌", "Could not detect snarkjs version")
        return False
        
    except Exception as e:
        log_progress("💥", f"Error: {e}")
        return False

def compile_circuit(circuit_path):
    """Compile a circuit with detailed output."""
    log_step("🔨", f"Compiling circuit {circuit_path}")
    cmd = [
        'circom',
        '--r1cs', '--wasm', '--sym',
        '--O2',
        circuit_path
    ]
    return run_command(cmd, f"compiling {circuit_path}")

def verify_circuit(circuit_path):
    """Verify that a circuit file is valid."""
    log_step("🔍", f"Verifying circuit file {circuit_path}")
    try:
        result = subprocess.run(['circom', '--help'],
                                capture_output=True,
                                text=True)
        if result.returncode != 0:
            log_progress("⚠️", "circom verification failed")
            return False
            
        with open(circuit_path, 'r') as f:
            content = f.read()
            if not content.strip():
                log_progress("❌", f"Circuit file {circuit_path} is empty")
                return False
            if "template" not in content or "component main" not in content:
                log_progress("❌", f"Missing required elements in {circuit_path}")
                return False
        log_progress("✅", f"Circuit {circuit_path} is valid")
        return True
    except Exception as e:
        log_progress("💥", f"Error: {e}")
        return False

def ensure_environment():
    """Verify all required environment variables, paths, and dependencies exist."""
    log_step("🔧", "Checking environment setup")
    circom_path = os.getenv('CIRCOM')
    if not circom_path:
        raise EnvironmentError("CIRCOM environment variable not set. Please run setup_zkp.sh and source your shell.")
    
    log_progress("✅", f"Found CIRCOM at: {circom_path}")
    
    # Verify snarkjs
    if not verify_snarkjs():
        raise EnvironmentError("snarkjs not found or not working. Please check your installation.")
    
    # Check for required files
    circuits_dir = Path("./circuits")
    if not circuits_dir.exists():
        raise FileNotFoundError("Circuits directory not found at ./circuits")
    
    required_files = [
        circuits_dir / "client.circom",
        circuits_dir / "aggregator.circom",
        circuits_dir / "mimc_hash.circom"
    ]
    
    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(f"Required circuit file not found: {file}")
        log_progress("📄", f"Found: {file}")

def clean_ptau_files():
    """Clean up any old .ptau files in the root directory."""
    log_step("🧹", "Cleaning up old ptau files")
    for file in Path().glob("*.ptau"):
        try:
            file.unlink()
            log_progress("🗑️", f"Removed: {file}")
        except Exception as e:
            log_progress("⚠️", f"Could not remove {file}: {e}")

def setup_build_directory():
    """Set up build directory for circuit artifacts."""
    log_step("📁", "Setting up build directory")
    build_dir = Path("./build/circuits")
    build_dir.mkdir(parents=True, exist_ok=True)
    log_progress("✅", f"Build directory ready: {build_dir}")
    return build_dir

def setup_ptau():
    """Set up Powers of Tau with detailed logging and progress monitoring."""
    print_banner("🚀 Powers of Tau Setup")

    ptau_file = "pot_0000.ptau"
    final_ptau = "final.ptau"
    
    # Check if final.ptau already exists
    if os.path.exists(final_ptau):
        log_progress("ℹ️", f"{final_ptau} already exists. Skipping PTau generation steps.")
        return final_ptau

    # If we don't have final.ptau, run the full ceremony
    log_step("1️⃣", "Starting new powers of tau ceremony")
    cmd = ["snarkjs", "powersoftau", "new", "bn128", "18", ptau_file]
    if not run_command(cmd, "starting new powers of tau", show_progress=True):
        raise RuntimeError("Failed to start powers of tau ceremony")

    log_step("2️⃣", "Contributing to ceremony")
    contrib_ptau = "pot_0001.ptau"
    cmd = ["snarkjs", "powersoftau", "contribute", ptau_file, contrib_ptau,
           "--name=\"First contribution\"", "--entropy=\"random\""]
    if not run_command(cmd, "contributing to ptau", show_progress=True):
        raise RuntimeError("Failed to contribute to ceremony")

    log_step("3️⃣", "Adding random beacon")
    beacon_ptau = "pot_beacon.ptau"
    cmd = ["snarkjs", "powersoftau", "beacon", contrib_ptau, beacon_ptau,
           "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f", "10"]
    if not run_command(cmd, "adding beacon", show_progress=True):
        raise RuntimeError("Failed to add beacon")

    log_step("4️⃣", "Preparing phase 2")
    log_progress("⏳", "This step may take several minutes...")
    cmd = ["snarkjs", "powersoftau", "prepare", "phase2", beacon_ptau, final_ptau]
    if not run_command(cmd, "preparing phase 2", show_progress=True):
        raise RuntimeError("Failed to prepare phase 2")

    if not os.path.exists(final_ptau):
        raise FileNotFoundError(f"PTau file not found at expected location: {final_ptau}")
    log_progress("✨", f"PTau file created: {final_ptau}")
    
    return final_ptau

def process_circuit(circuit_name, ptau_file, build_dir):
    """Process a single circuit with detailed progress tracking."""
    print_banner(f"🔄 Processing {circuit_name} circuit")
    
    circuit_path = f"./circuits/{circuit_name}.circom"
    if not verify_circuit(circuit_path):
        raise ValueError(f"Circuit validation failed: {circuit_path}")

    log_step("📜", "Compiling circuit")
    if not compile_circuit(circuit_path):
        raise ValueError(f"Compilation failed: {circuit_path}")

    r1cs_file = f"./{circuit_name}.r1cs"
    zkey_file = f"./{circuit_name}_0000.zkey"
    vkey_path = build_dir / f"{circuit_name}_vkey.json"

    log_step("🔑", "Generating zkey")
    cmd = ["snarkjs", "groth16", "setup", r1cs_file, ptau_file, zkey_file]
    if not run_command(cmd, f"{circuit_name} zkey setup", show_progress=True):
        raise ValueError(f"Failed to setup {circuit_name} zkey")

    log_step("📤", "Exporting verification key")
    cmd = ["snarkjs", "zkey", "export", "verificationkey", zkey_file, str(vkey_path)]
    if not run_command(cmd, f"{circuit_name} vkey export"):
        raise ValueError(f"Failed to export {circuit_name} verification key")

    log_progress("✅", f"Circuit processing complete: {circuit_name}")
    return vkey_path

def main():
    try:
        print("\n" + "=" * 80)
        print_banner("🛠️  ZKey Generator Starting")
        print("=" * 80 + "\n")

        log_step("🔍", "Verifying environment")
        ensure_environment()
        
        log_step("📁", "Setting up directories")
        build_dir = setup_build_directory()
        
        log_step("🧹", "Cleaning old files")
        clean_ptau_files()
        
        # Change to project root directory
        os.chdir(Path(__file__).parent.parent)
        
        # Setup PTau
        ptau_file = setup_ptau()

        # Process circuits
        client_vkey = process_circuit("client", ptau_file, build_dir)
        aggregator_vkey = process_circuit("aggregator", ptau_file, build_dir)

        print("\n" + "=" * 80)
        print_banner("🎉 ZKey Generation Complete! 🎉")
        log_progress("📦", f"Verification keys stored in: {build_dir}")
        log_progress("🔑", f"Client key: {client_vkey}")
        log_progress("🔑", f"Aggregator key: {aggregator_vkey}")
        print("=" * 80 + "\n")

    except Exception as e:
        print("\n" + "=" * 80)
        print_banner("❌ Error Occurred")
        print(f"💥 {str(e)}")
        print("=" * 80 + "\n")
        sys.exit(1)

if __name__ == "__main__":
    main()