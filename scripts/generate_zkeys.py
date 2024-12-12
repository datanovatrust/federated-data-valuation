#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path
from zkpy.ptau import PTau
from zkpy.circuit import Circuit, GROTH

def run_command(cmd, description):
    """Run a command and capture detailed output."""
    try:
        result = subprocess.run(cmd, 
                              capture_output=True, 
                              text=True,
                              check=False)
        if result.returncode != 0:
            print(f"Error during {description}:")
            print(f"Command: {' '.join(cmd)}")
            print(f"Return code: {result.returncode}")
            print("stdout:")
            print(result.stdout)
            print("stderr:")
            print(result.stderr)
            return False
        return True
    except Exception as e:
        print(f"Exception during {description}: {e}")
        return False

def verify_snarkjs():
    """Detailed verification of snarkjs installation."""
    try:
        # Check if snarkjs exists in PATH
        result = subprocess.run(['which', 'snarkjs'], 
                              capture_output=True, 
                              text=True)
        if result.returncode != 0:
            print("Debug: snarkjs not found in PATH")
            print(f"Current PATH: {os.environ.get('PATH', 'PATH not set')}")
            return False
        
        snarkjs_path = result.stdout.strip()
        print(f"Found snarkjs at: {snarkjs_path}")
        
        # Test snarkjs functionality by checking version string
        version_result = subprocess.run(['snarkjs', '--version'], 
                                      capture_output=True, 
                                      text=True)
        
        # Check if output contains version information
        if 'snarkjs@' in version_result.stdout:
            version = version_result.stdout.split('\n')[0].strip()
            print(f"snarkjs version detected: {version}")
            return True
            
        print("Could not detect snarkjs version in output")
        return False
        
    except Exception as e:
        print(f"Error checking snarkjs: {e}")
        return False

def compile_circuit(circuit_path):
    """Compile a circuit with detailed output."""
    cmd = [
        'circom',
        '--r1cs', '--wasm', '--sym',
        '--O2',
        circuit_path
    ]
    return run_command(cmd, f"compiling {circuit_path}")

def verify_circuit(circuit_path):
    """Verify that a circuit file is valid."""
    try:
        result = subprocess.run(['circom', '--help'],
                              capture_output=True,
                              text=True)
        if result.returncode != 0:
            print("Warning: circom verification failed")
            return False
            
        with open(circuit_path, 'r') as f:
            content = f.read()
            if not content.strip():
                print(f"Warning: Circuit file {circuit_path} is empty")
                return False
            if "template" not in content or "component main" not in content:
                print(f"Warning: Circuit file {circuit_path} may be missing required elements")
                return False
        return True
    except Exception as e:
        print(f"Error verifying circuit {circuit_path}: {e}")
        return False

def ensure_environment():
    """Verify all required environment variables, paths, and dependencies exist."""
    # Check for CIRCOM environment variable
    circom_path = os.getenv('CIRCOM')
    if not circom_path:
        raise EnvironmentError("CIRCOM environment variable not set. Please run setup_zkp.sh and source your shell.")
    
    print(f"Found CIRCOM at: {circom_path}")
    
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
        print(f"Found required file: {file}")

def clean_ptau_files():
    """Clean up any old .ptau files in the root directory."""
    for file in Path().glob("*.ptau"):
        try:
            file.unlink()
            print(f"Cleaned up old ptau file: {file}")
        except Exception as e:
            print(f"Warning: Could not remove {file}: {e}")

def setup_build_directory():
    """Set up build directory for circuit artifacts."""
    build_dir = Path("./build/circuits")
    build_dir.mkdir(parents=True, exist_ok=True)
    return build_dir

def main():
    try:
        print("Verifying environment...")
        ensure_environment()
        
        print("Setting up build directory...")
        build_dir = setup_build_directory()
        
        print("Cleaning up old ptau files...")
        clean_ptau_files()
        
        # Change to project root directory
        os.chdir(Path(__file__).parent.parent)
        
        # Prepare powers of tau
        ptau = PTau()
        print("Starting powers of tau...")
        ptau.start()
        print("Contributing to powers of tau...")
        ptau.contribute()
        ptau.beacon()
        ptau.prep_phase2()

        # Set up client circuit
        print("\nSetting up client circuit...")
        client_path = "./circuits/client.circom"
        if not verify_circuit(client_path):
            raise ValueError(f"Circuit validation failed for {client_path}")
            
        print("Compiling client circuit...")
        if not compile_circuit(client_path):
            raise ValueError(f"Circuit compilation failed for {client_path}")
            
        client_circuit = Circuit(client_path)
        try:
            print("Setting up client circuit with GROTH16...")
            setup_result = client_circuit.setup(GROTH, ptau)
            if not setup_result:
                raise ValueError("GROTH16 setup failed")
            client_circuit.export_vkey(str(build_dir / "client_vkey.json"))
            print("Client circuit setup completed.")
        except Exception as e:
            print(f"Detailed error processing client circuit:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            if hasattr(e, 'output'):
                print(f"Command output: {e.output}")
            raise

        # Set up aggregator circuit
        print("\nSetting up aggregator circuit...")
        aggregator_circuit = Circuit("./circuits/aggregator.circom")
        try:
            print("Compiling aggregator circuit...")
            aggregator_circuit.compile()
            print("Setting up aggregator circuit with GROTH16...")
            aggregator_circuit.setup(GROTH, ptau)
            aggregator_circuit.export_vkey(str(build_dir / "aggregator_vkey.json"))
            print("Aggregator circuit setup completed.")
        except Exception as e:
            print(f"Error processing aggregator circuit: {str(e)}")
            raise

        print("\nZKey generation completed successfully.")
        print(f"Verification keys stored in: {build_dir}")
        
    except FileNotFoundError as e:
        print(f"File Error: {e}")
        print("Please ensure all required circuit files are present.")
        sys.exit(1)
    except EnvironmentError as e:
        print(f"Environment Error: {e}")
        print("Please ensure setup_zkp.sh has been run and your shell has been sourced.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during compilation/setup: {str(e)}")
        print("Check that your circuits are valid and all dependencies are properly installed.")
        sys.exit(1)

if __name__ == "__main__":
    main()