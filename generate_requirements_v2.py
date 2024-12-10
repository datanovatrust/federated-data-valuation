import subprocess

packages = [
    "torch",
    "torchvision",
    "transformers",
    "datasets",
    "numpy",
    "pandas",
    "matplotlib",
    "jupyter",
    "PySyft",
    "pyyaml",
    "pytest",
    "syft",
    "scipy",
    "POT",
    "prv-accountant",
    "scipy",       # duplicate
    "pydantic",
    "opt_einsum",
    "ml-swissknife",
    "fairscale",
    "tqdm",
    "deepspeed",
    "jupyter",     # duplicate
    "jupyterlab",
    "eth-brownie",
    "web3"
]

# Remove duplicates while preserving order
seen = set()
unique_packages = []
for pkg in packages:
    if pkg not in seen:
        unique_packages.append(pkg)
        seen.add(pkg)

requirements = []

for pkg in unique_packages:
    try:
        # Run pip show and capture output
        result = subprocess.run(["pip", "show", pkg], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            # If pip show returns non-zero, package might not be installed
            # You can print a warning or continue
            print(f"Warning: Package '{pkg}' not found. Skipping.")
            continue
        
        # Parse the version line
        lines = result.stdout.splitlines()
        version_line = next((line for line in lines if line.startswith("Version:")), None)
        
        if version_line:
            version = version_line.split(":", 1)[1].strip()
            requirements.append(f"{pkg}=={version}")
        else:
            print(f"Warning: Could not find version for package '{pkg}'. Skipping.")
            
    except Exception as e:
        print(f"Error processing package '{pkg}': {e}")

# Write to requirements_v2.txt
with open("requirements_v2.txt", "w") as f:
    for req in requirements:
        f.write(req + "\n")

print("requirements_v2.txt has been successfully created.")
