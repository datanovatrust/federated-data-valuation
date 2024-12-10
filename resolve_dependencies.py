import re
import subprocess

input_file = "requirements.txt"
intermediate_file = "requirements.in"
output_file = "requirements_v2.txt"

# A function to relax exact version pins.
# For example, torch==2.2.2 becomes torch>=2.2.2,<3.0.0
def relax_version_pin(line):
    # Match patterns like `package==x.y.z`
    match = re.match(r"^([a-zA-Z0-9_\-]+)==([\d\.]+)$", line.strip())
    if match:
        package, version = match.groups()
        # Split the version by '.' and take the first major version part
        # We'll assume a simple next major version boundary (version+1)
        # This is a heuristic and may need manual tweaking.
        parts = version.split('.')
        if parts[0].isdigit():
            major = int(parts[0])
            next_major = major + 1
            # Use a <next_major.0.0 pin as upper bound
            relaxed_line = f"{package}>={version},<{next_major}.0.0"
        else:
            # If we can't parse major, fallback to no upper bound
            relaxed_line = f"{package}>={version}"
        return relaxed_line
    return line

# Read requirements_v2.txt
with open(input_file, "r") as f:
    lines = f.readlines()

relaxed_lines = []
for line in lines:
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    # Relax exact pins
    if "==" in line:
        line = relax_version_pin(line)
    # If line already has >= or other specifiers, leave it as is.
    relaxed_lines.append(line)

# Write to requirements.in
with open(intermediate_file, "w") as f:
    for l in relaxed_lines:
        f.write(l + "\n")

print("Running pip-compile to resolve dependencies...")

# Run pip-compile on the requirements.in file
# This will produce a fully resolved requirements_v3.txt if possible.
try:
    subprocess.run(["pip-compile", intermediate_file, "--output-file", output_file], check=True)
    print(f"Resolved dependencies written to {output_file}.")
    print("You can now install with:")
    print(f"pip install -r {output_file}")
except subprocess.CalledProcessError:
    print("pip-compile failed to find a compatible set of dependencies.")
    print("You may need to manually adjust the constraints in requirements.in.")
