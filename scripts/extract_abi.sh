# scripts/extract_abi.sh

# Exit on error
set -e

# Check if jq is installed
if ! command -v jq &> /dev/null
then
    echo "jq could not be found. Please install jq and rerun the script."
    exit 1
fi

# Paths (adjust if your contract name or paths differ)
BUILD_CONTRACTS_PATH="build/contracts/FLRegistry.json"
OUTPUT_ABI_PATH="src/config/FLRegistry_abi.json"

# Check if the build contracts file exists
if [ ! -f "$BUILD_CONTRACTS_PATH" ]; then
    echo "Error: $BUILD_CONTRACTS_PATH does not exist. Make sure you compiled the contract with Brownie."
    exit 1
fi

# Extract the 'abi' field from the contract JSON and write to the ABI file
jq '.abi' "$BUILD_CONTRACTS_PATH" > "$OUTPUT_ABI_PATH"

echo "✅ Extracted ABI from $BUILD_CONTRACTS_PATH to $OUTPUT_ABI_PATH successfully."
