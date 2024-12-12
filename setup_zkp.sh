#!/usr/bin/env bash
set -euo pipefail

DEBUG="${DEBUG:-false}" # Set DEBUG=true before running to see more output
log() {
    if [ "$DEBUG" = "true" ]; then
        echo "[DEBUG] $*"
    fi
}

PROFILE_FILES=("$HOME/.profile" "$HOME/.bash_profile" "$HOME/.zshenv")
USER_GROUP=$(id -gn "$USER")
RC_FILE="$HOME/.bashrc"  # Always use bash for consistency

# Ensure .bashrc exists
touch "$RC_FILE"

echo "╔════════════════════════════════════════"
echo "║ Setting up ZKP Environment"
echo "╚════════════════════════════════════════"
echo "Using RC file: $RC_FILE"

echo -e "\n➜ Ensuring write permissions on shell profiles..."
for file in "${PROFILE_FILES[@]}" "$RC_FILE"; do
    if [ -f "$file" ] && [ ! -w "$file" ]; then
        echo "  Fixing permissions for $file..."
        sudo chown "$USER":"$USER_GROUP" "$file" >/dev/null 2>&1 || true
        sudo chmod u+w "$file" >/dev/null 2>&1 || true
    fi
done

# Clean up any existing PATH entries first
if [ -f "$RC_FILE" ]; then
    sed -i.bak '/export PATH.*npm.*bin/d' "$RC_FILE" || true
    sed -i.bak '/export PATH.*cargo.*bin/d' "$RC_FILE" || true
fi

# Fix npm permissions first with better error handling
echo -e "\n➜ Fixing npm permissions..."
for dir in "$HOME/.npm" "/usr/local/lib/node_modules" "/usr/local/bin"; do
    if [ -d "$dir" ] || mkdir -p "$dir" 2>/dev/null; then
        sudo chown -R "$USER":"$USER_GROUP" "$dir" 2>/dev/null || {
            log "Warning: Failed to fix permissions for $dir"
            true
        }
    fi
done

# Check Python version before proceeding
echo -e "\n➜ Checking Python environment..."
if ! command -v python3 &>/dev/null; then
    echo "Error: Python 3 not found. Please install Python 3.7 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")
if ! python3 -c "import sys; assert sys.version_info >= (3,7), 'Python version too low'" 2>/dev/null; then
    echo "Error: Python 3.7 or higher is required (found: $PYTHON_VERSION)"
    exit 1
fi

# Install Rust & Cargo if not present
if ! command -v rustup &>/dev/null; then
    echo -e "\n➜ Installing Rust and Cargo..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --quiet >/dev/null 2>&1 || {
        echo "Error: Failed to install Rust and Cargo."
        exit 1
    }
fi

# Source cargo environment if available
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
    log "Sourced $HOME/.cargo/env"
else
    echo "Warning: $HOME/.cargo/env not found. Cargo might not be on PATH immediately."
fi

# Add cargo bin to PATH if not already done
if ! grep -q 'export PATH="$HOME/.cargo/bin:$PATH"' "$RC_FILE" 2>/dev/null; then
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> "$RC_FILE"
    log "Appended Cargo bin path to $RC_FILE"
fi

# Ensure Node.js and npm are installed
if ! command -v npm &>/dev/null; then
    echo "npm not found. Please install Node.js and npm first (e.g., 'brew install node')."
    exit 1
fi

# Uninstall any existing snarkjs to ensure clean installation
echo -e "\n➜ Setting up snarkjs..."
npm uninstall -g snarkjs >/dev/null 2>&1 || true

# Clear npm cache with proper permissions
npm cache clean --force >/dev/null 2>&1 || {
    log "Warning: Failed to clean npm cache. Attempting to continue..."
}

# Install snarkjs if not present
if ! command -v snarkjs &>/dev/null; then
    # Try installing without sudo first
    if ! npm install -g snarkjs >/dev/null 2>&1; then
        echo "Attempting installation with sudo..."
        sudo npm install -g snarkjs >/dev/null 2>&1 || {
            echo "Error: Failed to install snarkjs. Please check your npm setup."
            exit 1
        }
    fi
    log "Installed snarkjs globally."
fi

# Attempt to find npm global bin directory with multiple fallbacks
NPM_BIN_DIR=$(npm bin -g 2>/dev/null || true)
log "npm bin -g returned: '$NPM_BIN_DIR'"

if [ -z "$NPM_BIN_DIR" ] || [ ! -d "$NPM_BIN_DIR" ]; then
    # Fallback 1: Try npm prefix
    NPM_PREFIX=$(npm config get prefix 2>/dev/null || true)
    if [ -n "$NPM_PREFIX" ]; then
        NPM_BIN_DIR="$NPM_PREFIX/bin"
        log "Derived NPM_BIN_DIR from prefix: '$NPM_BIN_DIR'"
    fi
fi

if [ -z "$NPM_BIN_DIR" ] || [ ! -d "$NPM_BIN_DIR" ]; then
    # Fallback 2: Try to deduce from snarkjs location
    SNARKJS_PATH=$(which snarkjs 2>/dev/null || true)
    log "which snarkjs returned: '$SNARKJS_PATH'"
    if [ -n "$SNARKJS_PATH" ]; then
        NPM_BIN_DIR=$(dirname "$SNARKJS_PATH")
        log "Derived NPM_BIN_DIR from which snarkjs: '$NPM_BIN_DIR'"
    fi
fi

if [ -n "$NPM_BIN_DIR" ] && [ -d "$NPM_BIN_DIR" ]; then
    # Ensure npm global bin is in PATH if not already
    if ! echo "$PATH" | grep -q "$NPM_BIN_DIR"; then
        echo "export PATH=\"$NPM_BIN_DIR:\$PATH\"" >> "$RC_FILE"
        log "Appended NPM bin path to $RC_FILE"
    fi
else
    echo "Warning: Unable to determine global npm bin directory. snarkjs might not be accessible without a full path."
fi

# Install circom using official method
echo -e "\n➜ Setting up circom..."
if ! command -v circom &>/dev/null; then
    TEMP_DIR=$(mktemp -d)
    echo "Cloning circom repository..."
    git clone https://github.com/iden3/circom.git "$TEMP_DIR" >/dev/null 2>&1 || {
        echo "Error: Failed to clone circom repository."
        rm -rf "$TEMP_DIR"
        exit 1
    }
    
    echo "Building circom..."
    (cd "$TEMP_DIR" && cargo build --release) >/dev/null 2>&1 || {
        echo "Error: Failed to build circom."
        rm -rf "$TEMP_DIR"
        exit 1
    }
    
    echo "Installing circom..."
    (cd "$TEMP_DIR" && cargo install --path .) >/dev/null 2>&1 || {
        echo "Error: Failed to install circom."
        rm -rf "$TEMP_DIR"
        exit 1
    }
    
    rm -rf "$TEMP_DIR"
    log "Installed circom via official method."
fi

# Export paths for immediate use
export PATH="$HOME/.cargo/bin:$PATH"
[ -n "$NPM_BIN_DIR" ] && export PATH="$NPM_BIN_DIR:$PATH"

# Verify circom and snarkjs before installing zkpy
echo "Checking ZKP dependencies..."
if ! command -v circom &>/dev/null || ! command -v snarkjs &>/dev/null; then
    echo "Error: circom and/or snarkjs not found. These are required for zkpy."
    exit 1
fi

# Install zkpy if not present
if ! pip show zkpy >/dev/null 2>&1; then
    echo "Installing zkpy Python package..."
    pip install zkpy >/dev/null 2>&1 || {
        echo "Error: Failed to install zkpy. Please check your Python environment."
        exit 1
    }
fi

# Verify installations with comprehensive checks
echo -e "\n╔════════════════════════════════════════"
echo "║ Verifying Installations"
echo "╚════════════════════════════════════════"

if command -v snarkjs >/dev/null 2>&1; then
    SNARKJS_PATH=$(which snarkjs)
    echo "snarkjs location: $SNARKJS_PATH"
    if [ -x "$SNARKJS_PATH" ]; then
        SNARKJS_VERSION=$(snarkjs --version 2>/dev/null | head -n 1 || echo 'Unable to get version')
        echo "snarkjs version: $SNARKJS_VERSION"
    else
        echo "Warning: snarkjs exists but may not be executable"
        ls -l "$SNARKJS_PATH" || true
    fi
else
    echo "Warning: snarkjs installation could not be verified"
fi

if command -v circom >/dev/null 2>&1; then
    CIRCOM_PATH=$(which circom)
    echo "circom location: $CIRCOM_PATH"
    if [ -x "$CIRCOM_PATH" ]; then
        CIRCOM_VERSION=$(circom --version 2>/dev/null || echo 'Unable to get version')
        echo "circom version: $CIRCOM_VERSION"
    else
        echo "Warning: circom exists but may not be executable"
        ls -l "$CIRCOM_PATH" || true
    fi
else
    echo "Warning: circom installation could not be verified"
fi

# After installing circom, add these lines:
CIRCOM_PATH="$HOME/.cargo/bin/circom"
# Remove any existing CIRCOM exports
sed -i.bak '/export CIRCOM=/d' "$RC_FILE" || true
# Add new CIRCOM export
echo "export CIRCOM=\"$CIRCOM_PATH\"" >> "$RC_FILE"
# Export for immediate use
export CIRCOM="$CIRCOM_PATH"

# Verify basic functionality
echo -e "\n╔════════════════════════════════════════"
echo "║ Installation Summary"
echo "╚════════════════════════════════════════"
echo "✓ snarkjs: $(command -v snarkjs >/dev/null 2>&1 && echo 'Installed' || echo 'Not found')"
echo "✓ circom: $(command -v circom >/dev/null 2>&1 && echo 'Installed' || echo 'Not found')"
echo "✓ zkpy: $(pip show zkpy >/dev/null 2>&1 && echo 'Installed' || echo 'Not found')"
echo "✓ PATH setup: $(echo $PATH | grep -q "$NPM_BIN_DIR" && echo 'Configured' || echo 'Warning: npm bin not in PATH')"
echo "✓ Permissions: $(test -x "$(which snarkjs 2>/dev/null)" && test -x "$(which circom 2>/dev/null)" && echo 'OK' || echo 'Check permissions')"

echo -e "\n╔════════════════════════════════════════"
echo "║ Setup Complete"
echo "╚════════════════════════════════════════"
echo "Please run:"
echo "  source $RC_FILE"
echo "Then verify installation with:"
echo "  snarkjs --version"
echo "  circom --version"
echo "  python -c 'import zkpy; print(\"zkpy imported successfully\")'"