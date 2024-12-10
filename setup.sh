#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Colors and formatting
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Environment settings
ENV_NAME="fdv-env"
PYTHON_VERSION="3.11.8"

# Function for section headers
print_header() {
    printf "\n${BLUE}${BOLD}=== %s ===${NC}\n" "$1"
}

# Function for success messages
print_success() {
    printf "${GREEN}✓ %s${NC}\n" "$1"
}

# Function for progress messages
print_progress() {
    printf "${YELLOW}➜ %s${NC}\n" "$1"
}

# Function for error messages
print_error() {
    printf "${RED}✗ %s${NC}\n" "$1"
    exit 1
}

# Function to show spinner while a command is running
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

print_header "Federated Data Valuation Environment Setup"

# Check conda installation
print_progress "Checking conda installation..."
if ! command -v conda &> /dev/null; then
    print_error "conda not found. Please install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
fi
print_success "conda found"

# Create conda environment
print_progress "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y >/dev/null 2>&1
print_success "Environment created"

# Activate conda environment
print_progress "Activating conda environment..."
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    print_error "conda.sh not found. Conda activation might fail."
fi
conda activate "${ENV_NAME}"
print_success "Environment activated"

# Install dependencies
print_header "Installing Dependencies"

# Function to install packages quietly
install_packages() {
    local message=$1
    shift
    print_progress "$message"
    pip install --quiet "$@"
    print_success "Installation complete"
}

# Upgrade pip
install_packages "Upgrading pip..." --upgrade pip

# Install dependencies in stages
install_packages "Installing eth-brownie..." 'eth-brownie==1.20.6'
install_packages "Installing PyTorch..." 'torch>=2.2.2,<3.0.0' 'torchvision>=0.17.2,<0.18.0'

install_packages "Installing core ML packages..." \
    'transformers>=4.41.2,<4.42.0' \
    'numpy>=1.24.4,<1.25.0' \
    'pandas>=2.2.2,<2.3.0' \
    'matplotlib>=3.9.0,<3.10.0' \
    'pyyaml>=6.0.1,<6.1.0' \
    'pytest>=6.2.5,<6.3.0' \
    'tensorboard>=2.18.0,<2.19.0'

install_packages "Installing scientific computing packages..." \
    'scipy>=1.11.3,<1.12.0' \
    'scikit-learn>=1.6.0,<1.7.0' \
    'joblib>=1.4.2,<1.5.0' \
    'threadpoolctl>=3.5.0,<3.6.0' \
    'POT>=0.9.5,<0.10.0' \
    'pydantic>=2.6.0,<2.7.0' \
    'opt_einsum>=3.3.0,<3.4.0' \
    'tqdm>=4.66.2,<4.67.0' \
    'seaborn>=0.13.2,<0.14.0'

install_packages "Installing Web3 packages..." \
    'web3>=6.15.1,<6.16.0' \
    'eth-utils>=2.3.1,<2.4.0'

install_packages "Installing ML fine-tuning packages..." 'peft>=0.13.2,<0.14.0'
install_packages "Installing datasets..." 'datasets>=3.0.0,<3.2.0'

# Verify installation
print_header "Verification"
print_progress "Verifying core dependencies..."
if python -c "import torch; import transformers; import datasets; import tensorboard; import sklearn" >/dev/null 2>&1; then
    print_success "Core dependencies successfully installed"
else
    print_error "Verification failed. Please check the installation logs."
fi

# Print environment information
print_header "Environment Information"
printf "${BOLD}Python version:${NC} $(python --version)\n"
printf "${BOLD}Key packages:${NC}\n"
pip freeze | grep -E "torch|transformers|datasets|eth-brownie|tensorboard|scikit-learn" | sed 's/^/  /'

print_header "Setup Complete"
printf "${GREEN}${BOLD}Environment '${ENV_NAME}' is ready!${NC}\n"
printf "\nTo activate the environment, run:\n"
printf "${BOLD}conda activate ${ENV_NAME}${NC}\n\n"