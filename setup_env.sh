#!/bin/bash
# ============================================================
# Setup script for Fuel_Equilibrium conda environment
# ============================================================
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh
# ============================================================

set -e

ENV_NAME="Fuel_Equilibrium"

echo "============================================================"
echo "  Setting up conda environment: $ENV_NAME"
echo "============================================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH."
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing environment '$ENV_NAME'..."
    conda env remove -n "$ENV_NAME" -y
fi

# Create environment from yml file
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

echo ""
echo "============================================================"
echo "  Environment '$ENV_NAME' created successfully!"
echo "============================================================"
echo ""
echo "  To activate:"
echo "    conda activate $ENV_NAME"
echo ""
echo "  To run the program:"
echo "    python equilibrium.py"
echo ""
echo "  Or in batch mode:"
echo '    python equilibrium.py -r "2H2 + O2" -T 3000 -P "1 atm" -v'
echo ""
echo "  To deactivate:"
echo "    conda deactivate"
echo "============================================================"
